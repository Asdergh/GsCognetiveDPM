import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import (Optional, Tuple, Callable, Dict)
from einops.layers.torch import Rearrange
from ..utils.tensors import (seq2spatial, spatial2seq)



_ACTIVATIONS_ = {
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "softmax": nn.Softmax,
    "gelu": nn.GELU,
    "none": nn.Identity
}
get_activation = lambda act_type: (
    _ACTIVATIONS_["none"]() if act_type is None else
    _ACTIVATIONS_["none"]() if act_type.lower() not in _ACTIVATIONS_ else
    _ACTIVATIONS_[act_type](dim=-1) if act_type.lower() == "softmax" else 
    _ACTIVATIONS_[act_type]() 
)

class BasicLinear(nn.Module):
    def __init__(
        self,
        in_features: int, 
        out_features: int, 
        activation: str, 
        norm: Optional[bool]=True
    ):
        super().__init__()
        self.base_ = nn.Sequential(
            nn.Linear(in_features, out_features),
            get_activation(activation),
            (nn.Identity() if not norm else nn.LayerNorm(out_features))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_(x)

class BasicConv(nn.Module):
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        activation: str, 
        norm: Optional[bool]=True, 
        kernel_size: Optional[Tuple[int, int]]=(3, 3), 
        stride: Optional[int]=1, 
        padding: Optional[int]=1, 
        output_padding: Optional[int]=1,
        mode: Optional[str]="down", 
        sampler: Optional[str]="conv", 
        scale_factor: Optional[str]=2
    ):
    
        super().__init__()
        if sampler == "default":
            sampler_x = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor),
                nn.Conv2d(
                    in_channels=in_features,
                    out_channels=out_features, 
                    kernel_size=(3, 3), 
                    stride=1, 
                    padding=1
                )
            )
        
        elif sampler == "conv":
            if mode == "up":
                sampler_x = nn.ConvTranspose2d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding
                )
            if mode == "down":
                sampler_x = nn.Conv2d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )

        self.base_ = nn.Sequential(
            sampler_x,
            (nn.Identity() if not norm else nn.BatchNorm2d(out_features)),
            get_activation(activation)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_(x)


class Mlp(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int]=None,
        hiden_features: Optional[int]=None,
        activation_fn: Optional[str]="relu"
    ) -> None:
        
        super().__init__()
        hiden_features = (
            hiden_features 
            if hiden_features is not None
            else in_features
        )
        out_features = (
            out_features
            if out_features is not None
            else hiden_features
        )
        self.dense_ = nn.Sequential(
            nn.Linear(in_features, hiden_features),
            nn.LayerNorm(hiden_features),
            get_activation(activation_fn),
            nn.Linear(hiden_features, out_features)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       return self.dense_(x)


class ChannelWiseCrossAttention(nn.Module):

    def __init__(
        self,
        in_features: int,
        embedding_features: Optional[int]=None,
        out_features: Optional[int]=128,
        att_cross_mode: Optional[bool]=True,
        agr_activation: Optional[str]="relu",
        film_mode: Optional[bool]=False,
        film_activation: Optional[str]="sigmoid"
    ) -> None:
        
        super().__init__()
        out_features = (in_features if out_features is None else out_features)
        self.cross_mode = att_cross_mode
        self.film_ag = film_mode
        self.d = torch.sqrt(torch.tensor(out_features))

        self.split_forward_ = lambda x, split_part: torch.unbind(
            x.view(x.size(0), split_part, x.size(-1) // split_part),
            dim=1
        )
        self.spatial_agregation_ = BasicConv(in_features, out_features, agr_activation)
        if film_mode :
            self.filtration_ = BasicLinear(
                in_features=out_features, 
                out_features=out_features * 2, 
                activation=film_activation,
                norm=False
            )
        if att_cross_mode:
            self.qv_agregation_ = BasicLinear(
                in_features=in_features, 
                out_features=out_features * 2, 
                activation=agr_activation,
                norm=True
            )
            self.k_agregation_ = BasicLinear(
                in_features=embedding_features, 
                out_features=out_features, 
                activation=agr_activation,
                norm=True
            )
        
        else:
            self.qkv_agregation_ = BasicLinear(
                in_features=in_features, 
                out_features=out_features * 3, 
                activation=agr_activation,
                norm=True
            )
            
    
    def forward(self, x: torch.Tensor, tokens: Optional[torch.Tensor]=None) -> None:

        x = F.avg_pool2d(x, kernel_size=(x.size(-1), x.size(-2))).squeeze()
        if self.cross_mode:
            assert (tokens is not None), ("tokens are required in cross mode")
            x = self.qv_agregation_(x)
            (q, v) = self.split_forward_(x, 2)
            k = self.k_agregation_(tokens)
        
        else:
            x = self.qkv_agregation_(x)
            (q, k, v) = self.split_forward_(x, 3)
        
        qk = torch.sigmoid((q @ k.T) / self.d)
        x = qk @ v
        if self.film_ag:
            (scale, shift) = self.split_forward_(self.filtration_(x), 2)
            x = scale * x + shift
        
        return x
            
            
            
            
class InteractionAttention(nn.Module):
    
    def __init__(
        self, 
        input_features: int,
        patch_n_pr: Tuple[int, int],
        embed_features: Optional[int]=None,
        first_dim: Optional[int]=None,
        hiden_features: Optional[int]=None,
        output_features: Optional[int]=None,
        pooling_size: Optional[int]=3,
        format: Optional[str]="spatial", # [spatial, sequence]
        mode: Optional[str]="self", # [self, cross]
        latent_first_activation: Optional[str]="relu",
        latent_last_activation: Optional[str]="relu",
    ) -> None:
        
        super().__init__()
        self.mode = mode
        self.format = format
        assert (self.format in ["spatial", "sequence"])

        self.ps = pooling_size
        self.N, self.d = patch_n_pr
        self.L = self.d // self.ps

        self.ltf_d = (input_features if first_dim is None else first_dim)
        self.lti_d = (self.ltf_d if hiden_features is None else hiden_features)
        self.lto_d = (self.lti_d if output_features is None else output_features)

        if mode == "self":
            self.in_projection_ = nn.Sequential(
                nn.Linear(input_features, self.ltf_d * 3),
                get_activation(latent_first_activation)
            )
        
        else:
            assert (embed_features is not None), ("""
            in cross attention mode you must specify 
            embed_features as one of input args
            """)
            self.in_projection_ = nn.Sequential(
                nn.Linear(input_features, self.ltf_d * 2),
                get_activation(latent_first_activation)
            )
            self.v_projection_ = nn.Sequential(
                nn.Linear(embed_features, self.ltf_d),
                get_activation(latent_first_activation)
            )
        
        self.latent_inter_ = nn.Sequential(
            nn.Linear(self.ltf_d, self.lti_d),
            nn.Softmax(dim=-1)
        )
        self.latent_last_ = nn.Sequential(
            nn.Linear(self.lti_d, self.lto_d),
            get_activation(latent_last_activation)
        )
    
    def _project(self, x: torch.Tensor, N: int, d: int, layer: nn.Module, split_part: int=None) -> torch.Tensor:
        x = spatial2seq(x, input_format="BCWH", output_format="BNC")
        x = layer(x)
        x = seq2spatial(x, input_format="BNC", output_format="BCWH", patches_n=(N, d))
        if split_part is not None:
            x = x.view(x.size(0), split_part, x.size(1) // split_part, N, d)
            x = x.unbind(dim=1)
        return x
    
    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        
        C = x.size(1)
        x = x.transpose(-1, -2)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = F.avg_pool1d(x, self.ps)
        x = torch.unflatten(x, dim=1, sizes=(C, self.d))
        x = x.transpose(-1, -2)
        return x

    def forward(self, x: torch.Tensor, V: Optional[torch.Tensor]=None) -> torch.Tensor:
        if len(x.size()) == 3:
            x = seq2spatial(
                x=x, 
                input_format="BNC", 
                output_format="BCWH", 
                patches_n=(self.N, self.d)
            )

        if self.mode == "self":
            Q, K, V = self._project(
                x=x, 
                N=self.N, 
                d=self.d, 
                layer=self.in_projection_,
                split_part=3
            )
        elif self.mode == "cross":
            if V is not None:
                Q, K = self._project(
                    x=x, 
                    N=self.N, 
                    d=self.d, 
                    layer=self.in_projection_,
                    split_part=2
                )
                V = self._project(V, self.N, self.d, self.v_projection_)
            else:
                Q, K, V = self._project(
                    x=x, 
                    N=self.N, 
                    d=self.d, 
                    layer=self.in_projection_,
                    split_part=3
                )
                
        
        q = self._pool(Q)
        k = self._pool(K)

        Qk = Q @ k.transpose(-1, -2)
        Qk = self._project(Qk, self.N, self.L, self.latent_inter_)
        Qk = self._project(Qk, self.N, self.L, self.latent_last_)

        qK = q @ K.transpose(-1, -2)
        qK = self._project(qK, self.L, self.N, self.latent_inter_)
        qK = self._project(qK, self.L, self.N, self.latent_last_)

        V = self._project(V, self.N, self.d, self.latent_inter_)
        V = self._project(V, self.N, self.d, self.latent_last_)

        o = (Qk @ (qK @ V))
        return (
            o if self.format == "spatial" 
            else spatial2seq(
                x=o, 
                input_format="BCWH", 
                output_format="BNC"
            )
        )
        
        
class CrossAttention(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        c_dim: int,
        out_dim: Optional[int]=None,
        activation_fn: Optional[str]="relu"
    ) -> None:

        super().__init__()
        out_dim = (out_dim if out_dim is not None else latent_dim)
        
        self.d = torch.tensor(latent_dim)
        self.q_fn_ = Mlp(latent_dim, activation_fn=activation_fn)
        self.k_fn_ = Mlp(c_dim, latent_dim, activation_fn=activation_fn)
        self.v_fn_ = Mlp(c_dim, latent_dim, activation_fn=activation_fn)
        self.out_fn_ = nn.Sequential(
            nn.Linear(latent_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self._activations = None

    @property
    def activations(self) -> Dict[str, torch.Tensor]:
        if self._activations is not None:
            return self._activations
        else:
            return False

    
    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        
        B, N = z.size()[:-1]
        q = self.q_fn_(z)
        k = self.k_fn_(y).view(B, 1, int(self.d.item())).repeat(1, N, 1)
        v = self.v_fn_(y).view(B, 1, int(self.d.item())).repeat(1, N, 1)
        
        att_scores = (q @ k.transpose(-2, -1)) / torch.sqrt(self.d)
        att_weights = F.softmax(att_scores, dim=-1)
        attention = self.out_fn_(att_weights @ v)
        self._activations = att_weights

        return attention

class Block(nn.Module):

    def __init__(
        self,
        in_features: int,
        hiden_features: Optional[int]=None,
        out_features: Optional[int]=None,
        activation_fn: Optional[str]="softmax",
        apply_film: Optional[bool]=True,
        film_activation_fn: Optional[str]="sigmoid",
        depth_level: Optional[int]=3
    ) -> None:
        
        super().__init__()
        self.afilm = apply_film
        self.d_max = depth_level
        self.hiden_features = (
            hiden_features 
            if hiden_features is not None
            else in_features
        )
        self.out_features = (
            out_features
            if out_features is not None
            else hiden_features
        )
        self.weights_fn = lambda dt: F.softmax(torch.exp(torch.tensor(dt * 0.1)), dim=-1)
        self.blocks_ = nn.ModuleList([
            Mlp(
                in_features=(in_features if idx == 0 else self.hiden_features),
                hiden_features=hiden_features,
                out_features=(out_features if idx == (depth_level - 1) else self.hiden_features),
                activation_fn=activation_fn
            ) for idx in range(self.d_max)
        ])
        
        if self.afilm:
            self.film_h_ = nn.Sequential(
                nn.Linear(self.hiden_features, self.hiden_features * 2),
                get_activation(film_activation_fn)
            )
            self.film_o_ = nn.Sequential(
                nn.Linear(self.out_features, self.out_features * 3),
                get_activation(film_activation_fn)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for dt, block in enumerate(self.blocks_):
            x = block(x)
            if dt != self.d_max - 1:
                if self.afilm:
                    film = self.film_h_(x)
                    film = film.view(film.size(0), film.size(1), 2, self.hiden_features)
                    scale_h, shift_h = film.unbind(dim=-2)
                    x = scale_h * x + shift_h
            else:
                if self.afilm:
                    film = self.film_o_(x)
                    film = film.view(film.size(0), film.size(1), 3, self.out_features)
                    x, scale_o, shift_o = film.unbind(dim=-2)
                    x = scale_o * x + shift_o
            x = self.weights_fn(dt) * x

        return x
    

class ResidualConv(nn.Module):

    def __init__(
        self, features: int, 
        pj_features: Optional[int]=None,
        normalize: Optional[bool]=False,
        activation: Optional[str]="relu"
    ) -> None:
        
        super().__init__()
        h_features = (features if pj_features is None else pj_features)
        self.conv1 = BasicConv(features, 
                                      h_features,
                                      activation, 
                                      normalize)
        self.conv2 = BasicConv(h_features, 
                                      h_features,
                                      activation, 
                                      normalize)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.size(1)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        # print(x.size(), x1.size(), x2.size())
        return (x + x2 if C == x2.size(1) else x1 + x2)
    
    
        


if __name__ == "__main__":


    
    rearrange = Rearrange(
        "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
        p1=16, p2=16
    )
    test = torch.normal(0, 1, (100, 10, 32))
    images_test = torch.normal(0, 1, (100, 3, 128, 128))
    # images_test = F.unfold(images_test, kernel_size=(16, 16))
    images_test = rearrange(images_test)
    print(images_test.size())
    block = Block(
        in_features=32,
        hiden_features=None,
        out_features=128,
        apply_film=True,
        depth_level=12
    )
    
    print(block(test).size())
    # print(sum(p.numel() for p  ))
            

    