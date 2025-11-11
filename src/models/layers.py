import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import (Optional, Tuple, Callable, Dict)
from einops.layers.torch import Rearrange



_ACTIVATIONS_ = {
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "softmax": nn.Softmax,
    "gelu": nn.GELU
}
get_activation = lambda act_type: (
    _ACTIVATIONS_[act_type](dim=-1) if act_type.lower() == "softmax" else 
    _ACTIVATIONS_[act_type]() if act_type in _ACTIVATIONS_ else
    nn.Identity()
)
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
        apply_film: Optional[bool]=False,
        film_activation_fn: Optional[str]="sigmoid",
        depth_level: Optional[int]=6
    ) -> None:
        
        super().__init__()
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
                in_features=next(val for (val, cond) in [
                    (in_features, idx == 0),
                    (self.hiden_features, idx != 0)
                ] if cond),
                hiden_features=hiden_features,
                out_features=next(val for (val, cond) in [
                    (self.hiden_features, idx != (depth_level - 1)),
                    (self.out_features, idx == (depth_level - 1))
                ] if cond),
                activation_fn=activation_fn
            ) for idx in range(self.d_max)
        ])
        
        if apply_film:
            self.film_ = nn.Sequential(
                nn.Linear(self.hiden_features, self.hiden_features * 2),
                get_activation(film_activation_fn)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for dt, block in enumerate(self.blocks_):
            x = block(x)
            if dt != self.d_max - 1:
                if hasattr(self, "film_"):
                    film = self.film_(x)
                    scale, shift= film.split([self.hiden_features, self.hiden_features], dim=-1)
                    x = scale * x + shift
            
            x = self.weights_fn(dt) * x
        
        return x


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
        hiden_features=64,
        out_features=128,
        apply_film=True,
        depth_level=12
    )
    
    print(block(test).size())
            

    