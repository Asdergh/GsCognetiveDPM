import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (CrossAttention, Block, get_activation)
from typing import (Optional, Tuple, List, Callable, Dict)


class DenoisingUnet(nn.Module):

    def __init__(
        self,
        in_features: int,
        c_dim: int,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        latent_dims_per_level: Optional[List[int]]=[318, 128, 32],
        blocks_per_level: Optional[List[int]]=[3, 3, 3],
        acts_per_level: Optional[List[str]]=["relu", "relu", "gelu"],
        film_per_level: Optional[List[bool]]=[True, False, True],
        film_acts_per_level: Optional[List[str]]=["sigmoid", None, "sigmoid"],
    ) -> None:
        
        super().__init__()

        if not (len(latent_dims_per_level) == 
                len(blocks_per_level) == 
                len(acts_per_level)):
            raise ValueError("all input params list must be with same len")

        self.in_d = in_features
        self.patch_size = patch_size
        self.phN = int(img_size[0] / patch_size[0])
        self.pwN = int(img_size[1] / patch_size[1])

        self.latend_ds = latent_dims_per_level
        self.out_block_ = nn.Conv2d(latent_dims_per_level[0], in_features, 3, 1, 1)
        self.down_blocks_ = self._construct_net(
            in_features, c_dim,
            latent_dims_per_level, 
            blocks_per_level, acts_per_level,
            film_per_level, film_acts_per_level,
            "down"
        )
        self.up_blocks_ = self._construct_net(
            latent_dims_per_level[-1], c_dim,
            latent_dims_per_level[::-1], 
            blocks_per_level[::-1], acts_per_level[::-1],
            film_per_level[::-1], film_acts_per_level[::-1],
            "up"
        )
        self._activations = {}


    def _construct_net(self, 
        in_features, c_dim,
        latent_dims_per_level, 
        blocks_per_level, acts_per_level,
        film_per_level, film_acts_per_level,
        scaling_mode):
        

        blocks_ = []
        for idx, (latent_dim, blocks_n, act_fn, film, film_act_fn) in enumerate(zip(
            latent_dims_per_level,
            blocks_per_level,
            acts_per_level,
            film_per_level,
            film_acts_per_level
        )):
            
            in_f = (in_features if idx == 0 else latent_dims_per_level[idx - 1])  
            conv = nn.Sequential(
                (
                    nn.Conv2d(in_f, latent_dim, 1, 2, 0)
                    if scaling_mode.lower() == "down" 
                    else nn.ConvTranspose2d(in_f, latent_dim, 1, 2, 0, 1)
                ),
                nn.BatchNorm2d(latent_dim),
                get_activation(act_fn),
                nn.Flatten(start_dim=2)
            )          
            latent_blocks = nn.ModuleList([
                Block(
                    in_features=latent_dim,
                    activation_fn=act_fn,
                    apply_film=film,
                    film_activation_fn=film_act_fn
                ) for _ in range(blocks_n)
            ])
            attention = CrossAttention(latent_dim, c_dim, activation_fn="nan")
            block = {
                "conv": conv,
                "latent_blocks": latent_blocks,
                "attention": attention
            }
            if scaling_mode.lower() == "down":
                block.update({"conv_up": nn.ConvTranspose2d(latent_dim, latent_dim, 1, 2, 0, 1)})
            
            if scaling_mode.lower() == "up":
                block.update({
                    "film_skip": nn.Sequential(
                        nn.Linear(latent_dim, latent_dim * 2),
                        nn.LayerNorm(latent_dim * 2),
                        get_activation("sigmoid"),
                    ),
                    "alphas": nn.Sequential(
                        nn.Linear(latent_dim, 1),
                        nn.LayerNorm(1),
                        get_activation("sigmoid")
                    )
                })
            blocks_.append(nn.ModuleDict(block))
        
        return nn.ModuleList(blocks_)
    
    @property
    def activations(self) -> Dict:
        if self._activations:
            return self._activations
        else:
            return False

    def _call_net(
        self, x: torch.Tensor, c: torch.Tensor,
        in_dim: int, 
        blocks: nn.ModuleList, 
        scaling_mode: str, 
        phN: int, pwN: int,
        latent_ds: List,
        skip_connections: Optional[List[torch.Tensor]]=None
    ) -> torch.Tensor:
        
        if scaling_mode.lower() == "down":
            skips = []

        B = x.size()[0]
        x = x.view(B, in_dim, phN, pwN)
        for idx, block in enumerate(blocks):
            x = block["conv"](x).permute(0, 2, 1)
            tokens_ = []
            scaling = ((1 / (2 ** (idx + 1))) if scaling_mode.lower() == "down" else (2 ** (idx + 1)))
            for latent_block in block["latent_blocks"]:
                x = latent_block(x)
                tokens_.append(x)
            x = torch.stack(tokens_, dim=1).mean(dim=1)
            x = block["attention"](x, c)
            
            
            x = x.view(B, latent_ds[idx], int(phN * scaling), int(pwN * scaling))
            self._activations.update({f"{scaling_mode}/block_{idx}_latent_patch": x.detach()})
            self._activations.update({f"{scaling_mode}/block_{idx}_attention_weights": block["attention"].activations.detach()})

            if scaling_mode.lower() == "down":
                up_x = block["conv_up"](x)
                skips.append(up_x)

            if skip_connections is not None:
                x_flatten = torch.flatten(x, start_dim=-2).permute(0, 2, 1)
                scale, shift = block["film_skip"](x_flatten).split([latent_ds[idx], latent_ds[idx]], dim=-1)
                x_film = (scale * x_flatten + shift)
        
                skip = torch.flatten(skip_connections[idx], start_dim=-2).permute(0, 2, 1)
                alphas = block["alphas"](skip)
                x = (alphas * skip + (1 - alphas) * x_film).permute(0, 2, 1)
                x = x.view(B, latent_ds[idx], int(phN * scaling), int(pwN * scaling))
                self._activations.update({f"{scaling_mode}/block_{idx}_skip_patch": x.detach()})
            

            # print("latent activation per Unet depth", x.size())
            if idx == (len(blocks) - 1):
                if scaling_mode.lower() == "up":
                    x = self.out_block_(x)
                x = torch.flatten(x, start_dim=-2)
            
        if scaling_mode.lower() == "down":
            return (
                x, skips,
                (int(phN * scaling), int(pwN * scaling)), 
            )
    
        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x, skip, (phN, pwN) = self._call_net(
            x.permute(0, 2, 1), 
            c, self.in_d, 
            self.down_blocks_, "down", 
            self.phN, self.pwN,
            self.latend_ds
        )
        x = self._call_net(
            x, c, 
            self.latend_ds[-1], 
            self.up_blocks_, "up", 
            phN, pwN, 
            self.latend_ds[::-1],
            skip_connections=skip[::-1]
        )
        return x.permute(0, 2, 1)
    


if __name__ == "__main__":
    
    test = torch.normal(0, 1, (10, 64, 128))
    test_c = torch.normal(0, 1, (10, 32))
    DenUnet = DenoisingUnet(
        in_features=128,
        c_dim=32,
        patch_size=(28, 28),
        img_size=(224, 224)
    )
    print(DenUnet)
    out = DenUnet(test, test_c)
    activations = DenUnet.activations
    print(out.size())
            
             
            
            


