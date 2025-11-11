import torch 
import torch.nn as nn
import math as mt
from .layers import Block, get_activation
from typing import (Optional, List)
from einops.layers.torch import Rearrange

class PatchEncoder(nn.Module):

    def __init__(
        self, 
        in_channels: Optional[int]=3,
        width: Optional[int]=256,
        height: Optional[int]=256,
        patch_size: Optional[int]=16,
        latent_dim: Optional[int]=32,
        out_dim: Optional[int]=128,
        apply_film: Optional[bool]=True,
        blocks_n: Optional[int]=1,
        conv_activation_fn: Optional[str]="tanh",
        block_activation_fn: Optional[str]="sigmoid",
        film_activation_fn: Optional[str]="softmax",
        out_activation_fn: Optional[str]="softmax"
    ) -> None:
        
        super().__init__()
        self.w, self.h = (width, height)
        self.blocks_n = blocks_n
        self.patch_s = patch_size
        self.patch_per_row = int(mt.sqrt(self.w * self.h / (self.patch_s ** 2)))
        
        self.backbone_ = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, 3, 1, 1),
            nn.BatchNorm2d(latent_dim),
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_s,
                p2=self.patch_s
            ),
            get_activation(conv_activation_fn),
            nn.Linear(latent_dim * int(self.patch_s ** 2), latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.block_ = nn.ModuleList([
            Block(            
                in_features=latent_dim,
                activation_fn=block_activation_fn,
                apply_film=apply_film,
                film_activation_fn=film_activation_fn
            ) 
            for _ in range(blocks_n)
        ] + [nn.Sequential(
            nn.Linear(latent_dim, out_dim),
            nn.LayerNorm(out_dim),
            get_activation(out_activation_fn)
        )])
    
    def forward(self, images: torch.Tensor) -> None:

        x = self.backbone_(images)
        if self.blocks_n > 1:
            tokens = []
            for block in self.block_[:-1]:
                x = block(x)
                tokens.append(x)
            
            tokens = torch.stack(tokens, dim=1)
            tokens = torch.mean(tokens, dim=1)
            tokens = self.block_[-1](tokens)
            return tokens
        
        else:
            x = self.block_[0](x)
            x = self.block_[-1](x)
            return x



class SimpleDecoder(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_channels: Optional[int]=3,
        latent_channels: Optional[int]=None,
        patches_n: Optional[int]=16,
        result_img_size: Optional[int]=512,
        latent_activation_fn: Optional[str]="relu",
        out_activation_fn: Optional[str]="sigmoid"
    ) -> None:
        
        super().__init__()
        
        Dph = int(mt.log2(result_img_size) - mt.log2(patches_n))
        self.input_ = nn.ConvTranspose2d(in_features, latent_channels, 3, 2, 1, 1)
        self.output_ = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, out_channels, 3, 2, 1, 1),
            nn.BatchNorm2d(out_channels),
            get_activation(out_activation_fn)
        )
        self.latent_ = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(latent_channels, latent_channels, 3, 2, 1, 1),
                nn.BatchNorm2d(latent_channels),
                get_activation(latent_activation_fn)
            ) for _ in range(Dph - 1)
        ])
        self._activations = []
    
    
    @property
    def activations(self) -> List:
        if self._activations:
            return self._activations
        else:
            return False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #[B, N, C] -> [B, C, Pn, Pn]
        B, N, C = x.size()

        x = x.permute(0, 2, 1).view(B, C, int(mt.sqrt(N)), int(mt.sqrt(N)))

        x = self.input_(x)
        self._activations.append(x.detach())
        for block in self.latent_:
            x = block(x)
            self._activations.append(x.detach())

        x = self.output_(x)
        self._activations.append(x.detach())
        return x

if __name__ == "__main__":

    test = torch.normal(0, 1, (100, 3, 112, 112))
    model = PatchEncoder(
        width=112, height=112,
        patch_size=16,
        blocks_n=4,
        out_dim=718
    )
    decoder = SimpleDecoder(
        in_features=718,
        latent_channels=32,
        out_channels=3,
        result_img_size=128
    )
    print(model)
    out = model(test)
    print(out.size())
    out = decoder(out)
    print(out.size())

