import torch 
import torch.nn as nn
import math as mt
import torch.nn.functional as F
from .layers import Block, get_activation
from typing import (Optional, List, Tuple)
from einops.layers.torch import Rearrange

class VisualPatchEncoder(nn.Module):

    def __init__(
        self, 
        img_size: Tuple[int, int],
        in_channels: Optional[int]=3,
        patch_size: Optional[Tuple[int, int]]=(16, 16),
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
        self.blocks_n = blocks_n
        h, w = img_size
        ph, pw = patch_size
        # pN = int((w * h) / (pw * ph))
        self.backbone_ = nn.Sequential(
            nn.Conv2d(in_channels, latent_dim, 3, 1, 1),
            nn.BatchNorm2d(latent_dim),
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=ph, p2=pw
            ),
            get_activation(conv_activation_fn),
            nn.Linear(latent_dim * pw * ph, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.block_ = nn.ModuleList([
            Block(            
                in_features=latent_dim,
                out_features=(out_dim if idx == blocks_n else latent_dim),
                activation_fn=(out_activation_fn if idx == blocks_n else block_activation_fn),
                apply_film=apply_film,
                film_activation_fn=film_activation_fn
            ) 
            for idx in range(blocks_n + 1)
        ])
    
    def forward(self, images: torch.Tensor) -> None:

        x = self.backbone_(images)
        if self.blocks_n > 1:
            tokens = []
            for block in self.block_[:-1]:
                x = block(x)
                tokens.append(x)
            
            x = torch.stack(tokens, dim=1)
            x = self.block_[-1](torch.mean(x, dim=1))
            return x
        
        else:
            x = self.block_[0](x)
            x = self.block_[-1](x)
            return x



class SimpleDecoder(nn.Module):

    def __init__(
        self,
        in_features: int,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int] | int,
        out_channels: Optional[int]=3,
        latent_channels: Optional[int]=None,
        latent_activation_fn: Optional[str]="relu",
        out_activation_fn: Optional[str]="sigmoid"
    ) -> None:
        
        super().__init__()
        self.img_size = img_size
        self.phN = int(img_size[0] / patch_size[0])
        self.pwN = int(img_size[1] / patch_size[1])
        img_s_max = max(img_size)
        Dph = int(mt.log2(img_s_max) - mt.log2((patch_size[0] if img_size[0] == img_s_max else patch_size[1])))
        assert ((self.pwN % 1 == 0) and 
                (self.phN % 1 == 0)), ("img_size must be devideble by patch_size in vertical and hotizontal axises!!")
        
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
        x = x.permute(0, 2, 1).view(B, C, self.phN, self.pwN)
        print(x.size(), )
        x = self.input_(x)
        self._activations.append(x.detach())
        for block in self.latent_:
            x = block(x)
            self._activations.append(x.detach())

        print(x.size())
        x = self.output_(x)
        self._activations.append(x.detach())
        if x.size()[-2:] != self.img_size:
            print(f"before interpolation: {x.size()}")
            x = F.interpolate(x, self.img_size, mode="bilinear")
        return x

if __name__ == "__main__":

    test = torch.normal(0, 1, (100, 3, 112, 224))
    model = VisualPatchEncoder(
        img_size=(112, 224),
        patch_size=(14, 28),
        blocks_n=4,
        out_dim=718
    )
    decoder = SimpleDecoder(
        img_size=(112, 224),
        patch_size=(14, 28),
        in_features=718,
        latent_channels=32,
        out_channels=3,
    )
    print(model)
    out = model(test)
    print(out.size())
    out = decoder(out)
    print(out.size())

