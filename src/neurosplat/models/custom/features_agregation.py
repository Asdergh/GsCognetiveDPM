import torch 
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import (dataclass, field)
from .blocks import (BasicLinear, BasicConv, 
                     ChannelWiseCrossAttention,
                     InteractionAttention)
from typing import (Tuple, Optional, Dict, Any)
from omegaconf import OmegaConf
from ..utils.tensors import (spatial2seq, seq2spatial)


class ChannelWiseAgregation(nn.Module):

    @dataclass
    class Config:
        in_features: int
        embed_features: int
        hiden_features: Optional[int]=None
        readout_tokens_features: Optional[int]=32
        scales: Optional[list]=field(default_factory=lambda: [2, 4, 8])
        attention_actiavtion: Optional[str]="sigmoid"
        resample_actiavtion: Optional[str]="tanh"
        linear_activation: Optional[str]="relu"
        linear_gate_activation: Optional[str]="sigmoid"
    
    def __init__(self, cfg: Dict[int, Any]) -> None:
        
        super().__init__()
        self.cfg = self.Config(**cfg)
        self.embed_features = self.cfg.embed_features
        hiden_features = (
            self.cfg.in_features 
            if self.cfg.hiden_features is None 
            else self.cfg.hiden_features
        )
        self.readout_ = nn.Parameter(torch.rand(self.cfg.readout_tokens_features))
        self.spatial_projection_ = BasicConv(
            in_features=self.cfg.in_features,
            out_features=hiden_features,
            activation=self.cfg.resample_actiavtion
        )
        self.blocks_ = nn.ModuleList([
            nn.ModuleDict({
                "resample": BasicConv(
                    in_features=hiden_features,
                    out_features=hiden_features,
                    scale_factor=scale_factor,
                    sampler="default",
                    norm=True,
                    activation=self.cfg.resample_actiavtion
                ),
                "ch_attention": nn.ModuleList([
                    BasicLinear(
                        in_features=(self.cfg.embed_features 
                                     + self.cfg.readout_tokens_features),
                        out_features=(self.cfg.embed_features * 2),
                        activation=self.cfg.linear_activation,
                        norm=True,
                    ),
                    ChannelWiseCrossAttention(
                        in_features=hiden_features,
                        embedding_features=self.cfg.embed_features,
                        out_features=hiden_features,
                        att_cross_mode=True,
                        agr_activation=self.cfg.attention_actiavtion,
                        film_mode=False
                    )
                ]),
                "linear": nn.ModuleList([
                    BasicLinear(
                        in_features=hiden_features,
                        out_features=hiden_features,
                        activation=self.cfg.linear_activation,
                        norm=True
                    ),
                    BasicLinear(
                        in_features=hiden_features,
                        out_features=1,
                        activation=self.cfg.linear_gate_activation
                    )
                ])
            })
            for scale_factor in self.cfg.scales
        ])
    
    
    
    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:

        intermidiates = []
        readout_tokens = self.readout_[None].repeat(x.size(0), 1)
        initial_tokens = torch.cat([tokens, readout_tokens], dim=-1)
        x_spatial = self.spatial_projection_(x)
        for block in self.blocks_:
            x = block["resample"](x_spatial)

            tokens = block["ch_attention"][0](initial_tokens)
            readed_tokens = tokens[..., :self.embed_features] + tokens[..., -self.embed_features:]
            x = block["ch_attention"][1](x, readed_tokens)

            x_tmp = block["linear"][0](x)
            alpha = block["linear"][1](x)
            x = alpha * x_tmp
            intermidiates.append(x)
        
        x = torch.stack(intermidiates, dim=1)
        x = torch.sum(x, dim=1)
        x = x_spatial * x[..., None, None]
        
        return x


class SpatialAgregation(nn.Module):
    @dataclass
    class Config:
        in_features: int
        embed_features: int
        img_size: Tuple[int, int]
        patch_size: Tuple[int, int]
        hiden_features: Optional[int]=None
        output_features: Optional[int]=None
        readout_tokens_features: Optional[int]=32
        att_pooling_size: Optional[int]=3
        attention_activation: Optional[str]="relu"
        linear_activation: Optional[str]="relu"
        
    def __init__(self, cfg: Dict[str, Any]) -> None:

        super().__init__()
        self.cfg = self.Config(**cfg)
        self.pWn = self.cfg.img_size[0] // self.cfg.patch_size[0]
        self.pHn = self.cfg.img_size[1] // self.cfg.patch_size[1]

        self.hiden_features = (
            self.cfg.in_features
            if self.cfg.hiden_features is None
            else self.cfg.hiden_features
        )
        output_features = (
            self.hiden_features 
            if self.cfg.output_features is  None
            else self.cfg.output_features
        )
        self.readout_ = nn.Parameter(torch.rand(self.cfg.readout_tokens_features))
        self.process_tokens_ = BasicLinear(
            in_features=(self.cfg.embed_features + self.cfg.readout_tokens_features),
            out_features=(self.hiden_features * 2),
            activation=self.cfg.linear_activation,
            norm=False,
        )
        self.cross_attention_ = InteractionAttention(
            input_features=self.cfg.in_features,
            embed_features=self.cfg.hiden_features,
            hiden_features=self.cfg.hiden_features,
            latent_first_activation=self.cfg.attention_activation,
            latent_last_activation=self.cfg.attention_activation,
            patch_n_pr=(self.pWn, self.pHn),
            mode="cross"
        )
        self.transformer_part_ = nn.Sequential(
            BasicLinear(
                in_features=self.hiden_features,
                out_features=self.hiden_features,
                activation=self.cfg.linear_activation,
                norm=True,
            ),
            InteractionAttention(
                input_features=self.hiden_features,
                latent_first_activation=self.cfg.attention_activation,
                latent_last_activation=self.cfg.attention_activation,
                patch_n_pr=(self.pWn, self.pHn),
                mode="self",
                format="sequence"
            ),
            BasicLinear(
                in_features=self.hiden_features,
                out_features=output_features,
                activation=self.cfg.linear_activation,
                norm=False
            )
        )
    
    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:

        B = x.size(0)
        readout_tokens = self.readout_[None].repeat(B, 1)
        tokens = torch.cat([tokens, readout_tokens], dim=-1)
        tokens = tokens[..., None, None].repeat(1, 1, self.pWn, self.pHn)

        tokens = spatial2seq(tokens, input_format="BCWH", output_format="BNC")
        tokens = self.process_tokens_(tokens)

        readed_tokens = tokens[..., :self.hiden_features] + tokens[..., -self.hiden_features:]
        readed_tokens = seq2spatial(
            x=readed_tokens, 
            input_format="BNC", 
            output_format="BCWH",
            patches_n=(self.pWn, self.pHn)
        )
        x = self.cross_attention_(x, readed_tokens)
        x = spatial2seq(x, "BCWH", "BNC")
        x = seq2spatial(
            x=self.transformer_part_(x),
            input_format="BNC",
            output_format="BCWH",
            patches_n=(self.pWn, self.pHn)
        )
        
        return x
        



        
        


        
if __name__ == "__main__":

    INPUT_FEATURES = 256
    EMBEDDING_FEATURES = 328
    IMG_SIZE = (224, 224)
    PATCH_SIZE = (7, 7)
    HIDEN_FEATURES = 128

    cfg_spatial = OmegaConf.structured(
        SpatialAgregation.Config(
            in_features=INPUT_FEATURES,
            embed_features=EMBEDDING_FEATURES,
            hiden_features=HIDEN_FEATURES,
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE
        )
    )
    cfg_channel = OmegaConf.structured(
        ChannelWiseAgregation.Config(
            in_features=INPUT_FEATURES,
            embed_features=EMBEDDING_FEATURES,
            hiden_features=HIDEN_FEATURES
        )
    )
    spatial_model = SpatialAgregation(cfg_spatial)
    channel_model = ChannelWiseAgregation(cfg_channel)
    print("SPATIAL TOTAL PARAMS:", sum(p.numel() for p in spatial_model.parameters()))
    print("CHANNEL TOTAL PARAMS: ", sum(p.numel() for p in channel_model.parameters()))
    test0 = torch.rand(32, INPUT_FEATURES, 32, 32)
    test1 = torch.rand(32, EMBEDDING_FEATURES)
    print("spatial features size: ", spatial_model(test0, test1).size())
    print("channels features size: ", channel_model(test0, test1).size())


    