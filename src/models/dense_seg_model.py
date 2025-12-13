
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import (Tuple, Optional, List)
from dataclasses import (dataclass, field)
from .blocks import (Block, InteractionAttention, 
                    ResidualConv, get_activation,
                    build_conv_stack)
from ..configs.configs import parse_structured
from ..utils.tensors import (seq2spatial)

class SegmentationTransformer(nn.Module):

    def __init__(
        self,
        in_features: int,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        hiden_dim: Optional[int]=128,
        last_dim: Optional[int]=254,
        embeddings_dim: Optional[int]=768, 
        readout_token_size: Optional[int]=128,
        readout_tokens_n: Optional[int]=3,
        transformer_depth: Optional[int]=4,
        latent_blocks_depth: Optional[int]=1,
        att_pooling_size: Optional[int]=4,
        attention_activation: Optional[str]="relu",
        blocks_activation: Optional[str]="relu",
        film_activation: Optional[str]="sigmoid",
        return_intermediates: Optional[bool]=False
    ) -> None:
        
        super().__init__()
        self.embeds_dim = embeddings_dim
        self.r_int = return_intermediates
        self.t_depth = transformer_depth
        pwN = img_size[0] // patch_size[0]
        phN = img_size[1] // patch_size[1]
        self.patches_n = (pwN, phN)

        self.rt_size = readout_token_size
        self.rt_n = readout_tokens_n
        self.ts = self.rt_size * self.rt_n

        def build_agregator_(features):
            return nn.ModuleList([
                Block(
                    in_features=(features if idx == 0 else hiden_dim),
                    hiden_features=hiden_dim,
                    out_features=(
                        hiden_dim if idx != (transformer_depth - 1) 
                        else last_dim
                    ),
                    depth_level=latent_blocks_depth,
                    activation_fn=blocks_activation,
                    apply_film=True,
                    film_activation_fn=film_activation,
                )
                for idx in range(transformer_depth)
        ])
        self.readout_tokens_ = nn.Parameter(torch.rand(self.ts))
        self.visual_argregator_ = build_agregator_(in_features + self.ts)
        self.prompt_argregator_ = build_agregator_(embeddings_dim)
        self.att_blocks_ = nn.ModuleList([
            InteractionAttention(
                input_dim=(hiden_dim if idx != (transformer_depth - 1) else last_dim),
                hiden_dim=hiden_dim,
                patch_n_pr=self.patches_n,
                pooling_size=att_pooling_size,
                latent_first_act_fn=attention_activation,
                latent_last_act_fn=attention_activation,
                format="sequence",
                mode="cross"
            )
            for idx in range(transformer_depth)
        ])
    
    def forward(self, x: torch.Tensor, tokens: Optional[torch.Tensor]=None) -> torch.Tensor:
        
        readout_tokens = self.readout_tokens_.repeat(x.size(0), x.size(1), 1)
        tokens = tokens.repeat(x.size(1), 1, 1).transpose(0, 1)
        # print(tokens.size(), readout_tokens.size())
        x = torch.cat([x, readout_tokens], dim=-1)

        if self.r_int:
            intermediates = []

        for (vis_agr, prompt_agr, attention) in zip(
            self.visual_argregator_, 
            self.prompt_argregator_,
            self.att_blocks_
        ):
            x = vis_agr(x)
            tokens = prompt_agr(tokens)
            # print(x.size(), tokens.size())
            x = attention(x, tokens).transpose(-1, -2)
            # print(x.size())
            if self.r_int:
                x_spat = seq2spatial(x, format="BNC", patches_n=self.patches_n).permute(0, -1, 1, 2)
                intermediates.append(x_spat)

        x = seq2spatial(x, format="BNC", patches_n=self.patches_n).permute(0, -1, 1, 2)
        return (x if not self.r_int else intermediates)
    

class DenseSegmentationDecoder(nn.Module):

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        readout_token_size: Optional[int]=128,
        readout_tokens_n: Optional[int]=3,
        decoder_depth: Optional[int]=4,
        head_depth: Optional[int]=3,
        hiden_dim: Optional[int]=128,
        class_labels: Optional[int]=None,
        hiden_activation: Optional[str]="relu",
        out_activation: Optional[str]="tanh",
        fusion_alpha: Optional[float]=0.32,
        return_intermediates: Optional[bool]=False
    ) -> None:
        
        super().__init__()
        self.r_ints = return_intermediates
        self.alpha = fusion_alpha
        self.rt_size = readout_token_size
        self.rt_n = readout_tokens_n
        self.ts = (self.rt_size * self.rt_n)

        pwN = (img_size[0] // patch_size[0])
        phN = (img_size[1] // patch_size[1])
        self.pNg = (pwN, phN)
        
        self.head_ = nn.ModuleList([
            build_conv_stack(
                i_f=hiden_dim,
                o_f=(class_labels if idx == (head_depth - 1) else hiden_dim),
                activation=out_activation,
                norm=True,
                kernel_size=(3, 3),
                stride=2,
                mode="up",
                sampler="default",
                scale_factor=(4 * (2 ** idx))
            )
            for idx in range(head_depth)
        ])
        self.resample_ = nn.ModuleList([
            nn.ModuleList([
                nn.Upsample(scale_factor=(4 * (2 ** idx))),
                nn.Conv2d(hiden_dim, 
                          hiden_dim, 
                          (3, 3), 1, 1)
            ])
            for idx in range(decoder_depth)
        ])
        def build_residual_stack_(N):
            return nn.ModuleList([
                    ResidualConv(features=hiden_dim, 
                                normalize=True,
                                activation=hiden_activation)
                    for idx in range(N)
                ])
        
        self.residual_ = build_residual_stack_(decoder_depth)
        skips = build_residual_stack_(decoder_depth - 1)
        self.skip_residual_ = nn.ModuleList([nn.Identity(), ])
        self.skip_residual_.extend(skips)

        self.downsample_ = nn.ModuleList([
            build_conv_stack(
                i_f=hiden_dim,
                o_f=hiden_dim,
                activation=out_activation,
                norm=True,
                mode="down",
                sampler="conv",
                stride=2
            )
            for idx in range(head_depth)
        ])

    def forward(self, agregations: List[torch.Tensor]) -> torch.Tensor:

        
        resampled_tokens = []
        # print(len(self.resample_))
        for idx, resample in enumerate(self.resample_):
            x = agregations[idx]
            # print("agregates tokens", x.shape)
            x = resample[0](x)
            # print("Upsample", x.shape)
            x = resample[1](x)
            # print("Conv", x.shape)
            resampled_tokens.append(x)

        resampled_tokens = resampled_tokens[::-1]
        
        for idx, (sk_residual, residual, downsample) in enumerate(zip(
            self.skip_residual_,
            self.residual_,
            self.downsample_
        )):
        
            x_skip = resampled_tokens[idx]
            x_skip = sk_residual(x_skip)
            # print("residual", x.size())
            x = residual(x)
            # print("botch features", x.size(), x_skip.size())

            x = self.alpha * x + (1 - self.alpha) * x_skip

            # print("pre downsampler", x.size())
            x = downsample(x)
            # print("post downsample", x.size())
            
        
        if self.r_ints:
            ints = []
        for block in self.head_:
            x = block(x)
            if self.r_ints:
                ints.append(x)

        return (x if not self.r_ints else ints)


class DenseSegmentationModel(nn.Module):

    @dataclass
    class Config:
        in_features: int
        embeddings_dim: int
        img_size: Tuple[int, int]
        patch_size: Tuple[int, int]
        hiden_dim: Optional[int]=128
        last_transformer_dim: Optional[int]=32
        readout_token_size: Optional[int]=32
        readout_tokens_n: Optional[int]=3
        agregation_depth: Optional[int]=2
        latent_blocks_depth: Optional[int]=1
        attention_pooling_size: Optional[int]=4
        attention_activation: Optional[str]="relu"
        blocks_activation: Optional[str]="relu"
        film_activation: Optional[str]="sigmoid"
        decoder_hiden_activation: Optional[str]="sigmoid"
        decoder_out_activvation: Optional[str]="tanh"
        segmentation_head_depth: Optional[int]=3
        fusion_alpha: Optional[float]=0.32
    
    def __init__(self, cfg) -> None:

        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.transformer_ = SegmentationTransformer(
            img_size=self.cfg.img_size,
            patch_size=self.cfg.patch_size,
            in_features=self.cfg.in_features,
            hiden_dim=self.cfg.hiden_dim,
            last_dim=self.cfg.last_transformer_dim,
            embeddings_dim=self.cfg.embeddings_dim,
            readout_token_size=self.cfg.readout_token_size,
            readout_tokens_n=self.cfg.readout_tokens_n,
            transformer_depth=self.cfg.agregation_depth,
            latent_blocks_depth=self.cfg.latent_blocks_depth,
            att_pooling_size=self.cfg.attention_pooling_size,
            attention_activation=self.cfg.attention_activation,
            blocks_activation=self.cfg.blocks_activation,
            film_activation=self.cfg.film_activation,
            return_intermediates=True
        )
        self.segmentator_ = DenseSegmentationDecoder(
            img_size=self.cfg.img_size,
            patch_size=self.cfg.patch_size,
            hiden_dim=self.cfg.hiden_dim,
            readout_token_size=self.cfg.readout_token_size,
            readout_tokens_n=self.cfg.readout_tokens_n,
            decoder_depth=self.cfg.agregation_depth,
            head_depth=self.cfg.segmentation_head_depth,
            hiden_activation=self.cfg.decoder_hiden_activation,
            out_activation=self.cfg.decoder_out_activvation,
            fusion_alpha=self.cfg.fusion_alpha,
            class_labels=32,
            return_intermediates=True
        )
    
    def forward(self, visual_tokens: torch.Tensor, prompt_tokens: torch.Tensor) -> torch.Tensor:
        agregated_tokens = self.transformer_(visual_tokens, prompt_tokens)
        print(len(agregated_tokens))
        dense_segmentation_masks = self.segmentator_(agregated_tokens)
        return dense_segmentation_masks
    



if __name__ == "__main__":

    import tyro
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))

  
    args = DenseSegmentationModel.Config
    args = tyro.cli(args)
    model = DenseSegmentationModel(args).to("cuda")
    print(model.cfg.img_size, type(model.cfg.img_size))
    print(model.cfg.patch_size, type(model.cfg.patch_size))
    
    test = torch.rand((10, 49, 16)).to("cuda")
    tokens = torch.rand(10, 16).to("cuda")
    output = model(test, tokens)
    for act in output:
        print(act.shape)
    print(model)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    

    

        
        
                


                
            

        
        
        

            
        