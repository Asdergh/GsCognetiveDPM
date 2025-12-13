import torch 
import numpy as np
from typing import (Optional, Tuple)




def spatial2seq(x: torch.Tensor, format: Optional[str]="BCWH"):
    if format == "BCWH":
        x = torch.flatten(x, start_dim=-2)
    elif format == "BWHC":
        x = torch.flatten(x, start_dim=1, end_dim=-2)
        x = x.transpose(-1, -2)
    return x

def seq2spatial(
    x: torch.Tensor,
    img_size: Optional[Tuple[int, int]]=None,
    patch_size: Optional[Tuple[int, int]]=None,
    patches_n: Optional[Tuple[int, int]]=None,
    format: Optional[str]="BCN",
):
    
    def seq2sp(x, pNx, pNy):
        if format == "BCN":
            x = x.view(x.size(0), x.size(1), pNx, pNy)
        elif format == "BNC":
            x = x.view(x.size(0), pNx, pNy, x.size(-1))
        else:
            raise ValueError("unkown tensor format")
        return x
        
    assert (not(img_size is None
            and patch_size is None
            and patches_n is None)), (f"""
        provide one combination of parameters:
                1) [img_size, patch_size]
                2) [patches_n] # per each img axis
        """)
    
    if patches_n is not None:
        x = seq2sp(x, patches_n[0], patches_n[1])
    
    elif (img_size is not None
          and patch_size is not None):
        pNx = img_size[0] // patch_size[0]
        pNy = img_size[1] // patch_size[1]
        x = seq2sp(x, pNx, pNy)
    
    return x
