import torch
import cv2
import numpy as np
from typing import (Union, Optional)

def conver_img(
    img: torch.Tensor,
    in_format: Optional[str]="WHC", 
    out_format: Optional[str]="CWH",
    return_type: Optional[str]="pt",
    for_log: Optional[bool]=False
) -> Union[torch.Tensor, np.ndarray]:
    
    if out_format == "WHC":
        if in_format == "CWH":
            img = img.permute(1, 2, 0)
    
    if out_format == "CWH":
        if in_format == "WHC":
            img = img.permute(-1, 0, 1)
    
    if for_log:
        img = img.detach().cpu()
    
    if return_type == "np":
        img = img.detach().cpu().numpy()
    
    return img
    
    