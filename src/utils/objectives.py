import torch
import torch.nn as nn
import numpy as np
import math 
import torch.nn.functional as F
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math as mt
from torchvision.transforms import (
    InterpolationMode,
    functional as Fv
)
from typing import (
    Union,
    Optional,
    List,
    Tuple
)
from torchvision.models import vgg16
from torch.nn import (MSELoss, L1Loss)



#====================SSIM OBJECTIVE================================================================
def _gauss_kernel(kernel_size: int) -> Tuple:

    labels = np.linspace(-1, 1, kernel_size)
    exp = np.exp(-(labels ** 2 / 2))
    GoP = np.outer(exp, exp) 
    GoP /= GoP.sum()

    return torch.Tensor(GoP)

def _sobel_kernel(kernel_size: int, return_full: Optional[bool]=False) -> Tuple:

    b = []
    for i in range(kernel_size):
        C = math.comb(kernel_size - 1, i)
        b.append(C)


    k = (kernel_size - 1) / 2
    _neg_d = []
    _pos_d = []
    while k >= 0:

        _neg_d.append(-k)
        _pos_d.append(k)
        k -= 1
    
    d = _neg_d[:-1] + _pos_d[::-1]
    GxOp = torch.Tensor(np.outer(b, d))
    GyOp = torch.Tensor(np.outer(d, b))
    
    if return_full:
        return (GxOp @ GyOp)

    return (GxOp, GyOp)

class SSIM(nn.Module):

    def __init__(
        self,
        K1: Optional[float]=0.01,
        K2: Optional[float]=0.03,
        L: Optional[float]=255.0,
        kernel_size: Optional[int]=3,
        get_ssim_map: Optional[bool]=False,
        device: Optional[str]="cpu",
        kernel_type: Optional[str]="sobel", #[gauss, sobel]
        inverse: Optional[bool]=False
    ) -> Union[Tuple, torch.Tensor]:
    
        super().__init__()
        self.get_map = get_ssim_map
        self.inv = inverse
        if kernel_type == "gauss":
            GoP = _gauss_kernel(kernel_size)
        
        elif kernel_type == "sobel":
            GoP = _sobel_kernel(kernel_size, return_full=True)

        else:
            raise ValueError("unknown type kernel !!!")
    
        self.GoP = GoP.view(1, 1, *GoP.size())
        self.GoP = GoP.repeat(1, 3, 1, 1).to(device)
        self.C1 = (K1 * L) ** 2
        self.C2 = (K2 * L) ** 2

    def forward(self, Img1, Img2) -> Union[torch.Tensor, Tuple]:

        mu_x = F.conv2d(Img1, self.GoP)
        mu_y = F.conv2d(Img2, self.GoP)
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_xx = F.conv2d(Img1 * Img1, self.GoP) - mu_x_sq
        sigma_yy = F.conv2d(Img2 * Img2, self.GoP) - mu_y_sq
        sigma_xy = F.conv2d(Img1 * Img2, self.GoP) - mu_xy
     
        

        luminance = (2 * mu_xy + self.C1) / (mu_x_sq + mu_y_sq + self.C1)
        contrast_structure = (2 * sigma_xy + self.C2) / (sigma_xx + sigma_yy + self.C2)
        SSIM_map = luminance * contrast_structure 
        ssim_score = SSIM_map.mean()
        
        if self.get_map:
            return (ssim_score, SSIM_map)

        return (ssim_score if not self.inv else 1.0 - ssim_score)
#====================SSIM OBJECTIVE================================================================

#====================PERCEPTIVE OBJECTIVE==========================================================
class PerceptiveLoss(nn.Module):

    def __init__(self, max_degree: int=16, get_maps: bool=False) -> None:

        super().__init__()
        self.max_degree = max_degree
        self.get_maps = get_maps
        vgg16_base_ = vgg16(pretrained=True).features
        for param in vgg16_base_.parameters():
            param.requires_grad = False

        self.slices= nn.ModuleDict()
        for slice_idx in range(0, int(mt.log2(max_degree))):
            scale = 2 ** (slice_idx + 1)
            vgg_slice = vgg16_base_[slice_idx * 5: (slice_idx + 1) * 5]
            self.slices.update({f"slice_{scale}": nn.Sequential(*vgg_slice)})
    
        print(self.slices)

    def forward(self, x_gt: torch.Tensor, x_tar: torch.Tensor) -> torch.Tensor:
        
        Loss = 0.0
        B, C, H, W = x_gt.size()
        # if self.get_maps:
        #     features_gt_, features_tar_ = [], []
        for block in self.slices:

            x_gt = self.slices[block](x_gt)
            x_tar = self.slices[block](x_tar)
            print(x_gt.size(), x_tar.size())
            Loss += (1 / (x_gt.size(1) * x_gt.size(2) * x_gt.size(3))) * F.mse_loss(x_gt, x_tar)
            # if self.get_maps:
            #     x_gt_tmp = F.interpolate(x_gt, size=(H, W)).mean(dim=1)
            #     x_tar_tmp = F.interpolate(x_tar, size=(H, W)).mean(dim=1)
            #     x_gt_tmp = cm.inferno(x_gt_tmp.detach())[..., :3]
            #     x_tar_tmp = cm.inferno(x_tar_tmp.detach())[..., :3]
            #     features_gt_.append(torch.Tensor(x_gt_tmp).permute(0, 3, 1, 2))
            #     features_tar_.append(torch.Tensor(x_tar_tmp).permute(0, 3, 1, 2))
        
        # if self.get_maps:
        #     return (Loss, torch.stack(features_gt_).permute(1, 0, 2, 3, 4), torch.stack(features_tar_).permute(1, 0, 2, 3, 4))
        
        return Loss
#====================PERCEPTIVE OBJECTIVE==========================================================
            
        

def psnr(Img1: torch.Tensor, Img2: torch.Tensor) -> torch.Tensor:

    assert ((Img1.max() <= 1 and Img2.max() <= 1) or
            (Img1.max() <= 255.0 and Img2.max() <= 255)), ("images must me in one range")
    
    if Img1.max() <= 1:
        MAX = 20 * torch.log10(torch.tensor(1.0))
    
    else:
        MAX = 20 * torch.log10(torch.tensor(255.0))

    MSE = 10 * torch.log10(F.mse_loss(Img1, Img2))
    return MAX - MSE 
    


__OBJECTIVES__ = {
    "ssim": SSIM,
    "perceptive": PerceptiveLoss,
    "mse": MSELoss,
    "l1": L1Loss,
}

get_loss = lambda objective_str: (
    __OBJECTIVES__["ssim"](get_ssim_map=False, inverse=True, device="cuda") if objective_str == "d-ssim"
    else __OBJECTIVES__["mse"]() if objective_str not in __OBJECTIVES__ 
    else __OBJECTIVES__[objective_str]()
)


# if __name__ == "__main__":
    

#     from PIL import Image
#     from torchvision.transforms import (Compose, PILToTensor, Resize, Lambda)
#     from torchvision.utils import make_grid

#     target_size = (112, 224)
#     tf = Compose([
#         Lambda(lambda img_f: Image.open(img_f)),
#         PILToTensor(),
#         Resize(target_size),
#         Lambda(lambda img: ((img / 255.0).to(torch.float32) if img.max() > 1 else img))
#     ])

#     test1 = tf("/home/ram/Downloads/360_v2/bicycle/images_8/_DSC8719.JPG")[None].repeat(4, 1, 1, 1)
#     test2 = tf("/home/ram/Downloads/360_v2/bicycle/images_8/_DSC8717.JPG")[None].repeat(4, 1, 1, 1)
    
#     imgs = torch.cat([test1, test2], dim=0)
#     grid = make_grid(imgs)
#     perceptive_loss = PerceptiveLoss(4, True)
#     loss, map1, map2 = perceptive_loss(test1, test2)
    
#     print(map1.size())
#     map1_values = make_grid(map1[0])
#     map2_values = make_grid(map2[0])
#     plt.style.use("dark_background")
#     _, axis = plt.subplots(nrows=3)
#     print(map1_values.size(), map2_values.size())
#     axis[0].imshow(grid.permute(1, 2, 0))
#     axis[1].imshow(map1_values.permute(1, 2, 0))
#     axis[2].imshow(map2_values.permute(1, 2, 0))
#     plt.show()
    
    