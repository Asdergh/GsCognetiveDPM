import torch
import numpy as np
import math 
import torch.nn.functional as F
import matplotlib.cm as cm
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

def ssim(
    Img1: torch.Tensor,
    Img2: torch.Tensor,
    K1: Optional[float]=0.01,
    K2: Optional[float]=0.03,
    L: Optional[float]=255.0,
    kernel_size: Optional[int]=3,
    get_ssim_map: Optional[bool]=False,
    device: Optional[str]="cpu",
    kernel_type: Optional[str]="sobel" #[gauss, sobel]
) -> Union[Tuple, torch.Tensor]:
    


    if kernel_type == "gauss":
        GoP = _gauss_kernel(kernel_size)
    
    elif kernel_type == "sobel":
        GoP = _sobel_kernel(kernel_size, return_full=True)

    else:
        raise ValueError("unknown type kernel !!!")
  
    GoP = GoP.view(1, 1, *GoP.size())
    GoP = GoP.repeat(1, 3, 1, 1).to(device)

    mu_x = F.conv2d(Img1, GoP)
    mu_y = F.conv2d(Img2, GoP)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(Img1 * Img1, GoP) - mu_x_sq
    sigma_yy = F.conv2d(Img2 * Img2, GoP) - mu_y_sq
    sigma_xy = F.conv2d(Img1 * Img2, GoP) - mu_xy
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    

    luminance = (2 * mu_xy + C1) / (mu_x_sq + mu_y_sq + C1)
    contrast_structure = (2 * sigma_xy + C2) / (sigma_xx + sigma_yy + C2)
    SSIM_map = luminance * contrast_structure 
    ssim_score = SSIM_map.mean()
    
    if get_ssim_map:
        return (ssim_score, SSIM_map)

    return ssim_score


def psnr(Img1: torch.Tensor, Img2: torch.Tensor) -> torch.Tensor:

    assert ((Img1.max() <= 1 and Img2.max() <= 1) or
            (Img1.max() <= 255.0 and Img2.max() <= 255)), ("images must me in one range")
    
    if Img1.max() <= 1:
        MAX = 20 * torch.log10(torch.tensor(1.0))
    
    else:
        MAX = 20 * torch.log10(torch.tensor(255.0))

    MSE = 10 * torch.log10(F.mse_loss(Img1, Img2))
    return MAX - MSE 
    



# if __name__ == "__main__":
    
#     # img1 = torch.normal(0, 1, (10, 3, 448, 448))
#     # # img2 = torch.normal(0, 1, (10, 3, 256, 256))
#     # img2 = torch.zeros_like(img1)
#     import matplotlib.pyplot as plt
#     from PIL import Image
#     from torchvision.transforms import (
#         Compose,
#         Resize,
#         PILToTensor,
#         Lambda
#     )
#     plt.style.use("dark_background")
#     resolution = (448, 448)
#     tf = Compose([
#         PILToTensor(),
#         Resize(resolution),
#         Lambda(lambda rgb: ((rgb / 255.0).to(torch.float32) if rgb.max() > 1.0 else rgb))
#     ])
#     img1 = tf(Image.open("/media/ram/T7/2025-03-26_processed/map2/keyframe_map/zedx_front_left/rgb/000120.jpg"))
#     # img2 = tf(Image.open("/media/ram/T7/2025-03-26_processed/map1/keyframe_map/zedxone_right/rgb/000657.jpg"))
#     img2 = tf(Image.open("/media/ram/T7/2025-03-26_processed/map2/keyframe_map/zedx_front_left/rgb/000158.jpg"))
#     ssim_score, ssim_heatmap = ssim(img1, img2, get_ssim_map=True, kernel_type="gauss", L=256.0, kernel_size=11)
#     psnr_score = psnr(img1, img2)
    
#     print(ssim_score)
#     print(psnr_score)
#     print(ssim_heatmap.size())
#     _, axis = plt.subplots(ncols=3)
#     axis[0].imshow(img1.squeeze().permute(1, 2, 0))
#     axis[1].imshow(img2.squeeze().permute(1, 2, 0))
#     axis[2].imshow(ssim_heatmap.squeeze())
#     plt.show()
    
    