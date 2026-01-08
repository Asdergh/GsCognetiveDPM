import torch
import cv2
import numpy as np
import math
from PIL import Image
from dataclasses import (dataclass, field, fields)
from typing import (
    Union, 
    Optional, 
    Tuple, 
    NamedTuple,
    List
)
from torch.nn.functional import interpolate
from torchvision.transforms import PILToTensor, Resize, Compose
from torchvision.transforms.functional import pil_to_tensor


def quat2Rmat(q: Union[torch.Tensor, np.ndarray], in_format="xyzw") -> np.ndarray:
    
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()    

    if (q.ndim == 1):
        batched = False
        N = 1
        q = q.reshape(1, 4)
    elif (q.ndim == 2 and q.shape[-1] == 4):
        N = q.shape[0]
        batched = True
    else:
        raise ValueError(f"q is not quaternion vector or batch of vectors. Size: {q.shape}")
        
    if in_format == "xyzw":
        x = q[:, 0]
        y = q[:, 1]
        z = q[:, 2]
        w = q[:, 3]
    else:
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]
        w = q[:, 0]
    
    x2, y2, z2 = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    Rmat_batched = np.zeros((N, 3, 3))
    Rmat_batched[:, 0, 0] = 1 - 2*(y2 + z2)
    Rmat_batched[:, 0, 1] = 2*(xy - wz)
    Rmat_batched[:, 0, 2] = 2*(xz + wy)

    Rmat_batched[:, 1, 0] = 2*(xy + wz)
    Rmat_batched[:, 1, 1] = 1 - 2*(x2 + z2)
    Rmat_batched[:, 1, 2] = 2*(yz - wx)

    Rmat_batched[:, 2, 0] = 2*(xz - wy)
    Rmat_batched[:, 2, 1] = 2*(yz + wx)
    Rmat_batched[:, 2, 2] = 1 - 2*(x2 + y2)

    return (Rmat_batched[0] if not batched else Rmat_batched)

def Rmat2quat(Rmat: Union[torch.Tensor, np.ndarray], out_format="xyzw") -> np.ndarray:
    
    if isinstance(Rmat, torch.Tensor):
        Rmat = Rmat.detach().cpu().numpy() 

    if Rmat.ndim == 3:
        batched = True
        N = Rmat.shape[0]
    else:
        N = 1
        batched = False
        Rmat = Rmat.reshape(1, 3, 3)

    Tr = np.diagonal(Rmat, axis1=1, axis2=2).sum(axis=1)
    q_result = np.zeros((N, 4))
    max_diag = np.stack([Rmat[:, 0, 0], Rmat[:, 1, 1], Rmat[:, 2, 2]], axis=-1).max(axis=1)
    
    where_tr_pos = (Tr > 0)
    if where_tr_pos.sum() != 0.0:
        w = np.sqrt(1 + Tr) / 2
        x = (Rmat[where_tr_pos, 2, 1] - Rmat[where_tr_pos, 1, 2]) / (4 * w + 1e-6)
        y = (Rmat[where_tr_pos, 0, 2] - Rmat[where_tr_pos, 2, 0]) / (4 * w + 1e-6)
        z = (Rmat[where_tr_pos, 1, 0] - Rmat[where_tr_pos, 0, 1]) / (4 * w + 1e-6)
        q_result[where_tr_pos] = np.stack((
            [x, y, z, w] if out_format == "xyzw" 
            else [w, x, y, z]
        ), axis=-1)
    
    where_r11_max = (Rmat[:, 0, 0] == max_diag)
    if where_r11_max.sum() != 0.0:
        x = (np.sqrt(1 + 2 * Rmat[where_r11_max, 0, 0] - Tr[where_r11_max])) / 2
        w = (Rmat[where_r11_max, 2, 1] - Rmat[where_r11_max, 1, 2]) / (4 * x + 1e-6)
        y = (Rmat[where_r11_max, 0, 1] + Rmat[where_r11_max, 1, 0]) / (4 * x + 1e-6)
        z = (Rmat[where_r11_max, 0, 2] + Rmat[where_r11_max, 2, 0]) / (4 * x + 1e-6)
        q_result[where_r11_max, ...] = np.stack((
            [x, y, z, w] if out_format == "xyzw" 
            else [w, x, y, z]
        ), axis=-1)

    where_r22_max = (Rmat[:, 1, 1] == max_diag)
    if where_r22_max.sum() != 0.0:
        y = (np.sqrt(1 + 2 * Rmat[where_r22_max, 1, 1] - Tr[where_r22_max])) / 2
        w = (Rmat[where_r22_max, 0, 2] - Rmat[where_r22_max, 2, 0]) / (4 * y + 1e-6)
        x = (Rmat[where_r22_max, 0, 1] + Rmat[where_r22_max, 1, 0]) / (4 * y + 1e-6)
        z = (Rmat[where_r22_max, 1, 2] + Rmat[where_r22_max, 2, 1]) / (4 * y + 1e-6)
        q_result[where_r22_max, ...] = np.stack((
            [x, y, z, w] if out_format == "xyzw" 
            else [w, x, y, z]
        ), axis=-1)

    where_r33_max = (Rmat[:, 2, 2] == max_diag)
    if where_r33_max.sum() != 0.0:
        z = (np.sqrt(1 + 2 * Rmat[where_r33_max, 2, 2] - Tr[where_r33_max])) / 2
        w = (Rmat[where_r33_max, 1, 0] - Rmat[where_r33_max, 0, 1]) / (4 * z + 1e-6)
        x = (Rmat[where_r33_max, 0, 2] - Rmat[where_r33_max, 2, 0]) / (4 * z + 1e-6)
        y = (Rmat[where_r33_max, 1, 2] - Rmat[where_r33_max, 2, 1]) / (4 * z + 1e-6)
        q_result[where_r33_max, ...] = np.stack((
            [x, y, z, w] if out_format == "xyzw" 
            else [w, x, y, z]
        ), axis=-1)
    
    q_norms = np.linalg.norm(q_result, axis=-1, keepdims=True)
    q_result = (q_result / q_norms)
    return (q_result[0] if not batched else q_result)



def getProjectionMatrix2(znear, zfar, K, W, H):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    top = znear * cy / fy
    bottom = -znear * (H - cy) / fy
    right = znear * (W - cx) / fx
    left = -znear * cx / fx

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = -(right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0, w2c: bool=False):

    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = (t + translate) * scale
    return Rt
    # Rt[:3, :3] = (R if not w2c else R.T)
    # Rt[:3, 3] = (t if not w2c else -(R.T @ t))

    # cam_center = (Rt[:3, 3] + translate) * scale
    # Rt[:3, 3] = cam_center
    # Rt = ColmapNnerf_convertion(Rt)
    # W2C = np.linalg.inv(Rt)
    # return W2C
    

def transform_imgTensor(img: Union[np.ndarray, torch.Tensor], transform_order: str):
    axes_idx_map = {
        "WHC->CWH": (2, 0, 1),
        "WHC->CHW": (2, 1, 0),
        "WHC->HWC": (1, 0, 2),
        "CWH->WHC": (1, 2, 0),
        "CWH->HWC": (2, 1, 0),
        "CWH->CHW": (0, 2, 1),
        "CHW->WHC": (2, 0, 1),
        "CHW->CWH": (0, 2, 1),
        "CHW->HWC": (1, 2, 0),
        "HWC->WHC": (1, 0, 2),
        "HWC->CWH": (2, 0, 1),
        "HWC->CHW": (2, 1, 0),
        "WHC->WHC": (0, 1, 2),
        "CWH->CWH": (0, 1, 2),
        "CHW->CHW": (0, 1, 2),
        "HWC->HWC": (0, 1, 2),
    }
    permute_order = axes_idx_map[transform_order]
    if img.ndim > 3:
        dims_df = img.ndim - 3
        first_axes = list(range(dims_df))
        last_axes = [axis + dims_df for axis in permute_order]
        permute_order = tuple(first_axes + last_axes)
        print(permute_order)

    if transform_order not in axes_idx_map:
        raise ValueError(f"unrecognized convert type: {transform_order:}!!!")
    
    return (
        img.transpose(permute_order) 
        if isinstance(img, np.ndarray) 
        else img.permute(permute_order)
    )

def ColmapNnerf_convertion(
    input: Union[torch.Tensor, np.ndarray], 
    in_type: str, 
    in_mode: Optional[str]=None,
):

    convertion_Mat = np.eye(4)
    convertion_Mat[1, :] *= -1
    convertion_Mat[2, :] *= -1

    if in_type in ["points", "points3D", "pts"]:
        result = input @ convertion_Mat[:3, :3]
    elif in_type in ["view", "viewmat", "viewmatrix"]:
        if in_mode is not None:
            if in_mode == "c2w":
                result = convertion_Mat @ input
            elif in_mode == "w2c":
                input = convertion_Mat @ input
                result = np.linalg.inv(input) @ convertion_Mat
                # inv_conv = np.linalg.inv(convertion_Mat)
                # result = convertion_Mat @ c2w @ convertion_Mat
        else:
            result = convertion_Mat @ input
    
    return result

def C2W_pinhole(
    uv: Union[np.ndarray, torch.Tensor],
    Fx: float, Fy: float, 
    Cx: float, Cy:float
):
    
    tensor_type = "np"
    if isinstance(uv, torch.Tensor):
        tensor_type = "pt"
        uv = uv.detach().cpu().numpy()

    batched = True
    if uv.ndim == 1:
        batched = False
        uv = uv.reshape(1, 2)

    mx = (uv[:, 0] - Cx) / Fx
    my = (uv[:, 1] - Cy) / Fy
    Dir_xyz = np.concatenate([mx, my, np.ones_like(mx)], dim=-1)
    Dir_xyz = Dir_xyz / np.linalg.norm(Dir_xyz, dim=-1, keepdims=True)
    Dir_xyz = (Dir_xyz if batched else Dir_xyz[0])
    Dir_xyz = (Dir_xyz if tensor_type != "pt" else torch.Tensor(Dir_xyz))
    return Dir_xyz

def W2C_pinhole(
    xyz: Union[np.ndarray, torch.Tensor],
    Fx: float, Fy: float,
    Cx: float, Cy: float
):
    tensor_type = "np"
    if isinstance(xyz, torch.Tensor):
        tensor_type = "pt"
        xyz = xyz.detach().cpu().numpy()
    
    batched = True
    if xyz.ndim == 1:
        batched = False
        xyz = xyz.reshape(1, 3)
        
    u = (Fx / xyz[:, 2]) * xyz[:, 0] + Cx 
    v = (Fy / xyz[:, 2]) * xyz[:, 1] + Cy 
    UV = np.concatenate([u, v], dim=-1)
    UV = (UV if batched else UV[0])
    UV = (UV if tensor_type != "pt" else torch.Tensor(UV))
    return UV

