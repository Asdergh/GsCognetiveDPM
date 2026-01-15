import torch
import numpy as np
from ..types import *


def pix2ray_pinhole(
    uv: Union[np.ndarray, torch.Tensor],
    FocalX: float, FocalY: float, 
    CentX: float, CentY:float
):
    
    tensor_type = "np"
    if isinstance(uv, torch.Tensor):
        tensor_type = "pt"
        uv = uv.detach().cpu().numpy()

    batched = True
    if uv.ndim == 1:
        batched = False
        uv = uv.reshape(1, 2)

    mx = (uv[:, 0] - CentX) / FocalX
    my = (uv[:, 1] - CentY) / FocalY
    Dir_xyz = np.concatenate([mx, my, np.ones_like(mx)], dim=-1)
    Dir_xyz = Dir_xyz / np.linalg.norm(Dir_xyz, dim=-1, keepdims=True)
    Dir_xyz = (Dir_xyz if batched else Dir_xyz[0])
    Dir_xyz = (Dir_xyz if tensor_type != "pt" else torch.Tensor(Dir_xyz))
    return Dir_xyz

def ray2pix_pinhole(
    xyz: Union[np.ndarray, torch.Tensor],
    FocalX: float, FocalY: float,
    CentX: float, CentY: float
):
    tensor_type = "np"
    if isinstance(xyz, torch.Tensor):
        tensor_type = "pt"
        xyz = xyz.detach().cpu().numpy()
    
    batched = True
    if xyz.ndim == 1:
        batched = False
        xyz = xyz.reshape(1, 3)
        
    u = (FocalX / xyz[:, 2]) * xyz[:, 0] + CentX 
    v = (FocalY / xyz[:, 2]) * xyz[:, 1] + CentY 
    UV = np.concatenate([u, v], dim=-1)
    UV = (UV if batched else UV[0])
    UV = (UV if tensor_type != "pt" else torch.Tensor(UV))
    return UV

def get_cam2world_grid(
    FocalX: float, FocalY: float,
    CentX: float, CentY: float,
    target_size: Tuple[int, int],
    return_tensors: Optional[str]="np",
) -> None:
    
    (u, v) = np.meshgrid(np.arange(target_size[0]), np.arange(target_size[1]))
    grid = np.stack([u, v], axis=-1)

    mx = (grid[..., 0] - CentX) / FocalX
    my = (grid[..., 1] - CentY) / FocalY
    dir_norm = np.expand_dims(np.sqrt(mx ** 2 + my ** 2 + 1), axis=-1)
    Mxyz = np.stack([mx, my, np.ones_like(mx)], axis=-1)
    dir = (Mxyz / dir_norm)
    
    return (
        dir 
        if return_tensors not in ["pt", "tensor", "torch"] 
        else torch.from_numpy(dir).float()
    )

def get_world2cam_grid(
    points: Union[np.ndarray, torch.Tensor],
    FocalX: float, FocalY: float,
    CentX: float, CentY: float,
    target_size: Tuple[int, int],
    return_tensors: Optional[str]="np",
) -> None:
    
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    Npts = points.shape[0]
    assert (Npts <= (target_size[0] * target_size[1])), \
    (f"points number: {Npts} must fillable to: {target_size} grid size")
    pts_grid = points.reshape(target_size[0], target_size[1], 3)
    (u, v) = np.meshgrid(np.arange(target_size[0]), np.arange(target_size[1]))
    grid = np.stack([u, v], axis=-1)

    xy = (pts_grid[..., :2] / np.expand_dims(pts_grid[..., 2], axis=-1))
    u = (xy[..., 0] / FocalX) + CentX
    v = (xy[..., 1] / FocalY) + CentY
    grid[..., 0] = u
    grid[..., 1] = v
    grid = grid.astype(np.int32)

    return (
        grid
        if return_tensors not in ["pt", "tensor", "torch"]
        else torch.from_numpy(grid).float()
    )
    
    




