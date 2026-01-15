import torch 
import numpy as np
import math
import copy
from dataclasses import (dataclass, fields, field)
from ..types import *
from ..utils.graphics_utils import getProjectionMatrix2

@dataclass 
class PointCloudInfo:
    xyz: Union[np.ndarray, torch.Tensor]
    rgb: Union[np.ndarray, torch.Tensor]
    normals: Union[np.ndarray, torch.Tensor]
    view_ang_maps: Dict[str, Union[np.ndarray, torch.Tensor]]=None
    initial_scales: Optional[torch.Tensor]=None
    

    def __str__(self):
        report = 32*"=" + "[POINTS INFO]" + 32*"=" + "\n"
        report += f"shape: {self.xyz.shape}, mean: {self.xyz.mean(0)}" + "\n"
        report += f"rgb bounds: [{self.rgb.min(), self.rgb.max()}]" + "\n"
        report += 32*"=" + "[POINTS INFO]" + 32*"=" + "\n"*3
        return report

    def to(self, arg: str) -> Self:
        output = copy.deepcopy(self)
        def check_value(value, arg):
            if value is not None:
                if arg in ["numpy", "array"]:
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                elif arg in ["tensor", "pt"]:
                    if isinstance(value, np.ndarray):
                        value = torch.from_numpy(value).float()
                elif arg in ["cpu", "cuda"]:
                    if isinstance(value, np.ndarray):
                        value = torch.from_numpy(value).float()
                    value = value.to(arg)
                else:
                    raise ValueError(f"unsupported arg type: {arg}")
            return value
                    
        vars_list = ["xyz", "rgb", "normals"]
        for attr in vars_list:
            value = getattr(output, attr)
            value = check_value(value, arg)
            setattr(output, attr, value)
        
        if self.view_ang_maps is not None:
            for (key, value) in self.view_ang_maps.items():
                value = check_value(value, arg)
                self.view_ang_maps[key] = value
        output.view_ang_maps = self.view_ang_maps
        
        return output

    def get_anguler_map_by_view(self, view_idx: int, ret_tensors: str="pt") -> Union[torch.Tensor, np.ndarray]:
        if self.view_ang_maps is not None:
            map = self.view_ang_maps[view_idx]
            if ret_tensors in ["pt", "tensor"]:
                if isinstance(map, np.ndarray):
                    map = torch.from_numpy(map).float()
            elif ret_tensors in ["numpy", "np", "array"]:
                if isinstance(map, torch.Tensor):
                    map = map.detach().cpu().numpy()
            else:
                raise ValueError(f"unsupported tensors type: {ret_tensors}")
            return map
        else:
            raise ValueError("self.view_ang_maps is not set !!")
        
@dataclass
class CameraInfo:
    height: float; width: float
    FocalX: float; FocalY: float
    CentX: float; CentY: float
    R: Union[np.ndarray, torch.Tensor]
    t: Union[np.ndarray, torch.Tensor]
    near: float=0.01
    far: float=100.0
    in_viewformat: Optional[str]="world2cam"
    distortion: Optional[List[float]]=None

    def __post_init__(self) -> None:

        assert (self.R is not None and self.t is not None)
        assert (self.width is not None and self.height is not None)
        assert (self.FocalX is not None 
                and self.FocalY is not None
                and self.CentX is not None
                and self.CentY is not None)
        
        if (self.FocalX is not None
            and self.FocalY is not None
            and self.CentX is not None
            and self.CentY is not None):
            self.K = np.array([
                [self.FocalX, 0.0, self.CentX],
                [0.0, self.FocalY, self.CentY],
                [0.0, 0.0, 1.0]
            ])

        if self.in_viewformat in ["world2cam", "w2c", "W2C"]:
            self.world2cam = np.eye(4)
            self.world2cam[:3, :3] = self.R
            self.world2cam[:3, 3] = self.t
            self.cam2world = np.linalg.inv(self.world2cam)
            self.camcent = self.cam2world[:3, 3]
        elif self.in_viewformat in ["cam2world", "c2w", "C2W"]:
            self.cam2world = np.eye(4)
            self.cam2world[:3, :3] = self.R
            self.cam2world[:3, 3] = self.t
            self.world2cam = np.linalg.inv(self.world2cam)
            self.camcent = self.cam2world[:3, 3]
        else:
            raise ValueError(f"unsupported input view convention: {self.in_viewformat}")

            
        self.projmatrix = getProjectionMatrix2(
            znear=self.near,
            zfar=self.far,
            K=self.K,
            W=self.width, H=self.height
        )
        
        if (self.K is not None):
            self.FovX = 2.0 * math.atan2(self.width / 2.0, self.K[0, 0])
            self.FovY = 2.0 * math.atan2(self.height / 2.0, self.K[1, 1])
    
            
    def __str__(self) -> str:
        
        report = 32 * "=" + "!![CAMERA INFO REPORT]!!" + 32 * "=" + "\n"
        if self.K is not None:
            report += 32 * "=" + "[INTRINSICS]" + 32 * "=" + "\n"
            report += str(self.K) + "\n"
            report += 32 * "=" + "[INTRINSICS]" + 32 * "=" + "\n"
        if self.cam2world is not None:
            report += 32 * "=" + "[EXTRINSICS CAM2WORLD]" + 32 * "=" + "\n"
            report += str(self.cam2world) + "\n"
            report += 32 * "=" + "[EXTRINSICS CAM2WORLD]" + 32 * "=" + "\n"
        if self.world2cam is not None:
            report += 32 * "=" + "[EXTRINSICS WORLD2CAM]" + 32 * "=" + "\n"
            report += str(self.world2cam) + "\n"
            report += 32 * "=" + "[EXTRINSICS WORLD2CAM]" + 32 * "=" + "\n"
        if self.projmatrix is not None:
            report += 32 * "=" + "[PROJMATRIX]" + 32 * "=" + "\n"
            report += str(self.projmatrix) + "\n"
            report += 32 * "=" + "[PROJMATRIX]" + 32 * "=" + "\n"
        if (self.near is not None 
            and self.far is not None
            and self.width is not None
            and self.height is not None):
            report += 32 * "=" + "[OTHER]" + 32 * "=" + "\n"
            report += f"RESOLUTION: [{self.width}, {self.height}]\n"
            report += f"BOUNDS: [near: {self.near}, far: {self.far}]\n"
            report += 32 * "=" + "[OTHER]" + 32 * "=" + "\n"
        report += 32 * "=" + "!![CAMERA INFO REPORT]!!" + 32 * "="
        
        return report
 
    def to(self, arg: str) -> Self:
        output = copy.deepcopy(self)
        fields_to_convert_ = [
            "t", "R", 
            "cam2world","world2cam", 
            "K", "camcent"
        ]
        for field in fields_to_convert_:
            value = getattr(output, field)
            if value is not None:
                if arg in ["pt", "tensor"]:
                    if isinstance(value, np.ndarray):
                        setattr(output, field, torch.from_numpy(value).float())
                elif arg in ["np", "array"]:
                    if isinstance(value, torch.Tensor):
                        setattr(output, field, value.detach().cpu().numpy())
                elif arg in ["cpu", "cuda"]:
                    if isinstance(value, np.ndarray):
                        value = torch.from_numpy(value).float()
                    setattr(output, field, value.to(arg))
                else:
                    raise ValueError("unrecognized arg value !!!")
            
        return output
    


# class SceneInfo:
#     def __init__(
#         self, 
#         source: str, 
#         camr_params: Optional[Dict[str, Any]]=None,
#         ptsr_params: Optional[Dict[str, Any]]=None,
#         dataset: Optional[str]="mip-nerf",
#     ) -> None:
        
#         camr_params = (camr_params if camr_params is not None else {})
#         ptsr_params = (ptsr_params if ptsr_params is not None else {})
#         (self.cameras, self.images) = _CAMERA_READERS_[dataset](source, **camr_params)
#         self.points3D = _POINTS3D_READERS_[dataset](source, **ptsr_params)
#         self.N_views = len(self.cameras)
        
#         camcents = torch.stack([cam.camcent for cam in self.cameras], dim=0)
#         self.cameras_extent = get_camera_extent(camcents)

#         idxs = np.arange(0, self.N_views)
#         train_split, test_split, val_split = np.array_split(idxs, 3)
#         self.splits_ = {
#             "train": train_split,
#             "test": test_split,
#             "validation": val_split 
#         }
        
#     def get_view(self, idx: int) -> Tuple[CameraInfo, torch.Tensor]:
#         return (self.cameras[idx], self.images[idx].squeeze())
    
#     def get_split_views(self, split: Optional[str]=None) -> Tuple[List[CameraInfo], torch.Tensor]:
#         if split is None:
#             return (self.cameras, self.images)
#         else:
#             return (
#                 [self.cameras[idx] 
#                 for idx in self.splits_[split]],
#                 self.images[self.splits_[split], ...]
#             )
    
#     def sample_views(self, n: Optional[int]=2) -> List[CameraInfo]:
#         idxs = np.random.randint(0, len(self.cameras), (n, ))
#         return (
#             [self.cameras[idx] 
#             for idx in idxs],
#             self.images[idxs, ...]
#         )


    


    
