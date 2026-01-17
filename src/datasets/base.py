import torch 
import numpy as np
import os
from torch.utils.data import IterableDataset
from dataclasses import (dataclass, field)
from abc import ABC, abstractmethod
from open3d.utility import Vector3dVector as vec3d
from open3d.geometry import (
    PointCloud, 
    KDTreeSearchParamHybrid as kdtree_hyb
)
from ..types import *
from ..scene.scene_objects import (PointCloudInfo, CameraInfo)

_DATASETS_REGISTRY: Dict[str, IterableDataset] = {}
def register_dataset(name: str):
    def decorator(cls: Any):
        if not (IterableDataset in cls.__mro__):
            raise TypeError(f"cls: {cls.__mro__[2]} must be subclass of IterableDataset")
        if name in _DATASETS_REGISTRY:
            raise KeyError(f"class: {name} if already in registry: {_DATASETS_REGISTRY}")
        _DATASETS_REGISTRY[name] = cls
        return cls
    return decorator



@dataclass
class BaseIterableDatasetConfig:
    source: str
    type: Optional[str]=None
    target_size: Tuple[int, int]=None
    batch_size: Optional[int]=32
    device: Optional[str]="cpu"
    n_views: Optional[int]=None
    n_points_max: Optional[int]=None
    split_parts: Optional[List[int]]=None
    outlier_knn_trashhold: Optional[float]=100
    outlier_radii_trashhold: Optional[float]=0.05
    cameras_scale: Optional[float]=1.0
    points_scale: Optional[float]=1.0
    normals_searching_rad: Optional[float]=0.1
    normals_searching_nns: Optional[int]=30
    near: Optional[float]=0.1
    far: Optional[float]=100.0
    

@register_dataset("base")
class BaseIterableDataset(ABC, IterableDataset):
    
    def __init__(self, config) -> None:
        self.cfg = config

    def preprocess_point_cloud(self, xyz: np.ndarray) -> np.ndarray:

        pcd = PointCloud()
        pcd.points = vec3d(xyz)
        pcd.remove_radius_outlier(
            self.cfg.outlier_knn_trashhold, 
            self.cfg.outlier_radii_trashhold
        )
        pcd.estimate_normals(search_param=kdtree_hyb(
            self.cfg.normals_searching_rad,
            self.cfg.normals_searching_nns
        ))
        return np.asarray(pcd.points), np.asarray(pcd.normals)
    
    def build_splits(self, all_index: np.ndarray):
        if self.cfg.split_parts is not None:

            assert (sum(list(self.cfg.split_parts.values())) > 100.0), ("inputs pslit percents must be normalized to 100%")
            n = len(self.cfg.split_parts)
            names = [key for key in self.cfg.split_parts.keys()]

            split_parts = np.asarray([part for part in self.cfg.split_parts.values()])
            sort_idx = np.argsort(split_parts)
            used_volume = 0
            for idx in sort_idx:

                split_size = int((self.cfg.n_view2load * split_parts[idx]) / 100.0)
                index_stack = all_index[used_volume:(split_size + used_volume)]
                used_volume += split_size
                splits_index_stacks[names[idx]] = index_stack
            
            used_volume_part = int((used_volume / n) * 100)
            if used_volume_part != 100:
                names = set(["train", "validation", "test"] + names)
                if len(names) == 2:
                    (split1, split2) = np.array_split(all_index[used_volume:], 2)
                    splits_index_stacks[names[0]] = split1
                    splits_index_stacks[names[1]] = split2
                else:
                    splits_index_stacks[names[0]] = all_index[used_volume:]

        else:
            print(type(all_index), all_index)
            (train, val, test) = np.array_split(all_index, 3, axis=0)
            splits_index_stacks = {
                "train": train,
                "validation": val,
                "test": test
            }   
        return splits_index_stacks                  
                    

    @abstractmethod
    def collate(self) -> None:
        """
        Docstring for collate
        
        :param self: Description
        """


    def __iter__(self):
        while True:
            yield {}


   
    
   
    
            

            
            
        
    
    
    

