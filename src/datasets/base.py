import torch 
import numpy as np
import os
from torch.utils.data import IterableDataset
from dataclasses import (dataclass, field)
from abc import ABC, abstractmethod
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
    

@register_dataset("base")
class BaseIterableDataset(ABC, IterableDataset):
    
    def __init__(self, config) -> None:
        self.cfg = config
        self.load_points3D()
        self.load_cameras()
    
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
    def load_cameras(self) -> Dict[int, Any]:
        """
        This functions collects all the information 
        from dataset releated with cameras, intrinsics 
        and extrinsics collections as well as rgb images

        Returns
        -------
        Dict[str, Any]: The collection with all information 
        neede to formulate sampling batch
        """
        raise NotImplementedError


    @abstractmethod
    def load_points3D(self) -> PointCloudInfo:
        """
        This functions loads points cloud data
        from dataset, and calculate some geometrical
        information like: normals, view-dependent angle maps
        and e.t.c

        Returns
        -------
        PointCloudInfo: the collection with all 
        attributes listed in description
        """
        raise NotImplementedError
    
    @abstractmethod
    def collate(self, batch) -> Dict[str, Any]:
        """
        This functions collects all
        data from each dataset into 
        reuslting batch for training 
        
        :param self: 
        :return: The colleciton of all data that needs for 
                rasterization:
                    [cameras]: CameraInfo object storage 
                    [view_idx]: Stack of view indexes
                    [images]: Tensor of gt images in (B, C, W, H) shape format 
        :rtype: Dict[str, Any]
        """
    # def collate(self, batch) -> Dict[str, Any]:
        
    #     batch = {}
    #     batch["images"] = []
    #     batch["cameras"] = []
    #     batch["view_idx"] = []
    #     for _ in range(self.cfg.batch_size):
            
    #         idx = self.view_index_stack.pop()
    #         if not self.view_index_stack:
    #             self.view_index_stack = np.random.choice(
    #                 self.viewpoints_pkg["view_index_stack"], 
    #                 size=self.cfg.n_views
    #             ).tolist()
    #             idx = self.view_index_stack.pop()
            
    #         camera = self.viewpoints_pkg["cameras"][idx]
    #         image = self.viewpoints_pkg["images"][idx]
    #         batch["images"].append(image)
    #         batch["cameras"].append(camera)
    #         batch["view_idx"].append(idx)
        
    #     batch["images"] = torch.stack(batch["images"], dim=0)
    #     batch["view_idx"] = torch.Tensor(batch["view_idx"])
    #     return batch


    def __iter__(self):
        while True:
            yield {}


   
    
   
    
            

            
            
        
    
    
    

