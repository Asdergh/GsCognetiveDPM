import torch
import numpy as np
import pandas as pd
import os
import cv2
from dataclasses import (dataclass, field)
from .base import (
    BaseIterableDatasetConfig,
    BaseIterableDataset,
    register_dataset
)
from ..types import *
from ..scene.scene_objects import (PointCloudInfo, CameraInfo)
from ..utils.graphics_utils import quat2Rmat
from ..utils.projection_funcs import get_cam2world_grid

@dataclass
class ICSLDatasetConfig(BaseIterableDatasetConfig):
    target_size: Optional[Tuple[int, int]]=None



# TODO rethink the architecture of this dataset 

@register_dataset("icl-indoore")
class ICLDataset(BaseIterableDataset):

    def __init__(self, config) -> None:
        self.cfg = config
        self._load_poses()
        self._load_camparams()
        super().__init__(config)
    
    def _load_camparams(self) -> None:
        
        cammodelf = os.path.join(self.cfg.source, "camerainfo.txt")
        with open(cammodelf, "r") as file:
            data = file.readlines()
            (self.width, self.height) = map(int, data[1].replace("\n", "").split(" "))
            (self.FocalX, self.FocalY) = map(float, data[3].replace("\n", "").split(" "))
            (self.CentX, self.CentY) = map(float, data[5].replace("\n", "").split(" "))
        
        if self.cfg.target_size is not None:
            Sx = (self.width / self.cfg.target_size[0])
            Sy = (self.height / self.cfg.target_size[1])
            self.FocalX *= Sx; self.FocalY *= Sy
            self.CentX *= Sx; self.CentY *= Sx
            (self.width, self.height) = self.cfg.target_size
            
            
    def _load_poses(self) -> None:

        posesf = os.path.join(self.cfg.source, "poses.gt")
        assert (os.path.exists(posesf)), \
        (f"coun't  find any poses file at location: {posesf}")

        poses_df = pd.read_csv(posesf)
        quaternions = poses_df.iloc[:, 4:8].to_numpy()

        self.translations_ = poses_df.iloc[:, 1:4].to_numpy()
        self.Rmats_ = quat2Rmat(quaternions)
        self.timessptems = {key_t: idx for (idx, key_t) in enumerate(poses_df.iloc[:, 0])}

    def _load_data(self) -> None:

        def check_visual_sources(dsource: str) -> Tuple[str, str]:
            dsource = os.path.join(self.cfg.source, dsource)
            data = os.path.join(dsource, "data")
            annots_csv = os.path.join(dsource, "data.csv")
            for path in [dsource, data, annots_csv]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"coun'd find needed data at location: {path}")
            
            return (data, annots_csv)
                
        (depths_source, depths_annots) = check_visual_sources("depth0")
        (rgbs_source, rgbs_annots) = check_visual_sources("cam0")
        
        
        depth_df = pd.read_csv(depths_annots)
        timestamps = depth_df.iloc[:, 0]
    
        index_stack = np.asarray([self.timessptems[key_t] for key_t in timestamps])
        Rmats = self.Rmats_[index_stack, ...]
        translations = self.translations_[index_stack, np.newaxis, :]
        depths = np.stack([
                cv2.resize(
                    cv2.imread(
                    os.path.join(depths_source, depthf),
                    cv2.IMREAD_GRAYSCALE
                ),
                (self.width, self.height)
            )[..., np.newaxis]
            for depthf in depths_annots.iloc[:, 1]
        ], axis=0)
        images_rgb = np.stack([
                cv2.resize(
                    cv2.imread(
                    os.path.join(rgbs_source, rgbf)
                ),
                (self.width, self.height)
            )[..., np.newaxis]
            for rgbf in rgbs_annots[rgbs_annots.iloc[:, 0].isin(timestamps)].iloc[:, 0]
        ], axis=0)

        direction_grid = get_cam2world_grid(
            target_size=(self.width, self.height),
            FocalX=self.FocalX, FocalY=self.FocalY,
            CentX=self.CentX, CentY=self.CentY,
            return_tensors="np"
        )[np.newaxis, ...]
        points_xyz = (direction_grid * depths).reshape(-1, (self.width * self.height), 3)
        points_xyz += translations
        points_xyz = np.einsum("nij, npj -> npi", Rmats, points_xyz)
        points_rgb = (images_rgb.reshape(-1, (self.width * self.height), 3).astype(np.float32) / 255.0)
        
        
    def load_points3D(self):
        
        pass
        # return PointCloudInfo(xyz=points)

    def load_cameras(self):
        pass

    



if __name__ == "__main__":

    cfg = ICSLDatasetConfig(
        source="/media/ram/T71/deer_robot",
        target_size=(224, 224)
    )
    dataset = ICLDataset(cfg)
