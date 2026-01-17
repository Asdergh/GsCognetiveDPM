import torch
import numpy as np
import pandas as pd
import os
import cv2
import random as rd
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
    map_scale: Optional[float]=8.0
    frames_step: Optional[int]=10


@register_dataset("icl-indoore")
class ICLDataset(BaseIterableDataset):

    def __init__(self, config) -> None:
        self.cfg = config
        self.check_sources()
        super().__init__(config)
    

    def check_sources(self) -> None:
        self.paths: Dict[str, str] = {
            "depths_data": os.path.join(self.cfg.source, "depth0/data"),
            "depths_annotations": os.path.join(self.cfg.source, "depth0/data.csv"),
            "images_data": os.path.join(self.cfg.source, "cam0/data"),
            "images_annotations": os.path.join(self.cfg.source, "cam0/data.csv"),
            "poses": os.path.join(self.cfg.source, "poses.gt"),
            "camerainfo": os.path.join(self.cfg.source, "camerainfo.txt")
        }
        for (key, val) in self.paths.items():
            if not os.path.exists(val):
                raise FileNotFoundError(f"{key} data wasn't found at location: {val}")
        
        self._load_camparams()
        self._load_poses()
        depths_df = pd.read_csv(self.paths["depths_annotations"])
        rgbs_df = pd.read_csv(self.paths["images_annotations"])
        self.timestamps2depth_names = dict(zip(depths_df.iloc[:, 0], depths_df.iloc[:, 1]))
        self.timestamps2images_names = dict(zip(rgbs_df.iloc[:, 0], rgbs_df.iloc[:, 1]))
        
        

    def _load_camparams(self) -> None:
        
        cammodelf = self.paths["camerainfo"]
        with open(cammodelf, "r") as file:
            data = file.readlines()
            (self.width, self.height) = map(int, data[1].replace("\n", "").split(" "))
            (self.FocalX, self.FocalY) = map(float, data[3].replace("\n", "").split(" "))
            (self.CentX, self.CentY) = map(float, data[5].replace("\n", "").split(" "))
        
        if self.cfg.target_size is not None:
            Sx = (self.cfg.target_size[0] / self.width)
            Sy = (self.cfg.target_size[1] / self.height)
            self.FocalX *= Sx; self.FocalY *= Sy
            self.CentX *= Sx; self.CentY *= Sy
            (self.width, self.height) = self.cfg.target_size
        
            
    def _load_poses(self) -> None:

        posesf = self.paths["poses"]
        assert (os.path.exists(posesf)), \
        (f"coun't  find any poses file at location: {posesf}")
        poses_df = pd.read_csv(posesf)

        T1 = np.diag([1, 1, -0.5])
        quaternions = poses_df.iloc[:, 4:8].to_numpy()
        translations = poses_df.iloc[:, 1:4].to_numpy()
        Rmats = quat2Rmat(quaternions)

        self.timestamps_stack = poses_df.iloc[::self.cfg.frames_step, 0].to_numpy()
        self.current_timestamps_collection = self.timestamps_stack[:self.cfg.batch_size].tolist()
        self.current_timestamps_collection = self.current_timestamps_collection[::-1]
        self.c_tile = 0
        self.poses = {
            key_t: (T1 @ Rmats[idx], translations[idx]) 
            for (idx, key_t) in enumerate(poses_df.iloc[:, 0])
        }

    
    def get_map_at_timestamp(self, timestamp: int, map_type: Optional[str]="depths") -> np.ndarray:

        if not map_type in ["depths", "images"]:
            raise ValueError(f"unknown map type: {map_type}, only [depths, rgbs] allowed")
    
        if map_type == "depths":
            map_name = self.timestamps2depth_names[timestamp] 
            map_file = os.path.join(self.paths["depths_data"], map_name)
            map = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
            # map = (map.astype(np.float32) / 10.0)
        else:
            map_name = self.timestamps2images_names[timestamp] 
            map_file = os.path.join(self.paths["images_data"], map_name)
            map = cv2.imread(map_file)
            map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
            map = (map.astype(np.float32) / 255.0)
        
        map = cv2.resize(map, (self.width, self.height))
        return (map if map_type == "images" else map[..., np.newaxis])

    def get_viewstate_at_timestamp(self, timestamp: int) -> Dict[str, Any]:

        (Rmat, translation) = self.poses[timestamp]
        depth_map = self.get_map_at_timestamp(timestamp, "depths")
        rgb_map = self.get_map_at_timestamp(timestamp, "images")
        # direction_grid = get_cam2world_grid(
        #     FocalX=self.FocalX, FocalY=self.FocalY,
        #     CentX=self.CentX, CentY=self.CentY,
        #     target_size=(self.width, self.height),
        #     return_tensors="np"
        # )
        # points_xyz = (direction_grid * depth_map).reshape((self.width * self.height), 3)
        # points_xyz = np.einsum("ni, ij-> nj", points_xyz, Rmat)
        # points_xyz += translation[np.newaxis, ...]
        # points_rgb = rgb_map.copy().reshape((self.width * self.height), 3)
        return {
            "extrinsics": (Rmat, translation),
            "image": rgb_map,
            "depth": depth_map,
            # "point_cloud": (points_xyz, points_rgb)
        }
        
    def collate(self, batch) -> Dict[str, Any]:

        batch = {}
        batch["cameras"] = []
        batch["depths"] = []
        batch["images"] = []
        batch["points_xyz"] = []
        batch["points_rgb"] = []
        batch["view_index_stack"] = []
        for _ in range(self.cfg.batch_size):
            
            if not self.current_timestamps_collection:
                self.c_tile += 1
                if (self.c_tile + 1) * self.cfg.n_views >= self.timestamps_stack.shape[0]:
                    self.c_tile = 0
                self.current_timestamps_collection = \
                    self.timestamps_stack[
                        self.c_tile * self.cfg.batch_size: 
                        (self.c_tile + 1) * self.cfg.batch_size
                    ].tolist()[::-1]
                idx = self.current_timestamps_collection.pop()
            
            else:
                idx = self.current_timestamps_collection.pop()
            
            view_sample = self.get_viewstate_at_timestamp(idx)
            batch["depths"].append(torch.Tensor(view_sample["depth"]).permute(2, 0, 1))
            batch["images"].append(torch.Tensor(view_sample["image"]).permute(2, 0, 1))
            # batch["points_xyz"].append(torch.Tensor(view_sample["point_cloud"][0]))
            # batch["points_rgb"].append(torch.Tensor(view_sample["point_cloud"][1]))
            batch["cameras"].append(
                CameraInfo(
                    FocalX=self.FocalX,
                    FocalY=self.FocalY,
                    CentX=self.CentX,
                    CentY=self.CentY,
                    width=self.width, height=self.height,
                    near=self.cfg.near, far=self.cfg.far,
                    R=view_sample["extrinsics"][0], t=view_sample["extrinsics"][1]
                )
            )
            batch["view_index_stack"].append(idx)
        
        batch["depths"] = torch.stack(batch["depths"], dim=0)
        batch["images"] = torch.stack(batch["images"], dim=0)
        # batch["points_xyz"] = torch.flatten(torch.stack(batch["points_xyz"], dim=0), end_dim=-2)
        # batch["points_rgb"] = torch.flatten(torch.stack(batch["points_rgb"], dim=0), end_dim=-2)
        batch["view_index_stack"] = np.array(batch["view_index_stack"])
        return batch



        

if __name__ == "__main__":

    import rerun as rr
    import rerun.blueprint as rrb

    from torch.utils.data import DataLoader
    cfg = ICSLDatasetConfig(
        source="/media/ram/T71/deer_robot",
        n_views=1000,
        target_size=(224, 224),
        cameras_scale=0.1,
        batch_size=1000,
        frames_step=10
    )
    dataset = ICLDataset(cfg)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,
        num_workers=0,
        collate_fn=dataset.collate
    )

    
    sample = next(iter(dataloader))
    # print(sample["points_xyz"].size())
    # print(sample["points_rgb"].size())
    origin = "origin"
    rr.init(origin, spawn=True)
    # rr.log(
    #     f"{origin}/PointCloud",
    #     rr.Points3D(
    #         positions=sample["points_xyz"],
    #         colors=sample["points_rgb"],
    #         radii=[0.003]
    #     )
    # )
    for idx, cam in enumerate(sample["cameras"]):
        rr.log(
            f"{origin}/World2Cam-{idx}",
            rr.Transform3D(
                translation=cam.cam2world[:3, 3],
                mat3x3=cam.cam2world[:3, :3]
            ),
            rr.Pinhole(
                image_from_camera=cam.K,
                width=cam.width,
                height=cam.height,
                color=[0, 1., 0]
            ),
            rr.Image(sample["images"][idx].permute(1, 2, 0))
        )

    blueprint = rrb.Blueprint(rrb.Spatial3DView(origin=origin))
    rr.send_blueprint(blueprint)
    # sample = next(iter(dataloader))
    # print(sample["images"].size(), sample["view_idx"].size(), sample["cameras"][0])
    
