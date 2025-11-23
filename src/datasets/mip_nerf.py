import os 
import numpy as np
import torch
import torch.nn as nn
import rerun as rr
import open3d as o3d
import rerun.blueprint as rrb
import random as rd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from nerfstudio.data.utils.colmap_parsing_utils import (
    read_points3D_binary,
    read_cameras_binary,
    read_images_binary
)
from PIL import Image
from dataclasses import fields
from torchvision.transforms import (Compose, 
                                    PILToTensor, 
                                    Resize, Lambda)
from torchvision.transforms.functional import InterpolationMode
from scipy.spatial.transform import Rotation as R
from typing import (Optional, Tuple, Dict, Any)
from rerun import Quaternion
from open3d.geometry import (PointCloud, KDTreeSearchParam as knn_sr)
from open3d.utility import Vector3DVector as vec


_INTERPOLATION_TYPES_ = {
    "bilinear": InterpolationMode.BILINEAR,
    "nearest": InterpolationMode.NEAREST,
    "bicubic": InterpolationMode.BICUBIC,
    "box": InterpolationMode.BOX,
    "hamming": InterpolationMode.HAMMING,
    "lanczos": InterpolationMode.LANCZOS
}
class MipNerfDataset(Dataset):

    def __init__(
        self,
        path: str,
        target_size: Optional[Tuple[int, int]]=None,
        scene_type: Optional[str]="bicyle",
        images_scale: Optional[int]=1,
        interpolation_mode: Optional[str]="bilinear",
        pts_partition_size: Optional[int]=1000,
        pts_partitions_n: Optional[int]=40,
        pts_shuffle: Optional[bool]=True,
        normals_knn: Optional[int]=11
    ) -> None:
        
        super().__init__()

        assert scene_type in os.listdir(path), (f"""
        scene type must one of folders in data_path.
        scene_type: {scene_type}, data_path containment: {os.listdir(path)}
        """)
        path = os.path.join(path, scene_type)
        
        
        assert (images_scale == 1) or (f"images_{images_scale}" in os.listdir(path)), (f"""
        Image with provided scale factor couldn'e be found in data_path folder
        provided images_scale: {images_scale}, scene_path containment: {os.listdir(path)}
        """)
        images_f = (f"images_{images_scale}" if images_scale != 1 else "images")
        images_path = os.path.join(path, images_f)
        self.imgs_transform = Compose([
            Lambda(lambda img_f: Image.open(img_f)),
            PILToTensor(),
            (Resize(target_size) if target_size is not None else nn.Identity()),
            Lambda(lambda img: ((img / 255.0).to(torch.float32) if img.max > 1.0 else img))
        ])

        poses_bin_in = os.path.join(path, "poses_bounded.npy")
        assert os.path.exists(poses_bin_in), (f"CRITICAL ERROR: [bounded poses can't be loaded from {path}]")
        poses_bin_ou = np.load(poses_bin_in)
        self.poses = np.reshape(poses_bin_ou[:, :12], (-1, 3, 5))

        sparse_dpath = os.path.join(path, "sparse")
        assert os.path.exists(sparse_dpath), \
        (f"sparse path is not in scene_path: {os.listdir(path)}")

        cams_params = os.path.join(sparse_dpath, "0/cameras.bin")
        assert os.path.exists(cams_params), ("cameras params file is not int sparse_dpath")
        cams_params = read_cameras_binary(cams_params)
        
        imgs_annots = os.path.join(sparse_dpath, "0/images.bin")
        assert os.path.exists(imgs_annots), ("images annotations file is not int sparse_dpath")
        imgs_annots = read_images_binary(imgs_annots)

        assert os.path.exists(sparse_dpath), \
        (f"points3D.bin file is not in sparse_folder: {os.listdir(sparse_dpath)}")
        pts_f_bin = os.path.join(sparse_dpath, "0/points3D.bin")
        self._parse_points_and_cams(
            path=pts_f_bin,
            scene_path=path,
            images_path=images_path,
            parts_n=pts_partitions_n,
            part_size=pts_partition_size,
            imgs_annots=imgs_annots,
            cams_params=cams_params,
            normals_knn=normals_knn
        )
    
    @property
    def cameras_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.Ks, self.viewmats)
    
    @property
    def cameras_extent(self) -> float:
        cam_dists = torch.norm(self.viewmats[..., :3, 3], dim=-1)
        max_xyz_t = self.viewmats[torch.argmax(cam_dists), :3, 3]
        min_xyz_t = self.viewmats[torch.argmin(cam_dists), :3, 3]
        return torch.norm(max_xyz_t - min_xyz_t)
    
    @property
    def points_attrs(self) -> Dict[str, Any]:
        return {
            "xyz": torch.Tensor(self.o3d_pcd.points),
            "rgb": torch.Tensor(self.o3d_pcd.colors),
            "normals": torch.Tensor(self.o3d_pcd.normals)
        }
    
    def _parse_points_and_cams(self, path, scene_path, images_path,
                               part_size, parts_n, shuffle, 
                               imgs_annots, cams_params,
                               t_img_s, normals_knn) -> None:
        
        Np = part_size * parts_n
        pts_collection = read_points3D_binary(path)
        N = len(pts_collection)
        assert (N <= Np), (f"points lesst then partition_size * partitions_n")
        if shuffle:
            pts_ids = [idx for idx in range(0, N)]
        
        
        imgs_ids = set()
        xyz_buffer_ = np.zeros((Np, 3))
        rgb_buffer_ = np.zeros((Np, 3))
        current_pts_idx = 0
        with tqdm(
            total=parts_n,
            colour="white",
            desc="Collecting Pts Data..."
        ) as out_bar:
            for idx in range(parts_n):
                if shuffle:
                    idx = rd.choice(pts_ids)
                    del pts_ids[idx * part_size: (idx + 1) * part_size]
                with tqdm(
                    total=part_size,
                    desc=f"Partition: {idx}, from: {parts_n}...",
                    colour="green",
                    ascii=":>"
                ) as inner_bar:
                    for pts_idx in range(idx * part_size, (idx + 1) * part_size):
                        pts_3D = pts_collection[pts_idx]
                        xyz_buffer_[current_pts_idx] = pts_3D.xyz
                        rgb_buffer_[current_pts_idx] = pts_3D.rgb
                        imgs_ids.update(pts_3D.image_ids)
                        current_pts_idx += 1
                        inner_bar.update(1)
                out_bar.update(1)
        
        self.o3d_pcd = o3d.geometry.PointCloud()
        self.o3d_pcd.points = vec(xyz_buffer_)
        self.o3d_pcd.colors = vec(rgb_buffer_)
        self.o3d_pcd.estimate_normals(knn_sr(normals_knn))
        
        imgs_N = len(imgs_ids)
        self.Ks = torch.zeros((imgs_N, 3, 3))
        self.viewmats = torch.zeros(imgs_N, 4, 4)
        self.imgs_fs = []
        with tqdm(
            total=len(imgs_ids),
            desc="Collecting Images from annotations ...",
            colour="white"
        ) as annots_bar:
            for idx in imgs_ids:
                img_annot = imgs_annots[idx]
                cam_params = cams_params[img_annot.camera_id]
                camK = cam_params.params
                scale = (0.05 * (t_img_s[0] + t_img_s[1])) / (0.5 * (cam_params.height + cam_params.width))
                K = scale * torch.Tensor([
                    [camK[0], 0.0, camK[-2]],
                    [0.0, camK[1], camK[-2]],
                    [0.0, 0.0, 1.0]
                ])
                viewmat = torch.eye(4)
                viewmat[:3, :3] = self.poses[img_annot.id - 1, :3, :3]
                viewmat[:3, 3] = self.poses[img_annot.id - 1, :3, 3]

                self.Ks[idx - 1, ...] = K
                self.viewmats[idx - 1, ...] = viewmat
                self.imgs_fs.append(os.path.join(images_path, img_annot.name))
                annots_bar.update(1)
        
        def __len__(self) -> int:
            return len(self.imgs_fs)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

            img_f = self.imgs_fs[idx]

            



            
        

                
    
    


        
            

cameras_path = "/home/ram/Downloads/360_v2/bicycle/sparse/0/cameras.bin"
images_path = "/home/ram/Downloads/360_v2/bicycle/sparse/0/images.bin"
rgb_imgs = "/home/ram/Downloads/360_v2/bicycle/images_8"
rgb_imgs_f = os.listdir(rgb_imgs)

cams = read_cameras_binary(cameras_path)[1]
camsK = cams.params
imgs = read_images_binary(images_path)
target_size = (112, 224)
scale = (0.5 * (target_size[0] + target_size[1])) / (0.5 * (cams.height + cams.width))


img_tf = Compose([
    Lambda(lambda img_f: Image.open(img_f)),
    PILToTensor(),
    Resize(target_size),
    Lambda(lambda img: ((
        img / 255.0).to(torch.float32).permute(1, 2, 0) if img.max() > 1.0 
        else img.permute(1, 2, 0)
    ))
])
origin = "origin_world"
K = np.array([
    [camsK[0], 0.0, camsK[-2]],
    [0.0, camsK[1], camsK[-1]],
    [0.0, 0.0, 1.0]
]) * scale
rr.init(origin, spawn=True)
for idx, (k, img_f) in enumerate(zip(imgs, rgb_imgs_f)):
    img_f = os.path.join(rgb_imgs, img_f)
    img = img_tf(img_f).numpy()
    cam = imgs[k]
    print(cam.name, cam.camera_id, cam.id)
    break
    # print(type(cam.qvec2rotmat), type(cam.tvec))
    rr.set_time("time", sequence=idx)
    rr.log(
        f"{origin}/Frame{k}",
        rr.Transform3D(
            translation=cam.tvec,
            quaternion=Quaternion(xyzw=cam.qvec)
        ),
        rr.Pinhole(image_from_camera=K),
        rr.Image(img)
    )

blueprint = rrb.Blueprint(
    rrb.Spatial3DView(
        origin=origin
    )
)
rr.send_blueprint(blueprint)




                    
                    
            


