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
from open3d.geometry import (PointCloud, KDTreeSearchParamHybrid as knn_sr)
from open3d.utility import Vector3dVector as vec


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
        scene_type: Optional[str]="bicycle",
        images_scale: Optional[int]=1,
        pts_partition_size: Optional[int]=1000,
        pts_partitions_n: Optional[int]=40,
        pts_shuffle: Optional[bool]=False,
        pts_scale: Optional[float]=1.0,
        normal_knn: Optional[int]=11,
        normal_radii: Optional[float]=0.1
    ) -> None:
        
        super().__init__()
        self.pts_scale = pts_scale
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
            Lambda(lambda img: ((img / 255.0).to(torch.float32) if img.max() > 1.0 else img))
        ])

        poses_bin_in = os.path.join(path, "poses_bounds.npy")
        assert os.path.exists(poses_bin_in), (f"CRITICAL ERROR: [bounded poses can't be loaded from {path}]")
        poses_bin_ou = np.load(poses_bin_in)
        self.poses = np.reshape(poses_bin_ou[:, :15], (-1, 3, 5))

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
            images_path=images_path,
            parts_n=pts_partitions_n,
            shuffle=pts_shuffle,
            part_size=pts_partition_size,
            imgs_annots=imgs_annots,
            cams_params=cams_params,
            normal_knn=normal_knn,
            normal_radii=normal_radii,
            t_img_s=target_size
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
            "xyz": torch.Tensor(self.o3d_pcd.points) * self.pts_scale,
            "rgb": torch.Tensor(self.o3d_pcd.colors) / 255.0,
            "normals": torch.Tensor(self.o3d_pcd.normals),
            "cameras_extent": torch.tensor(self.cameras_extent),
            "viewpointsN": len(self.imgs_fs)
        }
    
    def _parse_points_and_cams(self, path, images_path,
                               part_size, parts_n, shuffle, 
                               imgs_annots, cams_params,
                               t_img_s, normal_knn, normal_radii) -> None:
        
        Np = part_size * parts_n
        pts_collection = read_points3D_binary(path)
        N = len(pts_collection)
        print(f"POINTS FOUND IN FILE: {N}")
        assert (N >= Np), (f"points lesst then partition_size * partitions_n")
        pts_ids = [idx for idx in range(0, int(N // part_size))]
        
        
        imgs_ids = set()
        xyz_buffer_ = np.zeros((Np, 3))
        rgb_buffer_ = np.zeros((Np, 3))
        current_pts_idx = 0
        with tqdm(
            total=parts_n,
            colour="white",
            desc="Collecting Pts..."
        ) as out_bar:
            for idx in pts_ids:
                if shuffle:
                    idx = rd.choice(pts_ids)
                    del pts_ids[pts_ids.index(idx)]

                with tqdm(
                    total=part_size,
                    desc=f"Partition: {idx}...",
                    colour="green",
                    ascii=":>"
                ) as inner_bar:
                    for pts_idx in range(idx * part_size, (idx + 1) * part_size):
                        try:
                            pts_3D = pts_collection[pts_idx]
                            xyz_buffer_[current_pts_idx] = pts_3D.xyz
                            rgb_buffer_[current_pts_idx] = pts_3D.rgb
                            imgs_ids.update(pts_3D.image_ids)
                            current_pts_idx += 1
                        except BaseException:
                            pass
                        inner_bar.update(1)
                out_bar.update(1)
        
        self.o3d_pcd = o3d.geometry.PointCloud()
        self.o3d_pcd.points = vec(xyz_buffer_)
        self.o3d_pcd.colors = vec(rgb_buffer_)
        self.o3d_pcd.estimate_normals(knn_sr(radius=normal_radii, max_nn=normal_knn))
        
        imgs_N = len(imgs_ids)
        self.Ks = np.zeros((imgs_N, 3, 3))
        self.viewmats = np.zeros((imgs_N, 4, 4))
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
                
                scale = (0.5 * (t_img_s[0] + t_img_s[1])) / (0.5 * (cam_params.height + cam_params.width))
                K = scale * np.array([
                    [camK[0], 0.0, camK[-2]],
                    [0.0, camK[1], camK[-2]],
                    [0.0, 0.0, 1.0]
                ])
                viewmat = np.eye(4)
                Rmat = self.poses[img_annot.id - 1, :3, :3]
                Rmat[:, -1] *= -1
                Rmat[:, 0] *= -1
                Translation = self.poses[img_annot.id - 1, :3, 3]
                
                viewmat[:3, :3] = Rmat
                viewmat[:3, 3] = Translation

                try:
                    self.Ks[idx - 1, ...] = K
                    self.viewmats[idx - 1, ...] = viewmat
                    self.imgs_fs.append(os.path.join(images_path, img_annot.name))
                    annots_bar.update(1)
                
                except:
                    pass

        self.viewmats = torch.Tensor(self.viewmats)
        self.Ks = torch.Tensor(self.Ks)
        
    def __len__(self) -> int:
        return len(self.imgs_fs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_f = self.imgs_fs[idx]
        img = self.imgs_transform(img_f)
        return {
            "gt-rgb": img,
            "viewmats": self.viewmats[idx],
            "Ks": self.Ks[idx]
        }

            



# if __name__ == "__main__":
    
#     from scipy.spatial.transform import Rotation as R
#     path = "/home/ram/Downloads/360_v2"
#     dataset = MipNerfDataset(
#         path=path,
#         target_size=(112, 224),
#         normal_knn=30,
#         pts_scale=1.0,
#         normal_radii=0.1,
#         scene_type="counter",
#         pts_partition_size=10000,
#         pts_partitions_n=10
#     )
#     loader = DataLoader(
#         dataset=dataset,
#         batch_size=1,
#         shuffle=False
#     )
#     sample = next(iter(loader))
#     print(sample["gt-rgb"].size(), sample["viewmats"].size(), sample["Ks"].size())
#     print(sample["Ks"], sample["viewmats"])
#     print(dataset.cameras_extent)
#     pts_attr = dataset.points_attrs
#     origin = "origin_main"
#     rr.init(f"{origin}", spawn=True)
#     Ox = np.array([1.0, 0.0, 0.0])
#     Oy = np.array([0.0, 1.0, 0.0])
#     Oz = np.array([0.0, 0.0, 1.0])
    
#     Rx = R.from_rotvec(Ox * 90.0, degrees=True).as_matrix()
#     Rmat = Rx
#     for idx, sample in enumerate(loader):
#         if idx == 20:
#             break
#         rr.set_time("time", sequence=idx)
#         rr.log(
#             f"{origin}/Frame{idx}",
#             rr.Transform3D(
#                 translation=sample["viewmats"].squeeze()[:3, 3],
#                 mat3x3=sample["viewmats"].squeeze()[:3, :3]
#             ),
#             rr.Pinhole(image_from_camera=sample["Ks"].squeeze()),
#             rr.Image(sample["gt-rgb"].squeeze().permute(1, 2, 0))
#         )
#     rr.log(
#         f"{origin}/RgbPts",
#         rr.Points3D(
#             positions=pts_attr["xyz"],
#             colors=pts_attr["rgb"],
#             radii=0.003
#         ),
#     )
#     rr.log(
#         f"{origin}/NormalPts",
#         rr.Points3D(
#             positions=pts_attr["xyz"],
#             colors=pts_attr["normals"],
#             radii=0.003
#         )
#     )
#     blueprint = rrb.Blueprint(
#         rrb.Spatial3DView(origin=origin)
#     )
#     rr.send_blueprint(blueprint=blueprint)
     

# cameras_path = "/home/ram/Downloads/360_v2/bicycle/sparse/0/cameras.bin"
# images_path = "/home/ram/Downloads/360_v2/bicycle/sparse/0/images.bin"
# rgb_imgs = "/home/ram/Downloads/360_v2/bicycle/images_8"
# rgb_imgs_f = os.listdir(rgb_imgs)

# cams = read_cameras_binary(cameras_path)[1]
# camsK = cams.params
# imgs = read_images_binary(images_path)
# target_size = (112, 224)
# scale = (0.5 * (target_size[0] + target_size[1])) / (0.5 * (cams.height + cams.width))


# img_tf = Compose([
#     Lambda(lambda img_f: Image.open(img_f)),
#     PILToTensor(),
#     Resize(target_size),
#     Lambda(lambda img: ((
#         img / 255.0).to(torch.float32).permute(1, 2, 0) if img.max() > 1.0 
#         else img.permute(1, 2, 0)
#     ))
# ])
# origin = "origin_world"
# K = np.array([
#     [camsK[0], 0.0, camsK[-2]],
#     [0.0, camsK[1], camsK[-1]],
#     [0.0, 0.0, 1.0]
# ]) * scale
# rr.init(origin, spawn=True)
# for idx, (k, img_f) in enumerate(zip(imgs, rgb_imgs_f)):
#     img_f = os.path.join(rgb_imgs, img_f)
#     img = img_tf(img_f).numpy()
#     cam = imgs[k]
#     print(cam.name, cam.camera_id, cam.id)
#     break
#     # print(type(cam.qvec2rotmat), type(cam.tvec))
#     rr.set_time("time", sequence=idx)
#     rr.log(
#         f"{origin}/Frame{k}",
#         rr.Transform3D(
#             translation=cam.tvec,
#             quaternion=Quaternion(xyzw=cam.qvec)
#         ),
#         rr.Pinhole(image_from_camera=K),
#         rr.Image(img)
#     )

# blueprint = rrb.Blueprint(
#     rrb.Spatial3DView(
#         origin=origin
#     )
# )
# rr.send_blueprint(blueprint)




                    
                    
            


