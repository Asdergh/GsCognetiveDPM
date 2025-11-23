import torch 
import numpy as np
import open3d 
import os
import rerun as rr
import pandas as pd

from moviepy.video.io.VideoFileClip import VideoFileClip
from typing import (
    Union,
    Optional,
    List,
    Dict,
    Tuple
)
from PIL import Image
from open3d.geometry import (
    PointCloud,
    KDTreeSearchParamKNN,
    KDTreeSearchParamHybrid
)
from open3d.io import write_point_cloud
from open3d.utility import Vector3dVector as vec

from torchvision.transforms import functional as Fv
from torchvision.io import read_video

from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm




class BasicPointCloudScene:

    def __init__(
        self, 
        width: Optional[int]=112, 
        height: Optional[int]=112,
        n_neighbors: Optional[int]=3
    ) -> None:

        self.w, self.h = (width, height)
        self.pts = torch.empty(0)
        self.normals = torch.empty(0)
        self.colors = torch.empty(0)     
        self.nn_searcher = NearestNeighbors(n_neighbors=n_neighbors)
        
    
    def _aply_tranform(self, pts, viewmat) -> Union[np.ndarray, torch.Tensor]:

        pts = (viewmat[:3, :3] @ pts.T).T
        pts[..., 0] += viewmat[0, -1]
        pts[..., 1] += viewmat[1, -1]
        pts[..., 2] += viewmat[3, -1]
        return pts

 
    def _handle_poses(
        self, 
        t, quats, 
        inv: Optional[bool]=False, 
    ) -> torch.Tensor:
        
        viewmats = torch.zeros(t.size()[0], 4, 4)
        for idx in range(t.size()[0]):

            tmp_t = t[idx, :]
            quat = quats[idx, :].numpy()
            Rmat = torch.Tensor(R.from_quat(quat, scalar_first=True).as_matrix())

            viewmat = torch.zeros(4, 4)
            viewmat[:3, :3] = Rmat
            viewmat[:3, -1] = tmp_t
            viewmat[-1, -1] = 1.0
            if inv:
                viewmat = torch.linalg.inv(viewmat)

            viewmats[idx, ...] = viewmat
            
        return viewmats
            

    def _parse_colmap_path(self, path: str) -> Tuple:

        imgs_path = os.path.join(path, "images")
        cameras_path = os.path.join(path, "sparse/images.txt")
        points_path = os.path.join(path, "sparse/points3D.txt")
        camera_params = os.path.join(path, "sparse/cameras.txt")

        cameras_info = pd.read_csv(cameras_path)
        K = None
        scale_factor = None
        if os.path.exists(camera_params):
            with open(camera_params, "r") as file:
                data = file.readlines()[-1].split(" ")
                params = [float(val) for val in data[2:6]]
                K = np.array([
                    [params[0], 0.0, params[2]],
                    [0.0, params[1], params[-1]],
                    [0.0, 0.0, 0.0]
                ])
                if (self.w is not None and 
                    self.h is not None):
                    scale_w = self.w / K[0, 0]
                    scale_h = self.h / K[1, 1]
                    scale_factor = 0.5 * (scale_w + scale_h)
                    K *= scale_factor


        return (
            imgs_path, 
            points_path, 
            cameras_info, 
            (K if K is not None else None),
            (scale_factor if scale_factor is None else None)
        )

    def create_from_colmap(
        self,
        path: str,
        partition_size: Optional[int]=1000,
        partitions_n: Optional[int]=32,
        shuffle: Optional[bool]=True,
        search_param: Optional[str | KDTreeSearchParamKNN | KDTreeSearchParamHybrid]="knn",
        k_nns: Optional[int]=30,
        radius: Optional[int]=0.1,
        max_radii: Optional[float]=None,
        inv_poses: Optional[float]=False,
        olr_radii: Optional[float]=0.32,
        olr_nbhs: Optional[float]=12,
        factor: Optional[float]=1.0
    ) -> None:

        imgs_path, points_f, cams, K, scale = self._parse_colmap_path(path)
        if K is not None:
            self.K = torch.Tensor(K)

        if isinstance(search_param, str):
            if search_param == "knn":
                search_param = KDTreeSearchParamKNN(knn=k_nns)
                
            elif search_param == "hybrid":
                search_param = KDTreeSearchParamHybrid(radius, k_nns)

        points_ = np.zeros((partition_size * partitions_n, 3))
        colors_ = np.zeros((partition_size * partitions_n, 3))
        imgs_fs = set()
        with tqdm(
            desc="Loading Collmap Partitions ...",
            colour="green",
            ascii=":>",
            total=(partition_size * partitions_n)
        ) as pbar:
            with open(points_f, "r") as file:
                data_strings = file.readlines()[3:]
                # print(len(data_strings))
                p_idx = 0
                for idx in range(partitions_n):
                    
                    if shuffle:
                        idx = np.random.randint(0, len(data_strings) - partition_size)
                        start_idx = idx
                        end_idx = start_idx + partition_size

                    else:
                        start_idx = idx * partition_size
                        end_idx = (idx + 1) * partition_size
                        
                    for raw_idx in range(start_idx, end_idx):
                        
                        raw = data_strings[raw_idx].split(" ")
                        xyz = np.asarray([float(val) for val in raw[1:4]])
                        rgb = np.asarray([int(val) for val in raw[4:7]])

                        cam_ids = [int(val) for val in raw[8::2]]
                        cam_fs = set([cams[cams["IMAGE_ID"] == id]["NAME"].item() for id in cam_ids])
                        

                        points_[p_idx, ...] = xyz
                        colors_[p_idx, ...] = rgb
                        imgs_fs.update(cam_fs)

                        pbar.update(1)
                        p_idx += 1
                    
                    if shuffle: 
                        del data_strings[idx:(idx + partition_size)]
                        idx = np.random.randint(0, len(data_strings) - partition_size)
                
        
        with tqdm(
            desc="Reading imgs ...",
            colour="green",
            ascii=":>",
            total=len(imgs_fs)
        ) as pbar:
            
            quats = []
            txyzs = []
            gt_imgs = []
            
            for idx, img_f in enumerate(imgs_fs):

                cam_raw = cams[cams["NAME"] == img_f]
                img_f = os.path.join(imgs_path, img_f)
                if os.path.exists(img_f):
                    img = Image.open(img_f)
                    # img = (Fv.pil_to_tensor(img) / 255.0).to(torch.float)
                    img = Fv.pil_to_tensor(img)
                    img = (img / 255.0).to(torch.float32)
                    img = Fv.resize(img, (self.w, self.h))
                    
                else:
                    img = torch.zeros((3, self.w, self.h))

                quat = torch.Tensor([float(val) for val in cam_raw.iloc[0, 1:5].tolist()])
                txyz = torch.Tensor([float(val) for val in cam_raw.iloc[0, 5:8].tolist()])

                gt_imgs.append(img)
                quats.append(quat)
                txyzs.append(txyz)

                pbar.update()
        

        quats = torch.stack(quats, dim=0)
        txyzs = torch.stack(txyzs, dim=0)

        pcd = PointCloud()

        if max_radii is not None:
            prune_mask = np.where(np.linalg.norm(points_, axis=-1) <= max_radii, True, False)
            points_ = np.delete(points_, ~prune_mask, axis=0)
            colors_ = np.delete(colors_, ~prune_mask, axis=0)

        # print(points_.max(), points_.min(), np.linalg.norm(points_, axis=-1).max())
        if scale is not None:
            points_ *= scale

        pcd.points = vec(points_ * factor)
        pcd.colors = vec(colors_)

        # print(points_[np.linalg.norm(points_, axis=-1) == 0].shape, points_.shape)
        pcd.estimate_normals(search_param)
        pcd.remove_radius_outlier(olr_nbhs, olr_radii)
        
        self.pcd = pcd
        self.gt_imgs = torch.stack(gt_imgs, dim=0)
        self.viewmats = self._handle_poses(
            txyzs, quats, 
            inv=inv_poses
        )
        self.viewmats[..., :3, -1] *= factor
        

    def save_ply(self, path: str) -> None:
        write_point_cloud(path, self.pcd)
            
        
    def show(self) -> None:
        
        path = "origin"
        rr.init(path, spawn=True)
        
        pcd_path = f"{path}/Scene"
        scene_items = self.attributes
        print(scene_items["colors"].shape, scene_items["colors"].min(), scene_items["colors"].mean(), scene_items["colors"].max())
        # print(scene_items["colors"].min(), scene_items["colors"].max())
        rr.log(
            f"{pcd_path}/rgb_pts",
            rr.Points3D(
                positions=scene_items["pts"],
                colors=scene_items["colors"],
                radii=[0.004]
            )
        )
        rr.log(
            f"{pcd_path}/normals_pts",
            rr.Points3D(
                positions=scene_items["pts"],
                colors=(scene_items["normals"] * 2) - 1,
                radii=[0.004]
            )
        )

        if ((scene_items["bbox_center"] is not None) and 
            (scene_items["bbox_extent"] is not None)):
            rr.log(
                f"{path}/bbox",
                rr.Boxes3D(
                    centers=[scene_items["bbox_center"]],
                    half_sizes=[scene_items["bbox_extent"] / 2],
                    quaternions=[rr.Quaternion(xyzw=scene_items["bbox_rotation"])],
                    colors=[(0, 255, 0)],
                    labels=f"Scene"
                )
            )
        for idx, viewmat in enumerate(scene_items["viewmats"]):

            gt_img = scene_items["gt_imgs"][idx]
            if gt_img.shape[0] == 3:
                gt_img = gt_img.permute(1, 2, 0)

            rr.log(
                f"{path}/Poses/Frame{idx}",
                rr.Transform3D(
                    mat3x3=viewmat[:3, :3],
                    translation=viewmat[:3, -1]
                )
            )
            rr.log(
                f"{path}/Poses/Frame{idx}/ImgRgb",
                rr.Pinhole(
                    focal_length=0.5 * (self.K[0, 0].item() + self.K[1, 1].item()),
                    width=self.w, 
                    height=self.h
                ),
                rr.Image(gt_img * 255.0)
            )
    @property
    def attributes(self) -> dict:

        try:
            bbox = self.pcd.get_oriented_bounding_box()

        except BaseException:
            bbox = None

        pts = np.asarray(self.pcd.points)
        self.nn_searcher.fit(pts)

        dists, _ = self.nn_searcher.kneighbors(pts)
        # print(dists.shape)
        # print(dists.min(), dists.mean(), dists.max())
        dists = np.exp(np.log(np.sqrt(dists.min(axis=-1)) + 1e-8))
        # print(dists.shape, dists.min(), dists.mean(), dists.max())
        # print(np.exp(dists).min(), np.exp(dists).mean(), np.exp(dists).max())
        initial_scales = np.abs(np.stack([dists, dists, dists], axis=-1))
        # print(f"INITIAL SCALES: {initial_scales.shape}, {initial_scales.min()}, {initial_scales.mean()}, {initial_scales.max()}")
        # central_camera_t = self.viewmats[..., :3, -1].mean(dim=0).unsqueeze(dim=0)
        # cameras_central_dist = torch.norm(central_camera_t - self.viewmats[..., :3, -1], dim=-1)
        # cameras_extent = torch.max(cameras_central_dist)
        cam_dists = torch.norm(self.viewmats[..., :3, -1], dim=1)
        max_cam = self.viewmats[torch.argmax(cam_dists), :3, -1]
        min_cam = self.viewmats[torch.argmin(cam_dists), :3, -1]
        cameras_dists_idx = torch.argsort(torch.norm(self.viewmats[..., :3, -1], dim=-1))
        top_10 = self.viewmats[cameras_dists_idx[:10], :3, -1]
        # print(top_10)
        # print(max_cam.size(), min_cam.size())
        cameras_extent = torch.norm(max_cam - min_cam)
        print(f"NEW CAMERAS EXTENT: {cameras_extent}")
        print(f"BBOX EXTENT: {np.asarray(bbox.extent)}")
        print(f"BBOX CENTER: {np.asarray(bbox.center)}")
        return {
            "pts": pts,
            "initial_scales": None,
            "colors": (np.asarray(self.pcd.colors) / 255.0).astype(np.float32),
            "normals": np.asarray(self.pcd.normals),
            "cameras_extent": cameras_extent,
            "bbox_center": (
                np.asarray(bbox.center)
                if bbox is not None
                else None
            ),
            "bbox_extent": (
                np.asarray(bbox.extent)
                if bbox is not None
                else None
            ),
            "bbox_rotation": (
                R.from_matrix(bbox.R).as_quat()
                if bbox is not None
                else None
            ),
            "gt_imgs": self.gt_imgs,
            "viewmats": self.viewmats,
            "Ks": self.K.view(1, 3, 3).repeat(self.viewmats.size()[0], 1, 1),
            "cameras_extent": cameras_extent
        }





if __name__ == "__main__":
    
    K = np.array([
        [200.0, 0.0, 64.0],
        [0.0, 200.0, 64.0],
        [0.0, 0.0, 1.0]
    ])
    pcd = BasicPointCloudScene()
    # pcd.create_from_video([video3], 5.0)
    pcd.create_from_colmap(
        "/media/ram/T7/ply_collection/gerrard-hall", 
        partition_size=100, 
        partitions_n=1, 
        factor=60.0,
    )
    pcd.save_ply("/media/test/T7/ply_collection/dummy_test.ply")
    pcd.show()

    
    
            
        

        
            
        

        
        
        
            
            
        
        