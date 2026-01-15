import torch 
import numpy as np
import os
import cv2
import torchvision.transforms.functional as Fv
import torch.nn.functional as F
import copy
import random as rd
from dataclasses import (dataclass, field)
from .base import (
    BaseIterableDataset,
    BaseIterableDatasetConfig,
    register_dataset
)
from nerfstudio.data.utils.colmap_parsing_utils import (
    read_points3D_binary,
    read_cameras_binary,
    read_images_binary
)
from open3d.geometry import (
    PointCloud, 
    KDTreeSearchParamHybrid as kdtree_hyb
)
from open3d.utility import Vector3dVector as vec3d
from ..utils.graphics_utils import ( 
    quat2Rmat,
    get_camera_extent,
    similarity_from_cameras,
    align_principal_axes,
    transform_cameras,
    transform_points
)
from ..scene.scene_objects import (CameraInfo, PointCloudInfo)
from ..types import *
from tqdm import tqdm





#TODO think about possible features that could make this dataset more rich in practical realization

@dataclass
class MipNerfDatasetConfig(BaseIterableDatasetConfig):
    images_split: Optional[str]="images"
    near: Optional[float]=0.1
    far: Optional[float]=100.0
    apply_worldspace_norm: Optional[bool]=True

@register_dataset("mip-nerf360")
class MipNerfDataset(BaseIterableDataset):
    
    def __init__(self, config: MipNerfDatasetConfig) -> None:        
        self.cfg = config
        super().__init__(self.cfg)
        if self.cfg.apply_worldspace_norm:
            self.normalize_world_space()
    
    def normalize_world_space(self) -> Dict[str, Any]:

        cam2worlds = np.stack([cam.cam2world for cam in self.cameras.values], dim=0)
        xyz = self.point_cloud.xyz
        
        T1 = similarity_from_cameras(cam2worlds)
        cam2worlds = transform_cameras(T1, cam2worlds)
        xyz = transform_points(T1, xyz)

        T2 = align_principal_axes(xyz)
        cam2worlds = transform_cameras(T2, cam2worlds)
        xyz = transform_points(T2, xyz)

        full_transform = T2 @ T1
        if np.median(xyz[:, 2]) > np.mean(xyz[:, 2]):
            T3 = np.diag([1.0, -1.0, -1.0, 1.0])
            cam2worlds = transform_cameras(T3, cam2worlds)
            xyz = transform_points(T3, xyz)
            full_transform = T2 @ full_transform
        
        self.point_cloud.xyz = xyz
        world2cams = np.linalg.inv(cam2worlds)
        for idx, k in enumerate(self.cameras.keys()):
            self.cameras[k].cam2world = cam2worlds[idx]
            self.cameras[k].world2cam = world2cams[idx]
        
    

    @property
    def get_cams_extent(self) -> float:
        camcents = torch.stack([
            cam.camcent 
            for cam in self.viewpoints_pkg["cameras"].values()
        ], dim=0).cpu().numpy()
        return get_camera_extent(camcents)
        
    def load_points3D(self) -> PointCloudInfo:
        
        points_f = os.path.join(self.cfg.source, "sparse/0/points3D.bin")
        points_annots = read_points3D_binary(points_f)
        N_points = (
            min(self.cfg.n_points_max, len(points_annots))
            if self.cfg.n_points_max is not None
            else len(points_annots)
        )
        points_xyz = np.array([p.xyz for p in points_annots.values()])[:N_points]
        points_rgb = np.array([p.rgb for p in points_annots.values()])[:N_points]
        points_rgb = points_rgb + 0.5
        points_rgb = (
            points_rgb 
            if points_rgb.max() <= 1.0 
            else points_rgb.astype(np.float32) / 255.0
        )

        pcd = PointCloud()
        pcd.points = vec3d(points_xyz)
        pcd.colors = vec3d(points_rgb)
        pcd.remove_radius_outlier(
            self.cfg.outlier_knn_trashhold, 
            self.cfg.outlier_radii_trashhold
        )
        pcd.estimate_normals(search_param=kdtree_hyb(
            self.cfg.normals_searching_rad,
            self.cfg.normals_searching_nns
        ))
        # points_xyz = ColmapNnerf_convertion(np.asarray(pcd.points), "pts", permute_order=(0, 1))
        points_xyz = np.asarray(pcd.points)
        points_xyz = points_xyz * self.cfg.points_scale
        points_rgb = np.asarray(pcd.colors)
        points_normals = np.asarray(pcd.normals)

        pts = PointCloudInfo(
            xyz=points_xyz,
            rgb=points_rgb,
            normals=points_normals,
        ).to(self.cfg.device)
        print("[POINT CLOUD DATA WAS LOADED WITH SUCCES!!]")
        print(pts)
        return pts
    
    def load_cameras(self) -> Dict[int, Any]:

        imagesf = os.path.join(self.cfg.source, self.cfg.images_split)
        viewpointsf = os.path.join(self.cfg.source, "sparse/0/images.bin")
        poses_boundsf = os.path.join(self.cfg.source, "poses_bounds.npy")
        camsf = os.path.join(self.cfg.source, "sparse/0/cameras.bin")
        paths = [imagesf, viewpointsf, camsf]
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Nothing found at location: {path}!!")
            
        if os.path.exists(camsf):
            cam_models = {}
            cam_models_bin = read_cameras_binary(camsf)
            if self.cfg.target_size is not None:
                
                for cam_id, cam in cam_models_bin.items():

                    (Fx, Fy, Cx, Cy) = cam.params
                    (width, height) = (cam.width, cam.height)

                    Sx = (cam.width / self.cfg.target_size[0])
                    Sy = (cam.height / self.cfg.target_size[1])
                    Fx *= Sx; Cx *= Sx
                    Fy *= Sy; Cy *= Sy
                    width = self.cfg.target_size[0]
                    height = self.cfg.target_size[1]

                    cam_models[cam_id] = {
                        "params": (Fx, Fy, Cx, Cy),
                        "width": width,
                        "height": height, 
                    }

        from_b = False
        if os.path.exists(poses_boundsf):
            from_b = True
            data = np.load(poses_boundsf)
            (near, far) = (data[:, -2], data[:, -1])
            print("NEAR FAR STATUS")
            print(near.shape, far.shape)
            print(np.mean(near), np.mean(far))
            print("NEAR FAR STATUS")
        
        N_views = (
            min(self.cfg.n_views, data.shape[0]) 
            if self.cfg.n_views is not None 
            else view_ids.shape[0]
        )
        viewpoints = read_images_binary(viewpointsf)
        view_idx2camera_idx = {k: v.camera_id for (k, v) in list(viewpoints.items())[:N_views]}
        view_idx2image_name = {k: v.name for (k, v) in list(viewpoints.items())[:N_views]}
        view_ids = np.array([idx for idx in list(viewpoints.keys())[:N_views]])

        Rt = {k: (quat2Rmat(v.qvec), v.tvec) for (k, v) in viewpoints.items()}
        cameras = {}
        images = {}
        with tqdm(
            desc="[Reading Cameras Parameters ...]",
            total=N_views,            
        ) as pbar:
            for (idx, cam_id) in enumerate(view_ids):
                img_name = view_idx2image_name[cam_id]
                cammodel = cam_models[view_idx2camera_idx[cam_id]]

                intrinsics = cammodel["params"]
                (R, t) = Rt[cam_id]
                cameras[cam_id] = CameraInfo(
                    FocalX=intrinsics[0], 
                    FocalY=intrinsics[1],
                    CentX=intrinsics[2], 
                    CentY=intrinsics[3],
                    R=R, t=(t * self.cfg.cameras_scale),
                    width=cammodel["width"], height=cammodel["height"],
                    near=(near[idx] if from_b else self.cfg.near), 
                    far=(far[idx] if from_b else self.cfg.far),
                    in_viewformat="w2c"
                ).to(self.cfg.device)

                imgf = os.path.join(imagesf, img_name)
                img_rgb = (cv2.imread(imgf).astype(np.float32) / 255.0)
                if self.cfg.target_size is not None:
                    img_rgb = cv2.resize(img_rgb, self.cfg.target_size)
                img_rgb = torch.from_numpy(img_rgb).float()
                images[cam_id] = img_rgb.permute(2, 0, 1).to(self.cfg.device)
                pbar.update(1)
        
        self.cameras = cameras
        self.view_index_stack = view_ids[:N_views]
        self.view_index_collection = np.random.choice(
            self.view_index_collection
        ).tolist()
        self.view_index2camera_id = view_idx2camera_idx
        self.view_index2image_name = view_idx2image_name
        self.gt_images_rgb = images
        

    def collate(self, batch) -> Dict[str, Any]:
        
        batch = {}
        batch["images"] = []
        batch["cameras"] = []
        batch["view_idx"] = []
        for _ in range(self.cfg.batch_size):
            
            idx = self.view_index_stack.pop()
            if not self.view_index_stack:
                self.view_index_stack = np.random.choice(
                    self.view_index_collection
                ).tolist()
                idx = self.view_index_stack.pop()
            
            camera = self.cameras[idx]
            image = self.gt_images_rgb[idx]
            batch["images"].append(image)
            batch["cameras"].append(camera)
            batch["view_idx"].append(idx)
        
        batch["images"] = torch.stack(batch["images"], dim=0)
        batch["view_idx"] = torch.Tensor(batch["view_idx"])
        return batch
    
    



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    plt.style.use("dark_background")
    from torchvision.utils import make_grid
    from torch.utils.data import DataLoader
    from ..scene.render import gsplat_render
    from ..scene.gaussian_model import GaussianModel
    from ..configs.configs import OptimizationConfig

    source = "/media/ram/T71/360_v2/kitchen"
    opt = OptimizationConfig()
    gs = GaussianModel(opt, 2)
    cfg = MipNerfDatasetConfig(
        source=source, 
        target_size=(224, 224), 
        n_views=15, 
        n_points_max=int(1e+5),
        batch_size=64,
        device="cuda",
        apply_worldspace_norm=True,
        cameras_scale=1.0,
        points_scale=1.0
    )
    dataset = MipNerfDataset(cfg)
    print(dataset.get_cams_extent)
    gs.create_from_pcd(dataset.point_cloud, 1.0, dataset.get_cams_extent, cfg.n_views)
    loader = DataLoader(
        dataset=dataset,
        batch_size=None,
        num_workers=0,
        collate_fn=dataset.collate
    )
    
    sample = next(iter(loader))
    gt_img_grid = make_grid(sample["images"], nrow=8).permute(1, 2, 0).detach().cpu()
    render_pkg = gsplat_render(sample["cameras"], gs)
    
    render_rgb = render_pkg["render_rgb"].detach().cpu()
    print(render_rgb.size())
    rgb_grid = make_grid(render_rgb, nrow=8).permute(1, 2, 0)
    
    render_depth = render_pkg["render_depth"].squeeze().detach().cpu().numpy()
    render_depth = torch.from_numpy(cm.turbo(render_depth)).float().permute(0, 3, 1, 2)
    render_depth = (render_depth[:, :3, ...] + render_depth[:, 3, ...].unsqueeze(dim=1))
    depth_grid = make_grid(render_depth, nrow=8).permute(1, 2, 0)
    
    _, axis = plt.subplots(ncols=3)
    axis[0].imshow(gt_img_grid)
    axis[1].imshow(rgb_grid)
    axis[2].imshow(depth_grid)
    plt.show()
    
        
    
    

        
        
        
        




 
    


