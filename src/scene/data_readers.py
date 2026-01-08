import torch 
import numpy as np
import math
import os
import torchvision.transforms.functional as Fv
import torch.nn.functional as F
import matplotlib.cm as cm 

from tqdm import tqdm
from dataclasses import (dataclass, fields)
from typing import (Tuple, Union, Optional, List)
from ..utils.graphics_utils import (
    getWorld2View2, 
    getProjectionMatrix2, 
    Rmat2quat,
    transform_imgTensor,
    ColmapNnerf_convertion,
    C2W_pinhole,
    quat2Rmat
)
from nerfstudio.data.utils.colmap_parsing_utils import (
    read_points3D_binary,
    read_cameras_binary,
    read_images_binary
)
from PIL import Image
from open3d.geometry import (PointCloud, KDTreeSearchParamHybrid as kdtree_hyb)
from open3d.utility import Vector3dVector as vec3d


@dataclass 
class PoinCloud:
    xyz: Union[np.array, torch.Tensor]
    rgb: Union[np.array, torch.Tensor]
    normals_estimation_radii: Optional[float]=0.1
    normasls_estimation_knn: Optional[float]=30
    

    def __post_init__(self) -> None:
        
        
        pcd = PointCloud()
        pcd.points = vec3d(
            self.xyz.copy().detach().cpu().numpy()
            if isinstance(self.xyz, torch.Tensor) 
            else self.xyz
        )
        pcd.colors = vec3d(
            self.rgb.copy().detach().cpu().numpy()
            if isinstance(self.rgb, torch.Tensor)
            else self.rgb
        )
        self.normals = torch.Tensor(pcd.estimate_normals(search_param=kdtree_hyb(
            self.normals_estimation_radii,
            self.normasls_estimation_knn
        )))
        del pcd

@dataclass
class CameraInfo:
    height: float
    width: float
    Fx: float
    Fy: float
    Cx: float
    Cy: float
    R: Union[np.ndarray, torch.Tensor]
    t: Union[np.ndarray, torch.Tensor]
    near: float=0.01
    far: float=100.0
    W2C: Optional[bool]=False
    base_img: Union[np.ndarray, torch.Tensor]=None
    depth_mask: Union[np.ndarray, torch.Tensor]=None
    segmentation_masks: Union[np.ndarray, torch.Tensor]=None


    def __post_init__(self) -> None:

        assert (self.R is not None and self.t is not None)
        assert (self.width is not None and self.height is not None)
        assert (self.Fx is not None 
                and self.Fy is not None
                and self.Cx is not None
                and self.Cy is not None)
        assert (self.base_img is not None)

        self.viewmatrix = getWorld2View2(self.R, self.t, w2c=self.W2C)
        self.viewmatrix = ColmapNnerf_convertion(self.viewmatrix, "view", "c2w")
        self.Quat = Rmat2quat(self.R, "xyzw")
        self.CamCent = self.viewmatrix[3, :3]
        
        if (self.Fx is not None
            and self.Fy is not None
            and self.Cx is not None
            and self.Cy is not None):
            self.K = torch.Tensor([
                [self.Fx, 0.0, self.Cx],
                [0.0, self.Fy, self.Cy],
                [0.0, 0.0, 1.0]
            ])

        self.projmatrix = getProjectionMatrix2(
            znear=self.near,
            zfar=self.far,
            K=self.K,
            W=self.width, H=self.height
        )
        
        if (self.K is not None):
            self.FovX = 2.0 * math.atan2(self.width / 2.0, self.K[0, 0])
            self.FovY = 2.0 * math.atan2(self.height / 2.0, self.K[1, 1])
        
        self.base_img = transform_imgTensor(self.base_img, "CWH->WHC")
        if self.depth_mask is not None:
            self.depth_mask = transform_imgTensor(self.depth_mask, "CWH->WHC")
        if self.segmentation_masks is not None:
            self.segmentation_masks = transform_imgTensor(self.segmentation_masks, "CWH->WHC")
            
    def __str__(self) -> str:
        
        report = 32 * "=" + "!!CAMERA INFO REPORT!!" + 32 * "=" + "\n"
        if self.K is not None:
            report += 32 * "=" + "INTRINSICS" + 32 * "=" + "\n"
            report += str(self.K) + "\n"
            report += 32 * "=" + "INTRINSICS" + 32 * "=" + "\n"
        if self.viewmatrix is not None:
            report += 32 * "=" + "EXTRINSICS" + 32 * "=" + "\n"
            report += str(self.viewmatrix) + "\n"
            report += 32 * "=" + "EXTRINSICS" + 32 * "=" + "\n"
        if self.projmatrix is not None:
            report += 32 * "=" + "PROJMATRIX" + 32 * "=" + "\n"
            report += str(self.projmatrix) + "\n"
            report += 32 * "=" + "PROJMATRIX" + 32 * "=" + "\n"
        if (self.near is not None 
            and self.far is not None
            and self.width is not None
            and self.height is not None):
            report += 32 * "=" + "OTHER" + 32 * "=" + "\n"
            report += f"RESOLUTION: [{self.width}, {self.height}]\n"
            report += f"BOUNDS: [near: {self.near}, far: {self.far}]\n"
            report += 32 * "=" + "OTHER" + 32 * "=" + "\n"
        report += 32 * "=" + "!!CAMERA INFO REPORT!!" + 32 * "="
        
        return report

    @property
    def rgb_img_cwh(self):
        return transform_imgTensor(self.base_img, "WHC->CWH")
    @property
    def rgb_img_whc(self):
        return transform_imgTensor(self.base_img, "WHC->WHC")
    @property
    def depth_img(self):
        return transform_imgTensor(self.depth_mask, "WHC->CWH")
    @property
    def segmentation_img(self):
        return transform_imgTensor(self.segmentation_masks, "WHC->CWH")
    
    def to(self, arg: str) -> None:
        fields_to_convert_ = ["translation", 
                              "quat", "Rmat", "viewmatrix", "K",
                              "base_img", "depth_mask", "segmentation_masks"]
        for field in fields(self):
            f_name = field.name
            value = getattr(self, f_name)
            if value is not None:
                if f_name in fields_to_convert_:
                    if arg in ["pt", "tensor"]:
                        if isinstance(value, np.ndarray):
                            setattr(self, f_name, torch.Tensor(value))
                    elif arg in ["np", "array"]:
                        if isinstance(value, torch.Tensor):
                            setattr(self, f_name, value.detach().cpu().numpy())
                    elif arg in ["cpu", "cuda"]:
                        if isinstance(value, np.ndarray):
                            value = torch.Tensor(value)
                        assert isinstance(value, torch.Tensor), (f"can only convert pytorch tensors to {arg}!!!")
                        setattr(self, f_name, value.to(arg))
                    else:
                        raise ValueError("unrecognized arg value !!!")
    
    def get_masks(self, cls_idx: list, colorized: bool=True) -> Union[np.ndarray, torch.Tensor]:
        if self.segmentation_masks is not None:
            if len(cls_idx) != 1:
                masks = torch.stack([
                    (
                        (self.segmentation_masks[..., idx] > 0.5) * idx
                        if colorized
                        else self.segmentation_masks[..., idx]
                    )  for idx in cls_idx
                ], dim=-1)
                return masks
            else:
                mask = (
                    (self.segmentation_masks[..., cls_idx[0] > 0.5]) * cls_idx[0]
                    if colorized
                    else self.segmentation_masks[..., cls_idx[0]]
                )
                return mask
        else:
            raise ValueError("specify segmentation masks !!!")
    
    def get_masked_img(
        self, 
        cls_idx: list, 
        color_pallete: Union[List[Tuple[int, int, int]], np.ndarray],
        alpha: Optional[float]=0.5, 
    ) -> Union[np.ndarray, torch.Tensor]:
        
        masks = self.get_masks(cls_idx)
        rgb_img = transform_imgTensor(self.base_img.copy(), "CWH->WHC")
        masks = transform_imgTensor(masks.copy(), "CWH->WHC")
        
        mask_labeld = torch.argmax(masks, dim=0)
        colored_mask = torch.stack([
            (torch.where(mask_labeld == cls, 1.0, 0.0) 
            * torch.Tensor(color_pallete[cls]).view(3, 1, 1))
            for cls in mask_labeld.unique()  
        ], dim=0).sum(dim=0)
        
        masked_rgb = alpha * rgb_img + (1 - alpha) * colored_mask
        return masked_rgb

    def set_projection_bounds(self, near: float, far: float) -> None:
        self.near = near
        self.far = far
        self.projmatrix = getProjectionMatrix2(
            near=self.near,
            far=self.far,
            K=self.K,
            W=self.width, H=self.height
        ).transpose(0, 1)
    


def load_MipNerf_Data(
    path: str,
    images_scale: Optional[int]=1,
    target_imgsize: Optional[Tuple[int, int]]=None,
    cameras_scale: Optional[float]=1.0,
    cam_model: Optional[str]="pinhole", #[pinhole, simple-pinhole]
    points_scale: Optional[float]=1.0,
    get_normals_dst_up2camcent: Optional[bool]=False,
    normals_searching_rad: Optional[float]=0.1,
    normals_searching_nns: Optional[int]=30,
    return_tensors: Optional[str]="pt",
    n_views2load: Optional[int]=None,
    near_standart: Optional[float]=0.1,
    far_standart: Optional[float]=100.0,
    outlier_radii_trashhold: Optional[float]=0.05,
    outlier_knn_trashhold: Optional[float]=100
):
    
    imgs_path = (f"images_{images_scale}" if images_scale != 1 else "images") 
    imgs_path = os.path.join(path, imgs_path)

    points_f = os.path.join(path, "sparse/0/points3D.bin")
    points_annots = read_points3D_binary(points_f)
    points_xyz = np.array([p.xyz for p in points_annots.values()])
    points_rgb = np.array([p.rgb for p in points_annots.values()])
    points_rgb = (
        points_rgb 
        if points_rgb.max() < 1.0 
        else points_rgb.astype(np.float32) / 255.0
    )

    pcd = PointCloud()
    pcd.points = vec3d(points_xyz)
    pcd.colors = vec3d(points_rgb)
    pcd.remove_radius_outlier(
        outlier_knn_trashhold, 
        outlier_radii_trashhold
    )
    del points_xyz, points_rgb

    pcd.estimate_normals(search_param=kdtree_hyb(
        normals_searching_rad,
        normals_searching_nns
    ))
    points_xyz = ColmapNnerf_convertion(np.asarray(pcd.points), "pts")
    points_xyz = points_xyz * points_scale
    points_rgb = np.asarray(pcd.colors)
    points_normals = np.asarray(pcd.normals)

    print(np.linalg.norm(points_normals[0]), np.linalg.norm(points_normals, axis=-1).mean())
    del pcd
    
    
    img_annot_f = os.path.join(path, "sparse/0/images.bin")
    cameras_f = os.path.join(path, "sparse/0/cameras.bin")
    img_annotations = read_images_binary(img_annot_f)
    cameras = read_cameras_binary(cameras_f)

    poses_f = os.path.join(path, "poses_bounds.npy")
    from_pbounds = False
    if os.path.exists(poses_f):
        from_pbounds = True
        poses_and_bounds = np.load(poses_f)
        poses = poses_and_bounds[:, :15].reshape(-1, 3, 5)
        bounds = poses_and_bounds[:, -2:]
        R = poses[:, :3, :3]
        T = poses[:, :3, 3] * cameras_scale
        (heights, widths, focal_lenghts) = np.split(poses[..., 4], 3, axis=1)
        nears, fars = (bounds[:, 0], bounds[:, 1])

    cameras_list = []
    psd_camcent_div_maps = None
    if get_normals_dst_up2camcent:
        psd_camcent_div_maps = {}
    
    n_views2load = (
        n_views2load 
        if n_views2load is not None 
        else len(img_annotations)
    )
    with tqdm(
        total=n_views2load,
        colour="green",
        desc="LOADING MIP NERF DATA"
    ) as pbar:
        for idx, img_annot in enumerate(img_annotations.values()):
            
            if (idx == n_views2load):
                break

            img_f = os.path.join(imgs_path, img_annot.name)
            img_rgb = Image.open(img_f)
            img_rgb = (Fv.pil_to_tensor(img_rgb).float() / 255.0)[None]

            Rc = (R[idx, ...] if from_pbounds else quat2Rmat(img_annot.qvec, "xyzw"))
            tc = (T[idx, :] if from_pbounds else img_annot.tvec * cameras_scale)
            near = (nears[idx] if from_pbounds else near_standart)
            far = (fars[idx] if from_pbounds else far_standart)
                
            camera_pkg = cameras[img_annot.camera_id]
            width = (widths[idx, 0] if from_pbounds else camera_pkg.width)
            height = (heights[idx, 0] if from_pbounds else camera_pkg.height)
            if (cam_model == "simple-pinhole"
                and from_pbounds):
                Fx = Fy = focal_lenghts[idx, 0] 
            elif (cam_model == "pinhole"):
                (Fx, Fy) = camera_pkg.params[:2]
            (Cx, Cy) = camera_pkg.params[-2:]
            
            if target_imgsize is not None:
                img_rgb = F.interpolate(img_rgb, target_imgsize, mode="bilinear")
                sx = target_imgsize[0] / width
                sy = target_imgsize[1] / height
                Fx *= sx; Fy *= sy
                Cx *= sx; Cy *= sy
                width = target_imgsize[0]
                height = target_imgsize[1]
                

            camera = CameraInfo(
                base_img=img_rgb.squeeze(),
                Fx=Fx, Fy=Fy,
                Cx=Cx, Cy=Cy,
                R=Rc, t=tc,
                near=near, far=far,
                width=width, height=height
            )
            camera.to(return_tensors)
            cameras_list.append(camera)

            if get_normals_dst_up2camcent:
                dir_camcent = C2W_pinhole(
                    np.array([Cx, Cy]),
                    Fx, Fy, Cx, Cy
                ).reshape(1, 3).repeat(points_normals.shape[0], axis=0)
                dir_pts2camcent = dir_camcent - points_xyz
                dot = np.einsum(dir_pts2camcent, points_normals)
                coss_div_map = dot / np.linalg.norm(points_normals)
                psd_camcent_div_maps[f"view{idx}"] = coss_div_map
            
            pbar.update(1)
    
    points_pkg = {
        "xyz": points_xyz,
        "rgb": points_rgb,
        "normals": points_normals,
        "view_div_maps": psd_camcent_div_maps
    }
    if return_tensors == "pt":
        for (key, value) in points_pkg.items():
            if key != "view_div_maps":
                points_pkg[key] = torch.from_numpy(value).float()
            elif value is not None:
                for (map_key, map_value) in value.items():
                    value[map_key] = torch.from_numpy(map_value).float()
                points_pkg[key] = value

    return (cameras_list, points_pkg)

        
        
    

        
        

    
    

import rerun as rr
import rerun.blueprint as rrb

path = "/media/ram/T71/360_v2/kitchen"
cameras_list, points_pkg = load_MipNerf_Data(
    path, 
    target_imgsize=(224, 224),
    cameras_scale=1.0,
    points_scale=1.0,
    
)

origin = "origin"
rr.init(f"{origin}", spawn=True)
rr.log(f"{origin}/PointsRGB", rr.Points3D(
    positions=points_pkg["xyz"],
    colors=points_pkg["rgb"],
    radii=[0.003]
))
print(cameras_list[0])
for idx, camera in enumerate(cameras_list):
    rr.log(
        f"{origin}/Frame{idx}",
        rr.Transform3D(
            translation=camera.viewmatrix[:3, 3],
            mat3x3=camera.viewmatrix[:3, :3]
        ),
        rr.Pinhole(
            image_from_camera=camera.K,
            width=camera.width,
            height=camera.height
        ),
        rr.Image(camera.rgb_img_whc)
    )

blueprint = rrb.Blueprint(
    rrb.Spatial3DView(origin=origin)
)
rr.send_blueprint(blueprint)



    
    
    
# def load_Cameras_MipNerf(path: str):
#     pass
    
# class LoadedScene:
#     def __init__(self, path: str, scene_type="mip_nerf"):

#         pass

    
    

# if __name__ == "__main__":
    
#     import matplotlib.pyplot as plt
#     plt.style.use("dark_background")
#     from PIL import Image
#     from torchvision.transforms import (Compose, PILToTensor, Resize, Lambda)

#     resolution = (224, 224)
#     tf = Compose([
#         PILToTensor(),
#         Resize(resolution),
#         Lambda(lambda img: (img if img.max() < 1 else img / 255.0))
#     ])
#     img = "/media/ram/T71/360_v2/bicycle/images_4/_DSC8681.JPG"
#     ints = "/media/ram/T71/360_v2/bicycle/sparse/0/images.bin"
#     cam = "/media/ram/T71/360_v2/bicycle/sparse/0/cameras.bin"

#     img = Image.open(img)
#     img = tf(img)   
#     ints = read_images_binary(ints)
#     cam = read_cameras_binary(cam)

#     print(type(ints[1].qvec), type(cam[1].paramss))
#     K = torch.Tensor([
#         [cam[1]]
#     ])
    # color_pallete = np.random.rand(160, 3)
    # segmentation_masks = torch.rand((150, *resolution))
    # masked_rgb, colored_mask = apply_masks(img, segmentation_masks, color_pallete)
    # print(masked_rgb.size())

    # _, axis = plt.subplots(ncols=2)
    # axis[0].imshow(masked_rgb.permute(1, 2, 0))
    # axis[1].imshow(colored_mask.permute(1, 2, 0))
    # plt.show()
    
    
