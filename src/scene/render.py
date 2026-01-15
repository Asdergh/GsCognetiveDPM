import math 
import torch
from typing import Optional
from contextlib import nullcontext
from sklearn.decomposition import PCA
from .gaussian_model import GaussianModel
from .scene_objects import CameraInfo
from ..utils.sh_utils import eval_sh
from diff_gaussian_rasterization import (GaussianRasterizationSettings, GaussianRasterizer)
from gsplat import rasterization
from ..types import *
from dataclasses import fields



def gsplat_render(
    viewpoints_cameras: Union[CameraInfo, List[CameraInfo]],
    gs: GaussianModel,
    features2render: Optional[torch.Tensor]=None,
    eval_shs: Optional[bool]=True
) -> Dict[str, Any]:
    
    if isinstance(viewpoints_cameras, list):

        B = len(viewpoints_cameras)
        (width, height) = (viewpoints_cameras[0].width, viewpoints_cameras[0].height)
        (far, near) = (viewpoints_cameras[0].far, viewpoints_cameras[1].near)
        viewmatrises = torch.stack([cam.cam2world for cam in viewpoints_cameras], dim=0)
        K = torch.stack([cam.K for cam in viewpoints_cameras], dim=0)
        
        xyz = gs.get_xyz
        opacities = gs.get_opacity.squeeze()
        scales = gs.get_scaling
        quats = gs.get_rotation
        if features2render is not None:
            colors = features2render
        elif eval_shs:
            features = gs.get_features.repeat(B, 1, 1, 1).transpose(-1, -2)
            dirs = torch.stack([(view[:3, 3].repeat(gs.N, 1) - xyz) for view in viewmatrises], dim=0)
            dirs_norm = dirs / torch.linalg.norm(dirs, keepdim=True)
            sh_deggre = gs.max_sh_degree
            print(features.size())
            colors = eval_sh(sh_deggre, features, dirs_norm)
            print(colors.size())
        else:
            colors = gs.get_features_dc.squeeze()
            print(colors.size())
        
        print(f"""
            opacities: {opacities.min(), opacities.mean(), opacities.max()},
            colors: {colors.min(), colors.mean(), colors.max()}
        """)
        
        (render, alphas, meta) = rasterization(
            means=xyz,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            Ks=K,
            viewmats=viewmatrises,
            width=width, height=height,
            far_plane=far, near_plane=near,
        )
        radii = meta["radii"]
        print(radii.size())
        radii = torch.max(radii, dim=0).values
        print(radii.size())
        return {
            "render_rgb": render.permute(0, 3, 1, 2),
            "render_depth": alphas.permute(0, 3, 1, 2),
            "radii": meta["radii"],
            "visibility_filter": (radii > 0.0),
            "screen_space_points": torch.zeros_like(xyz)
        }
        
        

        
    
    
    
def default_render(
    viewpoint_camera: CameraInfo, 
    gs: GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    separate_sh: bool=False, 
    override_color: Optional[torch.Tensor]=None,
    render_features: Optional[torch.Tensor]=None,
    use_trained_exp: bool=False
) -> Dict[str, Any]:
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gs.get_xyz, dtype=gs.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.fovx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.fovy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.resolution[1]),
        image_width=int(viewpoint_camera.resolution[0]),
        tanfovx=viewpoint_camera.tanfovx,
        tanfovy=viewpoint_camera.tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.viewmatrix,
        projmatrix=viewpoint_camera.projmatrix,
        sh_degree=gs.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gs.get_xyz
    print(means3D.size())
    means2D = screenspace_points
    opacity = gs.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = gs.get_covariance(scaling_modifier)
    else:
        scales = gs.get_scaling
        rotations = gs.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = gs.get_features.transpose(1, 2).view(-1, 3, (gs.max_sh_degree+1)**2)
            dir_pp = (gs.get_xyz - viewpoint_camera.camera_center.repeat(gs.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(gs.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = gs.get_features_dc, gs.get_features_rest
            else:
                shs = gs.get_features
    elif not render_features:
        colors_precomp = override_color
    
    else:
        pca = PCA(n_components=3)
        print(render_features.size())
        dc_features = render_features.copy().detach().cpu().numpy()
        dc_pca = pca.fit_transform(dc_features).to("cuda")
        colors_precomp = dc_pca

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if (separate_sh
        and render_features is None
        and override_color is None):
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        with (
            torch.no_grad() 
            if render_features is None 
            else nullcontext()
        ):
            rendered_image, radii = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp
            )
            
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = gs.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        # "visibility_filter" : (radii > 0),
        # "radii": radii,
        # "depth" : depth_image
    }
    
    return out





if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from ..datasets.data_loaders import (MipNerfDataset, get_cameras_from_batch)
    from ..configs.configs import (
        RenderingConfig,
        OptimizationConfig
    )
    from ..utils.graphics_utils import CameraInfo
    from ..scene.gaussian_model import GaussianModel

    opt = OptimizationConfig()
    render_cfg = RenderingConfig()
    gs = GaussianModel(opt, render_cfg.sh_degree)

    data_path = "/media/ram/T71/360_v2"
    dataset_cfg = MipNerfDataset.Config(
        path=data_path,
        scene_type="kitchen",
        scene_scale=32.0,
        cameras_scale=32.6
    )
    dataset = MipNerfDataset(dataset_cfg)
    dataset.data_preview()
    gs.create_from_pcd(dataset.points_attrs, 1.0)
    loader = DataLoader(
        dataset=dataset,
        batch_size=12,
        shuffle=False
    )
    sample = next(iter(loader))
    cameras = get_cameras_from_batch(render_cfg.resolution, sample)
    print(len(cameras), cameras[0].viewmatrix, cameras[0].projmatrix)
    
    viewpoint_camera = cameras[0]
    viewpoint_camera.to("cuda")
    bg_color = torch.zeros(3).to("cuda")
    render_pkg = render(
        viewpoint_camera=viewpoint_camera,
        gs=gs,
        pipe=render_cfg,
        bg_color=bg_color
    )

    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    render_rgb = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
    _, axis = plt.subplots()
    axis.imshow(render_rgb)
    plt.show()


    
    