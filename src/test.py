import torch 
import torch.nn as nn
from typing import (Optional, Tuple, List)
from tqdm import tqdm
from gsplat import rasterization
from gsplat.utils import save_ply
from torch.optim import (Adam, Optimizer)
from torch.utils.tensorboard import SummaryWriter
from .models.layers import get_activation
from .utils.metrics import ssim
from .utils.sh_utils import eval_sh
from .configs import OptimizationConfig
from .scene.basic_pcd import BasicPointCloudScene
from .scene.gaussian_model import GaussianModel
import matplotlib.cm as cm




class SimpleTrainer:

    def __init__(
        self,
        colmap_path: str,
        resolution: Optional[Tuple[int, int]]=(224, 224),
        opt: Optional[OptimizationConfig]=OptimizationConfig(),
        device: Optional[str]="cuda",
        log_dir: Optional[str]="runs",
        alpha: Optional[float]=0.08,
        beta: Optional[float]=0.2,
        sh_degree: Optional[int]=3,
        checkpoint_steps: Optional[List]=[100, 1000, 2000, 3000, 4000, 5000],
    ):
        
        self.opt = opt
        self.device = device
        self.img_res = resolution
        self.checkpoint_steps = checkpoint_steps

        self.pcd = BasicPointCloudScene(width=self.img_res[0], height=self.img_res[1])
        self.pcd.create_from_colmap(colmap_path, partition_size=40, partitions_n=100)
        self.gt_imgs = self.pcd.gt_imgs.to(self.device)
        self.gaussians = GaussianModel(sh_degree=sh_degree, device=self.device)
        self.gaussians.create_from_pcd(self.pcd, 1)
        self.gaussians.training_setup(opt)

        self.epochs = opt.iterations
        self.device = device
        self.sh_degree = sh_degree

        self.loss_fn = nn.L1Loss()
        self.alpha = torch.tensor(alpha).to(self.device)
        self.beta = torch.tensor(beta).to(self.device)

        self.writer = SummaryWriter(log_dir)
        # self.noise = self._check_device(torch.normal(0, 1, (gs_points, noise_dim)))
        

    def train(self) -> torch.Tensor:        
        with tqdm(
            desc="SimpleGSTrainer",
            total=self.epochs,
            ascii=":>",
            colour="green"
        ) as pbar:
            losses = []
            for step in range(self.epochs):
                
                
                xyz = self.gaussians.get_xyz
                quats = self.gaussians.get_rotation
                scales = self.gaussians.get_scaling
                opacities = self.gaussians.get_opacity.squeeze()
                features_ds = self.gaussians.get_features_dc
                features_rest = self.gaussians.get_features_rest

                cat_rgbsh = torch.cat([features_ds, features_rest], dim=1).transpose(1, 2)
                colors = eval_sh(
                    deg=self.sh_degree,
                    sh=cat_rgbsh,
                    dirs=xyz
                )
                
                # print(f"""
                # xyz: {xyz.dtype}, {xyz.is_cuda},
                # quats: {quats.dtype}, {quats.is_cuda}
                # scales: {scales.dtype}, {scales.is_cuda},
                # opacities: {opacities.dtype}, {opacities.is_cuda}
                # colors: {colors.dtype}, {colors.size()}, {colors.is_cuda},
                # Ks: {self.gaussians.Ks.dtype}, {self.gaussians.Ks.size()}, {self.gaussians.Ks.is_cuda}
                # viewmats: {self.gaussians.viewmats.dtype}, {self.gaussians.viewmats.size()}, {self.gaussians.viewmats.is_cuda}
                # """)
                rendered_rgb, alphas, meta = rasterization(
                    means=xyz,
                    quats=quats,
                    scales=scales, 
                    opacities=opacities,
                    colors=colors,
                    Ks=self.gaussians.Ks, 
                    viewmats=self.gaussians.viewmats,
                    width=self.img_res[0],
                    height=self.img_res[1],
                    packed=False
                )
                print(self.gaussians.get_scaling.min(), self.gaussians.get_scaling.mean(), self.gaussians.get_scaling.max())
                rendered_rgb = rendered_rgb.permute(0, -1, 1, 2)
                # print(self.gt_imgs.size(), rendered_rgb.size(), alphas.size())
                L1 = self.loss_fn(rendered_rgb, self.gt_imgs)
                Dssim = 1.0 - ssim(rendered_rgb, self.gt_imgs, device=self.device)
                loss = self.alpha * L1 + self.beta * Dssim
                loss.backward()
    
                
                losses.append(loss.item())
                # print((torch.Tensor(cm.jet(alphas[0, ...].detach().cpu().squeeze(dim=-1))) / 255.0).size())
                self.writer.add_scalar("img_loss", loss, step)
                self.writer.add_scalar("L1_loss", L1, step)
                self.writer.add_scalar("Dssim", Dssim, step)
                self.writer.add_image("rendered_img", rendered_rgb[0, ...], step)
                self.writer.add_image("rendered_alphas", torch.Tensor(cm.jet(alphas[0, ...].detach().cpu().squeeze(dim=-1))).permute(-1, 0, 1)[:-1, ...] / 255.0, step)
                self.writer.add_image("gt_img", self.gt_imgs[0, ...], 0)
                
                radii = torch.max(torch.max(meta["radii"], dim=-1).values, dim=0).values
                visibility_filter = (radii > 0.0)
                with torch.no_grad():
                    
                    #densification
                    if (step < self.opt.densify_until_iter):
                        
                        
                        # print(radii.size())
                        # print(self.gaussians.max_radii2D.size(), visibility_filter.size())
                        self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], 
                                                                                    radii[visibility_filter])
                        self.gaussians.add_densification_stats(xyz, visibility_filter)

                        if ((step > self.opt.densify_from_iter) and 
                            (step % self.opt.densification_interval) == 0):
                            size_trashhold = 40
                            self.gaussians.densify_and_prune(
                                self.opt.densify_grad_threshold, 0.005, 
                                self.gaussians.cameras_extent, 
                                size_trashhold,
                                radii
                            )
                            print(f"Densification was applied |--> New Gaussians Size: {self.gaussians.get_xyz.size()}")
                        
                        # if (step % self.opt.densification_interval) == 0:
                        #     self.gaussians.reset_opacity()
                    
                    #optimization step
                    self.gaussians.exposure_optimizer.step()
                    self.gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    
                    if (step in self.checkpoint_steps):
                        save_ply({
                            "means": self.gaussians.get_xyz.detach().cpu(),
                            "quats": self.gaussians.get_rotation.detach().cpu(),
                            "scales":  self.gaussians.get_scaling.detach().cpu(),
                            "opacities":  self.gaussians.get_opacity.squeeze().detach().cpu(),
                            "sh0":  self.gaussians.get_features_dc.detach().cpu(),
                            "shN":  self.gaussians.get_features_rest.detach().cpu()
                        }, f"result_on_{step}.ply")
                
                pbar.update(1)
        
        return torch.Tensor(losses)
    
    def _check_device(self, v):
        if isinstance(v, nn.Module):
            if next(v.parameters()).device.type != self.device:
                v = v.to(self.device)

        else:
            if v.device.type != self.device:
                v = v.to(self.device)
        
        return v

                
    
class GsGeneration3D(nn.Module):

    def __init__(
        self,
        in_features: int,
        sh_degree: Optional[int]=3,
        splats_n: Optional[int]=1000,
        geom_activation_fn: Optional[str]="relu",
        colors_activation_fn: Optional[str]="sigmoid",
    ) -> None:
        
        super().__init__()
        self.sh_degree = sh_degree
        self._means = self._get_layer(in_features, 3, geom_activation_fn)
        self._quats = self._get_layer(in_features, 4, geom_activation_fn)
        self._scales = self._get_layer(in_features, 3, geom_activation_fn)
        self._colors = self._get_layer(in_features, 3, colors_activation_fn)
        self._sh_features = self._get_layer(in_features, 3 * (sh_degree + 1) ** 2, colors_activation_fn)
        self._opactities = self._get_layer(in_features, 1, colors_activation_fn)


    def _get_layer(self, in_f, out_f, act) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_f, out_f),
            get_activation(act),
            # nn.LayerNorm(out_f)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _ = x.size()
        return {
            "means": self._means(x),
            "quats": self._quats(x),
            "scales": self._scales(x),
            "sh0": self._colors(x).unsqueeze(dim=-1),
            "shN": self._sh_features(x).view(N, 3, (self.sh_degree + 1) ** 2),
            "opacities": self._opactities(x).squeeze()
        }
    



if __name__ == "__main__":

    from torchvision.transforms import (Compose, PILToTensor, Resize, Lambda)
    from PIL import Image

    log_dir = "runs"
    colmap_dir = "/media/ram/T7/ply_collection/gerrard-hall"
    trainer = SimpleTrainer(
        log_dir=log_dir,
        colmap_path=colmap_dir
    )
    losses = trainer.train()