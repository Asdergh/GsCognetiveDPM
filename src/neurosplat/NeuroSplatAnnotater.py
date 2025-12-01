import torch
import torch.nn as nn
import lightning as l
import os
import matplotlib.cm as cm
from typing import (Optional, Tuple, Dict, Any, List, Union)
from dataclasses import (dataclass, field)
from gsplat import rasterization
from ..scene.gaussian_model import GaussianModel
# from ..submodules.DG_rasterization.diff_gaussian_rasterization import (GaussianRasterizer, GaussianRasterizationSettings)
from ..utils.objectives import get_loss
from ..utils.sh_utils import eval_sh
from ..utils.system_utils import make_dir
from ..configs.configs import (save_cfg, load_cfg, parse_structured)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

class NeuroSplatAnnotater(l.LightningModule, GaussianModel):
    @dataclass
    class Config(GaussianModel.Config):
        #aux
        resolution: Optional[Tuple[int, int]]=(112, 224)
        sh_degree: Optional[int]=1
        features2log: List=field(default_factory=lambda: ["renders", "splats", "splats-normal"])
        log_from_step: Optional[int]=0
        log_until_step: Optional[int]=10000
        log_per_steps: Optional[int]=200
        n_views2log: Optional[int]=2
        steps2log: Optional[List]=field(default_factory=lambda: [])
        
        #optimization 
        train_mode: Optional[str]="both", #[gaussians, models, both]
        losses: Optional[List[str]]=field(default_factory=lambda: ["d-ssim", "l1", "mse", "perceptive"])
        metrics: Optional[List[str]]=field(default_factory=lambda: ["ssim"])
        #render params
        background: Optional[List]=field(default_factory=lambda: [0.0, 0.0, 0.0])
        scaling_modifier: Optional[float]=1.0
        densify_until_step: Optional[int]=10000
        densify_from_step: Optional[int]=100
        densification_interval: Optional[int]=500

    def __init__(self, cfg, steps: int) -> None:
        
        self._steps = steps
        if cfg is not None:
            self.cfg = parse_structured(self.Config, cfg)
        else:
            self.cfg = self.Config()

        l.LightningModule.__init__(self)
        GaussianModel.__init__(self)

        self.ply_storage = os.path.join(self.cfg.loging_path, "splats_collection")
        tb_writer_storage =  os.path.join(self.cfg.loging_path, "tensorboard_logs")
        make_dir(self.ply_storage)
        make_dir(tb_writer_storage)
        self._tb_writer = SummaryWriter(tb_writer_storage)

        self.Losses = {
            loss_str: get_loss(loss_str) 
            for loss_str in self.cfg.losses
        }
        if "perceptive" in self.Losses:
            self.Losses["perceptive"].to(self.cfg.device)
        self.render_cache_ = None

    def on_train_start(self):
        print("START TRAINING PROCEDURE !!!")
        initial_pkg = self.trainer.train_dataloader.dataset.points_attrs
        self.create_from_pcd(initial_pkg, 1.0)
        self.training_setup(self._steps)

    def on_train_epoch_end(self):

        assert (self.render_cache_ is not None), ("densification before training_step !!")
        radii, vis_filter, view_pts = self.render_cache_
        current_step = self.trainer.current_epoch
        with torch.no_grad():
            if (current_step < self.cfg.densify_until_step):
                
                self.max_radii2D[vis_filter] = torch.max(self.max_radii2D[vis_filter], radii[vis_filter])
                self.add_densification_stats(view_pts, vis_filter)
                if (current_step > self.cfg.densify_from_step and 
                    current_step % self.cfg.densification_interval == 0):
                    print("RUNNING DENSIFICATION !!!")
                    self.densify_and_prune(
                        self.cfg.densify_grad_trashhold, 0.005,
                        self.cfg.size_trashold,
                    )
                    print("DENSIFICATION COMPLITED!!!")
                    print(f"N GAUSSIANS AFTER DENSIFICATION: {self.get_xyz.size()}")

            self.exposure_optimizer.step()
            self.exposure_optimizer.zero_grad(set_to_none=True)
            self.gs_optimizer.step()
            self.gs_optimizer.zero_grad(set_to_none=True)

            if (current_step >= self.cfg.log_from_step
                and current_step < self.cfg.log_until_step
                and (current_step % self.cfg.log_per_steps) == 0):
                print("must have logg the data")
                path = os.path.join(self.ply_storage, f"Splats{current_step}.ply")
                self.save_ply(path)
        
    def render(self, viewmats, Ks, rgb_map: torch.Tensor=None):

        B = viewmats.size(0)
        if rgb_map is None:
            shs = self.get_features.transpose(1, 2)
            cameras_C = torch.stack([
                viewmats[idx_B, :3, -1][None].repeat(shs.size(0), 1)
                for idx_B in range(B) 
            ])
            dir_cp = self.get_xyz[None].repeat(B, 1, 1) - cameras_C
            dir_cp_normalized = dir_cp / torch.norm(dir_cp, dim=-1, keepdim=True)
            rgb = eval_sh(self.max_sh_degree, shs.repeat(B, 1, 1, 1), dir_cp_normalized)

        else:
            rgb = rgb_map

        render, alphas, meta = rasterization(
            means=self.get_xyz.repeat(B, 1, 1),
            quats=self.get_rotation.repeat(B, 1, 1),
            scales=self.get_scaling.repeat(B, 1, 1),
            colors=rgb,
            opacities=self.get_opacity.squeeze().repeat(B, 1),
            width=self.cfg.resolution[1], 
            height=self.cfg.resolution[0],
            viewmats=torch.transpose(viewmats[None], 0, 1),
            Ks=torch.transpose(Ks[None], 0, 1),
            packed=False
        )
       
        radii = torch.max(torch.max(meta["radii"].squeeze(), dim=-1).values, dim=0).values
        return {
            "render": render.squeeze().permute(0, -1, 1, 2),
            "radii": radii,
            "visibility_filter": (radii > 0.0),
            "viewspace_points": self.get_xyz,
            "depth": alphas.squeeze()
        }
    
    
    def _step(self, batch, mode="train") -> float:

        gts = batch["gt-rgb"]
        viewmats = batch["viewmats"]
        Ks = batch["Ks"]
        B = viewmats.size(0)
        render_pkg = self.render(viewmats, Ks)
        self.render_cache_ = (
            render_pkg["radii"], 
            render_pkg["visibility_filter"], 
            render_pkg["viewspace_points"]
        )
        render, alphas =  (render_pkg["render"], render_pkg["depth"])
        alphas = torch.Tensor(cm.inferno(alphas.cpu().detach().numpy()))
        alphas = alphas[..., :-1].permute(0, -1, 1, 2)

        Dssim = (0.0 if "d-ssim" not in self.Losses
                    else self.Losses["d-ssim"](render, gts))
        L1 = (0.0 if "l1" not in self.Losses
                else self.Losses["l1"](render, gts))
        L2 = (0.0 if "l1" not in self.Losses
                else self.Losses["mse"](render, gts))
        # PerceptiveLoss = (0.0 if "perceptive" not in self.Losses
        #       else self.Losses["perceptive"](render, gt))
        
        loss = (Dssim + L1 + L2)

        #loging into LightninLogger to monitor
        self.log(f"{mode}/d-ssim-loss", Dssim)
        self.log(f"{mode}/l1-loss", L1)
        self.log(f"{mode}/l2-loss", L2)
        self.log(f"{mode}/general-loss", loss)

        # loging to tb_writer
        if mode == "train":
            gs = self.global_step
            self._tb_writer.add_scalar("Dssim", Dssim, gs)
            self._tb_writer.add_scalar("L1", L1, gs)
            self._tb_writer.add_scalar("L2", L2, gs)
            # self._tb_writer.add_scalar("L1", L1, gs)

            idx_bf = torch.randint(0, B, (self.cfg.n_views2log, ))
            self._tb_writer.add_image("gt_rgb", make_grid(gts[idx_bf, ...]), gs)
            self._tb_writer.add_image("render_rgb", make_grid(render[idx_bf, ...]), gs)
            self._tb_writer.add_image("depth", make_grid(alphas[idx_bf, ...]), gs)

        return loss
        
    def training_step(self, batch, batch_idx) -> float:
        return self._step(batch, mode="train")
        
    
    def validation_step(self, batch, batch_idx) -> float:
        with torch.no_grad():
            return self._step(batch, mode="val")
            
    def configure_optimizers(self):
        return self.gs_optimizer
        
        


        

            



# if __name__ == "__main__":
    
#     from omegaconf import OmegaConf
#     gs_opt = GaussianOptimizerConfig()
 
#     cfg_f = "/home/ram/Desktop/own_projects/tmp/GsCognetiveDPM/src/configs/GsBaseModel.yaml"
#     initial_pkg = {
#         "gt_imgs": torch.zeros((10, 3, 10, 10)),
#         "pts": torch.rand((10, 3)),
#         "colors": torch.rand((10, 3)),
#         "cameras_extent": torch.tensor(4.98)
#     }
#     cfg = load_cfg(cfg_f)
#     pipeline = NeuroSplatAnnotater(cfg)
#     print(pipeline.cfg)
#     save_cfg(pipeline.cfg, "GsBaseModel.yaml")
        

    


        
        
        
        
        
