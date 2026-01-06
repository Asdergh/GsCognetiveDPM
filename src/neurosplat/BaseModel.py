import torch
import torch.nn as nn
import lightning as l
import os
import matplotlib.cm as cm
from abc import (ABC, abstractmethod)
from typing import (Optional, Tuple, Dict, Any, List, Union)
from dataclasses import (dataclass, field)
from gsplat import rasterization
from ..scene.gaussian_model import GaussianModel
# from ..submodules.DG_rasterization.diff_gaussian_rasterization import (GaussianRasterizer, GaussianRasterizationSettings)
from ..utils.objectives import get_loss
from ..utils.sh_utils import eval_sh
from ..utils.system_utils import make_dir
from ..configs.configs import (
    LogingConfig, 
    TrainingConfig,
    RenderingConfig, 
    OptimizationConfig
)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

class GaussianBase(l.LightningModule, ABC):

    def __init__(
        self,  
        opt_cfg: OptimizationConfig, 
        log_cfg: LogingConfig, 
        render_cfg: RenderingConfig,
        train_cfg: TrainingConfig
    ) -> None:
        
        self.opt_cfg = opt_cfg
        self.log_cfg = log_cfg
        self.train_cfg = train_cfg
        self.render_cfg = render_cfg

        l.LightningModule.__init__(self)
        
        self.gs = GaussianModel(self.opt_cfg, self.render_cfg.sh_degree)
        self.ply_storage = os.path.join(self.log_cfg.loging_path, "splats_collection")
        tb_writer_storage =  os.path.join(self.log_cfg.loging_path, "tensorboard_logs")
        make_dir(tb_writer_storage)
        make_dir(self.ply_storage)
        self.spalt_paths = {}
        for features2log in self.log_cfg.features2log:
            if "splats" in features2log:
                path = os.path.join(self.ply_storage, features2log)
                make_dir(path)
                self.splat_paths[features2log] = path

        self._tb_writer = SummaryWriter(tb_writer_storage)

        self.Losses = {
            loss_str: get_loss(loss_str) 
            for loss_str in self.train_cfg.losses
        }
        if "perceptive" in self.Losses:
            self.Losses["perceptive"].to(self.opt_cfg.device)
        self.render_cache_ = None

    def on_train_start(self):
        print("START TRAINING PROCEDURE !!!")
        initial_pkg = self.trainer.train_dataloader.dataset.points_attrs   
        self.gs.create_from_pcd(initial_pkg, 1.0)
        self.gs.training_setup(self.opt_cfg.steps)

    def on_train_epoch_end(self):

        assert (self.render_cache_ is not None), ("densification before training_step !!")
        radii, vis_filter, view_pts = self.render_cache_
        current_step = self.trainer.current_epoch
        with torch.no_grad():
            if (current_step < self.opt_cfg.densify_until_step):
                
                self.gs.max_radii2D[vis_filter] = torch.max(self.max_radii2D[vis_filter], radii[vis_filter])
                self.gs.add_densification_stats(view_pts, vis_filter)
                if (current_step >= self.opt_cfg.densify_from_step and 
                    (current_step % self.opt_cfg.densification_interval) == 0):
                    print("RUNNING DENSIFICATION !!!")
                    self.gs.densify_and_prune(
                        self.opt_cfg.grad_trashold, 0.005,
                        self.opt_cfg.size_trashold, radii
                    )
                    print("DENSIFICATION COMPLITED!!!")
                    print(f"N GAUSSIANS AFTER DENSIFICATION: {self.get_xyz.size()}")

            self.gs.exposure_optimizer.step()
            self.gs.exposure_optimizer.zero_grad(set_to_none=True)
            self.gs.gs_optimizer.step()
            self.gs.gs_optimizer.zero_grad(set_to_none=True)

            if (current_step >= self.log_cfg.log_from_step
                and current_step < self.log_cfg.log_until_step
                and (current_step % self.log_cfg.log_per_steps) == 0):

                if "splats" in self.spalt_paths:    
                    path = os.path.join(
                        self.splats_paths["splats"], 
                        f"Splats{current_step}.ply"
                    )
                    self.save_ply(path)
                
                # if "splats-normal" in self.spalt_paths:
                #     path = os.path.join(
                #         self.spalt_paths["splats-normal"], 
                #         f"Splats{current_step}.ply"
                #     )
                #     self.save_ply()

    # def render(self, viewmats, Ks, rgb_map: torch.Tensor=None):

    #     B = viewmats.size(0)
    #     if rgb_map is None:
    #         shs = self.get_features.transpose(1, 2)
    #         cameras_C = torch.stack([
    #             viewmats[idx_B, :3, -1][None].repeat(shs.size(0), 1)
    #             for idx_B in range(B) 
    #         ])
    #         dir_cp = self.get_xyz[None].repeat(B, 1, 1) - cameras_C
    #         dir_cp_normalized = dir_cp / torch.norm(dir_cp, dim=-1, keepdim=True)
    #         rgb = eval_sh(self.max_sh_degree, shs.repeat(B, 1, 1, 1), dir_cp_normalized)

    #     else:
    #         rgb = rgb_map

    #     render, alphas, meta = rasterization(
    #         means=self.get_xyz,
    #         quats=self.get_rotation,
    #         scales=self.get_scaling,
    #         colors=rgb,
    #         opacities=self.get_opacity.squeeze(),
    #         width=self.cfg.resolution[1], 
    #         height=self.cfg.resolution[0],
    #         viewmats=viewmats,
    #         Ks=Ks,
    #         packed=False
    #     )
       
    #     radii = torch.max(torch.max(meta["radii"].squeeze(), dim=-1).values, dim=0).values
    #     return {
    #         "render": render.squeeze().permute(0, -1, 1, 2),
    #         "radii": radii,
    #         "visibility_filter": (radii > 0.0),
    #         "viewspace_points": self.get_xyz,
    #         "depth": alphas.squeeze()
    #     }
    
    def log_datapkg(self, loging_pkg: Tuple[Dict[str, Any]]) -> None:
        gs = self.global_step
        for (k, value) in loging_pkg.item():
            if ("lightning" in value[0]
                and value[1] == "scalar"):
                self.log(k, value[2], gs)
            if ("tensorboard" in value[0]
                or "tb" in value[0]):
                if value[1] == "scalar":
                    self._tb_writer(k, value[2], gs)
                elif value[1] in ["img", "image"]:
                    image = value[2]
                    if value[-1]:
                        idx = torch.randint(0, image.size(0), (self.cfg.n_views2log, ))
                        image = make_grid(image[idx, ...])
                    self._tb_writer.image(k, image, gs)
                else:
                    raise ValueError(f"tried to log unrecognized type: {value[1]}")


    def _render_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        pass
        
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
        #       else self.Losses["perceptive"](render, gts))
        
        loss = (Dssim + L1 + L2)

        # loging into LightninLogger to monitor
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
            self._tb_writer.add_scalar("opacity_mean", self.get_opacity.mean(), gs)
            # self._tb_writer.add_scalar("L1", L1, gs)

            idx_bf = torch.randint(0, B, (self.cfg.n_views2log, ))
            self._tb_writer.add_image("gt_rgb", make_grid(gts[idx_bf, ...]), gs)
            self._tb_writer.add_image("render_rgb", make_grid(render[idx_bf, ...]), gs)
            self._tb_writer.add_image("depth", make_grid(alphas[idx_bf, ...]), gs)

        log_pkg = {
            f"{mode}_d-ssim-loss": (["lightning", "tb"], "scalar", Dssim),
            f"{mode}_l1-loss": (["lightning", "tb"], "scalar", Dssim),
            f"{mode}_l2-loss": (["lightning", "tb"], "scalar", Dssim),
        }
        
        if "gt_rgb" in self.log_cfg.features2log:
            log_pkg.update({f"{mode}_gt_rgb": (["tb"], "image", gts, True)})
        if "render_rgb" in self.log_cfg.features2log:
            log_pkg.update({f"{mode}_render_rgb": (["tb"], "image", render, True)})

        if "alphas" in self.log_cfg.features2log:
            log_pkg.update({f"{mode}_alphas": (["tb"], "image", alphas, True)})
        
        self.log_datapkg(log_pkg)
        return loss
    
    @abstractmethod
    def training_step(self, batch, batch_idx) -> float:
        """
        Docstring for training_step
        
        :param batch: input batch for training step
        :param batch_idx: general idx of train batch
        :return: loss during training step
        :rtype: float
        """
        # return self._step(batch, mode="train")
        
    
    @abstractmethod
    def validation_step(self, batch, batch_idx) -> float:
        """
        Docstring for validation_step
        
        :param batch: input batch for validation step
        :param batch_idx: general idx of validation batch
        :return: loss during validation step
        :rtype: float
        """
        # with torch.no_grad():
        #     return self._step(batch, mode="val")
        
    @abstractmethod
    def configure_optimizers(self):
        """
        Docstring for configure_optimizers
        method for optimization settings initializations
        """
        # return self.gs_optimizer
        
        


        

            



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
        

    


        
        
        
        
        
