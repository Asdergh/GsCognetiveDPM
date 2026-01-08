import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass, field
from pydantic import (BaseModel, Field)
from typing import (Optional, Tuple, List, Dict, Any, Dict, Union)

curr_path = Path(__file__)
loc_path = str(curr_path.parent)
default_logging_path = os.path.join(str(curr_path.parents[2]), "log_dir")



class LogingConfig(BaseModel):
    logging_path: Optional[str]=default_logging_path
    features2log: Optional[List[str]]=Field(default_factory=lambda: [
        "render_rgb", 
        "gt_rgb",
        "alphas",
        "splats", 
        # "splats-normal",
        # "splats-features",
        # "splats-masks"
    ])
    log_from_step: Optional[int]=0
    log_until_step: Optional[int]=10000
    log_per_step: Optional[int]=1000
    steps2log: Optional[List[int]]=None
    n_views2log: Optional[int]=2

class RenderingConfig(BaseModel):
    resolution: Optional[Tuple[int, int]]=(224, 224)
    sh_degree: Optional[int]=2
    convert_SHs_python: Optional[bool]=False
    compute_cov3D_python: Optional[bool]=False
    bg_color: Optional[Tuple[int, int, int]]=(0, 0, 0) #RGB in [0, 1]
    znear_plane: Optional[float]=0.01
    zfar_plane: Optional[float]=100.0
    scaling_trashhold: Optional[float]=20.0
    debug: Optional[bool]=False
    

class OptimizationConfig(BaseModel):
    steps: Optional[int]=30_000
    position_lr_init: Optional[float]=0.00016
    position_lr_final: Optional[float]=0.0000016
    position_lr_delay_mult: Optional[float]=0.01
    position_lr_max_steps: Optional[int]=30_000
    feature_lr: Optional[float]=0.0025
    opacity_lr: Optional[float]=0.025
    scaling_lr: Optional[float]=0.005
    rotation_lr: Optional[float]=0.001
    exposure_lr_init: Optional[float]=0.01
    exposure_lr_final: Optional[float]=0.001
    exposure_lr_delay_steps: Optional[float]=0
    exposure_lr_delay_mult: Optional[float]=0.0
    percent_dense: Optional[float]=0.01
    lambda_dssim: Optional[float]=0.2
    densification_interval: Optional[int]=100
    opacity_reset_interval: Optional[int]=3000
    densify_from_iter: Optional[int]=500
    densify_until_iter: Optional[int]=15_000
    densify_grad_threshold: Optional[int]=0.0002
    depth_l1_weight_init: Optional[int]=1.0
    depth_l1_weight_final: Optional[int]=0.01
    random_background: Optional[bool]=False
    optimizer_type: Optional[str]="default"
    device: Optional[str]="cuda"
    segmentation: Optional[bool]=False

class TrainingConfig(BaseModel):
    losses: Optional[List[str]]=Field(default_factory=[
        "d-ssim",
        "mse",
        "l1"
    ])
    

def save_cfg(cfg, path: str):
    if "/" not in path:
        path = os.path.join(loc_path, path)
    OmegaConf.save(cfg, path)


def parse_structured(fields, cfg: Union[Dict, DictConfig]):
    cfg = OmegaConf.structured(cfg)
    return fields(**cfg) 

def load_cfg(path: str) -> Dict[str, Any]:
    cfg = OmegaConf.load(path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return cfg

# trial_cfg = TrailConfig(
#     data_sampler="${load_config: '/home/ram/Desktop/own_projects/tmp/GsCognetiveDPM/src/configs/data_sampler.yaml'}",
#     pipeline="${load_config: '/home/ram/Desktop/own_projects/tmp/GsCognetiveDPM/src/configs/GsBaseModel.yaml'}",
# )
# trial_cfg = OmegaConf.structured(trial_cfg)
# trial_cfg = OmegaConf.to_container(trial_cfg, resolve=True)
# OmegaConf.save(trial_cfg, "/home/ram/Desktop/own_projects/tmp/GsCognetiveDPM/src/configs/trial_cfg_demo.yaml")
# cfg = OmegaConf.structured(opt)

# print(type(cfg))
# save_cfg(opt, "test_opt_config.yaml")



