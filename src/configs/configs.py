import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass, field
from typing import (Optional, Tuple, List, Dict, Any, Dict, Union)

loc_path = str(Path(__file__).parent)
OmegaConf.register_new_resolver("load_config", lambda cfg_f: load_cfg(cfg_f))




@dataclass 
class TrailConfig:
    data_sampler: Dict[str, Any]
    pipeline: Dict[str, Any]
    
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



