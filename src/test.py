import torch 
import math as mt
import numpy
from dataclasses import dataclass
from pprint import pprint
# from torch.utils.data import DataLoader
# from .datasets.mip_nerf import MipNerfDataset
# from .scene.gaussian_model import GaussianModel
# from diff_gaussian_rasterization import (GaussianRasterizationSettings, GaussianRasterizer)


# dataloader = DataLoader(
#     dataset=MipNerfDataset(
#         path="/media/ram/T71/360_v2",
#         scene_type="kitchen",
#         pts_partition_size=10000,
#         pts_shuffle=True,
#         pts_partitions_n=10,
#         scene_scale=32.0,
#         cameras_scale=6.0
#     ),
#     shuffle=True,
#     batch_size=1
# )

# initial_pkg = dataloader.dataset.points_attrs
# dataloader.dataset.data_preview()
# viewpoint_sample = next(iter(dataloader))
# gs = GaussianModel()
# gs.create_from_pcd(initial_pkg, 1.0)

# render = 

@dataclass
class Base:
    a: float
    b: float
    c: float

params = {"d": 0, "a": 1, "b": 2, "c": 5}
params = {k: v for (k, v) in params.items() if k in vars(Base)["__annotations__"]}
pprint(vars(Base))
base = Base(**params)
pprint(base)

