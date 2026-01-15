# import torch
# import rerun as rr
# import rerun.blueprint as rrb
# from .gaussian_model import GaussianModel
# from .render import gsplat_render
# # from .scene_objects import SceneInfo
# from ..configs.configs import (OptimizationConfig)
# from torchvision.utils import make_grid
# import matplotlib.cm as cm 
# import matplotlib.pyplot as plt
# import math 
# import numpy as np




import numpy as np
from itertools import (accumulate, product)
A = np.random.rand(1000)
splits = [60, 20, 20]
splits.sort()
split_sizes = [int((A.shape[0] * split_per) / 100) for split_per in splits]
split_sizes = list(accumulate(split_sizes))
combinations = np.asarray(list(product(split_sizes, repeat=2)))
print(combinations.shape)
combinations = [comb for comb in combinations if (comb[0] <= comb[1])]

for comb in combinations:
    print(A[comb[0]: comb[1]].shape)


#===========================================================================
# from nerfstudio.data.utils.colmap_parsing_utils import (
#     read_points3D_binary,
#     read_cameras_binary,
#     read_images_binary
# )


# # from dataclasses import fields
# path = "/media/ram/T71/360_v2/garden/sparse/0/cameras.bin"
# cameras = read_cameras_binary(path)
# print(type(cameras), (list(cameras.keys()) if isinstance(cameras, dict) else None),)
# print(getattr(cameras[1], "id"), setattr())

# R = np.random.rand(10, 3, 3)
# print((R * np.array([0, -1, 0])).shape)
# up = np.sum(R * np.array([0, -1, 0]), axis=-1)

# cross = np.cross(np.array([0, -1, 0]), up)
# print(cross.shape)

# a = np.random.rand(3, 3)
# vec = np.array([0, -1, 0])
# print(a, a * vec)

# Nv = 64
# opt = OptimizationConfig()
# gs = GaussianModel(opt, 2)
# path = "/media/ram/T71/360_v2/kitchen"
# scene = SceneInfo(
#     path, 
#     camr_params={
#         "n_views": Nv, 
#         "return_tensors": opt.device,
#         "cameras_scale": 1.0
#     },
#     ptsr_params={"points_scale": 4.0}
# )
# points_scene = scene.points3D
# gs.create_from_pcd(points_scene, 1.0, scene.cameras_extent, Nv)
# # cameras_list, rgb_imgs = scene.get_split_views()
# cameras_list, rgb_imgs = scene.sample_views(Nv)
# render_pkg = gsplat_render(
#     viewpoints_cameras=cameras_list,
#     gs=gs,
#     eval_shs=True
# )
# print(render_pkg["render_rgb"].size(), render_pkg["render_depth"].size())

# print(rgb_imgs.size())
# gt_grid = make_grid(rgb_imgs, nrow=int(math.sqrt(Nv))).permute(1, 2, 0).detach().cpu().numpy()
# render_rgb = render_pkg["render_rgb"].detach().cpu().permute(0, 3, 1, 2)
# rgb_grid = make_grid(render_rgb, nrow=int(math.sqrt(Nv))).permute(1, 2, 0).numpy()

# render_depth = render_pkg["render_depth"].detach().cpu().squeeze().numpy()
# render_depth = torch.from_numpy(cm.turbo(render_depth)).permute(0, 3, 1, 2)
# render_depth = (render_depth[:, :3, ...] * render_depth[:, 3, ...].unsqueeze(dim=1))
# depth_grid = make_grid(render_depth, nrow=int(math.sqrt(Nv))).permute(1, 2, 0).numpy()
 
# _, axis = plt.subplots(ncols=3)
# axis[0].imshow(gt_grid)
# axis[1].imshow(rgb_grid.astype(np.float32) / 255.0)
# axis[2].imshow(depth_grid) 
# plt.show()

# gs.save_ply("test_file.ply")