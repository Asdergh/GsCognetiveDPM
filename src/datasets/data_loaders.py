import os 
import numpy as np
import torch
import torch.nn as nn
import rerun as rr
import open3d as o3d
import rerun.blueprint as rrb
import random as rd
import matplotlib.pyplot as plt

from rerun import Quaternion
from dataclasses import dataclass
from pydantic import BaseModel
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from nerfstudio.data.utils.colmap_parsing_utils import (
    read_points3D_binary,
    read_cameras_binary,
    read_images_binary
)
from PIL import Image
from dataclasses import fields
from torchvision.transforms import (Compose, 
                                    PILToTensor, 
                                    Resize, Lambda)

from typing import (Optional, Tuple, Dict, Any, List)
from open3d.geometry import (PointCloud, KDTreeSearchParamHybrid as knn_sr)
from open3d.utility import Vector3dVector as vec
from ..utils.graphics_utils import (CameraInfo, Rmat2quat, getWorld2View2)
from torch.utils.data import IterableDataset

















            








                    
                    
            


