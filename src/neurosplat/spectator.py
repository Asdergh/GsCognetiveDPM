import torch
import os
import matplotlib.cm
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


class Spectator:
    
    _save_dir: Optional[str]=None
    _writer = SummaryWriter(_save_dir)
    
    def save_grid_rgb(self, rgb_grid, mode, file) -> None:
        pass
        