from lightning import LightningDataModule
from torch.utils.data import DataLoader
from .mip_nerf import MipNerfDataset
from ..configs.configs import *


__DATASETS__ = {
    "mip-nerf": MipNerfDataset
}


@dataclass
class AttributesCollection:

    #mip nerf dataset attributes
    path: str
    target_size: Tuple[int, int]=None
    scene_type: str="bicycle"
    images_scale: int=1
    pts_partition_size: int=1000
    pts_partitions_n: int=40
    pts_shuffle: bool=False
    scene_scale: float=1.0
    cameras_scale: float=1.0
    normal_knn: int=11
    normal_radii: float=0.1


@dataclass
class DataLoaderConfig:
    name: Optional[str]=None
    params: AttributesCollection=AttributesCollection
    batch_size: Optional[int]=32
    shuffle: Optional[bool]=False
    num_workers: Optional[int]=0


class NeuroSplatDataModule(LightningDataModule):
    @dataclass
    class NeuroSplatDataModuleConfig:
        train: DataLoaderConfig
        validation: Optional[DataLoaderConfig]=None
        test: Optional[DataLoaderConfig]=None

    cfg: NeuroSplatDataModuleConfig=None

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = parse_structured(self.NeuroSplatDataModuleConfig, cfg)
        print(self.cfg.train)
    
    def _readset(self, set_cfg: DataLoaderConfig):
        splitset = __DATASETS__[set_cfg.name]
        splitset = splitset(**{
            key: value 
            for (key, value) in vars(set_cfg.params)["__annotations__"].items() 
            if key in vars(splitset.Config)["__annotations__"]
        })
        return splitset
    
    def setup(self, stage: str) -> None:
        
        if stage == "fit":
            assert (self.cfg.train is not None), ("train data split is required in data module")
            self.trainset = self._readset(self.cfg.train)
            if (self.cfg.validation is not None):
                print("LOADING THE VALIDATION SET")
                self.valset = self._readset(self.cfg.validation)
        if stage == "test":
            if (self.cfg.test is not None):
                self.testset = self._readset(self.cfg.test)
            

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.cfg.train.batch_size,
            shuffle=self.cfg.train.shuffle,
            num_workers=self.cfg.train.num_workers
        )
    
    def val_dataloader(self) -> DataLoader:
        assert (self.cfg.validation is not None), \
        ("""
        tried to get validation set 
        from data module while one is not set
        """)

        return DataLoader(
            dataset=self.valset,
            batch_size=self.cfg.validation.batch_size,
            shuffle=self.cfg.validation.shuffle,
            num_workers=self.cfg.validation.num_workers
        )
    
    def test_dataloader(self) -> DataLoader:
        assert (self.cfg.test is not None), \
        ("""
        tried to get test set 
        from data module while one is not set
        """)

        return DataLoader(
            dataset=self.trainset,
            batch_size=self.cfg.test.batch_size,
            shuffle=self.cfg.test.shuffle,
            num_workers=self.cfg.test.num_workers
        )

# if __name__ == "__main__":
    
#     config = {
#         "train": {
#             "name": "mip-nerf",
#             "batch_size": 32,
#             "shuffle": False,
#             "params": {
#                 "path": "/home/ram/Downloads/360_v2",
#                 "target_size": (112, 224),
#                 "scene_type": "counter",
#                 "images_scale": 1,
#                 "pts_partition_size": 10000,
#                 "pts_partitions_n": 10,
#                 "pts_shuffle": False,
#                 "normal_knn": 30,
#                 "normal_radii": 0.1
#             }
#         },
#         "validation": {
#             "name": "mip-nerf",
#             "batch_size": 32,
#             "shuffle": False,
#             "params": {
#                 "path": "/home/ram/Downloads/360_v2",
#                 "target_size": (112, 224),
#                 "scene_type": "counter",
#                 "images_scale": 1,
#                 "pts_partition_size": 10000,
#                 "pts_partitions_n": 5,
#                 "pts_shuffle": True,
#                 "normal_knn": 30,
#                 "normal_radii": 0.1
#             }
#         }
#     }
#     # cfg = OmegaConf.(config)
#     OmegaConf.save(config, "/home/ram/Desktop/own_projects/tmp/GsCognetiveDPM/src/configs/data_sampler.yaml")
#     dm = NeuroSplatDataModule(config)
#     dm.setup("fit")
#     train_loader = dm.train_dataloader()
#     print(train_loader.dataset)

    