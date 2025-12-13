from lightning import LightningDataModule
from torch.utils.data import DataLoader
from .mip_nerf import MipNerfDataset
from ..configs.configs import *


__DATASETS__ = {
    "mip-nerf": MipNerfDataset
}


@dataclass
class DataLoaderConfig:
    name: Optional[str]=None
    params: Optional[DictConfig]=None
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
    
    def setup(self, stage: str) -> None:
        
        if stage == "fit":
            assert (self.cfg.train is not None), ("train data split is required in data module")
            train_cfg = self.cfg.train
            self.trainset = __DATASETS__[train_cfg.name](**train_cfg.params)
            if (self.cfg.validation is not None):
                print("LOADING THE VALIDATION SET")
                val_cfg = self.cfg.validation
                self.valset = __DATASETS__[val_cfg.name](**val_cfg.params)

        if stage == "test":
            if (self.cfg.test is not None):
                test_cfg = self.cfg.test
                self.testest = __DATASETS__[test_cfg.name](**test_cfg.params)
            

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

if __name__ == "__main__":
    
    config = {
        "train": {
            "name": "mip-nerf",
            "batch_size": 32,
            "shuffle": False,
            "params": {
                "path": "/home/ram/Downloads/360_v2",
                "target_size": (112, 224),
                "scene_type": "counter",
                "images_scale": 1,
                "pts_partition_size": 10000,
                "pts_partitions_n": 10,
                "pts_shuffle": False,
                "normal_knn": 30,
                "normal_radii": 0.1
            }
        },
        "validation": {
            "name": "mip-nerf",
            "batch_size": 32,
            "shuffle": False,
            "params": {
                "path": "/home/ram/Downloads/360_v2",
                "target_size": (112, 224),
                "scene_type": "counter",
                "images_scale": 1,
                "pts_partition_size": 10000,
                "pts_partitions_n": 5,
                "pts_shuffle": True,
                "normal_knn": 30,
                "normal_radii": 0.1
            }
        }
    }
    # cfg = OmegaConf.(config)
    OmegaConf.save(config, "/home/ram/Desktop/own_projects/tmp/GsCognetiveDPM/src/configs/data_sampler.yaml")
    dm = NeuroSplatDataModule(config)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    print(train_loader.dataset)

    