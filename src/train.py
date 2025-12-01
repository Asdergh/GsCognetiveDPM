import lightning as l
import os
from pathlib import Path
from argparse import ArgumentParser
from .neurosplat import NeuroSplatAnnotater
from .datasets import NeuroSplatDataModule
from .configs.configs import load_cfg
from lightning.pytorch.callbacks import (ModelCheckpoint, EarlyStopping)

abs_path = str(Path(__file__).parent)
def main() -> None:
    
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(abs_path, "configs/experiment_config.yaml"))
    parser.add_argument("--pipline_config", type=str, default=os.path.join(abs_path, "configs/NeuroSplatPipeilne.yaml"))
    parser.add_argument("--data_sampler_config", type=str, default=os.path.join(abs_path, "configs/data_sampler.yaml"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--objective2monitor", type=str, default="d-ssim")
    parser.add_argument("--k_top_ckpt", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--epochs_per_val_step", type=int, default=3)
    args = parser.parse_args()

    print(args.config)
    if not os.path.exists(args.config):
        assert (os.path.exists(args.pipline_config)
                and os.path.exists(args.data_sampler_config)
                and os.path.exists(args.general_config)), ("the is no configuratioon files in the project. Can't start any process")

        cfg = load_cfg(args.general_utils)
        cfg.update({"pipeline": load_cfg(args.pipline_config)})
        cfg.update({"data_sampler": load_cfg(args.data_sampler_config)})
            
    else:
        cfg = load_cfg(args.config)

    dm_cfg, pipe_cfg = (cfg["data_sampler"], cfg["pipeline"])
    pipe_cfg["densify_until_step"] = min(args.epochs, pipe_cfg["densify_until_step"])
    pipe_cfg["log_until_step"] = min(args.epochs, pipe_cfg["log_until_step"])
   

    ckpts_dir = os.path.join(pipe_cfg["loging_path"], "NeuroSplatAnnotater_ckpts")
    if not os.path.exists(ckpts_dir):
        os.mkdir(ckpts_dir)

    dm = NeuroSplatDataModule(dm_cfg)
    pipeline = NeuroSplatAnnotater(pipe_cfg, args.epochs)
    ckpt_mode = ("min" if args.objective2monitor in pipe_cfg["losses"] else "max")
    ckpt_monitor = (f"{args.objective2monitor}-loss" 
                    if args.objective2monitor in pipe_cfg["losses"]
                    else f"{args.objective2monitor}-metric")
    trainer = l.Trainer(
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.epochs_per_val_step,
        num_sanity_val_steps=0,
        callbacks=[
            ModelCheckpoint(
                dirpath=ckpts_dir,
                filename="{epoch}-{val/d-ssim-loss: .2f}-{val/l1-loss: .2f}",
                monitor=f"val/{ckpt_monitor}",
                mode=ckpt_mode,
                save_last=True,
                save_top_k=args.k_top_ckpt
            ),
            EarlyStopping(
                monitor="train/general-loss",
                check_on_train_epoch_end=True,
                mode="min",
                patience=args.patience
            )
        ]
    )
    # print(dm.train_dataloader().dataset)
    # print(dm.val_dataloader().dataset)
    trainer.fit(pipeline, datamodule=dm)
    
    
    


if __name__ == "__main__":
    main()
    
        






