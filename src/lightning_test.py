import torch 
import torch.nn as nn
import lightning as l
import os
import torch.nn.functional as F
from torchvision.datasets import MNIST
from typing import (Optional, Tuple, Union)
from .models.layers import get_activation, Mlp
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, Resize, Lambda, PILToTensor)
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score
)
from lightning.pytorch.callbacks import (ModelCheckpoint, EarlyStopping)
from torch.optim import (Optimizer, Adam)



#TODO research lightning module training configurations possibilities
class SimpleModule(l.LightningModule):

    def __init__(
        self,
        class_labels: int,
        img_size: Optional[Tuple[int, int]]=(128, 256),
        conv_heads: Optional[int]=3,
        hconv_features: Optional[int]=32,
        oconv_features: Optional[int]=32,
        hdense_features: Optional[int]=32,
        conv_act_fn: Optional[str]="tanh",
        dense_act_fn: Optional[str]="relu"
    ) -> None:
        
        super().__init__()
        self.class_labels = class_labels
        embed_size = [int(size / (2 ** conv_heads)) for size in img_size]
        self.CE_loss = nn.CrossEntropyLoss()
        self.ConvBase = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=(
                        1 if idx == 0 
                        else hconv_features if idx != 0 and idx != (conv_heads - 1) 
                        else oconv_features 
                    ),
                    out_channels=(
                        hconv_features if idx != (conv_heads - 1) 
                        else oconv_features
                    ),
                    stride=2,
                    padding=1,
                    kernel_size=(3, 3)
                ),
                nn.BatchNorm2d(hconv_features if idx != (conv_heads - 1) else oconv_features),
                get_activation(conv_act_fn)
            )
            for idx in range(conv_heads)
        ])
        self.DenseHead = nn.Sequential(
            Mlp(
                in_features=embed_size[0] * embed_size[1] * oconv_features,
                hiden_features=hdense_features,
                out_features=hdense_features,
                activation_fn=dense_act_fn
            ),
            Mlp(hdense_features, out_features=class_labels, activation_fn=dense_act_fn)
        )
    

    def validation_step(self, batch, batch_idx=None):
        with torch.no_grad():
            x, y = batch
            for layer in self.ConvBase:
                x = layer(x)
            
            x = torch.flatten(x, start_dim=1)
            logits = self.DenseHead(x)
            loss = self.CE_loss(logits, y)
            logits = torch.argmax(logits, dim=-1)

            logits = logits.cpu()
            y = y.cpu()
            accuracy = accuracy_score(logits, y)
            precision = precision_score(logits, y, average="weighted")
            recall = recall_score(logits, y, average="weighted")

            self.log("val/accuracy", accuracy)
            self.log("val/precision", precision)
            self.log("val/recall", recall)
            self.log("val/loss", loss)
    
    def training_step(self, batch, batch_idx) -> float:
        
        x, y = batch[0]
        for layer in self.ConvBase:
            x = layer(x)

        x = torch.flatten(x, start_dim=1)
        logits = self.DenseHead(x)
        loss = self.CE_loss(logits, y)
        self.log("train/loss", loss)

        return loss
    
    def configure_optimizers(self) -> Union[list, Optimizer]:
        return Adam(params=[
            {"params": self.ConvBase.parameters(), 
            "name": "conv_base", 
            "lr": 0.01},
            {"params": self.DenseHead.parameters(), 
            "naem": "dense_head", 
            "lr": 0.1}
        ])
    


if __name__ == "__main__":

    EPOCHS = 10
    EPOCHS_PER_VAL = 1
    METRICS2MONITOR = "accuracy"
    K_TOP_RESULTS = 3
    LOSS_PATIENCE = 3


    base_path = "/home/ram/Desktop/own_projects/tmp/GsCognetiveDPM/meta"
    root_dataset = os.path.join(base_path, "MNIST_data")
    model_ckpts = os.path.join(base_path, "MODEL_ckpts")
    if not os.path.exists(root_dataset):
        os.mkdir(root_dataset)

    img_size = (128, 256)
    dataset = MNIST(
        root=root_dataset,
        train=True,
        download=True,
        transform=Compose([
            PILToTensor(),
            Resize(img_size),
            Lambda(lambda img: (img / 255.0 if img.max() > 1 else img))
        ])
    )
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False
    )
    model = SimpleModule(class_labels=32)

    trainer = l.Trainer(
        max_epochs=EPOCHS,
        check_val_every_n_epoch=EPOCHS_PER_VAL,
        callbacks=[
            ModelCheckpoint(
                dirpath=model_ckpts,
                filename="{epoch}-{accuracy:.2f}-{recall:.2f}",
                monitor=f"val/{METRICS2MONITOR}",
                save_top_k=K_TOP_RESULTS,   
            ),
            EarlyStopping(
                monitor="train/loss",
                check_on_train_epoch_end=False,
                mode="min",
                patience=LOSS_PATIENCE
            )
        ]
    )
    trainer.fit(
        model=model,
        train_dataloaders=[train_loader],
        val_dataloaders=[val_loader]
    )
    


        