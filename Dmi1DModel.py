import functools
import torch
import torch.nn as nn
from torch import Tensor

from DmiConfig import DmiConfig
from EctLayer import EctLayer

from torch_geometric.data import Batch

import functools
import operator

from lightning import LightningModule


hidden = 32

class Dmi1DModel(LightningModule):
    def __init__(self, config: DmiConfig):
        super().__init__()
        self.loss_fn = nn.MSELoss()

        self.ectlayer = EctLayer(config)
        self.ectconfig = config

        self.conv = nn.Sequential(
                #nn.Conv2d(1,8, 3, padding=1, bias=False),
                #nn.Conv2d(1,8, kernel_size=(1,3), padding=1, bias=False),
                nn.Conv2d(1,8, kernel_size=(3,1), padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d(2),
                #nn.Conv2d(8, 16, 3, padding=1, bias=False),
                #nn.Conv2d(8, 16, kernel_size=(1,3), padding=1, bias=False),
                nn.Conv2d(8, 16, kernel_size=(3,1), padding=1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ).to(config.device)

        num_features = functools.reduce(
                operator.mul,
                list(self.conv(
                    torch.rand(1, 1, config.bump_steps, config.num_directions, device = config.device)
                ).shape
            ))


        self.linear = nn.Sequential(
                nn.Linear(num_features, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden, 1),
            ).to(config.device)


    def training_step(self, batch):
        x = self(batch)
        loss = self.loss_fn(x, batch.y)
        self.log("training_loss", loss, batch_size = len(batch), on_epoch = True, on_step = False, prog_bar = True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


    def validation_step(self, batch):
        predicted = self(batch)
        loss = self.loss_fn(predicted, batch.y)
        self.log("val_loss", loss, batch_size = len(batch), on_epoch = True, prog_bar = True)
        return loss


    def test_step(self, batch):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("test_loss", loss, batch_size = len(batch), on_epoch = True, on_step = False)
        return loss


    def forward(self, batch: Batch):
        x: Tensor = self.ectlayer(batch).unsqueeze(1)
        x = self.conv(x).view(x.size(0), -1)
        x = self.linear(x).squeeze(1)
        return x