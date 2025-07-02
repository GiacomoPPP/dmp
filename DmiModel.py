import torch
import torch.nn as nn
from torch import Tensor

from DmiConfig import DmiConfig
from EctLayer import EctLayer

from torch_geometric.data import Batch

from lightning import LightningModule

from SequentialSmoothing import SequentialSmoothing


hidden = 32

class DmiModel(LightningModule):
    def __init__(self, config: DmiConfig):
        super().__init__()

        self.ectlayer = EctLayer(config)
        self.config = config

        self.conv = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1))
            ).to(config.device)


        self.linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(p=config.dropout_rate),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(p=config.dropout_rate),
                nn.Linear(hidden, 1),
            ).to(config.device)


    def training_step(self, batch):
        x = self(batch)
        loss = self.penalized_loss(x, batch.y)
        self.log("training_loss", loss, batch_size = len(batch), on_epoch = True, on_step = False, prog_bar = True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay = self.config.optimizer_weight_decay)
        return optimizer


    def validation_step(self, batch):
        predicted = self(batch)
        loss_fn = nn.MSELoss()
        loss = loss_fn(predicted, batch.y)
        self.log("val_loss", loss, batch_size = len(batch), on_epoch = True, prog_bar = True)
        return loss


    def test_step(self, batch):
        y_hat = self(batch)
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_hat, batch.y)
        self.log("test_loss", loss, batch_size = len(batch), on_epoch = True, on_step = False)
        return loss


    def forward(self, batch: Batch):
        x: Tensor = self.ectlayer(batch).unsqueeze(1)
        x = self.conv(x).view(x.size(0), -1)
        x = self.linear(x).squeeze(1)
        return x


    def penalized_loss(self, predicted: Tensor, target: Batch) -> Tensor:
        lossFn = self.config.training_loss()

        sequentialSmoothing = SequentialSmoothing()

        ect_param_list: Tensor = self._get_ect_param_list()

        return lossFn(predicted, target) + sequentialSmoothing(ect_param_list, self.global_step, self.config)


    def _get_ect_param_list(self) -> Tensor:
        parameter_list: Tensor = list(self.ectlayer.parameters())[0]

        parameter_list = parameter_list.T

        return parameter_list