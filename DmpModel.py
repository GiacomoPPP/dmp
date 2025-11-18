import torch
import torch.nn as nn
from torch import Tensor

from DmpConfig import DmpConfig
from EctLayer import EctLayer

from torch_geometric.data import Batch

from lightning import LightningModule

from SequentialSmoothing import SequentialSmoothing

config = DmpConfig()


class DmpModel(LightningModule):
    def __init__(self, root_mean_square: float):
        super().__init__()

        self.ectlayer = EctLayer()
        self.config = config

        self.root_mean_square = root_mean_square

        self.register_buffer("geometric_scale", torch.tensor(1.0))

        self.conv = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(4, 8, kernel_size=3, padding = 1),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1))
            ).to(config.device)

        hidden_neurons = 16

        self.linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(8, hidden_neurons),
                nn.LayerNorm(hidden_neurons),
                nn.ReLU(),
                nn.Dropout(p=config.dropout_rate),
                nn.Linear(hidden_neurons, 1),
            ).to(config.device)


    def training_step(self, batch):
        x = self(batch)
        loss = self.penalized_loss(x, batch.y)
        self.log("training_pmse", loss, batch_size = len(batch), on_epoch = True, on_step = False, prog_bar = True)
        relative_loss: float = loss / self.root_mean_square
        self.log("relative_training_pmse", relative_loss, batch_size = len(batch), on_epoch = True, on_step = False)
        return relative_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay = self.config.optimizer_weight_decay)
        return optimizer


    def validation_step(self, batch):
        predicted = self(batch)
        loss_fn = nn.MSELoss()
        loss = loss_fn(predicted, batch.y)
        self.log("validation_pmse", loss, batch_size = len(batch), on_epoch = True, prog_bar = True)
        relative_loss: float = loss / self.root_mean_square
        self.log("validation_relative_pmse", relative_loss, batch_size = len(batch), on_epoch = True, on_step = False)
        return relative_loss


    def test_step(self, batch):
        y_hat = self(batch)
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_hat, batch.y)
        self.log("test_mse", loss, batch_size = len(batch), on_epoch = True, on_step = False)
        relative_loss: float = loss / self.root_mean_square
        self.log("test_relative_mse", relative_loss, batch_size = len(batch), on_epoch = True, on_step = False)
        rmse = torch.sqrt(loss)
        self.log("test_rmse", rmse, batch_size = len(batch), on_epoch = True, on_step = False)
        return relative_loss


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