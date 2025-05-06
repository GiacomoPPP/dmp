import torch
import torch.nn as nn

import torch.nn.functional as functional

import functools
import operator

from config import EctConfig
from layer import EctLayer

from torch_geometric.data import Batch


hidden = 100

class EctCnnModel(nn.Module):
    def __init__(self, ectconfig: EctConfig):
        super().__init__()
        self.ectlayer = EctLayer(ectconfig)
        self.ectconfig = ectconfig

        self.conv = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3),
                nn.MaxPool2d(2)
            ).to(ectconfig.device)

        num_features = functools.reduce(
                operator.mul,
                list(self.conv(
                    torch.rand(1, ectconfig.bump_steps, ectconfig.num_thetas, device = ectconfig.device)
                ).shape
            ))

        self.linear = nn.Sequential(
                nn.Linear(num_features, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
                nn.Linear(hidden, 1),
            ).to(ectconfig.device)

    def forward(self, batch: Batch):
        batch.x = functional.normalize(batch.x, p = 2, dim = 1)
        x = self.ectlayer(batch).unsqueeze(1)
        x = self.conv(x).view(x.size(0), -1)
        x = self.linear(x).squeeze(1)
        return x