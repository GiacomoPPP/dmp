import torch
import torch.nn as nn
from torch import Tensor

from DmiConfig import DmiConfig

class SequentialSmoothing(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: Tensor, step: int, config: DmiConfig):
        smoothness = (output[1:] - output[:-1]).norm(dim=1).sum()
        step_coefficient = 1/(step + 1)
        return config.sequential_smoothing * smoothness * step_coefficient
