import torch
import torch.nn as nn
from torch import Tensor

class SequentialSmoothing(nn.Module):
    def __init__(self, lambda_smooth=1e-1):
        super().__init__()
        self.lambda_smooth = lambda_smooth

    def forward(self, output: Tensor, step: int):
        smoothness = torch.sum((output[1:] - output[:-1])**2)
        step_coefficient = 1/(step + 1)
        return self.lambda_smooth * smoothness * step_coefficient
