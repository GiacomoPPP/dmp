from dataclasses import dataclass

from RunType import RunType

import torch.nn as nn

@dataclass(frozen=True)
class DmiConfig:
    run_type: RunType = RunType.TRAIN,
    seed = 42
    num_thetas: int = 128
    bump_steps: int = 128
    R: float = 1.01
    ect_type: str = "edges"
    device: str = 'mps:0'
    n_samples = 2**10
    n_minibatch = 2**5
    n_epochs = 2000
    dropout_rate = 0.20
    log_every_n_steps = 50
    training_loss = nn.MSELoss
    sequential_smoothing = 1e-3
    include_hydrogens_in_training = False
    fast_run = False