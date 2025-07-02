from dataclasses import dataclass


from DmpDataset import DmpDataset

from RunType import RunType

import torch.nn as nn

@dataclass(frozen=True)
class DmiConfig:
    run_type: RunType = RunType.TRAIN,
    dataset: DmpDataset = DmpDataset.ADRA1A
    seed = 42
    num_thetas: int = 64
    bump_steps: int = 128
    R: float = 1.1
    ect_type: str = "edges"
    device: str = 'mps:0'
    n_samples = 2**8
    n_minibatch = 2**6
    n_epochs = 300
    dropout_rate = 0.20
    log_every_n_steps = 50
    check_val_every_n_epoch = 10
    training_loss = nn.MSELoss
    sequential_smoothing = 0
    optimizer_weight_decay = 1e-4
    include_hydrogens_in_training = False
    fast_run = False