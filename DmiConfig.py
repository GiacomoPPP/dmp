from dataclasses import dataclass

@dataclass(frozen=True)
class DmiConfig:
    num_thetas: int = 64
    bump_steps: int = 32
    R: float = 1.1
    ect_type: str = "edges"
    device: str = 'mps:0'
    num_features: int = 3
    n_samples = 2**10
    n_minibatch = 2**5
    n_epochs = 200
    assessment_rate = 5
    log_every_n_steps = 50
    fast_run = True