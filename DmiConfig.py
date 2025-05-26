from dataclasses import dataclass

from RunType import RunType

@dataclass(frozen=True)
class DmiConfig:
    run_type: RunType = RunType.ANALYZE_DIRECTIONS,
    num_thetas: int = 128
    bump_steps: int = 64
    R: float = 1.1
    ect_type: str = "edges"
    device: str = 'mps:0'
    num_features: int = 3
    n_samples = 2**10
    n_minibatch = 2**5
    n_epochs = 500
    assessment_rate = 5
    log_every_n_steps = 50
    fast_run = True