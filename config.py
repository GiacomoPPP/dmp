from dataclasses import dataclass

@dataclass(frozen=True)
class EctConfig:
    num_thetas: int = 64
    bump_steps: int = 32
    R: float = 1.1
    ect_type: str = "edges"
    device: str = 'mps:0'
    num_features: int = 3