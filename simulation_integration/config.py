"""Simulation-only configuration; importing this creates no directories."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIMULATOR = ROOT.parent.parent / 'reset_policy_simulation' / 'v4_noelastic_unchanged'
GOALS = ((-0.119, -0.528), (-0.296, -0.366), (-0.225, -0.427),
         (-0.365, -0.488), (-0.142, -0.347))

@dataclass
class TrainingConfig:
    episodes: int = 50
    max_steps: int = 400
    seed: int = 42
    batch_size: int = 256
    warmup_steps: int = 5000
    her_k: int = 4
    device: str = 'cpu'
    torch_threads: int = 1
    success_threshold: float = 0.005
    distance_scale: float = 10.0
