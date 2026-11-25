"""Simulation-only configuration; importing this creates no directories."""
from dataclasses import dataclass
from pathlib import Path
from v4_noelastic_unchanged.params import BOARD

ROOT = Path(__file__).resolve().parent
SIMULATOR = ROOT / 'v4_noelastic_unchanged'
# Raw-coordinate bounds match src/reset_policy/config.py:
# x in [-0.50, -0.07], y in [-0.71, -0.12] metres.
GOAL_LOW = (BOARD.x_min, BOARD.y_min)
GOAL_HIGH = (BOARD.x_max, BOARD.y_max)

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
    distance_scale: float = 50.0  # Inverse of the 0.02 m scale in the log reward.
