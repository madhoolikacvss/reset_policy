"""
Analyze the effect of different safety penalty scaling factors
using logged per-episode reward components.

Usage:
    python analyze_penalty_scaling.py path/to/episodes.csv

Outputs:
    - A table of total reward vs scaling factor (mean over episodes)
    - A per-episode CSV for further analysis
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

SCALING_FACTORS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

# The single hard clamp in _compute_safety_penalty
PER_STEP_PENALTY_CLAMP = -1.0

# Which columns represent the penalty components (sums over the episode)
SAFETY_REASON_COLUMNS = [
    "safety_penalty_current_aware_scaling",
    "safety_penalty_temperature_high",
    "safety_penalty_temperature_critical",
    "safety_penalty_voltage_low",
    "safety_penalty_voltage_critical",
    "safety_penalty_tension_safety",
    "safety_penalty_opposing_motors",
    "safety_penalty_tension_too_low",
    "safety_penalty_high_horizontal_current",
    "safety_penalty_high_vertical_current",
    "safety_penalty_single_motor_over_current",
    "safety_penalty_position_limit_pull",
    "safety_penalty_position_limit_release",
    "safety_penalty_motor_stuck",
]

# Non-safety reward terms (positive or negative, but not scaled)
FIXED_REWARD_COLUMNS = [
    "distance_reward_sum",
    "current_change_penalty_sum",
    "hardware_error_penalty_sum",
    "tension_penalty_sum",
]


# ============================================================
# Load and prepare
# ============================================================

def load_episodes(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} episodes from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    return df


def total_reward_under_scaling(df: pd.DataFrame, scale: float,
                               include_oob: bool = True) -> np.ndarray:
    """
    Recompute total_reward for each episode assuming the safety
    penalty sums are multiplied by `scale`.

    Notes
    -----
    - We only scale the *safety penalty* terms, not the distance reward
      or the other fixed penalties.
    - Out-of-bounds penalty is not logged separately. It's part of
      `safety_penalty_sum` in the current env code
      (safety_penalty + out_of_bound_penalty). If you want to exclude
      OOB, set include_oob=False; otherwise it's included implicitly
      via safety_penalty_sum.
    - If any single-step penalty was clamped at -1.0, scaling it by
      k>1 doesn't increase it. We approximate by clamping the
      per-episode contribution to [-steps, 0] — see `_scale_penalty_sum`.
    """
    fixed = df[FIXED_REWARD_COLUMNS].sum(axis=1).values
    safety_total = df[SAFETY_REASON_COLUMNS].sum(axis=1).values
    steps = df["steps"].values

    scaled_safety = _scale_penalty_sum(safety_total, steps, scale)

    total = fixed + scaled_safety

    # Clip to [0, 1]? No — the env clips only the reward-per-step, not
    # the episode sum. We leave the sum unclipped here. The env's clip
    # to [0, 1] applies to the per-step reward; the total reward is a
    # sum of clipped per-step rewards. Reconstructing that exactly
    # requires per-step data, which we don't have. This analysis is an
    # approximation of the *overall* effect.
    return total


def _scale_penalty_sum(safety_sum: np.ndarray, steps: np.ndarray,
                       scale: float) -> np.ndarray:
    """
    Approximate the new episode safety-penalty sum after scaling.
    Bounded below by -1 * steps (the hard per-step clamp).
    """
    scaled = safety_sum * scale
    lower_bound = -1.0 * steps
    # Safety sum is negative; clamp upper bound at 0
    return np.clip(scaled, lower_bound, 0.0)


# ============================================================
# Analysis
# ============================================================

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale in SCALING_FACTORS:
        totals = total_reward_under_scaling(df, scale)
        rows.append({
            "scale": scale,
            "mean_total_reward": totals.mean(),
            "median_total_reward": np.median(totals),
            "std_total_reward": totals.std(),
            "mean_safety_penalty_sum": df[SAFETY_REASON_COLUMNS].sum(axis=1).mean() * scale,
        })
    return pd.DataFrame(rows)


def per_goal_breakdown(df: pd.DataFrame) -> None:
    """Break down mean safety penalty per goal."""
    print("\nPer-goal safety penalty sum (from logs):")
    df = df.copy()
    df["goal_id"] = df["goal_x"].round(3).astype(str) + "_" + df["goal_y"].round(3).astype(str)
    grouped = df.groupby("goal_id").agg(
        episodes=("episode", "count"),
        mean_safety=("safety_penalty_sum", "mean"),
        mean_distance=("distance_reward_sum", "mean"),
        mean_total=("total_reward", "mean"),
    )
    print(grouped)


def plot_totals(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        summary["scale"],
        summary["mean_total_reward"],
        yerr=summary["std_total_reward"],
        marker="o", capsize=4,
    )
    ax.set_xlabel("Safety penalty scaling factor")
    ax.set_ylabel("Mean total reward (episode)")
    ax.set_title("Effect of safety penalty scaling on total reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_penalty_scaling.py <episodes.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    df = load_episodes(csv_path)

    summary = summarize(df)
    print("\nSummary across scaling factors:")
    print(summary.to_string(index=False))

    per_goal_breakdown(df)

    out_plot = csv_path.with_name("penalty_scaling_plot.png")
    plot_totals(summary, out_plot)

    out_csv = csv_path.with_name("penalty_scaling_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"Saved summary to {out_csv}")


if __name__ == "__main__":
    main()