"""
Plot v_error across all episodes in motor_logs/.

- X-axis: global step count (cumulative across episodes)
- Y-axis: v_error (m/s)
- One line per episode (colored by episode index)

Usage:
    python plot_verror.py
    python plot_verror.py --log-dir /path/to/motor_logs
    python plot_verror.py --overlay   # plot all episodes on the same axes, no offset
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_all_episodes(log_dir: Path):
    """Load all episode CSVs and return a list of dataframes sorted by episode."""
    files = sorted(log_dir.glob("episode_*_motors.csv"))
    if not files:
        raise FileNotFoundError(f"No episode CSVs found in {log_dir}")

    print(f"Found {len(files)} episode files")

    episodes = []
    for f in files:
        df = pd.read_csv(f)
        if "v_error" not in df.columns:
            print(f"WARNING: {f.name} has no v_error column, skipping")
            continue
        episodes.append((f.stem, df))
    return episodes


def plot_verror_vs_step(episodes, out_path: Path, overlay: bool = False):
    """
    Plot v_error vs step.

    If overlay=False (default): X-axis is the global step count across all
    episodes (episode 1's steps are 0..N1-1, episode 2's are N1..N1+N2-1, ...).

    If overlay=True: each episode starts at step 0 (X-axis is step within episode).
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlabel("Global Step" if not overlay else "Step within Episode")
    ax.set_ylabel("v_error (m/s)")
    ax.set_title(f"Velocity Error across {len(episodes)} Episodes")
    ax.grid(True, alpha=0.3)

    cmap = plt.get_cmap("viridis")
    n = len(episodes)

    global_offset = 0
    for i, (name, df) in enumerate(episodes):
        v_error = df["v_error"].values
        steps = np.arange(len(v_error))

        if not overlay:
            x = steps + global_offset
            global_offset += len(v_error)
        else:
            x = steps

        color = cmap(i / max(1, n - 1))
        ax.plot(x, v_error, color=color, linewidth=0.8, alpha=0.75)

    # Optional: median line across all data as a guide
    all_v = np.concatenate([df["v_error"].values for _, df in episodes])
    median = float(np.median(all_v))
    ax.axhline(median, color="red", linestyle="--", linewidth=1.2,
               label=f"Overall median = {median:.5f} m/s")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.close(fig)


def print_summary(episodes):
    """Print per-episode and overall v_error stats."""
    print("\nPer-episode v_error summary:")
    print(f"{'Episode':<20} {'Steps':>6} {'Mean':>10} {'Median':>10} {'Std':>10} {'Max':>10}")
    all_v = []
    for name, df in episodes:
        v = df["v_error"].dropna().values
        if len(v) == 0:
            continue
        all_v.append(v)
        print(f"{name:<20} {len(v):>6} {v.mean():>10.5f} {np.median(v):>10.5f} "
              f"{v.std():>10.5f} {v.max():>10.5f}")

    if all_v:
        combined = np.concatenate(all_v)
        print(f"\nOverall: {len(combined)} steps")
        print(f"  mean   = {combined.mean():.5f}")
        print(f"  median = {np.median(combined):.5f}")
        print(f"  std    = {combined.std():.5f}")
        print(f"  max    = {combined.max():.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/motor_logs",
        help="Path to motor_logs/ directory (default: logs/motor_logs)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: <log-dir>/v_error_vs_step.png)",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Plot each episode starting at step 0 (default: global step count)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {log_dir}")

    episodes = load_all_episodes(log_dir)
    print_summary(episodes)

    out_path = Path(args.out) if args.out else log_dir / "v_error_vs_step.png"
    plot_verror_vs_step(episodes, out_path, overlay=args.overlay)


if __name__ == "__main__":
    main()