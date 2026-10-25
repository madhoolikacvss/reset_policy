"""
Load all logged episodes into a HER-augmented ReplayBuffer.

Supports two log formats:
  - New: includes per-step reward components (total_reward, current_change_penalty,
         tension_penalty, hardware_error_penalty, safety_filtering_penalty).
  - Old: only distance_reward and total_safety_penalty. Penalties are approximated.

Output:
  - buffer_from_logs.npz

Usage:
    python load_her.py \
        --motor-log-dir /path/to/motor_logs \
        --episodes-csv /path/to/episodes.csv \
        --out /path/to/buffer_from_logs.npz \
        --k 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from reset_policy.rl.replay_buffer import ReplayBuffer
from reset_policy.rl.her import HERBuffer
from reset_policy.config import config


# -------- Config --------

OBS_COLS = [
    "obs_cube_x_norm", "obs_cube_y_norm", "obs_cube_yaw_norm",
    "obs_goal_x_norm", "obs_goal_y_norm",
    "obs_motor_16_pos_delta_norm", "obs_motor_17_pos_delta_norm",
    "obs_motor_18_pos_delta_norm", "obs_motor_19_pos_delta_norm",
    "obs_motor_16_current_norm", "obs_motor_17_current_norm",
    "obs_motor_18_current_norm", "obs_motor_19_current_norm",
    "obs_horizontal_tension_norm", "obs_vertical_tension_norm",
    "obs_total_tension_norm",
    "obs_motor_16_target_error_norm", "obs_motor_17_target_error_norm",
    "obs_motor_18_target_error_norm", "obs_motor_19_target_error_norm",
]
ACTION_COLS = ["action_m16", "action_m17", "action_m18", "action_m19"]

GOAL_DIMS = slice(3, 5)
GOAL_SUCCESS_THRESHOLD = 0.005

X_MIN = config.environment.x_min
X_MAX = config.environment.x_max
Y_MIN = config.environment.y_min
Y_MAX = config.environment.y_max


# -------- Helpers --------

def normalize_goal(goal_physical) -> np.ndarray:
    gx, gy = float(goal_physical[0]), float(goal_physical[1])
    return np.array([
        (gx - X_MIN) / (X_MAX - X_MIN),
        (gy - Y_MIN) / (Y_MAX - Y_MIN),
    ], dtype=np.float32)


def recover_goal_from_obs(df: pd.DataFrame) -> np.ndarray:
    gx_norm = float(df["obs_goal_x_norm"].iloc[0])
    gy_norm = float(df["obs_goal_y_norm"].iloc[0])
    gx = gx_norm * (X_MAX - X_MIN) + X_MIN
    gy = gy_norm * (Y_MAX - Y_MIN) + Y_MIN
    return np.array([gx, gy], dtype=np.float32)


def load_episodes_summary(episodes_csv: Path) -> dict[int, dict]:
    """Load episodes.csv, keeping the LAST row for each episode.
    Handles ragged CSVs by reading only the needed columns and skipping bad rows.
    """
    if not episodes_csv.exists():
        print(f"WARNING: episodes.csv not found at {episodes_csv}")
        return {}

    needed = ["episode", "hardware_error", "termination_reason", "truncated"]

    try:
        df = pd.read_csv(
            episodes_csv,
            usecols=lambda c: c in needed,
            on_bad_lines="skip",
        )
    except Exception as e:
        print(f"WARNING: failed to read episodes.csv with usecols ({e}); "
              f"falling back to full read with skip")
        df = pd.read_csv(episodes_csv, on_bad_lines="skip", engine="python")

    for col in needed:
        if col not in df.columns:
            if col == "episode":
                print("ERROR: 'episode' column missing from episodes.csv")
                return {}
            print(f"WARNING: '{col}' missing; defaulting")
            df[col] = False if col in ("hardware_error", "truncated") else ""

    df = df.drop_duplicates(subset="episode", keep="last")
    df = df.set_index("episode")

    out: dict[int, dict] = {}
    for ep, row in df.iterrows():
        try:
            ep_int = int(ep)
        except (ValueError, TypeError):
            continue
        hw_raw = row.get("hardware_error", False)
        tr_raw = row.get("truncated", False)
        out[ep_int] = {
            "hardware_error": bool(hw_raw) if pd.notna(hw_raw) else False,
            "termination_reason": str(row.get("termination_reason", "")),
            "truncated": bool(tr_raw) if pd.notna(tr_raw) else False,
        }

    print(f"Loaded {len(out)} episode summaries from episodes.csv")
    return out


# -------- Episode to transitions --------

def load_episode_to_transitions(
    df: pd.DataFrame,
    goal_xy: np.ndarray,
    hw_error: bool,
) -> dict:
    """Convert one episode's motor log into transition arrays.

    Supports both new and old log formats. Returns a dict of arrays.
    """
    T = len(df)
    new_format = all(
        c in df.columns for c in [
            "total_reward", "current_change_penalty", "tension_penalty",
            "hardware_error_penalty", "safety_filtering_penalty",
        ]
    )

    # --- Observations (20-dim) ---
    obs_arr = df[OBS_COLS].to_numpy(dtype=np.float32)
    states = obs_arr
    next_states = np.empty_like(states)
    next_states[:-1] = obs_arr[1:]
    next_states[-1] = obs_arr[-1]

    # --- Actions ---
    actions = df[ACTION_COLS].to_numpy(dtype=np.float32)
    executed_actions = actions.copy()  # no filter reconstruction for now

    # --- Rewards ---
    distance_rewards = df["distance_reward"].to_numpy(dtype=np.float32)

    if new_format:
        change_penalties = df["current_change_penalty"].to_numpy(dtype=np.float32)
        hw_penalties = df["hardware_error_penalty"].to_numpy(dtype=np.float32)
        tension_penalties = df["tension_penalty"].to_numpy(dtype=np.float32)
        safety_penalties = df["safety_filtering_penalty"].to_numpy(dtype=np.float32)
        final_rewards = df["total_reward"].to_numpy(dtype=np.float32)
    else:
        # Old format: approximate
        if "total_safety_penalty" in df.columns:
            safety_penalties = df["total_safety_penalty"].to_numpy(dtype=np.float32)
        else:
            safety_penalties = np.zeros(T, dtype=np.float32)

        change_penalties = np.zeros(T, dtype=np.float32)
        hw_penalties = np.zeros(T, dtype=np.float32)
        tension_penalties = np.zeros(T, dtype=np.float32)

        goal_reached = (df["dist_to_goal"].to_numpy(dtype=np.float32)
                        < GOAL_SUCCESS_THRESHOLD)
        goal_bonus = goal_reached.astype(np.float32)
        final_rewards = (
            np.clip(distance_rewards + goal_bonus, 0.0, 1.0) + safety_penalties
        ).astype(np.float32)

        if hw_error and T > 0:
            final_rewards[-1] -= 1.0

    # --- Dones ---
    dones = np.zeros(T, dtype=np.float32)
    if T > 0:
        dones[-1] = 1.0

    # --- Goals ---
    desired_goals = np.tile(goal_xy.astype(np.float32), (T, 1))
    achieved_goals = df[["cube_x", "cube_y"]].to_numpy(dtype=np.float32)

    return {
        "states": states,
        "actions": actions,
        "executed_actions": executed_actions,
        "rewards": final_rewards,
        "next_states": next_states,
        "dones": dones,
        "achieved_goals": achieved_goals,
        "desired_goals": desired_goals,
        "distance_rewards": distance_rewards,
        "change_penalties": change_penalties,
        "hw_penalties": hw_penalties,
        "tension_penalties": tension_penalties,
        "safety_penalties": safety_penalties,
    }


# -------- Main loader --------

def load_all_episodes_into_her(motor_log_dir, episodes_csv, out_path, k):
    episode_summary = load_episodes_summary(episodes_csv)

    motor_files = sorted(motor_log_dir.glob("episode_*_motors.csv"))
    if not motor_files:
        raise FileNotFoundError(f"No motor logs found in {motor_log_dir}")

    base = ReplayBuffer(state_dim=20, action_dim=4, goal_dim=2)
    her = HERBuffer(
        base_buffer=base,
        goal_dims=GOAL_DIMS,
        k=k,
        strategy="future",
        normalize_goal=normalize_goal,
    )

    # Common required columns (present in both formats)
    required_common = OBS_COLS + ACTION_COLS + [
        "cube_x", "cube_y", "distance_reward", "dist_to_goal",
    ]

    n_loaded = 0
    n_skipped = 0
    n_transitions = 0
    n_new_format = 0
    n_old_format = 0

    for f in motor_files:
        try:
            ep_num = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            print(f"WARNING: cannot parse episode number from {f.name}; skipping.")
            n_skipped += 1
            continue

        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"WARNING: failed to read {f.name}: {e}; skipping.")
            n_skipped += 1
            continue

        if len(df) < 2:
            n_skipped += 1
            continue

        missing = [c for c in required_common if c not in df.columns]
        if missing:
            print(f"WARNING: {f.name} missing {missing}; skipping.")
            n_skipped += 1
            continue

        # Detect format
        is_new = all(
            c in df.columns for c in [
                "total_reward", "current_change_penalty", "tension_penalty",
                "hardware_error_penalty", "safety_filtering_penalty",
            ]
        )
        if is_new:
            n_new_format += 1
        else:
            n_old_format += 1

        goal_xy = recover_goal_from_obs(df)
        summary = episode_summary.get(ep_num, {})
        hw_error = summary.get("hardware_error", False)
        is_truncated = summary.get("truncated", False)

        episode = load_episode_to_transitions(df, goal_xy, hw_error)

        if is_truncated and len(episode["dones"]) > 0:
            episode["dones"][-1] = 0.0

        T = len(episode["rewards"])
        for t in range(T):
            her.add(
                state=episode["states"][t],
                action=episode["actions"][t],
                executed_action=episode["executed_actions"][t],
                reward=float(episode["rewards"][t]),
                next_state=episode["next_states"][t],
                done=float(episode["dones"][t]),
                achieved_goal=episode["achieved_goals"][t],
                desired_goal=episode["desired_goals"][t],
                distance_reward=float(episode["distance_rewards"][t]),
                current_change_penalty=float(episode["change_penalties"][t]),
                hardware_error_penalty=float(episode["hw_penalties"][t]),
                tension_penalty=float(episode["tension_penalties"][t]),
                safety_filtering_penalty=float(episode["safety_penalties"][t]),
            )

        n_relabeled = her.finalize_episode()
        n_loaded += 1
        n_transitions += T

        if n_loaded % 25 == 0 or n_loaded == len(motor_files):
            print(f"  [{n_loaded}/{len(motor_files)}] ep {ep_num}: "
                  f"{T} orig + {n_relabeled} relabeled (buffer={len(her)})")

    print(f"\nLoaded {n_loaded} episodes ({n_skipped} skipped).")
    print(f"  New format: {n_new_format}, Old format: {n_old_format}")
    print(f"Original transitions: {n_transitions}")
    print(f"Total buffer size:    {len(her)}")
    print(f"Effective multiplier: {len(her) / max(1, n_transitions):.2f}×")
    print("\nBuffer stats:")
    for key, val in her.stats().items():
        print(f"  {key}: {val}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    her.base.save(out_path)
    print(f"\nSaved buffer to {out_path}")


# -------- CLI --------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motor-log-dir", type=Path,
        default=Path("/home/madhoolika/workspace/reset_policy/src/logs/motor_logs"),
    )
    parser.add_argument(
        "--episodes-csv", type=Path,
        default=Path("/home/madhoolika/workspace/reset_policy/src/logs/episodes.csv"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("/home/madhoolika/workspace/reset_policy/src/logs/buffer_from_logs.npz"),
    )
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    load_all_episodes_into_her(
        motor_log_dir=args.motor_log_dir,
        episodes_csv=args.episodes_csv,
        out_path=args.out,
        k=args.k,
    )


if __name__ == "__main__":
    main()