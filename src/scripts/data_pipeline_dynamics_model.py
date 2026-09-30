# scripts/check_dynamics_data.py
from pathlib import Path
import pandas as pd
import numpy as np

MOTOR_LOG_DIR = Path("/home/madhoolika/workspace/reset_policy/src/logs/motor_logs")

files = sorted(MOTOR_LOG_DIR.glob("episode_*_motors.csv"))
print(f"Found {len(files)} episodes")

# Inspect first file
df = pd.read_csv(files[0])
print(f"\nColumns: {list(df.columns)}")
print(f"Rows: {len(df)}")

# Verify transition construction
n_transitions = 0
n_skipped = 0
for t in range(len(df) - 1):
    s_t = df.iloc[t][["cube_x", "cube_y",
                       "action_m16", "action_m17", "action_m18", "action_m19"]].to_numpy(dtype=np.float32)
    cube_next = df.iloc[t + 1][["cube_x", "cube_y"]].to_numpy(dtype=np.float32)
    cube_t = df.iloc[t][["cube_x", "cube_y"]].to_numpy(dtype=np.float32)
    delta = cube_next - cube_t

    if not (np.isfinite(s_t).all() and np.isfinite(delta).all()):
        n_skipped += 1
        continue
    n_transitions += 1

    if t < 5:
        print(f"  t={t}: cube_t={cube_t}, action={s_t[2:]}, delta={delta * 1000} mm")

print(f"\nTransitions: {n_transitions}  (skipped {n_skipped} for NaN)")

# Aggregate across all files
total_trans = 0
total_skip = 0
for f in files:
    df = pd.read_csv(f)
    for t in range(len(df) - 1):
        s_t = df.iloc[t][["cube_x", "cube_y",
                           "action_m16", "action_m17", "action_m18", "action_m19"]].to_numpy(dtype=np.float32)
        cube_next = df.iloc[t + 1][["cube_x", "cube_y"]].to_numpy(dtype=np.float32)
        cube_t = df.iloc[t][["cube_x", "cube_y"]].to_numpy(dtype=np.float32)
        delta = cube_next - cube_t
        if not (np.isfinite(s_t).all() and np.isfinite(delta).all()):
            total_skip += 1
        else:
            total_trans += 1

print(f"\nTotal across all episodes:")
print(f"  Transitions: {total_trans}")
print(f"  Skipped for NaN: {total_skip}")
print(f"  Expected (152 episodes × ~399 transitions ≈ 60648): {152 * 399}")