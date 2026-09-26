import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# ---- Load all episode CSVs in order ----
folder = r"C:\Users\madho\OneDrive - UC San Diego\Desktop\ys_lab\reset_policy\src\logs\motor_logs"
files = glob.glob(os.path.join(folder, "episode_*_motors.csv"))

def episode_num(path):
    m = re.search(r"episode_(\d+)_motors\.csv", os.path.basename(path))
    return int(m.group(1)) if m else -1

files = sorted(files, key=episode_num)

frames = []
offset = 0
for f in files:
    ep = pd.read_csv(f)
    if ep.empty:
        print(f"Skipping empty file: {os.path.basename(f)}")
        continue
    ep = ep.sort_values("step").reset_index(drop=True)
    ep["global_step"] = ep["step"] + offset
    offset = ep["global_step"].max() + 1
    frames.append(ep)

if not frames:
    raise RuntimeError("No non-empty CSV files found.")

df = pd.concat(frames, ignore_index=True)

# ---- Plot 1: Distance to goal ----
fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(df["global_step"], df["dist_to_goal"], color="tab:red")
ax1.set_xlabel("Step")
ax1.set_ylabel("Distance to Goal")
ax1.set_title("Distance to Goal Across All Episodes")
ax1.grid(True)
fig1.tight_layout()

# ---- Plot 2: Distance reward ----
fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(df["global_step"], df["distance_reward"], color="tab:blue")
ax2.set_xlabel("Step")
ax2.set_ylabel("Distance Reward")
ax2.set_title("Distance Reward Across All Episodes")
ax2.grid(True)
fig2.tight_layout()

plt.show()