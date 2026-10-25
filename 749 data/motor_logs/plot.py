import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

folder = "/home/madhoolika/workspace/reset_policy/src/logs/motor_logs"
out_dir = os.path.join(folder, "../plots")
os.makedirs(out_dir, exist_ok=True)

files = glob.glob(os.path.join(folder, "episode_*_motors.csv"))

def episode_num(path):
    m = re.search(r"episode_(\d+)_motors\.csv", os.path.basename(path))
    return int(m.group(1)) if m else -1

files = sorted(files, key=episode_num)

# ---- Load all episodes ----
frames = []
for f in files:
    ep = pd.read_csv(f)
    if ep.empty:
        continue
    ep = ep.sort_values("step").reset_index(drop=True)
    ep["episode_id"] = episode_num(f)
    frames.append(ep)

if not frames:
    raise RuntimeError("No non-empty CSV files found.")

df = pd.concat(frames, ignore_index=True)

goals = [
    (-0.119, -0.528),
    (-0.296, -0.366),
    (-0.225, -0.427),
    (-0.365, -0.488),
    (-0.142, -0.347),
]
TOL = 1e-4
EPISODE_CUTOFF = 400


def make_plot(sub_df, y_col, ylabel, color, title, save_path,
              xlabel="Step", hlines=None, add_average=False):
    """Build one continuous-step plot and save it.

    hlines: optional list of (y_value, style_kwargs) tuples.
    add_average: if True, overlay a per-episode-averaged curve.
    """
    if sub_df.empty:
        print(f"No data for: {title}")
        return

    sub_df = sub_df.sort_values(["episode_id", "step"]).reset_index(drop=True)
    global_step = []
    offset = 0
    for ep_id, group in sub_df.groupby("episode_id", sort=True):
        g = group.sort_values("step").reset_index(drop=True)
        global_step.extend((g["step"] + offset).tolist())
        offset = g["step"].iloc[-1] + offset + 1
    sub_df["global_step"] = global_step

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sub_df["global_step"], sub_df[y_col], color=color,
            alpha=0.5, label="Raw")

        # Average curve across episodes, binned every BIN steps
    if add_average:
        BIN = 200
        df_avg = sub_df.copy()
        df_avg["bin"] = (df_avg["global_step"] // BIN) * BIN
        binned = (
            df_avg.groupby("bin")[y_col]
            .mean()
            .sort_index()
        )
        ax.plot(binned.index, binned.values,
                color="blue", linewidth=1.5, label=f"Average (per {BIN} steps)")

    if hlines:
        for y_val, kwargs in hlines:
            ax.axhline(y=y_val, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

# ============================================================
# Group 1: Episodes < 400 (all goals combined)
# ============================================================
df_pre = df[df["episode_id"] < EPISODE_CUTOFF]

make_plot(
    df_pre, "dist_to_goal", "Distance to Goal", "tab:red",
    f"Distance to Goal: Episodes < {EPISODE_CUTOFF} (fixed goal))",
    os.path.join(out_dir, "01_dist_to_goal_pre400.png"),
    hlines=[(0.05, dict(color="black", linestyle="--", linewidth=1.2,
                        label="0.05 m"))]
)

make_plot(
    df_pre, "distance_reward", "Distance Reward", "tab:blue",
    f"Distance Reward: Episodes < {EPISODE_CUTOFF} (fixed goal)",
    os.path.join(out_dir, "02_distance_reward_pre400.png")
)

# ============================================================
# Groups 2-6: Per-goal, episodes >= 400
# ============================================================
df_post = df[df["episode_id"] >= EPISODE_CUTOFF].copy()


make_plot(
    df_post, "distance_reward", "Distance Reward", "tab:blue",
    f"Distance Reward: Episodes > {EPISODE_CUTOFF} (changing goals)",
    os.path.join(out_dir, "02_distance_reward_post400.png"),
    add_average=True
)


if "goal_x" in df_post.columns and "goal_y" in df_post.columns:
    df_post["goal_x_ep"] = df_post["goal_x"]
    df_post["goal_y_ep"] = df_post["goal_y"]
else:
    df_post["goal_x_ep"] = df_post["cube_x"] - df_post["dist_to_goal_x"]
    df_post["goal_y_ep"] = df_post["cube_y"] - df_post["dist_to_goal_y"]

ep_goal = (
    df_post.groupby("episode_id")[["goal_x_ep", "goal_y_ep"]]
    .median()
    .reset_index()
)

for i, (gx, gy) in enumerate(goals, start=1):
    matched = ep_goal[
        (ep_goal["goal_x_ep"].sub(gx).abs() < TOL) &
        (ep_goal["goal_y_ep"].sub(gy).abs() < TOL)
    ]["episode_id"].tolist()

    sub = df_post[df_post["episode_id"].isin(matched)]
    print(f"Goal ({gx}, {gy}): {len(matched)} episodes, {len(sub)} rows")

    tag = f"goal{i}_{gx}_{gy}".replace(".", "p").replace("-", "m")

    make_plot(
        sub, "dist_to_goal", "Distance to Goal", "tab:red",
        f"Distance to Goal: Goal ({gx}, {gy}), Episodes {EPISODE_CUTOFF}-513",
        os.path.join(out_dir, f"{i:02d}_dist_to_goal_{tag}.png"),
        hlines=[
            (0.10, dict(color="black", linestyle="--", linewidth=1.2,
                        label="0.10 m")),
            (0.05, dict(color="black", linestyle=":",  linewidth=1.2,
                        label="0.05 m")),
        ]
    )

    make_plot(
        sub, "distance_reward", "Distance Reward", "tab:blue",
        f"Distance Reward: Goal ({gx}, {gy}), Episodes {EPISODE_CUTOFF}-513",
        os.path.join(out_dir, f"{i:02d}_distance_reward_{tag}.png")
    )

print(f"\nAll plots saved to: {out_dir}")