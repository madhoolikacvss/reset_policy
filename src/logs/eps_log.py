import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("episodes.csv")

# Preset goals
goals = [
    (-0.296, -0.366),
    (-0.225, -0.427),
    (-0.365, -0.488),
    (-0.142, -0.347),
    (-0.119, -0.528),
]

TOL = 1e-4

fig, axes = plt.subplots(len(goals), 2, figsize=(12, 14), squeeze=False, sharex=False)

for i, (gx, gy) in enumerate(goals):
    mask = (df["goal_x"].sub(gx).abs() < TOL) & (df["goal_y"].sub(gy).abs() < TOL)
    sub = df[mask].sort_values("episode")

    # ---- Left column: Final distance to goal ----
    ax = axes[i][0]
    ax.plot(sub["episode"], sub["final_distance_to_goal"],
            marker="o", color="tab:red")
    ax.grid(True)
    ax.set_title(f"Goal ({gx}, {gy})", loc="left", fontsize=10)

    # ---- Right column: Total reward ----
    ax = axes[i][1]
    ax.plot(sub["episode"], sub["total_reward"],
            marker="o", color="tab:blue")
    ax.grid(True)
    ax.set_title(f"Goal ({gx}, {gy})", loc="left", fontsize=10)

    # Only label bottom row's x-axis
    if i == len(goals) - 1:
        axes[i][0].set_xlabel("Episode")
        axes[i][1].set_xlabel("Episode")

# Only label y-axis on the first row
axes[0][0].set_ylabel("Final Distance to Goal")
axes[0][1].set_ylabel("Total Reward")

# Column headers (once, at the top)
axes[0][0].annotate("Final Distance to Goal", xy=(0.5, 1.15),
                    xycoords="axes fraction", ha="center", fontsize=12, fontweight="bold")
axes[0][1].annotate("Total Reward", xy=(0.5, 1.15),
                    xycoords="axes fraction", ha="center", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()