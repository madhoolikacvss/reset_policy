import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("episodes.csv")

# Preset goals
goals = [
#     (-0.296, -0.366),
#     (-0.225, -0.427),
#     (-0.365, -0.488),
#     (-0.142, -0.347),
    (-0.119, -0.528),
]

TOL = 1e-4

# ---------- Figure 1: Final distance to goal ----------
fig1, axes1 = plt.subplots(len(goals), 1, figsize=(10, 3 * len(goals)),
                           squeeze=False, sharex=False)

for i, (gx, gy) in enumerate(goals):
    mask = (df["goal_x"].sub(gx).abs() < TOL) & (df["goal_y"].sub(gy).abs() < TOL)
    sub = df[mask].sort_values("episode")

    ax = axes1[i][0]
    ax.plot(sub["episode"], sub["final_distance_to_goal"],
            marker="o", color="tab:red")
    ax.grid(True)
    ax.set_title(f"Goal ({gx}, {gy})", loc="left", fontsize=10)

    if i == len(goals) - 1:
        ax.set_xlabel("Episode")

axes1[0][0].set_ylabel("Final Distance to Goal")
fig1.suptitle("Final Distance to Goal", fontsize=13, fontweight="bold", y=0.995)
fig1.tight_layout()

# ---------- Figure 2: Total reward ----------
fig2, axes2 = plt.subplots(len(goals), 1, figsize=(10, 3 * len(goals)),
                           squeeze=False, sharex=False)

for i, (gx, gy) in enumerate(goals):
    mask = (df["goal_x"].sub(gx).abs() < TOL) & (df["goal_y"].sub(gy).abs() < TOL)
    sub = df[mask].sort_values("episode")

    ax = axes2[i][0]
    ax.plot(sub["episode"], sub["total_reward"],
            marker="o", color="tab:blue")
    ax.grid(True)
    ax.set_title(f"Goal ({gx}, {gy})", loc="left", fontsize=10)

    if i == len(goals) - 1:
        ax.set_xlabel("Episode")

axes2[0][0].set_ylabel("Total Reward")
fig2.suptitle("Total Reward", fontsize=13, fontweight="bold", y=0.995)
fig2.tight_layout()

# ----------------- figure 3: distance_reward_sum-----------
fig3, axes3 = plt.subplots(len(goals), 1, figsize=(10, 3 * len(goals)),
                           squeeze=False, sharex=False)

for i, (gx, gy) in enumerate(goals):
        mask = (df["goal_x"].sub(gx).abs() < TOL) & (df["goal_y"].sub(gy).abs() < TOL)
        sub = df[mask].sort_values("episode")

        ax = axes3[i][0]
        ax.plot(sub["episode"], sub["velocity_reward_sum"]/sub["steps"],
                marker="o", color="tab:green")
        ax.grid(True)
        ax.set_title(f"Goal ({gx}, {gy})", loc="left", fontsize=10)

        if i == len(goals) - 1:
                ax.set_xlabel("Episode")

axes3[0][0].set_ylabel("velocity_reward (sum/#steps))")
fig3.suptitle("velocity_reward per episode", fontsize=13, fontweight="bold", y=0.995)
fig3.tight_layout()     


plt.show()