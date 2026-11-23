import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def _save(fig, out_dir, name):
    if out_dir is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


def plot_training_metrics(csv_file_path, out_dir=None):
    """Plot q1_loss, q2_loss, actor_loss, alpha, target_q_mean vs global_step."""
    df = pd.read_csv(csv_file_path)

    required_columns = ['global_step', 'q1_loss', 'q2_loss', 'actor_loss', 'alpha', 'target_q_mean']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(df['global_step'], df['q1_loss'], 'b-', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('Q1 Loss', fontsize=12)
    axes[0].set_title('Q1 Loss vs Global Step', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df['global_step'], df['q2_loss'], 'r-', linewidth=1.5, alpha=0.7)
    axes[1].set_ylabel('Q2 Loss', fontsize=12)
    axes[1].set_title('Q2 Loss vs Global Step', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df['global_step'], df['actor_loss'], 'm-', linewidth=1.5, alpha=0.7)
    axes[2].set_ylabel('Actor Loss', fontsize=12)
    axes[2].set_title('Actor Loss vs Global Step', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(df['global_step'], df['alpha'], 'g-', linewidth=1.5, alpha=0.7)
    axes[3].set_ylabel('Alpha', fontsize=12)
    axes[3].set_title('Alpha vs Global Step', fontsize=14, fontweight='bold')
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(df['global_step'], df['target_q_mean'], 'k-', linewidth=1.5, alpha=0.7)
    axes[4].set_xlabel('Global Step', fontsize=12)
    axes[4].set_ylabel('Target Q Mean', fontsize=12)
    axes[4].set_title('Target Q Mean vs Global Step', fontsize=14, fontweight='bold')
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, out_dir, "training_metrics.png")
    plt.show()
    return df


def plot_with_rolling_average(csv_file_path, window=10, out_dir=None):
    """Same plots, with rolling averages overlaid."""
    df = pd.read_csv(csv_file_path)

    metrics = ['q1_loss', 'q2_loss', 'actor_loss', 'alpha', 'target_q_mean']
    for m in metrics:
        df[f'{m}_ma'] = df[m].rolling(window=window, min_periods=1).mean()

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    colors = ['b', 'r', 'm', 'g', 'k']

    for ax, m, c in zip(axes, metrics, colors):
        ax.plot(df['global_step'], df[m], color=c, linewidth=0.5, alpha=0.3, label='Raw')
        ax.plot(df['global_step'], df[f'{m}_ma'], color=c, linewidth=2, label=f'{window}-step MA')
        ax.set_ylabel(m.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{m.replace("_", " ").title()} vs Global Step (with Moving Average)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel('Global Step', fontsize=12)

    plt.tight_layout()
    _save(fig, out_dir, "training_metrics_rolling.png")
    plt.show()
    return df


def plot_combined(csv_file_path, out_dir=None):
    """All metrics normalized on the same axis."""
    df = pd.read_csv(csv_file_path)

    metrics = ['q1_loss', 'q2_loss', 'actor_loss', 'alpha', 'target_q_mean']
    colors  = ['b', 'r', 'm', 'g', 'k']

    fig, ax = plt.subplots(figsize=(12, 6))

    for m, c in zip(metrics, colors):
        norm = (df[m] - df[m].min()) / (df[m].max() - df[m].min() + 1e-10)
        ax.plot(df['global_step'], norm, color=c, linewidth=1.5,
                label=f'{m} (norm)', alpha=0.7)

    ax.set_xlabel('Global Step', fontsize=12)
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('Training Metrics Comparison (Normalized)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    _save(fig, out_dir, "training_metrics_combined.png")
    plt.show()
    return df


def plot_all_in_one(csv_file_path, out_dir=None):
    """2x3 grid: 5 individual plots + 1 combined."""
    df = pd.read_csv(csv_file_path)

    metrics = ['q1_loss', 'q2_loss', 'actor_loss', 'alpha', 'target_q_mean']
    colors  = ['b', 'r', 'm', 'g', 'k']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (m, c) in enumerate(zip(metrics, colors)):
        axes[i].plot(df['global_step'], df[m], color=c, linewidth=1.5)
        axes[i].set_xlabel('Global Step', fontsize=10)
        axes[i].set_ylabel(m.replace('_', ' ').title(), fontsize=10)
        axes[i].set_title(m.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)

    # Last panel: all combined (raw)
    for m, c in zip(metrics, colors):
        axes[5].plot(df['global_step'], df[m], color=c, linewidth=1, label=m, alpha=0.7)
    axes[5].set_xlabel('Global Step', fontsize=10)
    axes[5].set_ylabel('Value', fontsize=10)
    axes[5].set_title('All Metrics Combined', fontsize=12, fontweight='bold')
    axes[5].grid(True, alpha=0.3)
    axes[5].legend(fontsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "training_metrics_grid.png")
    plt.show()
    return df


if __name__ == "__main__":
    csv_file = "sac_train_metrics.csv"
    out_dir = "/home/madhoolika/workspace/reset_policy/src/logs/sac/plots"

    try:
        print("Creating basic plots...")
        df = plot_training_metrics(csv_file, out_dir=out_dir)

        print("\nCreating plots with rolling averages...")
        df = plot_with_rolling_average(csv_file, window=20, out_dir=out_dir)

        print("\nCreating combined plot...")
        df = plot_combined(csv_file, out_dir=out_dir)

        print("\nCreating grid layout...")
        df = plot_all_in_one(csv_file, out_dir=out_dir)

        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Total steps: {len(df)}")
        for m in ['q1_loss', 'q2_loss', 'actor_loss', 'alpha', 'target_q_mean']:
            print(f"\n{m}:")
            print(f"  Mean: {df[m].mean():.4f}")
            print(f"  Std:  {df[m].std():.4f}")
            print(f"  Min:  {df[m].min():.4f}")
            print(f"  Max:  {df[m].max():.4f}")

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found. Please check the file path.")
    except KeyError as e:
        print(f"Error: Column '{e}' not found in the CSV file.")
        print(f"Available columns: {pd.read_csv(csv_file).columns.tolist()}")