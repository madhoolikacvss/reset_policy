import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_training_metrics(csv_file_path):
    """
    Plot actor_loss, critic_loss, and entropy vs episode number.
    
    Parameters:
    csv_file_path: Path to your CSV file
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Check if the required columns exist
    required_columns = ['episode', 'actor_loss', 'critic_loss', 'entropy']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Actor Loss
    ax1.plot(df['episode'], df['actor_loss'], 'b-', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('Actor Loss', fontsize=12)
    ax1.set_title('Actor Loss vs Episode', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Critic Loss
    ax2.plot(df['episode'], df['critic_loss'], 'r-', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('Critic Loss', fontsize=12)
    ax2.set_title('Critic Loss vs Episode', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Entropy
    ax3.plot(df['episode'], df['entropy'], 'g-', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Episode Number', fontsize=12)
    ax3.set_ylabel('Entropy', fontsize=12)
    ax3.set_title('Entropy vs Episode', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df

def plot_with_rolling_average(csv_file_path, window=10):
    """
    Plot with rolling averages to smooth the curves.
    
    Parameters:
    csv_file_path: Path to your CSV file
    window: Rolling window size for smoothing (default: 10)
    """
    df = pd.read_csv(csv_file_path)
    
    # Calculate rolling averages
    df['actor_loss_ma'] = df['actor_loss'].rolling(window=window, min_periods=1).mean()
    df['critic_loss_ma'] = df['critic_loss'].rolling(window=window, min_periods=1).mean()
    df['entropy_ma'] = df['entropy'].rolling(window=window, min_periods=1).mean()
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Actor Loss with moving average
    ax1.plot(df['episode'], df['actor_loss'], 'b-', linewidth=0.5, alpha=0.3, label='Raw')
    ax1.plot(df['episode'], df['actor_loss_ma'], 'b-', linewidth=2, label=f'{window}-episode MA')
    ax1.set_ylabel('Actor Loss', fontsize=12)
    ax1.set_title('Actor Loss vs Episode (with Moving Average)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Critic Loss with moving average
    ax2.plot(df['episode'], df['critic_loss'], 'r-', linewidth=0.5, alpha=0.3, label='Raw')
    ax2.plot(df['episode'], df['critic_loss_ma'], 'r-', linewidth=2, label=f'{window}-episode MA')
    ax2.set_ylabel('Critic Loss', fontsize=12)
    ax2.set_title('Critic Loss vs Episode (with Moving Average)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Entropy with moving average
    ax3.plot(df['episode'], df['entropy'], 'g-', linewidth=0.5, alpha=0.3, label='Raw')
    ax3.plot(df['episode'], df['entropy_ma'], 'g-', linewidth=2, label=f'{window}-episode MA')
    ax3.set_xlabel('Episode Number', fontsize=12)
    ax3.set_ylabel('Entropy', fontsize=12)
    ax3.set_title('Entropy vs Episode (with Moving Average)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    return df

def plot_combined(csv_file_path):
    """
    Plot all three metrics on the same graph for comparison.
    """
    df = pd.read_csv(csv_file_path)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize each metric to [0, 1] for fair comparison on same scale
    # (Optional - remove if you want raw values)
    actor_norm = (df['actor_loss'] - df['actor_loss'].min()) / (df['actor_loss'].max() - df['actor_loss'].min() + 1e-10)
    critic_norm = (df['critic_loss'] - df['critic_loss'].min()) / (df['critic_loss'].max() - df['critic_loss'].min() + 1e-10)
    entropy_norm = (df['entropy'] - df['entropy'].min()) / (df['entropy'].max() - df['entropy'].min() + 1e-10)
    
    ax.plot(df['episode'], actor_norm, 'b-', linewidth=1.5, label='Actor Loss (norm)', alpha=0.7)
    ax.plot(df['episode'], critic_norm, 'r-', linewidth=1.5, label='Critic Loss (norm)', alpha=0.7)
    ax.plot(df['episode'], entropy_norm, 'g-', linewidth=1.5, label='Entropy (norm)', alpha=0.7)
    
    ax.set_xlabel('Episode Number', fontsize=12)
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('Training Metrics Comparison (Normalized)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return df

def plot_all_in_one(csv_file_path):
    """
    Create a single figure with all plots arranged in a grid.
    """
    df = pd.read_csv(csv_file_path)
    
    # Create a 2x2 grid of plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Actor Loss
    ax1.plot(df['episode'], df['actor_loss'], 'b-', linewidth=1.5)
    ax1.set_xlabel('Episode', fontsize=10)
    ax1.set_ylabel('Actor Loss', fontsize=10)
    ax1.set_title('Actor Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Critic Loss
    ax2.plot(df['episode'], df['critic_loss'], 'r-', linewidth=1.5)
    ax2.set_xlabel('Episode', fontsize=10)
    ax2.set_ylabel('Critic Loss', fontsize=10)
    ax2.set_title('Critic Loss', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Entropy
    ax3.plot(df['episode'], df['entropy'], 'g-', linewidth=1.5)
    ax3.set_xlabel('Episode', fontsize=10)
    ax3.set_ylabel('Entropy', fontsize=10)
    ax3.set_title('Entropy', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: All three combined (raw values)
    ax4.plot(df['episode'], df['actor_loss'], 'b-', linewidth=1, label='Actor Loss', alpha=0.7)
    ax4.plot(df['episode'], df['critic_loss'], 'r-', linewidth=1, label='Critic Loss', alpha=0.7)
    ax4.plot(df['episode'], df['entropy'], 'g-', linewidth=1, label='Entropy', alpha=0.7)
    ax4.set_xlabel('Episode', fontsize=10)
    ax4.set_ylabel('Value', fontsize=10)
    ax4.set_title('All Metrics Combined', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    return df

# Example usage
if __name__ == "__main__":
    # Replace with your actual CSV file path
    csv_file = "training_metrics.csv"
    
    try:
        # Option 1: Basic separate plots
        print("Creating basic plots...")
        df = plot_training_metrics(csv_file)
        
        # Option 2: With rolling averages (smoothing)
        print("\nCreating plots with rolling averages...")
        df = plot_with_rolling_average(csv_file, window=20)
        
        # Option 3: Combined plot
        print("\nCreating combined plot...")
        df = plot_combined(csv_file)
        
        # Option 4: All in one grid
        print("\nCreating grid layout...")
        df = plot_all_in_one(csv_file)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Total episodes: {len(df)}")
        print(f"\nActor Loss:")
        print(f"  Mean: {df['actor_loss'].mean():.4f}")
        print(f"  Std: {df['actor_loss'].std():.4f}")
        print(f"  Min: {df['actor_loss'].min():.4f}")
        print(f"  Max: {df['actor_loss'].max():.4f}")
        print(f"\nCritic Loss:")
        print(f"  Mean: {df['critic_loss'].mean():.4f}")
        print(f"  Std: {df['critic_loss'].std():.4f}")
        print(f"  Min: {df['critic_loss'].min():.4f}")
        print(f"  Max: {df['critic_loss'].max():.4f}")
        print(f"\nEntropy:")
        print(f"  Mean: {df['entropy'].mean():.4f}")
        print(f"  Std: {df['entropy'].std():.4f}")
        print(f"  Min: {df['entropy'].min():.4f}")
        print(f"  Max: {df['entropy'].max():.4f}")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found. Please check the file path.")
    except KeyError as e:
        print(f"Error: Column '{e}' not found in the CSV file.")
        print(f"Available columns: {pd.read_csv(csv_file).columns.tolist()}")