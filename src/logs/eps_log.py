# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_reward_and_penalties(csv_file_path, window=20):
#     """
#     Plot total reward, safety penalty sum, and tension too low penalty vs episode.
    
#     Parameters:
#     csv_file_path: Path to episodes.csv
#     window: Rolling window size for averaging (default: 20)
#     """
#     # Read the CSV file
#     df = pd.read_csv(csv_file_path)
    
#     # Strip any whitespace from column names
#     df.columns = df.columns.str.strip()
    
#     # Check if required columns exist
#     required_cols = ['episode', 'total_reward', 'safety_penalty_sum', 'safety_penalty_tension_too_low']
#     missing_cols = [col for col in required_cols if col not in df.columns]
    
#     if missing_cols:
#         print(f"Error: Missing columns: {missing_cols}")
#         print(f"Available columns: {df.columns.tolist()}")
#         return
    
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(14, 8))
    
#     # Plot total reward
#     ax.plot(df['episode'], df['total_reward'], 'b-', linewidth=1.5, alpha=0.7, 
#             label='Total Reward')
    
#     # Plot safety penalty sum
#     ax.plot(df['episode'], df['safety_penalty_sum'], 'r-', linewidth=1.5, alpha=0.7, 
#             label='Safety Penalty Sum')
    
#     # Plot tension too low penalty
#     ax.plot(df['episode'], df['safety_penalty_tension_too_low'], 'g-', linewidth=1.5, alpha=0.7, 
#             label='Tension Too Low Penalty')
    
#     # Add zero line for reference
#     ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
#     # Labels and title
#     ax.set_xlabel('Episode Number', fontsize=14)
#     ax.set_ylabel('Value', fontsize=14)
#     ax.set_title('Total Reward, Safety Penalty, and Tension Too Low Penalty vs Episode', 
#                  fontsize=16, fontweight='bold')
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc='upper right', fontsize=12)
#     ax.tick_params(labelsize=12)
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print summary statistics
#     print("\n" + "="*60)
#     print("SUMMARY STATISTICS")
#     print("="*60)
#     print(f"Total Reward:")
#     print(f"  Mean: {df['total_reward'].mean():.2f}")
#     print(f"  Min: {df['total_reward'].min():.2f}")
#     print(f"  Max: {df['total_reward'].max():.2f}")
#     print(f"\nSafety Penalty Sum:")
#     print(f"  Mean: {df['safety_penalty_sum'].mean():.2f}")
#     print(f"  Min: {df['safety_penalty_sum'].min():.2f}")
#     print(f"  Max: {df['safety_penalty_sum'].max():.2f}")
#     print(f"\nTension Too Low Penalty:")
#     print(f"  Mean: {df['safety_penalty_tension_too_low'].mean():.2f}")
#     print(f"  Min: {df['safety_penalty_tension_too_low'].min():.2f}")
#     print(f"  Max: {df['safety_penalty_tension_too_low'].max():.2f}")
    
#     return df

# def plot_reward_and_penalties_with_avg(csv_file_path, window=20):
#     """
#     Plot total reward, safety penalty sum, and tension too low penalty 
#     with moving averages.
    
#     Parameters:
#     csv_file_path: Path to episodes.csv
#     window: Rolling window size for averaging (default: 20)
#     """
#     # Read the CSV file
#     df = pd.read_csv(csv_file_path)
    
#     # Strip any whitespace from column names
#     df.columns = df.columns.str.strip()
    
#     # Check if required columns exist
#     required_cols = ['episode', 'total_reward', 'safety_penalty_sum', 'safety_penalty_tension_too_low']
#     missing_cols = [col for col in required_cols if col not in df.columns]
    
#     if missing_cols:
#         print(f"Error: Missing columns: {missing_cols}")
#         print(f"Available columns: {df.columns.tolist()}")
#         return
    
#     # Calculate moving averages
#     df['total_reward_ma'] = df['total_reward'].rolling(window=window, min_periods=1).mean()
#     df['safety_penalty_sum_ma'] = df['safety_penalty_sum'].rolling(window=window, min_periods=1).mean()
#     df['tension_too_low_ma'] = df['safety_penalty_tension_too_low'].rolling(window=window, min_periods=1).mean()
    
#     # Create the plot with two subplots (raw and averaged)
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
#     # Plot 1: Raw values
#     ax1.plot(df['episode'], df['total_reward'], 'b-', linewidth=1.0, alpha=0.5, 
#             label='Total Reward')
#     ax1.plot(df['episode'], df['safety_penalty_sum'], 'r-', linewidth=1.0, alpha=0.5, 
#             label='Safety Penalty Sum')
#     ax1.plot(df['episode'], df['safety_penalty_tension_too_low'], 'g-', linewidth=1.0, alpha=0.5, 
#             label='Tension Too Low Penalty')
#     ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
#     ax1.set_xlabel('Episode Number', fontsize=12)
#     ax1.set_ylabel('Value', fontsize=12)
#     ax1.set_title('Raw Values', fontsize=14, fontweight='bold')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend(loc='upper right', fontsize=10)
#     ax1.tick_params(labelsize=11)
    
#     # Plot 2: Averaged values
#     ax2.plot(df['episode'], df['total_reward_ma'], 'b-', linewidth=2.0, 
#             label=f'Total Reward ({window}-episode avg)')
#     ax2.plot(df['episode'], df['safety_penalty_sum_ma'], 'r-', linewidth=2.0, 
#             label=f'Safety Penalty Sum ({window}-episode avg)')
#     ax2.plot(df['episode'], df['tension_too_low_ma'], 'g-', linewidth=2.0, 
#             label=f'Tension Too Low Penalty ({window}-episode avg)')
#     ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
#     ax2.set_xlabel('Episode Number', fontsize=12)
#     ax2.set_ylabel('Value', fontsize=12)
#     ax2.set_title(f'Averaged Values ({window}-episode moving average)', fontsize=14, fontweight='bold')
#     ax2.grid(True, alpha=0.3)
#     ax2.legend(loc='upper right', fontsize=10)
#     ax2.tick_params(labelsize=11)
    
#     plt.tight_layout()
#     plt.show()
    
#     return df

# # Example usage
# if __name__ == "__main__":
#     # Replace with your actual file path
#     csv_file = "episodes.csv"
    
#     try:
#         # Option 1: Simple plot
#         print("Creating simple plot...")
#         df = plot_reward_and_penalties(csv_file, window=20)
        
#         # Option 2: Plot with moving averages (raw + averaged)
#         print("\nCreating plot with moving averages...")
#         df = plot_reward_and_penalties_with_avg(csv_file, window=20)
        
#     except FileNotFoundError:
#         print(f"Error: File '{csv_file}' not found. Please check the file path.")
#     except Exception as e:
#         print(f"Error: {e}")

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_safety_penalty_pie_chart(csv_file_path):
#     """
#     Create a pie chart showing the breakdown of safety penalties with a legend.
#     Penalties are negative values - the more negative, the bigger the contribution.
#     """
#     # Read the CSV file
#     df = pd.read_csv(csv_file_path)
    
#     # Strip any whitespace from column names
#     df.columns = df.columns.str.strip()
    
#     # Define the safety penalty columns
#     safety_penalty_columns = [
#         'safety_penalty_current_aware_scaling',
#         'safety_penalty_temperature_high',
#         'safety_penalty_temperature_critical',
#         'safety_penalty_voltage_critical',
#         'safety_penalty_tension_safety',
#         'safety_penalty_opposing_motors',
#         'safety_penalty_tension_too_low',
#         'safety_penalty_high_horizontal_current',
#         'safety_penalty_high_vertical_current',
#         'safety_penalty_single_motor_over_current',
#         'safety_penalty_position_limit_pull',
#         'safety_penalty_position_limit_release',
#         'safety_penalty_motor_stuck'
#     ]
    
#     # Check which columns exist
#     existing_columns = [col for col in safety_penalty_columns if col in df.columns]
    
#     if not existing_columns:
#         print("Error: None of the safety penalty columns found")
#         print(f"Available columns: {df.columns.tolist()}")
#         return
    
#     print(f"Found {len(existing_columns)} safety penalty columns")
    
#     # Calculate total sum for each penalty type
#     penalty_sums = {}
#     for col in existing_columns:
#         penalty_sums[col] = df[col].sum()
#         print(f"  {col}: {penalty_sums[col]:.2f}")
    
#     # Clean up column names for better readability
#     clean_names = {
#         'safety_penalty_current_aware_scaling': 'Current Scaling',
#         'safety_penalty_temperature_high': 'Temp High',
#         'safety_penalty_temperature_critical': 'Temp Critical',
#         'safety_penalty_voltage_critical': 'Voltage Critical',
#         'safety_penalty_tension_safety': 'Tension Safety',
#         'safety_penalty_opposing_motors': 'Opposing Motors',
#         'safety_penalty_tension_too_low': 'Tension Too Low',
#         'safety_penalty_high_horizontal_current': 'High Horiz Current',
#         'safety_penalty_high_vertical_current': 'High Vert Current',
#         'safety_penalty_single_motor_over_current': 'Single Motor Over Current',
#         'safety_penalty_position_limit_pull': 'Position Limit Pull',
#         'safety_penalty_position_limit_release': 'Position Limit Release',
#         'safety_penalty_motor_stuck': 'Motor Stuck'
#     }
    
#     # Create labels and values
#     labels = [clean_names.get(col, col.replace('safety_penalty_', '')) for col in existing_columns]
#     values = list(penalty_sums.values())
    
#     # For penalties, we want the absolute values (magnitude of penalty)
#     # More negative = bigger contribution
#     values_abs = [abs(v) for v in values]
    
#     # Filter out zero or near-zero values
#     non_zero_data = [(label, abs_val) for label, abs_val in zip(labels, values_abs) if abs_val > 0.001]
    
#     if not non_zero_data:
#         print("\nAll safety penalties are zero. No pie chart to display.")
#         return
    
#     labels, values_abs = zip(*non_zero_data)
    
#     # Calculate total safety penalty
#     total_safety_penalty_abs = sum(values_abs)
#     total_safety_penalty_actual = df['safety_penalty_sum'].sum() if 'safety_penalty_sum' in df.columns else -total_safety_penalty_abs
    
#     # Create the pie chart with legend
#     fig, ax = plt.subplots(figsize=(14, 10))
    
#     # Use a color palette
#     colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
#     # Create pie chart without labels on the slices
#     wedges, texts, autotexts = ax.pie(values_abs, 
#                                        labels=None,  # No labels on slices
#                                        autopct=lambda pct: f'{pct:.1f}%',  # Show only percentages
#                                        colors=colors,
#                                        startangle=90,
#                                        textprops={'fontsize': 10, 'weight': 'bold'})
    
#     # Create legend with labels and values
#     legend_labels = [f'{label}: {pct:.1f}%' for label, pct in zip(labels, [v/total_safety_penalty_abs*100 for v in values_abs])]
#     ax.legend(wedges, legend_labels, 
#              title="Penalty Types",
#              loc='center left',
#              bbox_to_anchor=(1, 0, 0.5, 1),
#              fontsize=10,
#              title_fontsize=12)
    
#     # Add title with actual total (negative value)
#     ax.set_title(f'Safety Penalty Breakdown\nTotal Safety Penalty: {total_safety_penalty_actual:.2f}', 
#                  fontsize=16, fontweight='bold')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print summary statistics
#     print("\n" + "="*60)
#     print("SAFETY PENALTY BREAKDOWN")
#     print("="*60)
#     print(f"Total safety penalty: {total_safety_penalty_actual:.2f}")
#     print("\nPenalty breakdown (by magnitude):")
    
#     # Sort by magnitude (largest penalty first)
#     sorted_data = sorted(zip(labels, values_abs), key=lambda x: x[1], reverse=True)
    
#     for label, abs_val in sorted_data:
#         percentage = (abs_val / total_safety_penalty_abs) * 100 if total_safety_penalty_abs > 0 else 0
#         actual_val = -abs_val  # Convert back to negative
#         print(f"  {label}: {actual_val:.2f} ({percentage:.1f}%)")
    
#     return df

# # Example usage
# if __name__ == "__main__":
#     # Replace with your actual file path
#     csv_file = "episodes.csv"
    
#     try:
#         df = plot_safety_penalty_pie_chart(csv_file)
        
#     except FileNotFoundError:
#         print(f"Error: File '{csv_file}' not found. Please check the file path.")
#     except Exception as e:
#         print(f"Error: {e}")

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_steps_per_episode(csv_file_path, window=20):
#     """
#     Simple plot of steps per episode with moving average.
    
#     Parameters:
#     csv_file_path: Path to episodes.csv
#     window: Rolling window size for averaging (default: 20)
#     """
#     # Read the CSV file
#     df = pd.read_csv(csv_file_path)
    
#     # Check if required columns exist
#     if 'episode' not in df.columns:
#         print("Error: 'episode' column not found")
#         return
    
#     if 'steps' not in df.columns:
#         print("Error: 'steps' column not found")
#         print(f"Available columns: {df.columns.tolist()}")
#         return
    
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(14, 8))
    
#     # Plot raw steps
#     ax.plot(df['episode'], df['steps'], 'b-', linewidth=1.0, alpha=0.4, 
#              label='Steps per Episode')
    
#     # Calculate and plot moving average
#     steps_ma = df['steps'].rolling(window=window, min_periods=1).mean()
#     ax.plot(df['episode'], steps_ma, 'r-', linewidth=2.5, 
#              label=f'{window}-episode Moving Average')
    
#     # Add horizontal line for overall average
#     avg_steps = df['steps'].mean()
#     ax.axhline(y=avg_steps, color='green', linestyle='--', alpha=0.7, 
#                 label=f'Overall Average: {avg_steps:.1f}')
    
#     # Labels and title
#     ax.set_xlabel('Episode Number', fontsize=14)
#     ax.set_ylabel('Number of Steps', fontsize=14)
#     ax.set_title('Steps per Episode', fontsize=16, fontweight='bold')
#     ax.grid(True, alpha=0.3)
    
#     # Put legend on top right
#     ax.legend(loc='upper right', fontsize=12)
    
#     ax.tick_params(labelsize=12)
    
#     plt.tight_layout()
#     plt.show()
    
#     return df

# # Example usage
# if __name__ == "__main__":
#     # Replace with your actual file path
#     csv_file = "episodes.csv"
    
#     try:
#         df = plot_steps_per_episode(csv_file, window=20)
        
#         # Print some basic statistics
#         print(f"\nSteps Statistics:")
#         print(f"  Total episodes: {len(df)}")
#         print(f"  Mean steps: {df['steps'].mean():.1f}")
#         print(f"  Median steps: {df['steps'].median():.1f}")
#         print(f"  Min steps: {df['steps'].min()}")
#         print(f"  Max steps: {df['steps'].max()}")
#         print(f"  Std dev: {df['steps'].std():.1f}")
        
#     except FileNotFoundError:
#         print(f"Error: File '{csv_file}' not found. Please check the file path.")
#     except KeyError as e:
#         print(f"Error: Column '{e}' not found in the CSV file.")
#         print(f"Available columns: {pd.read_csv(csv_file).columns.tolist()}")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def plot_episode_metrics(csv_file_path, window=20):
    """
    Create 4 separate plots from episodes.csv data.
    
    Parameters:
    csv_file_path: Path to episodes.csv
    window: Rolling window size for averaging (default: 20)
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Check required columns
    required_cols = ['episode', 'total_reward', 'coverage_reward_sum', 'current_reward_sum',
                     'current_change_penalty_sum', 'hardware_error_penalty_sum', 
                     'tension_penalty_sum', 'safety_penalty_sum']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Define color scheme for sub-rewards/penalties
    reward_colors = {
        'coverage_reward_sum': '#1f77b4',      # Blue
        'current_reward_sum': '#ff7f0e',       # Orange
        'current_change_penalty_sum': '#2ca02c', # Green
        'hardware_error_penalty_sum': '#d62728', # Red
        'tension_penalty_sum': '#9467bd',      # Purple
        'safety_penalty_sum': '#8c564b'        # Brown
    }
    
    # Calculate rolling averages
    df_ma = df.copy()
    for col in ['total_reward'] + list(reward_colors.keys()):
        if col in df.columns:
            df_ma[f'{col}_ma'] = df[col].rolling(window=window, min_periods=1).mean()
    
    # ================================================================
    # PLOT 1: Raw values - Total Reward with sub-components
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot total reward in bold
    ax1.plot(df['episode'], df['total_reward'], 'k-', linewidth=2.5, 
             label='Total Reward', alpha=1.0)
    
    # Plot sub-components with faint lines
    for col, color in reward_colors.items():
        if col in df.columns:
            ax1.plot(df['episode'], df[col], '-', color=color, linewidth=1.0, 
                    alpha=0.4, label=col.replace('_sum', '').replace('_penalty', ' Penalty'))
    
    ax1.set_xlabel('Episode', fontsize=14)
    ax1.set_ylabel('Reward / Penalty Value', fontsize=14)
    ax1.set_title('Raw Values: Total Reward vs Components', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10, ncol=2)
    ax1.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # ================================================================
    # PLOT 2: Averaged values - Total Reward with sub-components
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    # Plot averaged total reward in bold
    ax2.plot(df['episode'], df_ma['total_reward_ma'], 'k-', linewidth=2.5, 
             label='Total Reward (avg)', alpha=1.0)
    
    # Plot averaged sub-components with faint lines
    for col, color in reward_colors.items():
        if f'{col}_ma' in df_ma.columns:
            ax2.plot(df['episode'], df_ma[f'{col}_ma'], '-', color=color, linewidth=1.2, 
                    alpha=0.5, label=f"{col.replace('_sum', '').replace('_penalty', ' Penalty')} (avg)")
    
    ax2.set_xlabel('Episode', fontsize=14)
    ax2.set_ylabel('Reward / Penalty Value', fontsize=14)
    ax2.set_title(f'Averaged ({window}-episode): Total Reward vs Components', 
                  fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10, ncol=2)
    ax2.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # ================================================================
    # PLOT 3: Comparison of total reward raw vs averaged
    # ================================================================
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    ax3.plot(df['episode'], df['total_reward'], 'b-', linewidth=1.0, alpha=0.3, 
             label='Raw Total Reward')
    ax3.plot(df['episode'], df_ma['total_reward_ma'], 'r-', linewidth=2.5, 
             label=f'{window}-episode Average')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add shaded region for std deviation if enough data
    if len(df) > window:
        std = df['total_reward'].rolling(window=window, min_periods=1).std()
        ax3.fill_between(df['episode'], 
                         df_ma['total_reward_ma'] - std,
                         df_ma['total_reward_ma'] + std,
                         alpha=0.2, color='red', label='±1 STD')
    
    ax3.set_xlabel('Episode', fontsize=14)
    ax3.set_ylabel('Total Reward', fontsize=14)
    ax3.set_title('Total Reward: Raw vs Averaged', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=12)
    ax3.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # ================================================================
    # PLOT 4: All counts (split into logical groups)
    # ================================================================
    # Identify count columns
    count_columns = ['safety_interventions', 'hardware_error', 'current_aware_scaling_count',
                     'temperature_high_count', 'temperature_critical_count', 'voltage_low_count',
                     'voltage_critical_count', 'tension_safety_count', 'opposing_motors_count',
                     'tension_too_low_count', 'high_horizontal_current_count', 
                     'high_vertical_current_count', 'single_motor_over_current_count',
                     'position_limit_pull_count', 'position_limit_release_count', 
                     'motor_stuck_count']
    
    # Filter to columns that actually exist
    existing_counts = [col for col in count_columns if col in df.columns]
    
    if len(existing_counts) > 0:
        # Split counts into logical groups
        safety_cols = ['safety_interventions', 'tension_safety_count', 'opposing_motors_count',
                      'position_limit_pull_count', 'position_limit_release_count']
        current_cols = ['current_aware_scaling_count', 'high_horizontal_current_count',
                       'high_vertical_current_count', 'single_motor_over_current_count']
        temp_cols = ['temperature_high_count', 'temperature_critical_count',
                    'voltage_low_count', 'voltage_critical_count']
        error_cols = ['hardware_error', 'motor_stuck_count', 'tension_too_low_count']
        
        # Filter to existing columns
        safety_existing = [c for c in safety_cols if c in df.columns]
        current_existing = [c for c in current_cols if c in df.columns]
        temp_existing = [c for c in temp_cols if c in df.columns]
        error_existing = [c for c in error_cols if c in df.columns]
        
        # Determine how many subplots we need
        groups = []
        if safety_existing:
            groups.append(('Safety Events', safety_existing))
        if current_existing:
            groups.append(('Current Events', current_existing))
        if temp_existing:
            groups.append(('Temperature & Voltage', temp_existing))
        if error_existing:
            groups.append(('Errors & Stuck', error_existing))
        
        # Create individual plots for each group
        for group_name, cols in groups:
            fig4, ax4 = plt.subplots(figsize=(14, 8))
            
            colors = plt.cm.tab20(np.linspace(0, 1, len(cols)))
            for i, col in enumerate(cols):
                ax4.plot(df['episode'], df[col], '-', linewidth=2.0, 
                        color=colors[i], label=col, alpha=0.8)
            
            # Add rolling average for each count
            for i, col in enumerate(cols):
                ma_col = f'{col}_ma'
                df[ma_col] = df[col].rolling(window=window, min_periods=1).mean()
                ax4.plot(df['episode'], df[ma_col], '--', linewidth=1.5, 
                        color=colors[i], alpha=0.5, label=f'{col} (avg)')
            
            ax4.set_xlabel('Episode', fontsize=14)
            ax4.set_ylabel('Count', fontsize=14)
            ax4.set_title(f'{group_name} Counts (Raw and {window}-episode Avg)', 
                         fontsize=16, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='best', fontsize=10, ncol=2)
            ax4.tick_params(labelsize=12)
            
            plt.tight_layout()
            plt.show()
    
    return df

def plot_termination_analysis(csv_file_path):
    """
    Create separate plots for termination analysis.
    """
    df = pd.read_csv(csv_file_path)
    
    if 'termination_reason' not in df.columns:
        print("No termination_reason column found")
        return
    
    # ================================================================
    # PLOT: Termination reasons over episodes
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    unique_reasons = df['termination_reason'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_reasons)))
    
    for i, reason in enumerate(unique_reasons):
        mask = df['termination_reason'] == reason
        episodes = df.loc[mask, 'episode']
        # Add small jitter for visibility
        jitter = np.random.normal(0, 0.1, len(episodes))
        ax1.scatter(episodes, [i + jitter for jitter in jitter], 
                   c=[colors[i]], label=reason, alpha=0.6, s=50)
    
    ax1.set_yticks(range(len(unique_reasons)))
    ax1.set_yticklabels(unique_reasons, fontsize=11)
    ax1.set_xlabel('Episode', fontsize=14)
    ax1.set_ylabel('Termination Reason', fontsize=14)
    ax1.set_title('Termination Reasons Over Episodes', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=11)
    ax1.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # ================================================================
    # PLOT: Termination reason distribution (Pie chart)
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    reason_counts = df['termination_reason'].value_counts()
    ax2.pie(reason_counts.values, labels=reason_counts.index, autopct='%1.1f%%',
            colors=colors[:len(reason_counts)], textprops={'fontsize': 12})
    ax2.set_title('Termination Reason Distribution', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # ================================================================
    # PLOT: Termination reasons over episodes (stacked bar)
    # ================================================================
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    # Group episodes into bins
    bin_size = 50
    max_episode = df['episode'].max()
    bins = list(range(0, max_episode + bin_size, bin_size))
    df['episode_bin'] = pd.cut(df['episode'], bins=bins, right=False)
    
    # Create cross-tabulation
    cross_tab = pd.crosstab(df['episode_bin'], df['termination_reason'])
    
    # Plot stacked bars
    cross_tab.plot(kind='bar', stacked=True, ax=ax3, 
                   color=colors[:len(unique_reasons)], width=0.8)
    
    ax3.set_xlabel('Episode Range', fontsize=14)
    ax3.set_ylabel('Count', fontsize=14)
    ax3.set_title('Termination Reasons by Episode Range', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(loc='best', fontsize=11)
    ax3.tick_params(labelsize=11, rotation=45)
    
    plt.tight_layout()
    plt.show()

def create_reward_component_breakdown(csv_file_path, window=20):
    """
    Create individual plots for reward components breakdown.
    """
    df = pd.read_csv(csv_file_path)
    
    # Calculate rolling averages
    for col in ['total_reward', 'coverage_reward_sum', 'current_reward_sum',
                'current_change_penalty_sum', 'hardware_error_penalty_sum',
                'tension_penalty_sum', 'safety_penalty_sum']:
        if col in df.columns:
            df[f'{col}_ma'] = df[col].rolling(window=window, min_periods=1).mean()
    
    # ================================================================
    # PLOT: Penalty breakdown
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    penalty_cols = ['current_change_penalty_sum', 'hardware_error_penalty_sum', 
                    'tension_penalty_sum', 'safety_penalty_sum']
    penalty_cols = [col for col in penalty_cols if col in df.columns]
    
    penalty_colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, col in enumerate(penalty_cols):
        ax1.plot(df['episode'], df[col], '-', linewidth=1.0, alpha=0.4, 
                color=penalty_colors[i], label=col)
        ax1.plot(df['episode'], df[f'{col}_ma'], '-', linewidth=2.5, 
                color=penalty_colors[i], label=f'{col} (avg)')
    
    ax1.set_xlabel('Episode', fontsize=14)
    ax1.set_ylabel('Penalty Value', fontsize=14)
    ax1.set_title('Penalty Breakdown (Raw and Averaged)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=11, ncol=2)
    ax1.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # ================================================================
    # PLOT: Reward sources (coverage vs current)
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    reward_sources = ['coverage_reward_sum', 'current_reward_sum']
    reward_sources = [col for col in reward_sources if col in df.columns]
    reward_colors = ['#1f77b4', '#ff7f0e']
    
    for i, col in enumerate(reward_sources):
        ax2.plot(df['episode'], df[col], '-', linewidth=1.0, alpha=0.4, 
                color=reward_colors[i], label=col)
        ax2.plot(df['episode'], df[f'{col}_ma'], '-', linewidth=2.5, 
                color=reward_colors[i], label=f'{col} (avg)')
    
    ax2.set_xlabel('Episode', fontsize=14)
    ax2.set_ylabel('Reward Value', fontsize=14)
    ax2.set_title('Reward Sources: Coverage vs Current', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=12)
    ax2.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    csv_file = "episodes.csv"
    
    try:
        # Plot 1: Raw values
        print("Creating Plot 1: Raw values...")
        df = plot_episode_metrics(csv_file, window=20)
        
        # Plot 2: Termination analysis (3 separate plots)
        print("\nCreating Termination Analysis plots...")
        plot_termination_analysis(csv_file)
        
        # Plot 3: Penalty breakdown
        print("\nCreating Penalty Breakdown plot...")
        create_reward_component_breakdown(csv_file, window=20)
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found. Please check the file path.")
    except KeyError as e:
        print(f"Error: Column '{e}' not found in the CSV file.")
        print(f"Available columns: {pd.read_csv(csv_file).columns.tolist()}")