"""
Centralized logging for reset policy training.

Log structure:
    logs/
    ├── episodes.csv              # Episode-level summary
    ├── motor_logs/               # Per-step motor + observation data (max 200 files)
    │   └── episode_XXXX_motors.csv
    └── training_metrics.csv      # PPO training metrics
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np


class TrainingLogger:
    """Centralized logging for training."""

    def __init__(self, log_dir: Path = None):
        """Initialize logger with directory structure."""
        if log_dir is None:
            log_dir = Path("/home/madhoolika/workspace/reset_policy/src/logs")

        self.log_dir = Path(log_dir)
        self.motor_log_dir = self.log_dir / "motor_logs"

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.motor_log_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.episodes_file = self.log_dir / "episodes.csv"
        self.training_metrics_file = self.log_dir / "training_metrics.csv"

        # Motor log management
        self.max_motor_logs = 200
        self.current_motor_file = None
        self.current_motor_writer = None

        # Initialize CSV files
        self._initialize_episodes_log()
        self._initialize_training_metrics_log()

        print(f"Logging to: {self.log_dir}")

    def _initialize_episodes_log(self):
        """Initialize episodes CSV."""
        if not self.episodes_file.exists():
            with open(self.episodes_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "total_reward",
                    "steps",
                    "termination_reason",
                    "terminated",
                    "truncated",
                    # Goal tracking
                    "goal_x",
                    "goal_y",
                    "final_distance_to_goal",
                    "goal_reached",
                    # Reward components
                    "distance_reward_sum",
                    "velocity_reward_sum",
                    "current_change_penalty_sum",
                    "hardware_error_penalty_sum",
                    "tension_penalty_sum",
                    "safety_penalty_sum",
                    # Diagnostics
                    "max_current",
                    "safety_interventions",
                    "hardware_error",
                    "hardware_error_ids",
                    # Safety penalty sums (by reason)
                    "safety_penalty_current_aware_scaling",
                    "safety_penalty_temperature_high",
                    "safety_penalty_temperature_critical",
                    "safety_penalty_voltage_low",
                    "safety_penalty_voltage_critical",
                    "safety_penalty_tension_safety",
                    "safety_penalty_opposing_motors",
                    "safety_penalty_tension_too_low",
                    "safety_penalty_high_horizontal_current",
                    "safety_penalty_high_vertical_current",
                    "safety_penalty_single_motor_over_current",
                    "safety_penalty_position_limit_pull",
                    "safety_penalty_position_limit_release",
                    "safety_penalty_motor_stuck",
                ])

    def _initialize_training_metrics_log(self):
        """Initialize training metrics CSV."""
        if not self.training_metrics_file.exists():
            with open(self.training_metrics_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "actor_loss",
                    "critic_loss",
                    "entropy",
                    "buffer_size",
                ])

    def start_episode(self, episode_num):
        """Start logging for a new episode."""
        # Close previous motor log if open
        self.close_motor_log()

        # Create new motor log
        motor_filename = f"episode_{episode_num:04d}_motors.csv"
        motor_filepath = self.motor_log_dir / motor_filename

        self.current_motor_file = open(motor_filepath, "w", newline="")
        self.current_motor_writer = csv.writer(self.current_motor_file)

        # Write header with cube state + velocity + observation + actions
        self.current_motor_writer.writerow([
            "step",
            "action_count",

            # Cube state (the "output" for dynamics)
            "cube_x", "cube_y",

            # Distance to goal (per step)
            "dist_to_goal_x", "dist_to_goal_y", "dist_to_goal",

            # --- Velocity (NEW) ---
            "vx_actual", "vy_actual", "v_error",

            # Observation dim 0-2: normalized cube position
            "obs_cube_x_norm", "obs_cube_y_norm", "obs_cube_yaw_norm",

            # Observation dim 3-4: normalized goal position
            "obs_goal_x_norm", "obs_goal_y_norm",

            # Observation dim 5-8: motor position deltas (normalized)
            "obs_motor_16_pos_delta_norm", "obs_motor_17_pos_delta_norm",
            "obs_motor_18_pos_delta_norm", "obs_motor_19_pos_delta_norm",

            # Observation dim 9-12: normalized motor currents
            "obs_motor_16_current_norm", "obs_motor_17_current_norm",
            "obs_motor_18_current_norm", "obs_motor_19_current_norm",

            # Observation dim 13-15: normalized tensions
            "obs_horizontal_tension_norm", "obs_vertical_tension_norm",
            "obs_total_tension_norm",

            # Observation dim 16-19: normalized target errors
            "obs_motor_16_target_error_norm", "obs_motor_17_target_error_norm",
            "obs_motor_18_target_error_norm", "obs_motor_19_target_error_norm",

            # Actions (4-dim)
            "action_m16", "action_m17", "action_m18", "action_m19",
        ])

        # Clean up old motor logs
        self._cleanup_old_motor_logs()

    def log_step(self, step_num, action_count, motor_data, obs_data=None):
        """
        Log one step.

        Args:
            step_num: Step number within episode
            action_count: Global action count
            motor_data: Dictionary with keys 'actions', etc.
            obs_data: Dictionary with keys:
                'cube_x', 'cube_y'          # labels (post-step)
                'goal_x', 'goal_y'
                'vx_actual', 'vy_actual', 'v_error'  # velocity debug
                'obs'                        # 20-dim observation array
                'actions'                    # 4-dim action array
        """
        if self.current_motor_writer is None:
            return

        if obs_data is None:
            obs_data = {}

        # Unpack cube state
        cube_x = obs_data.get('cube_x')
        cube_y = obs_data.get('cube_y')

        # Unpack goal
        goal_x = obs_data.get('goal_x')
        goal_y = obs_data.get('goal_y')

        # Compute distances
        if cube_x is not None and goal_x is not None:
            dist_x = cube_x - goal_x
            dist_y = cube_y - goal_y
            dist_total = float(np.sqrt(dist_x**2 + dist_y**2))
        else:
            dist_x = dist_y = dist_total = None

        # Velocity debug fields
        vx_actual = obs_data.get('vx_actual', None)
        vy_actual = obs_data.get('vy_actual', None)
        v_error = obs_data.get('v_error', None)

        # Observation (20 dims)
        obs = obs_data.get('obs', [None] * 20)
        if obs is None:
            obs = [None] * 20

        # Actions (4 dims)
        actions = obs_data.get('actions', motor_data.get('actions', [None] * 4))

        self.current_motor_writer.writerow([
            step_num,
            action_count,

            # Cube state
            cube_x, cube_y,

            # Distance to goal
            dist_x, dist_y, dist_total,

            # Velocity
            vx_actual, vy_actual, v_error,

            # Observation (20 dims)
            obs[0], obs[1], obs[2],
            obs[3], obs[4],
            obs[5], obs[6], obs[7], obs[8],
            obs[9], obs[10], obs[11], obs[12],
            obs[13], obs[14], obs[15],
            obs[16], obs[17], obs[18], obs[19],

            # Actions
            actions[0], actions[1], actions[2], actions[3],
        ])

        # Flush periodically
        if step_num % 10 == 0:
            self.current_motor_file.flush()

    def log_episode(self, episode_num, episode_reward, info, steps,
                    terminated, truncated, episode_stats):
        """Log episode summary."""
        safety_penalties = episode_stats.get('safety_penalty_by_reason', {})

        row = [
            episode_num,
            episode_reward,
            steps,
            info.get("termination_reason", ""),
            terminated,
            truncated,
            # Goal tracking
            episode_stats.get('goal_x', np.nan),
            episode_stats.get('goal_y', np.nan),
            episode_stats.get('final_distance_to_goal', np.nan),
            episode_stats.get('goal_reached', False),
            # Reward components
            episode_stats.get('distance_reward', 0.0),
            episode_stats.get('velocity_reward', 0.0),
            episode_stats.get('current_change_penalty', 0.0),
            episode_stats.get('hardware_error_penalty', 0.0),
            episode_stats.get('tension_penalty', 0.0),
            episode_stats.get('safety_penalty', 0.0),
            # Diagnostics
            episode_stats.get('max_current', 0.0),
            episode_stats.get('safety_interventions', 0),
            bool(episode_stats.get('hardware_error_ids', set())),
            ",".join(map(str, sorted(episode_stats.get('hardware_error_ids', set())))),
            # Safety penalty sums
            safety_penalties.get('current_aware_scaling', 0.0),
            safety_penalties.get('temperature_high', 0.0),
            safety_penalties.get('temperature_critical', 0.0),
            safety_penalties.get('voltage_low', 0.0),
            safety_penalties.get('voltage_critical', 0.0),
            safety_penalties.get('tension_safety', 0.0),
            safety_penalties.get('opposing_motors', 0.0),
            safety_penalties.get('tension_too_low', 0.0),
            safety_penalties.get('high_horizontal_current', 0.0),
            safety_penalties.get('high_vertical_current', 0.0),
            safety_penalties.get('single_motor_over_current', 0.0),
            safety_penalties.get('position_limit_pull', 0.0),
            safety_penalties.get('position_limit_release', 0.0),
            safety_penalties.get('motor_stuck', 0.0),
        ]

        with open(self.episodes_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_training_metrics(self, episode_num, losses, buffer_size):
        """Log PPO training metrics."""
        row = [
            episode_num,
            losses.get('actor_loss', np.nan),
            losses.get('critic_loss', np.nan),
            losses.get('entropy', np.nan),
            buffer_size,
        ]

        with open(self.training_metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def close_motor_log(self):
        """Close current motor log file."""
        if self.current_motor_file is not None:
            self.current_motor_file.flush()
            self.current_motor_file.close()
            self.current_motor_file = None
            self.current_motor_writer = None

    def _cleanup_old_motor_logs(self):
        """Keep only the most recent N motor logs."""
        motor_files = sorted(self.motor_log_dir.glob("episode_*_motors.csv"))

        if len(motor_files) > self.max_motor_logs:
            files_to_delete = motor_files[:-self.max_motor_logs]
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    print(f"Deleted old motor log: {file_path.name}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    def close(self):
        """Close all open files."""
        self.close_motor_log()


# Singleton instance
logger = TrainingLogger()