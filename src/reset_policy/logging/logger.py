"""
Centralized logging for reset policy training.

Log structure:
    logs/
    ├── episodes.csv              # Episode-level summary
    ├── motor_logs/               # Per-step motor data (max 200 files)
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
                    "coverage",
                    "steps",
                    "termination_reason",
                    "terminated",
                    "truncated",
                    # Reward components
                    "coverage_reward_sum",
                    "current_reward_sum",
                    "current_change_penalty_sum",
                    "hardware_error_penalty_sum",
                    "tension_penalty_sum",
                    "safety_penalty_sum",
                    # Diagnostics
                    "max_current",
                    "safety_interventions",
                    "hardware_error",
                    "hardware_error_ids",
                    # Safety intervention breakdown
                    "current_aware_scaling_count",
                    "temperature_high_count",
                    "temperature_critical_count",
                    "voltage_low_count",
                    "voltage_critical_count",
                    "tension_safety_count",
                    "opposing_motors_count",
                    "tension_too_low_count",
                    "high_horizontal_current_count",
                    "high_vertical_current_count",
                    "single_motor_over_current_count",
                    "position_limit_pull_count",
                    "position_limit_release_count",
                    "motor_stuck_count",
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
        
        # Write header with target and current positions
        self.current_motor_writer.writerow([
            "step",
            "action_count",
            # Target positions
            "motor_16_target", "motor_17_target", "motor_18_target", "motor_19_target",
            # Current positions
            "motor_16_position", "motor_17_position", "motor_18_position", "motor_19_position",
            # Position error (target - current)
            "motor_16_error", "motor_17_error", "motor_18_error", "motor_19_error",
            # Currents (mA)
            "motor_16_current", "motor_17_current", "motor_18_current", "motor_19_current",
            # Voltages (V)
            "motor_16_voltage", "motor_17_voltage", "motor_18_voltage", "motor_19_voltage",
            # Temperatures (°C)
            "motor_16_temp", "motor_17_temp", "motor_18_temp", "motor_19_temp",
            # Actions
            "motor_16_action", "motor_17_action", "motor_18_action", "motor_19_action",
        ])
        
        # Clean up old motor logs
        self._cleanup_old_motor_logs()
    
    def log_step(self, step_num, action_count, motor_data):
        """
        Log motor data for one step.
        
        Args:
            step_num: Step number within episode
            action_count: Global action count
            motor_data: Dictionary with keys:
                'targets': Target positions for motors [16, 17, 18, 19]
                'positions': Current positions for motors [16, 17, 18, 19]
                'currents': Current readings for motors [16, 17, 18, 19]
                'voltages': Voltage readings for motors [16, 17, 18, 19]
                'temperatures': Temperature readings for motors [16, 17, 18, 19]
                'actions': RL actions for motors [16, 17, 18, 19]
        """
        if self.current_motor_writer is None:
            return
        
        targets = motor_data.get('targets', [None]*4)
        positions = motor_data.get('positions', [None]*4)
        currents = motor_data.get('currents', [None]*4)
        voltages = motor_data.get('voltages', [None]*4)
        temperatures = motor_data.get('temperatures', [None]*4)
        actions = motor_data.get('actions', [None]*4)
        
        # Calculate position errors (target - current)
        errors = []
        for i in range(4):
            if targets[i] is not None and positions[i] is not None:
                errors.append(targets[i] - positions[i])
            else:
                errors.append(None)
        
        self.current_motor_writer.writerow([
            step_num,
            action_count,
            # Targets
            targets[0], targets[1], targets[2], targets[3],
            # Positions
            positions[0], positions[1], positions[2], positions[3],
            # Position errors
            errors[0], errors[1], errors[2], errors[3],
            # Currents
            currents[0], currents[1], currents[2], currents[3],
            # Voltages
            voltages[0], voltages[1], voltages[2], voltages[3],
            # Temperatures
            temperatures[0], temperatures[1], temperatures[2], temperatures[3],
            # Actions
            actions[0], actions[1], actions[2], actions[3],
        ])
        
        # Flush periodically
        if step_num % 10 == 0:
            self.current_motor_file.flush()
    
    def log_episode(self, episode_num, episode_reward, info, steps, 
                    terminated, truncated, episode_stats):
        """Log episode summary."""
        row = [
            episode_num,
            episode_reward,
            info.get("coverage", np.nan),
            steps,
            info.get("termination_reason", ""),
            terminated,
            truncated,
            # Reward components
            episode_stats.get('coverage_reward', 0.0),
            episode_stats.get('current_reward', 0.0),
            episode_stats.get('current_change_penalty', 0.0),
            episode_stats.get('hardware_error_penalty', 0.0),
            episode_stats.get('tension_penalty', 0.0),
            episode_stats.get('safety_penalty', 0.0),
            # Diagnostics
            episode_stats.get('max_current', 0.0),
            episode_stats.get('safety_interventions', 0),
            bool(episode_stats.get('hardware_error_ids', set())),
            ",".join(map(str, sorted(episode_stats.get('hardware_error_ids', set())))),
            # Safety intervention breakdown
            episode_stats.get('safety_reasons', {}).get('current_aware_scaling', 0),
            episode_stats.get('safety_reasons', {}).get('temperature_high', 0),
            episode_stats.get('safety_reasons', {}).get('temperature_critical', 0),
            episode_stats.get('safety_reasons', {}).get('voltage_low', 0),
            episode_stats.get('safety_reasons', {}).get('voltage_critical', 0),
            episode_stats.get('safety_reasons', {}).get('tension_safety', 0),
            episode_stats.get('safety_reasons', {}).get('opposing_motors', 0),
            episode_stats.get('safety_reasons', {}).get('tension_too_low', 0),
            episode_stats.get('safety_reasons', {}).get('high_horizontal_current', 0),
            episode_stats.get('safety_reasons', {}).get('high_vertical_current', 0),
            episode_stats.get('safety_reasons', {}).get('single_motor_over_current', 0),
            episode_stats.get('safety_reasons', {}).get('position_limit_pull', 0),
            episode_stats.get('safety_reasons', {}).get('position_limit_release', 0),
            episode_stats.get('safety_reasons', {}).get('motor_stuck', 0),
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