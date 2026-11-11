"""
SAC-specific logger.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class SACLogger:
    def __init__(self, log_dir: Path, max_motor_logs: int = 10000):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.motor_log_dir = self.log_dir / "motor_logs_sac"
        self.motor_log_dir.mkdir(parents=True, exist_ok=True)

        self.episodes_file = self.log_dir / "sac_episodes.csv"
        self.metrics_file = self.log_dir / "sac_train_metrics.csv"

        self.max_motor_logs = max_motor_logs
        self.current_motor_file = None
        self.current_motor_writer = None

        self._initialize_episodes_log()
        self._initialize_metrics_log()

        print(f"[SACLogger] Logging to: {self.log_dir}")

    # ---------- Initialization ----------

    def _initialize_episodes_log(self) -> None:
        if self.episodes_file.exists():
            return

        with open(self.episodes_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "episode", "steps", "total_reward",
                "termination_reason", "terminated", "truncated",
                "goal_x", "goal_y",
                "final_distance_to_goal", "goal_reached",
                "buffer_size", "alpha",
                "q1_loss_mean", "q2_loss_mean", "actor_loss_mean",
                "safety_interventions", "safety_penalty_sum",
                "distance_reward_sum",
                "current_change_penalty_sum",
                "hardware_error_penalty_sum",
                "tension_penalty_sum",
                "max_current",
                "hardware_error", "hardware_error_ids",
                "wall_time_s",
                # Per-reason safety penalties
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

    def _initialize_metrics_log(self) -> None:
        if self.metrics_file.exists():
            return

        with open(self.metrics_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "global_step", "q1_loss", "q2_loss",
                "actor_loss", "alpha", "target_q_mean",
            ])

    # ---------- Per-episode step log ----------

    def start_episode(self, episode_num: int) -> None:
        """Open a new per-step CSV for the given episode."""
        self.close_motor_log()

        motor_filename = f"episode_{episode_num:04d}_motors.csv"
        motor_filepath = self.motor_log_dir / motor_filename

        self.current_motor_file = open(motor_filepath, "w", newline="")
        self.current_motor_writer = csv.writer(self.current_motor_file)

        # Header — matches PPO's motor log column layout (plus penalty components)
        self.current_motor_writer.writerow([
            "step", "action_count",
            "cube_x", "cube_y",
            "dist_to_goal_x", "dist_to_goal_y", "dist_to_goal",
            "distance_reward",
            "current_change_penalty",
            "hardware_error_penalty",
            "tension_penalty",
            "safety_filtering_penalty",
            "total_reward",
            # Observation (20 dims)
            "obs_cube_x_norm", "obs_cube_y_norm", "obs_cube_yaw_norm",
            "obs_goal_x_norm", "obs_goal_y_norm",
            "obs_motor_16_pos_delta_norm", "obs_motor_17_pos_delta_norm",
            "obs_motor_18_pos_delta_norm", "obs_motor_19_pos_delta_norm",
            "obs_motor_16_current_norm", "obs_motor_17_current_norm",
            "obs_motor_18_current_norm", "obs_motor_19_current_norm",
            "obs_horizontal_tension_norm", "obs_vertical_tension_norm",
            "obs_total_tension_norm",
            "obs_motor_16_target_error_norm", "obs_motor_17_target_error_norm",
            "obs_motor_18_target_error_norm", "obs_motor_19_target_error_norm",
            # Actions
            "action_m16", "action_m17", "action_m18", "action_m19",
            "executed_m16", "executed_m17", "executed_m18", "executed_m19",
            # Diagnostics
            "action_modified",
            "safety_reason",
            "modification_magnitude",
            "hardware_error",
        ])

        self._cleanup_old_motor_logs()

    def log_step(
        self,
        step_num: int,
        action_count: int,
        obs_before: np.ndarray,
        action_raw: np.ndarray,
        step_info: dict,
    ) -> None:
        """Write one row for a single step."""
        if self.current_motor_writer is None:
            return

        cube_pos = step_info.get("cube_position", (None, None))
        goal_pos = step_info.get("goal_position", (None, None))
        cube_x, cube_y = cube_pos
        goal_x, goal_y = goal_pos

        if cube_x is not None and goal_x is not None:
            dist_x = cube_x - goal_x
            dist_y = cube_y - goal_y
            dist_total = float(np.sqrt(dist_x ** 2 + dist_y ** 2))
        else:
            dist_x = dist_y = dist_total = None

        executed = step_info.get("executed_actions")
        if executed is None:
            executed = [None] * 4

        obs_list = obs_before.tolist() if obs_before is not None else [None] * 20

        def _safe(v, default=None):
            if v is None:
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        row = [
            step_num, action_count,
            cube_x, cube_y,
            dist_x, dist_y, dist_total,
            _safe(step_info.get("distance_reward")),
            _safe(step_info.get("current_change_penalty")),
            _safe(step_info.get("hardware_error_penalty")),
            _safe(step_info.get("tension_penalty")),
            _safe(step_info.get("safety_filtering_penalty")),
            _safe(step_info.get("total_reward")),
            # Observation
            *obs_list,
            # Raw actions
            _safe(action_raw[0]), _safe(action_raw[1]),
            _safe(action_raw[2]), _safe(action_raw[3]),
            # Executed actions
            _safe(executed[0]), _safe(executed[1]),
            _safe(executed[2]), _safe(executed[3]),
            # Diagnostics
            bool(step_info.get("action_modified", False)),
            step_info.get("safety_reason", ""),
            _safe(step_info.get("modification_magnitude"), 0.0),
            bool(step_info.get("hardware_error", False)),
        ]

        self.current_motor_writer.writerow(row)

        if step_num % 10 == 0:
            self.current_motor_file.flush()

    def close_motor_log(self) -> None:
        if self.current_motor_file is not None:
            try:
                self.current_motor_file.flush()
                self.current_motor_file.close()
            except Exception:
                pass
            self.current_motor_file = None
            self.current_motor_writer = None

    def _cleanup_old_motor_logs(self) -> None:
        motor_files = sorted(self.motor_log_dir.glob("episode_*_motors.csv"))
        if len(motor_files) > self.max_motor_logs:
            for f in motor_files[:-self.max_motor_logs]:
                try:
                    f.unlink()
                except Exception as e:
                    print(f"[SACLogger] Failed to delete {f}: {e}")

    # ---------- Episode log ----------

    def log_episode(self, row: dict) -> None:
        by_reason = row.get("safety_penalty_by_reason", {}) or {}

        with open(self.episodes_file, "a", newline="") as f:
            csv.writer(f).writerow([
                row["episode"], row["steps"], row["total_reward"],
                row["termination_reason"], row["terminated"], row["truncated"],
                row["goal_x"], row["goal_y"],
                row["final_distance_to_goal"], row["goal_reached"],
                row["buffer_size"], row["alpha"],
                row["q1_loss_mean"], row["q2_loss_mean"], row["actor_loss_mean"],
                row["safety_interventions"], row["safety_penalty_sum"],
                row.get("distance_reward_sum", 0.0),
                row.get("current_change_penalty_sum", 0.0),
                row.get("hardware_error_penalty_sum", 0.0),
                row.get("tension_penalty_sum", 0.0),
                row.get("max_current", 0.0),
                row["hardware_error"], row["hardware_error_ids"],
                row["wall_time_s"],
                # Per-reason safety penalties
                by_reason.get("current_aware_scaling", 0.0),
                by_reason.get("temperature_high", 0.0),
                by_reason.get("temperature_critical", 0.0),
                by_reason.get("voltage_low", 0.0),
                by_reason.get("voltage_critical", 0.0),
                by_reason.get("tension_safety", 0.0),
                by_reason.get("opposing_motors", 0.0),
                by_reason.get("tension_too_low", 0.0),
                by_reason.get("high_horizontal_current", 0.0),
                by_reason.get("high_vertical_current", 0.0),
                by_reason.get("single_motor_over_current", 0.0),
                by_reason.get("position_limit_pull", 0.0),
                by_reason.get("position_limit_release", 0.0),
                by_reason.get("motor_stuck", 0.0),
            ])

    # ---------- Training metrics ----------

    def log_metric(self, row: dict) -> None:
        with open(self.metrics_file, "a", newline="") as f:
            csv.writer(f).writerow([
                row["global_step"], row["q1_loss"], row["q2_loss"],
                row["actor_loss"], row["alpha"], row["target_q_mean"],
            ])

    # ---------- Cleanup ----------

    def close(self) -> None:
        self.close_motor_log()