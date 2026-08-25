from __future__ import annotations

import time
import gymnasium as gym
import numpy as np

from gymnasium import spaces

import sys
from pathlib import Path

from reset_policy.control.safety_filter import SafetyFilter

sys.path.append(str(Path(__file__).resolve().parent))

from observation import ObservationBuilder
from reward import RewardFunction
from occupancy_grid import OccupancyGrid
from renderer import BoardRenderer


class ResetPolicyEnv(gym.Env):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array"
        ]
    }

    def __init__(
        self,
        executor,
        observation_builder: ObservationBuilder,
        cube_tracker,
        occupancy_grid: OccupancyGrid,
        reward_function: RewardFunction,

        max_motor_delta=400,
        action_duration: float = 0.8,

        render_mode=None,

        max_steps=100,
        target_coverage=0.95,

        # =====================================================
        # Current thresholds
        # =====================================================

        safe_current_threshold=500.0,
        moderate_current_threshold=1800.0,
        high_current_threshold=2500.0,
        
        # =====================================================
        # Safety Filter
        # =====================================================
        safety_filter: SafetyFilter = None,
        safety_penalty_weight: float = 2.0,

    ):

        super().__init__()

        self.executor = executor
        self.obs_builder = observation_builder
        self.cube_tracker = cube_tracker

        self.grid = occupancy_grid
        self.reward_fn = reward_function

        self.action_duration = action_duration

        # Updated observation space to 14 dimensions (added tension features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.render_mode = render_mode

        if self.render_mode in ("human", "rgb_array"):
            self.renderer = BoardRenderer(
                board_width=(self.grid.x_max - self.grid.x_min) * 100,
                board_height=(self.grid.y_max - self.grid.y_min) * 100,
                cell_size=self.grid.cell_size * 100,
            )
        else:
            self.renderer = None

        self.max_steps = max_steps
        self.target_coverage = target_coverage

        self.safe_current_threshold = safe_current_threshold
        self.moderate_current_threshold = moderate_current_threshold
        self.high_current_threshold = high_current_threshold

        self.step_count = 0
        self.episode_count = 0

        self.safety_filter = safety_filter
        self.safety_penalty_weight = safety_penalty_weight
        self.safety_interventions_total = 0
        self.safety_interventions_episode = 0
        
        # Cache for last valid observation
        self.last_valid_observation = None
        
        # Track if we need recovery on next reset
        self.needs_recovery = False
        self.needs_reposition = False

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        time.sleep(1)
        
        print("\n================ ENV RESET ================ ")
        self.reward_fn.prev_currents = None

        # 1. Recover from hardware error if needed
        if self.needs_recovery:
            print("Recovering from hardware error...")
            self.executor.recovery()
            self.needs_recovery = False
        else:
            print("No hardware error - skipping recovery")

        # 2. Reposition cube to center if needed
        if self.needs_reposition:
            print("Repositioning cube to center...")
            self.reposition_cube_to_center(attempt=0)
            self.needs_reposition = False
            
            # ====================================================
            # CRITICAL FIX: Sync targets after repositioning
            # ====================================================
            print("Syncing executor targets with actual motor positions...")
            self._sync_targets_with_actual_positions()
        
        # 3. Reset episode bookkeeping
        self.grid.reset()
        self.step_count = 0
        self.safety_interventions_episode = 0
        
        # 4. Establish initial motor positions AFTER reposition and sync
        self.obs_builder.reset()

        # 5. Get first observation
        result = self.obs_builder.get_observation_result()

        if result.hardware_error:
            raise RuntimeError(
                "Hardware error occurred during environment reset: "
                f"{result.error_message}"
            )

        if result.observation is None:
            raise RuntimeError(
                "Failed to obtain observation during reset: "
                f"{result.error_message}"
            )

        observation = result.observation
        
        # 6. Mark starting cube position
        self.grid.visit(
            observation.cube_x,
            observation.cube_y,
        )
        self.last_valid_observation = observation

        return observation.as_numpy(), {}
    # STEP
    def step(self, action):
        # Clamp action to valid range
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1
        
        # =====================================================
        # 1. APPLY SAFETY FILTER (BEFORE EXECUTION)
        # =====================================================
        
        # Track safety modifications
        action_modified = False
        safety_reason = "no_filter"
        safety_detail = ""
        modification_magnitude = 0.0
        safety_result = None
        
        # Get current motor state for safety filter
        positions = self.executor.read_positions()
        currents = self.executor.read_currents()
        voltages = self.executor.read_voltages()
        temperatures = self.executor.read_temperatures()    
        initial_positions = self.obs_builder.initial_motor_positions
        
        if self.safety_filter is not None and positions is not None and currents is not None:
            if initial_positions is not None:
                safety_result = self.safety_filter.filter(
                    action=action,
                    currents=np.array(currents),
                    positions=np.array(positions),
                    initial_positions=initial_positions,
                    voltages=np.array(voltages) if voltages is not None else None,
                    temperatures=np.array(temperatures) if temperatures is not None else None,  # <-- ADD THIS
                    action_count=self.step_count,
                )
                
                # Use the safe action
                action = safety_result.safe_action
                
                # Track if action was modified
                action_modified = safety_result.modified
                safety_reason = safety_result.reason.value if safety_result.reason else "safe"
                safety_detail = safety_result.detail
                
                if action_modified:
                    self.safety_interventions_total += 1
                    self.safety_interventions_episode += 1
                    
                    # Calculate how much the action was modified
                    modification_magnitude = np.mean(
                        np.abs(safety_result.raw_action - safety_result.safe_action)
                    )
            else:
                print("WARNING: initial_positions is None, skipping safety filter")

        # =====================================================
        # 2. Execute RL action
        # =====================================================

        execution_result = self.executor.execute(action)

        # -----------------------------------------------------
        # Hardware error during action execution
        # -----------------------------------------------------

        if execution_result.hardware_error:
            self.needs_recovery = True
            
            print("Hardware error during action execution.")
            print(f"Hardware error IDs: {execution_result.hardware_error_ids}")
            print(f"Hardware error status: {execution_result.hardware_error_status}")

            # Use the last known-good observation
            observation = self.last_valid_observation

            # If last_valid_observation is None, create a zero observation
            if observation is None:
                print("WARNING: No valid observation available, using zeros")
                from observation import Observation
                observation = Observation(
                    cube_x=0.0,
                    cube_y=0.0,
                    cube_yaw=0.0,
                    motor_positions=np.zeros(4, dtype=np.float32),
                    motor_currents=np.zeros(4, dtype=np.float32),
                    initial_motor_positions=np.zeros(4, dtype=np.float32),
                    cube_x_norm=0.0,
                    cube_y_norm=0.0,
                )

            # Compute reward with hardware error penalty
            reward_info = self.reward_fn.compute(
                visitation_count=0,
                motor_currents=observation.motor_currents if observation else np.zeros(4),
                hardware_error=True,
            )
            
            reward = reward_info.total
            
            # Apply safety penalty even on hardware errors
            safety_penalty = 0.0
            if action_modified and self.safety_penalty_weight > 0:
                safety_penalty = -self.safety_penalty_weight * (2.0 + modification_magnitude)
                reward += safety_penalty
                print(
                    f"  [SAFETY PENALTY ON HARDWARE ERROR] {safety_penalty:.2f} "
                    f"(reason: {safety_reason}, magnitude: {modification_magnitude:.3f})"
                )

            info = {
                "coverage": self.grid.coverage(),
                "visits": 0,
                "hardware_error": True,
                "hardware_error_ids": execution_result.hardware_error_ids,
                "hardware_error_status": execution_result.hardware_error_status,
                "execution_success": False,
                "execution_error_message": execution_result.error_message,
                "termination_reason": "hardware_error",
                # Safety info
                "action_modified": action_modified,
                "safety_reason": safety_reason,
                "safety_detail": safety_detail,
                "safety_penalty": safety_penalty,
                "modification_magnitude": modification_magnitude,
                "tension_penalty": reward_info.tension_penalty,
            }

            return (
                observation.as_numpy() if observation else np.zeros(14),
                reward,
                True,  # terminated
                False,  # truncated
                info,
            )

        # -----------------------------------------------------
        # Non-hardware execution failure
        # -----------------------------------------------------

        if not execution_result.success:
            raise RuntimeError(execution_result.error_message)

        # =====================================================
        # 3. Wait for physical movement
        # =====================================================

        time.sleep(self.action_duration)

        # =====================================================
        # 4. Read observation
        # =====================================================

        observation_result = self.obs_builder.get_observation_result()

        # -----------------------------------------------------
        # Hardware error while reading observation
        # -----------------------------------------------------

        if observation_result.hardware_error:
            self.needs_recovery = True
            
            print("Hardware error while reading observation.")
            print(f"Hardware error IDs: {observation_result.hardware_error_ids}")
            print(f"Hardware error status: {observation_result.hardware_error_status}")

            # Use the last valid observation
            observation = self.last_valid_observation

            # If last_valid_observation is None, create a zero observation
            if observation is None:
                print("WARNING: No valid observation available, using zeros")
                from observation import Observation
                observation = Observation(
                    cube_x=0.0,
                    cube_y=0.0,
                    cube_yaw=0.0,
                    motor_positions=np.zeros(4, dtype=np.float32),
                    motor_currents=np.zeros(4, dtype=np.float32),
                    initial_motor_positions=np.zeros(4, dtype=np.float32),
                    cube_x_norm=0.0,
                    cube_y_norm=0.0,
                )

            # Compute reward with hardware error penalty
            reward_info = self.reward_fn.compute(
                visitation_count=0,
                motor_currents=observation.motor_currents if observation else np.zeros(4),
                hardware_error=True,
            )
            
            reward = reward_info.total
            
            # Apply safety penalty even on hardware errors
            safety_penalty = 0.0
            if action_modified and self.safety_penalty_weight > 0:
                safety_penalty = -self.safety_penalty_weight * (2.0 + modification_magnitude)
                reward += safety_penalty
                print(
                    f"  [SAFETY PENALTY ON HARDWARE ERROR] {safety_penalty:.2f} "
                    f"(reason: {safety_reason}, magnitude: {modification_magnitude:.3f})"
                )

            info = {
                "coverage": self.grid.coverage(),
                "visits": 0,
                "hardware_error": True,
                "hardware_error_ids": observation_result.hardware_error_ids,
                "hardware_error_status": observation_result.hardware_error_status,
                "execution_success": True,
                "execution_error_message": observation_result.error_message,
                "termination_reason": "hardware_error",
                # Safety info
                "action_modified": action_modified,
                "safety_reason": safety_reason,
                "safety_detail": safety_detail,
                "safety_penalty": safety_penalty,
                "modification_magnitude": modification_magnitude,
                "tension_penalty": reward_info.tension_penalty,
            }

            return (
                observation.as_numpy() if observation else np.zeros(14),
                reward,
                True,  # terminated
                False,  # truncated
                info,
            )

        # -----------------------------------------------------
        # Observation succeeded - but check if observation is None
        # -----------------------------------------------------

        if observation_result.observation is None:
            self.needs_recovery = True
            
            print("ERROR: Observation result is None!")
            # Use the last valid observation if available
            observation = self.last_valid_observation
            if observation is None:
                # Fallback to zeros
                print("WARNING: No valid observation available, using zeros")
                from observation import Observation
                observation = Observation(
                    cube_x=0.0,
                    cube_y=0.0,
                    cube_yaw=0.0,
                    motor_positions=np.zeros(4, dtype=np.float32),
                    motor_currents=np.zeros(4, dtype=np.float32),
                    initial_motor_positions=np.zeros(4, dtype=np.float32),
                    cube_x_norm=0.0,
                    cube_y_norm=0.0,
                )
            
            # Get cube position
            x = observation.cube_x
            y = observation.cube_y
            
            # Compute reward with zero visitation
            visits = 0
            reward_info = self.reward_fn.compute(
                visitation_count=visits,
                motor_currents=observation.motor_currents,
                hardware_error=True,
            )
            reward = reward_info.total
            
            info = {
                "coverage": self.grid.coverage(),
                "visits": 0,
                "coverage_reward": 0.0,
                "current_reward": reward_info.current_reward,
                "current_change_penalty": reward_info.current_change_penalty,
                "hardware_error_penalty": reward_info.hardware_error_penalty,
                "max_current": 0.0,
                "motor_currents": np.zeros(4),
                "hardware_error": False,
                "hardware_error_ids": [],
                "hardware_error_status": {},
                "execution_success": True,
                "cube_position": (x, y),
                "termination_reason": "observation_failed",
                "action_modified": action_modified,
                "safety_reason": safety_reason,
                "safety_detail": safety_detail,
                "safety_penalty": 0.0,
                "modification_magnitude": modification_magnitude,
                "tension_penalty": reward_info.tension_penalty,
            }
            
            return (
                observation.as_numpy(),
                reward,
                True,  # terminate on observation failure
                False,
                info,
            )

        # -----------------------------------------------------
        # Observation succeeded
        # -----------------------------------------------------

        observation = observation_result.observation
        
        # Update renderer with trajectory
        if self.renderer is not None and observation is not None:
            cube_x_cm = (observation.cube_x - self.grid.x_min) * 100
            cube_y_cm = (observation.cube_y - self.grid.y_min) * 100
            self.renderer.update_step(self.step_count, cube_x_cm, cube_y_cm)

        # Cache this as the newest known-good state
        self.last_valid_observation = observation

        # =====================================================
        # 5. Get physical cube position
        # =====================================================

        x = observation.cube_x
        y = observation.cube_y

        # =====================================================
        # 6. Check out-of-bounds
        # =====================================================

        out_of_bounds = (
            x < self.grid.x_min
            or x > self.grid.x_max
            or y < self.grid.y_min
            or y > self.grid.y_max
        )

        # =====================================================
        # 7. Update occupancy grid (only if in bounds)
        # =====================================================
        out_of_bound_penalty = 0
        if out_of_bounds:
            visits = 0
            print(f"Cube out of bounds at ({x:.3f}, {y:.3f}) - will reposition on reset")
            self.needs_reposition = True  # Flag for repositioning on reset
            out_of_bound_penalty = -20
        else:
            visits = self.grid.visit(x, y)

        # =====================================================
        # 8. Compute base reward
        # =====================================================

        reward_info = self.reward_fn.compute(
            visitation_count=visits,
            motor_currents=observation.motor_currents,
            hardware_error=False,
        )

        reward = reward_info.total + out_of_bound_penalty

        # =====================================================
        # 9. Apply Safety Penalty (Learning Signal!)
        # =====================================================

        safety_penalty = 0.0
        
        if action_modified and self.safety_penalty_weight > 0:
            # Penalty proportional to how much the action was modified
            safety_penalty = -self.safety_penalty_weight * (1.0 + modification_magnitude)
            
            # Add extra penalty for dangerous actions
            dangerous_keywords = ["over_current", "tension", "opposing"]
            if any(kw in safety_reason for kw in dangerous_keywords):
                safety_penalty *= 1.5
            
            reward += safety_penalty
            
            print(
                f"  [SAFETY PENALTY] {safety_penalty:.2f} "
                f"(reason: {safety_reason}, magnitude: {modification_magnitude:.3f})"
            )

        # =====================================================
        # 10. Episode conditions
        # =====================================================

        coverage = self.grid.coverage()
        coverage_done = coverage >= self.target_coverage

        max_current = max(abs(float(i)) for i in observation.motor_currents)
        high_current = max_current > self.high_current_threshold

        # =====================================================
        # 11. Termination
        # =====================================================

        terminated = False
        truncated = False
        termination_reason = None

        if high_current:
            terminated = True
            termination_reason = "high_current"
        elif out_of_bounds:
            terminated = True
            termination_reason = "out_of_bounds"
        elif coverage_done:
            terminated = True
            termination_reason = "coverage"
        elif self.step_count >= self.max_steps:
            truncated = True
            termination_reason = "max_steps"

        # =====================================================
        # 12. Info
        # =====================================================

        info = {
            "coverage": coverage,
            "visits": visits,
            "coverage_reward": reward_info.coverage_reward,
            "current_reward": reward_info.current_reward,
            "current_change_penalty": reward_info.current_change_penalty,
            "hardware_error_penalty": reward_info.hardware_error_penalty,
            "max_current": max_current,
            "motor_currents": observation.motor_currents.copy(),
            "hardware_error": False,
            "hardware_error_ids": [],
            "hardware_error_status": {},
            "execution_success": True,
            "cube_position": (x, y),
            "termination_reason": termination_reason,
            "out_of_bounds_penalty": out_of_bound_penalty,
            # Safety info
            "action_modified": action_modified,
            "safety_reason": safety_reason,
            "safety_detail": safety_detail,
            "safety_penalty": safety_penalty,
            "modification_magnitude": modification_magnitude,
            "tension_penalty": reward_info.tension_penalty,
        }

        # =====================================================
        # 13. Return
        # =====================================================

        return (
            observation.as_numpy(),
            reward,
            terminated,
            truncated,
            info,
        )

    # =========================================================
    # REPOSITION CUBE TO CENTER
    # =========================================================
    def reposition_cube_to_center(self, attempt=0):
        """
        Simple deterministic repositioning.
        Pull the appropriate motor to bring cube back to center.
        """
        print("\n========== REPOSITIONING CUBE ==========")

        max_attempt = 3
        
        # 1. Get current cube position
        result = self.obs_builder.get_observation_result()
        if result.observation is None:
            print("ERROR: Cannot get cube position")
            return False
        
        x = result.observation.cube_x
        y = result.observation.cube_y
        
        print(f"Current position: ({x:.3f}, {y:.3f})")
        print(f"Board bounds: x=[{self.grid.x_min:.3f}, {self.grid.x_max:.3f}], y=[{self.grid.y_min:.3f}, {self.grid.y_max:.3f}]")
        
        
        # Determine which motor to pull based on which direction we need to go
        # Priority: whichever direction is further from center
        motor_to_pull = None
        direction = 0  # +1 for pull, -1 for release
        steps = 30
        step_delta = 50
        
        # Check which axis is further from center
        if x < self.grid.x_min:
        # X-axis is further off
            motor_to_pull = 18
            direction = 1
            print(f"x={x:.3f} moving (Motor 18)")
        elif x > self.grid.x_max:
            # X-axis is further off
            motor_to_pull = 19
            direction = 1
            print(f"x={x:.3f} moving (Motor 19)")
        elif y < self.grid.y_min:
            motor_to_pull = 17
            direction = 1
            print(f"y={y:.3f} moving (Motor 17")
        elif y > self.grid.y_max:
            motor_to_pull = 16
            direction = 1
            print(f"y={y:.3f} moving (Motor 16")
        else: 
            # In bounds but off-center - pull toward center
            center_x = (self.grid.x_min + self.grid.x_max) / 2
            center_y = (self.grid.y_min + self.grid.y_max) / 2
            dx = center_x - x
            dy = center_y - y
            
            if abs(dx) > abs(dy):
                if dx > 0:
                    motor_to_pull = 18
                    direction = 1
                    print(f"Moving  toward center (Motor 18)")
                else:
                    motor_to_pull = 19
                    direction = 1
                    print(f" (Motor 19)")
            else:
                if dy > 0:
                    motor_to_pull = 17
                    direction = 1
                    print(f"Motor 17)")
                else:
                    motor_to_pull = 16
                    direction = 1
                    print(f" (Motor 16)")


        print(f"Releasing other motors (especially the opposite of Motor {motor_to_pull})...")
        released_motors = []
        for motor in self.executor.motor_ids:
            if motor != motor_to_pull:
                try:
                    self.executor.write1(motor, 64, 0)  # TORQUE_DISABLE
                    released_motors.append(motor)
                    print(f"  Released Motor {motor}")
                except Exception as e:
                    print(f"  Failed to release motor {motor}: {e}")
        time.sleep(0.05)
        
        print(f"Motors released: {released_motors}")
        print(f"Active motor: {motor_to_pull} (pulling {direction})")
                    
        # Move in small steps
        print(f"Moving Motor {motor_to_pull}: {steps} steps of {step_delta * direction} ticks")
        
        for step in range(steps):
            print(step)
            self.executor.move_motor_by_delta(motor_to_pull, step_delta * direction)
            time.sleep(0.05)

        print(f"Re-enabling all motors...")
        for motor in self.executor.motor_ids:
            try:
                self.executor.write1(motor, 64, 1)
            except Exception as e:
                print(f"  Failed to enable motor {motor}: {e}")
        time.sleep(0.5)
        print("Syncing targets after reposition...")
        self._sync_targets_with_actual_positions()
    

        result = self.obs_builder.get_observation_result()
        if result.observation is not None:
            new_x = result.observation.cube_x
            new_y = result.observation.cube_y
            print(f"New position: ({new_x:.3f}, {new_y:.3f})")
            if x < self.grid.x_min or x > self.grid.x_max or y < self.grid.y_min or y > self.grid.y_max:
                print("still needs repositioning, calling again")
                self.reposition_cube_to_center(attempt + 1)
        
        print("========== REPOSITION COMPLETE ==========")
        return True


    # In environment.py

    def _sync_targets_with_actual_positions(self):
        """
        Sync the executor's target positions with actual motor positions.
        This prevents the RL policy from using old/stale target values.
        """
        print("  Syncing targets with actual positions...")
        positions = self.executor.read_positions()
        if positions is not None:
            for motor_id, pos in zip(self.executor.motor_ids, positions):
                old_target = self.executor.targets.get(motor_id, pos)
                self.executor.targets[motor_id] = pos
                print(f"    Motor {motor_id}: target {old_target} → {pos}")
        else:
            print("  WARNING: Could not read positions for sync!")
    # =========================================================
    # RENDER
    # =========================================================

    def render(self):
        if self.renderer is None:
            return None

        observation = self.obs_builder.get_observation()
        
        # Convert to cm AND offset to board origin
        cube_x_cm = (observation.cube_x - self.grid.x_min) * 100
        cube_y_cm = (observation.cube_y - self.grid.y_min) * 100

        print(f"[RENDER] cube_x={observation.cube_x:.3f}m → {cube_x_cm:.1f}cm")
        print(f"[RENDER] cube_y={observation.cube_y:.3f}m → {cube_y_cm:.1f}cm")
        print(f"[RENDER] board: {self.renderer.width:.1f}cm x {self.renderer.height:.1f}cm")

        return self.renderer.render(
            cube_x=cube_x_cm,
            cube_y=cube_y_cm,
            occupancy_grid=self.grid.as_numpy(),
            coverage=self.grid.coverage(),
            mode=self.render_mode,
        )

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):
        """Close the environment."""
        
        # Only shutdown executor if it hasn't been shut down yet
        if hasattr(self, 'executor') and self.executor is not None:
            try:
                # Check if executor has already been shut down
                if not hasattr(self.executor, '_shutdown_done'):
                    self.executor.shutdown()
                    self.executor._shutdown_done = True
            except Exception as e:
                print(f"Executor shutdown error in environment: {e}")
        
        if self.renderer is not None:
            self.renderer.close()
        
        print("Environment closed")