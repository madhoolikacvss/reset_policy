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
        action_duration: float = 0.4,

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

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
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
            self.renderer = BoardRenderer()
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

    # =========================================================
    # RESET
    # =========================================================
    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)
        self.episode_count += 1

        print("\n================ ENV RESET ================ ")
        self.reward_fn.prev_currents = None

        # 1. Recover physical hardware
        self.executor.recovery()
        
        # 2. Reset episode bookkeeping
        self.grid.reset()
        self.step_count = 0
        self.safety_interventions_episode = 0
        
        # 3. Establish initial motor positions AFTER recovery
        self.obs_builder.reset()

        # 4. Get first observation
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
        
        # 5. Mark starting cube position
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
        initial_positions = self.obs_builder.initial_motor_positions
        
        if self.safety_filter is not None and positions is not None and currents is not None:
            if initial_positions is not None:
                safety_result = self.safety_filter.filter(
                    action=action,
                    currents=np.array(currents),
                    positions=np.array(positions),
                    initial_positions=initial_positions,
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
            print("Hardware error during action execution.")
            print(f"Hardware error IDs: {execution_result.hardware_error_ids}")
            print(f"Hardware error status: {execution_result.hardware_error_status}")

            # Use the last known-good observation
            observation = self.last_valid_observation

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
            }

            return (
                observation.as_numpy() if observation else np.zeros(11),
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
            print("Hardware error while reading observation.")
            print(f"Hardware error IDs: {observation_result.hardware_error_ids}")
            print(f"Hardware error status: {observation_result.hardware_error_status}")

            # Use the last valid observation
            observation = self.last_valid_observation

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
            }

            return (
                observation.as_numpy() if observation else np.zeros(11),
                reward,
                True,  # terminated
                False,  # truncated
                info,
            )

        # -----------------------------------------------------
        # Observation succeeded
        # -----------------------------------------------------

        observation = observation_result.observation

        # Cache this as the newest known-good state
        self.last_valid_observation = observation

        # =====================================================
        # 5. Get physical cube position
        # =====================================================

        x = observation.cube_x
        y = observation.cube_y

        # =====================================================
        # 6. Update occupancy grid
        # =====================================================

        visits = self.grid.visit(x, y)

        # =====================================================
        # 7. Compute base reward
        # =====================================================

        reward_info = self.reward_fn.compute(
            visitation_count=visits,
            motor_currents=observation.motor_currents,
            hardware_error=False,
        )

        reward = reward_info.total

        # =====================================================
        # 8. Apply Safety Penalty (Learning Signal!)
        # =====================================================

        safety_penalty = 0.0
        
        if action_modified and self.safety_penalty_weight > 0:
            # Penalty proportional to how much the action was modified
            # This teaches PPO: "Don't output actions that need filtering"
            safety_penalty = -self.safety_penalty_weight * (1.0 + modification_magnitude)
            
            # Add extra penalty for dangerous actions
            dangerous_keywords = ["over_current", "tension", "opposing"]
            if any(kw in safety_reason for kw in dangerous_keywords):
                safety_penalty *= 1.5  # Extra penalty for dangerous actions
            
            reward += safety_penalty
            
            # Log the penalty
            print(
                f"  [SAFETY PENALTY] {safety_penalty:.2f} "
                f"(reason: {safety_reason}, magnitude: {modification_magnitude:.3f})"
            )

        # =====================================================
        # 9. Episode conditions
        # =====================================================

        coverage = self.grid.coverage()
        coverage_done = coverage >= self.target_coverage

        out_of_bounds = (
            x < self.grid.x_min
            or x > self.grid.x_max
            or y < self.grid.y_min
            or y > self.grid.y_max
        )

        max_current = max(abs(float(i)) for i in observation.motor_currents)
        high_current = max_current > self.high_current_threshold

        # =====================================================
        # 10. Termination
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
        # 11. Info
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
            # Safety info
            "action_modified": action_modified,
            "safety_reason": safety_reason,
            "safety_detail": safety_detail,
            "safety_penalty": safety_penalty,
            "modification_magnitude": modification_magnitude,
        }

        # =====================================================
        # 12. Return
        # =====================================================

        return (
            observation.as_numpy(),
            reward,
            terminated,
            truncated,
            info,
        )

    # =========================================================
    # RENDER
    # =========================================================

    def render(self):
        if self.renderer is None:
            return None

        observation = self.obs_builder.get_observation()
        return self.renderer.render(
            cube_x=observation.cube_x,
            cube_y=observation.cube_y,
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