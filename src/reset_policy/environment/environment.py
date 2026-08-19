from __future__ import annotations

import time
import gymnasium as gym
import numpy as np

from gymnasium import spaces

import sys
from pathlib import Path

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

        self.safe_current_threshold = (
            safe_current_threshold
        )

        self.moderate_current_threshold = (
            moderate_current_threshold
        )

        self.high_current_threshold = (
            high_current_threshold
        )

        self.step_count = 0
        self.episode_count = 0


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

        self.executor.logger.update_context(
            episode=self.episode_count,
            step=0,
            action=None,
            cube_x=observation.cube_x if hasattr(observation, 'cube_x') else None,
            cube_y=observation.cube_y if hasattr(observation, 'cube_y') else None,
        )

        return observation.as_numpy(), {}


 
    # STEP
    def step(self, action):
        # Clamp action to valid range
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1


        # Log action before execution (for debugging)
        # But we don't have cube position yet, so useing last known

        self.executor.logger.update_context(
        episode=self.episode_count,
        step=self.step_count,
        action=action,
        cube_x=self.last_valid_observation.cube_x if self.last_valid_observation else None,
        cube_y=self.last_valid_observation.cube_y if self.last_valid_observation else None,
        )   

        # =====================================================
        # 1. Execute RL action
        # =====================================================

        execution_result = self.executor.execute(action)

        # -----------------------------------------------------
        # Hardware error during action execution
        # -----------------------------------------------------

        if execution_result.hardware_error:

            print("Hardware error during action execution.")
            print(
                f"Hardware error IDs: "
                f"{execution_result.hardware_error_ids}"
            )
            print(
                f"Hardware error status: "
                f"{execution_result.hardware_error_status}"
            )

            # Use the last known-good observation.
            observation = self.last_valid_observation

            reward_info = self.reward_fn.compute(
                visitation_count=0,
                motor_currents=observation.motor_currents,
                hardware_error=True,
            )

            info = {
                "coverage": self.grid.coverage(),
                "visits": 0,

                "hardware_error": True,
                "hardware_error_ids":
                    execution_result.hardware_error_ids,
                "hardware_error_status":
                    execution_result.hardware_error_status,

                "execution_success": False,
                "execution_error_message":
                    execution_result.error_message,

                "termination_reason":
                    "hardware_error",
            }

            return (
                observation.as_numpy(),
                reward_info.total,
                True,
                False,
                info,
            )

        # -----------------------------------------------------
        # Non-hardware execution failure
        # -----------------------------------------------------

        if not execution_result.success:

            raise RuntimeError(
                execution_result.error_message
            )

        # =====================================================
        # 2. Wait for physical movement
        # =====================================================

        time.sleep(self.action_duration)

        # =====================================================
        # 3. Read observation
        # =====================================================

        observation_result = (
            self.obs_builder.get_observation_result()
        )

        # -----------------------------------------------------
        # Hardware error while reading observation
        # -----------------------------------------------------

        if observation_result.hardware_error:

            print(
                "Hardware error while reading observation."
            )

            print(
                f"Hardware error IDs: "
                f"{observation_result.hardware_error_ids}"
            )

            print(
                f"Hardware error status: "
                f"{observation_result.hardware_error_status}"
            )

            # IMPORTANT:
            # Do NOT use the failed observation.
            # Use the last valid one.
            observation = self.last_valid_observation

            reward_info = self.reward_fn.compute(
                visitation_count=0,
                motor_currents=observation.motor_currents,
                hardware_error=True,
            )

            info = {
                "coverage": self.grid.coverage(),
                "visits": 0,

                "hardware_error": True,

                "hardware_error_ids":
                    observation_result.hardware_error_ids,

                "hardware_error_status":
                    observation_result.hardware_error_status,

                "execution_success": True,

                "execution_error_message":
                    observation_result.error_message,

                "termination_reason":
                    "hardware_error",
            }


            return (
                observation.as_numpy(),
                reward_info.total,
                True,
                False,
                info,
            )

        # -----------------------------------------------------
        # Observation succeeded
        # -----------------------------------------------------

        observation = observation_result.observation

        # Cache this as the newest known-good state.
        self.last_valid_observation = observation

        self.executor.logger.update_context(
            episode=self.episode_count,
            step=self.step_count,
            action=action,
            cube_x=observation.cube_x if hasattr(observation, 'cube_x') else None,
            cube_y=observation.cube_y if hasattr(observation, 'cube_y') else None,
        )
        # =====================================================
        # 4. Get physical cube position
        # =====================================================

        x = observation.cube_x
        y = observation.cube_y

        # =====================================================
        # 5. Update occupancy grid
        # =====================================================

        visits = self.grid.visit(x, y)

        # =====================================================
        # 6. Compute reward
        # =====================================================

        reward_info = self.reward_fn.compute(
            visitation_count=visits,
            motor_currents=observation.motor_currents,
            hardware_error=False,
        )

        reward = reward_info.total

        # =====================================================
        # 7. Episode conditions
        # =====================================================

        coverage = self.grid.coverage()

        coverage_done = (
            coverage >= self.target_coverage
        )

        out_of_bounds = (
            x < self.grid.x_min
            or x > self.grid.x_max
            or y < self.grid.y_min
            or y > self.grid.y_max
        )

        max_current = max(
            abs(float(i))
            for i in observation.motor_currents
        )

        high_current = (
            max_current >
            self.high_current_threshold
        )

        # =====================================================
        # 8. Termination
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
        # 9. Info
        # =====================================================

        info = {
            "coverage": coverage,
            "visits": visits,

            "coverage_reward":
                reward_info.coverage_reward,

            "current_reward":
                reward_info.current_reward,

            "current_change_penalty":
                reward_info.current_change_penalty,

            "hardware_error_penalty":
                reward_info.hardware_error_penalty,

            "max_current":
                max_current,

            "motor_currents":
                observation.motor_currents.copy(),

            "hardware_error":
                False,

            "hardware_error_ids":
                [],

            "hardware_error_status":
                {},

            "execution_success":
                True,

            "cube_position":
                (x, y),

            "termination_reason":
                termination_reason,
        }

        # =====================================================
        # 10. Return
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

        observation = (
            self.obs_builder.get_observation()
        )

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

        self.executor.shutdown()

        if self.renderer is not None:
            self.renderer.close()