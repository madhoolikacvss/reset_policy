from __future__ import annotations

import time
import gymnasium as gym
import numpy as np

from gymnasium import spaces

from observation import ObservationBuilder
from reward import RewardFunction
from occupancy_grid import OccupancyGrid


class ResetPolicyEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        executor,
        observation_builder: ObservationBuilder,
        cube_tracker,
        occupancy_grid: OccupancyGrid,
        reward_function: RewardFunction,
        max_motor_delta: int = 400,
        action_duration: float = 0.4,
    ):

        super().__init__()

        self.executor = executor

        self.obs_builder = observation_builder

        self.cube_tracker = cube_tracker

        self.grid = occupancy_grid

        self.reward_fn = reward_function

        self.max_motor_delta = max_motor_delta

        self.action_duration = action_duration

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-max_motor_delta,
            high=max_motor_delta,
            shape=(4,),
            dtype=np.float32,
        )


    def reset(self,*,seed=None,options=None,):

        super().reset(seed=seed)
        self.grid.reset()
        observation = self.obs_builder.get_observation()
        self.grid.visit(observation.cube_x,observation.cube_y,)

        return observation.as_numpy(), {}

    def step(
        self,
        action,
    ):
        
        # execute motor command
        self.executor.execute_motor_action(action)

        # allow motion to settle
        time.sleep(self.action_duration)

        # read new obs
        observation = self.obs_builder.get_observation()

        # update occupancy grid
        visits = self.grid.visit(
            observation.cube_x,
            observation.cube_y,
        )

        reward = self.reward_fn.compute(
            visitation_count=visits,
            motor_currents=observation.motor_currents,
        )

        terminated = False
        truncated = False

        info = {

            "coverage": self.grid.coverage(),
            "visits": visits,
            "coverage_reward":
                reward.coverage_reward,
            "current_penalty":
                reward.current_penalty,
        }

        return (

            observation.as_numpy(),
            reward.total,
            terminated,
            truncated,
            info,
        )

    def render(self):
        pass

    def close(self):
        self.executor.shutdown()