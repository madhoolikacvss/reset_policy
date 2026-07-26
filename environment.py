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
        action_duration: float = 0.4,
    ):

        super().__init__()

        self.executor = executor
        self.obs_builder = observation_builder
        self.cube_tracker = cube_tracker

        self.grid = occupancy_grid
        self.reward_fn = reward_function

        self.action_duration = action_duration

        # Observation:
        #
        # cube x,y,yaw
        # 4 motor positions
        # 4 motor currents
        #
        # total = 11

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32,
        )

        # Four Dynamixels
        #
        # each action:
        # -1 = release max
        # +1 = pull max

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )



    def reset(self,*,seed=None,options=None,):

        super().reset(seed=seed)
        self.grid.reset()
        observation = self.obs_builder.get_observation()
        # cm -> meters
        x = observation.cube_x / 100.0
        y = observation.cube_y / 100.0

        self.grid.visit(x,y,)
        return observation.as_numpy(), {}



    def step(self,action,):
        self.executor.execute(action)
        time.sleep(self.action_duration)
        observation = self.obs_builder.get_observation()

        # Update occupancy grid
        x = observation.cube_x / 100.0
        y = observation.cube_y / 100.0

        visits = self.grid.visit(x,y,)

        reward_info = self.reward_fn.compute(
            visitation_count=visits,
            motor_currents=observation.motor_currents,
        )

        reward = reward_info.total
        terminated = False
        truncated = False

        info = {
            "coverage":
                self.grid.coverage(),
            "visits":
                visits,
            "coverage_reward":
                reward_info.coverage_reward,
            "current_penalty":
                reward_info.current_penalty,
            "cube_position_cm":
                (
                    observation.cube_x,
                    observation.cube_y,
                ),
        }

        return (
            observation.as_numpy(),
            reward,
            terminated,
            truncated,
            info,
        )



    def render(self):
        pass


    def close(self):
        self.executor.shutdown()