from __future__ import annotations

import time
import gymnasium as gym
import numpy as np

from gymnasium import spaces

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
        max_current=800,
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
        self.render_mode = render_mode
        if self.render_mode in ("human", "rgb_array"):
            self.renderer = BoardRenderer()
        else:
            self.renderer = None

        self.max_steps = max_steps
        self.target_coverage = target_coverage
        self.max_current = max_current

        self.step_count = 0
                



    def reset(self,*,seed=None,options=None,):

        super().reset(seed=seed)
        self.grid.reset()
        observation = self.obs_builder.get_observation()
        x = observation.cube_x
        y = observation.cube_y
        self.step_count = 0
        self.grid.visit(x,y,)
        return observation.as_numpy(), {}



    def step(self, action):
        self.step_count += 1

        self.executor.execute(action)

        time.sleep(self.action_duration)

        observation = self.obs_builder.get_observation()

        x = observation.cube_x
        y = observation.cube_y

        visits = self.grid.visit(x, y)

        reward_info = self.reward_fn.compute(
            visitation_count=visits,
            motor_currents=observation.motor_currents,
        )

        if self.render_mode == "human":
            self.render()

        info = {
            "coverage": self.grid.coverage(),
            "visits": visits,
            "coverage_reward": reward_info.coverage_reward,
            "current_penalty": reward_info.current_penalty,
            "cube_position_cm": (
                observation.cube_x,
                observation.cube_y,
            ),
        }
        # TODO: move the penalty logic to reward.py
        truncated = terminated = False
        truncated = self.step_count >= self.max_steps
        coverage_done = info["coverage"] >= self.target_coverage
        if coverage_done:
            reward+=20

        x_pos = info["cube_position_cm"][0]
        y_pos = info["cube_position_cm"][1]

        out_of_bounds = (
            x_pos < 0
            or x_pos > self.grid.board_width
            or y_pos < 0
            or y_pos > self.grid.board_height
        )
        if out_of_bounds:
            reward = -10

        current_limit = any(
            abs(i) >= self.max_current
            for i in observation.motor_currents
        )
        if current_limit:
            reward-=10

        if coverage_done or out_of_bounds or current_limit:
            terminated = True

        if coverage_done:
            info["termination_reason"] = "coverage"

        elif out_of_bounds:
            info["termination_reason"] = "out_of_bounds"

        elif current_limit:
            info["termination_reason"] = "over_current"

        elif truncated:
            info["termination_reason"] = "max_steps"

        return (
            observation.as_numpy(),
            reward_info.total,
            terminated,
            truncated,
            info,
        )



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


    def close(self):

        self.executor.shutdown()
        if self.renderer is not None:
            self.renderer.close()