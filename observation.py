from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from cube_tracker import CubeTracker
from dynamixel_executor import DynamixelExecutor


@dataclass
class Observation:

    cube_x: float
    cube_y: float

    motor_positions: np.ndarray
    motor_currents: np.ndarray

    cube_yaw: float = 0.0


    def as_numpy(self) -> np.ndarray:

        return np.concatenate(
            [
                np.array([self.cube_x,self.cube_y,self.cube_yaw,],dtype=np.float32,),
                self.motor_positions.astype(np.float32),
                self.motor_currents.astype(np.float32),
            ]
        )

class ObservationBuilder:
    def __init__(self,cube_tracker: CubeTracker,executor: DynamixelExecutor,):

        self.cube_tracker = cube_tracker
        self.executor = executor



    def get_observation(self) -> Observation:

        cube_state = self.cube_tracker.get_state()

        if not cube_state.detected:
            raise RuntimeError("Cube not detected")


        motor_positions = np.array(self.executor.get_positions())
        motor_currents = np.array(self.executor.get_currents())


        return Observation(
            cube_x=cube_state.x,
            cube_y=cube_state.y,
            cube_yaw=cube_state.yaw,
            motor_positions=motor_positions,
            motor_currents=motor_currents,
        )