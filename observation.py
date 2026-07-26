from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Observation:

    cube_x: float
    cube_y: float

    cube_yaw: float = 0.0 # to use when we have movable motors

    motor_positions: np.ndarray
    motor_currents: np.ndarray

    def as_numpy(self) -> np.ndarray:
        """
        Returns a flat observation vector.
        """

        return np.concatenate([
            np.array(
                [
                    self.cube_x,
                    self.cube_y,
                ],
                dtype=np.float32,
            ),
            self.motor_positions.astype(np.float32),
            self.motor_currents.astype(np.float32),
        ])

class ObservationBuilder:

    """
    Creates an Observation from camera + Dynamixels.
    """

    def __init__(self,cube_tracker,horizontal_axis,vertical_axis,):

        self.cube_tracker = cube_tracker
        self.horizontal = horizontal_axis
        self.vertical = vertical_axis

    def get_observation(self,) -> Observation:

        state = self.cube_tracker.get_state()

        if not state.detected:
            raise RuntimeError("Cube not detected.")

        motor_positions = np.array([
            self.horizontal.read_pull_position(),
            self.horizontal.read_release_position(),
            self.vertical.read_pull_position(),
            self.vertical.read_release_position(),
        ])

        motor_currents = np.array([
            self.horizontal.read_pull_current(),
            self.horizontal.read_release_current(),
            self.vertical.read_pull_current(),
            self.vertical.read_release_current(),
        ])

        return Observation(
            cube_x=state.position.x,
            cube_y=state.position.y,
            motor_positions=motor_positions,
            motor_currents=motor_currents,
        )