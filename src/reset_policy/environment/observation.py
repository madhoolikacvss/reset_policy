from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from perception.cube_tracker import CubeTracker
from control.dynamixel_executor import DynamixelExecutor


# ==============================
# Normalization constants
# ==============================

# Board bounds (meters)
X_MIN = 0.035
X_MAX = 0.830

Y_MIN = 0.140
Y_MAX = 0.654


# Dynamixel current limit (mA)
CURRENT_LIMIT = 1750.0


# Maximum expected motor displacement
# Tune this based on your task.
# Example:
# 400 counts/action * ~10 actions = 4000 counts
MAX_POSITION_DELTA = 4000.0



@dataclass
class Observation:

    # Cube state
    cube_x: float
    cube_y: float
    cube_yaw: float

    # Raw motor states
    motor_positions: np.ndarray
    motor_currents: np.ndarray

    # Stored initial motor positions
    initial_motor_positions: np.ndarray


    def as_numpy(self) -> np.ndarray:
        """
        Return normalized observation vector for PPO.

        Output:
        [
            cube_x_norm,
            cube_y_norm,
            cube_yaw_norm,

            motor1_position_delta_norm,
            motor2_position_delta_norm,
            motor3_position_delta_norm,
            motor4_position_delta_norm,

            motor1_current_norm,
            motor2_current_norm,
            motor3_current_norm,
            motor4_current_norm,
        ]
        """

        # -----------------------------
        # Normalize cube position
        # -----------------------------

        cube_x_norm = (
            self.cube_x - X_MIN
        ) / (
            X_MAX - X_MIN
        )

        cube_y_norm = (
            self.cube_y - Y_MIN
        ) / (
            Y_MAX - Y_MIN
        )


        # Clamp in case of small calibration errors
        cube_x_norm = np.clip(
            cube_x_norm,
            0.0,
            1.0,
        )

        cube_y_norm = np.clip(
            cube_y_norm,
            0.0,
            1.0,
        )


        # -----------------------------
        # Normalize yaw
        # -----------------------------

        cube_yaw_norm = (
            self.cube_yaw / np.pi
        )

        cube_yaw_norm = np.clip(
            cube_yaw_norm,
            -1.0,
            1.0,
        )


        # -----------------------------
        # Normalize motor positions
        # Relative displacement from start
        # -----------------------------

        position_delta = (
            self.motor_positions
            -
            self.initial_motor_positions
        )

        position_delta_norm = (
            position_delta
            /
            MAX_POSITION_DELTA
        )

        position_delta_norm = np.clip(
            position_delta_norm,
            -1.0,
            1.0,
        )


        # -----------------------------
        # Normalize currents
        # -----------------------------

        current_norm = (
            self.motor_currents
            /
            CURRENT_LIMIT
        )

        current_norm = np.clip(
            current_norm,
            -1.0,
            1.0,
        )


        obs = np.concatenate(
            [
                np.array(
                    [
                        cube_x_norm,
                        cube_y_norm,
                        cube_yaw_norm,
                    ],
                    dtype=np.float32,
                ),

                position_delta_norm.astype(
                    np.float32
                ),

                current_norm.astype(
                    np.float32
                ),
            ]
        )


        return obs.astype(np.float32)



class ObservationBuilder:

    def __init__(
        self,
        cube_tracker: CubeTracker,
        executor: DynamixelExecutor,
    ):

        self.cube_tracker = cube_tracker
        self.executor = executor

        # Store motor starting position
        self.initial_motor_positions = None



    def reset(self):
        """
        Called at the beginning of every episode.
        """

        self.initial_motor_positions = np.array(
            self.executor.read_positions(),
            dtype=np.float32,
        )



    def get_observation(self) -> Observation:

        cube_state = self.cube_tracker.get_state()

        if not cube_state.detected:
            raise RuntimeError(
                "Cube not detected"
            )


        motor_positions = np.array(
            self.executor.read_positions(),
            dtype=np.float32,
        )


        motor_currents = np.array(
            self.executor.read_currents(),
            dtype=np.float32,
        )


        # Safety check
        if self.initial_motor_positions is None:
            self.initial_motor_positions = (
                motor_positions.copy()
            )


        return Observation(

            cube_x=cube_state.x,

            cube_y=cube_state.y,

            cube_yaw=cube_state.yaw,

            motor_positions=motor_positions,

            motor_currents=motor_currents,

            initial_motor_positions=
                self.initial_motor_positions.copy(),
        )