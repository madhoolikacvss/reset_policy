from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import sys
from pathlib import Path
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from perception.cube_tracker import CubeTracker
from control.dynamixel_executor import DynamixelExecutor


# ============================================================
# Normalization constants
# ============================================================

CURRENT_LIMIT = 1750.0

MAX_POSITION_DELTA = 4000.0


# ============================================================
# Observation
# ============================================================

@dataclass
class Observation:

    # --------------------------------------------------------
    # Physical cube state
    # --------------------------------------------------------

    # meters
    cube_x: float
    cube_y: float

    # radians
    cube_yaw: float

    # --------------------------------------------------------
    # Motor state
    # --------------------------------------------------------

    motor_positions: np.ndarray
    motor_currents: np.ndarray

    # Motor positions at beginning of episode
    initial_motor_positions: np.ndarray

    # --------------------------------------------------------
    # Normalized observation
    # --------------------------------------------------------

    cube_x_norm: float
    cube_y_norm: float

    # --------------------------------------------------------
    # Convert to PPO observation
    # --------------------------------------------------------

    def as_numpy(self) -> np.ndarray:

        cube_yaw_norm = (
            self.cube_yaw / np.pi
        )

        cube_yaw_norm = np.clip(
            cube_yaw_norm,
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # Motor position normalization
        # ----------------------------------------------------

        position_delta = (
            self.motor_positions
            -
            self.initial_motor_positions
        )

        position_delta_norm = (
            position_delta /
            MAX_POSITION_DELTA
        )

        position_delta_norm = np.clip(
            position_delta_norm,
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # Current normalization
        # ----------------------------------------------------

        current_norm = (
            self.motor_currents /
            CURRENT_LIMIT
        )

        current_norm = np.clip(
            current_norm,
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # Final PPO observation
        # ----------------------------------------------------

        obs = np.concatenate(
            [
                np.array(
                    [
                        self.cube_x_norm,
                        self.cube_y_norm,
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


# ============================================================
# Observation Result
# ============================================================

@dataclass
class ObservationResult:

    observation: Observation | None

    hardware_error: bool = False

    hardware_error_ids: list | None = None

    hardware_error_status: dict | None = None

    error_message: str = ""

    def __post_init__(self):

        if self.hardware_error_ids is None:
            self.hardware_error_ids = []

        if self.hardware_error_status is None:
            self.hardware_error_status = {}


# ============================================================
# Observation Builder
# ============================================================

class ObservationBuilder:

    def __init__(
        self,
        cube_tracker: CubeTracker,
        executor: DynamixelExecutor,

        # Board bounds in meters
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ):

        self.cube_tracker = cube_tracker
        self.executor = executor

        self.x_min = x_min
        self.x_max = x_max

        self.y_min = y_min
        self.y_max = y_max

        self.initial_motor_positions = None


    # ========================================================
    # Reset
    # ========================================================

    def reset(self):

        positions = self.executor.read_positions()

        if positions is None:

            (
                hardware_error,
                error_ids,
                error_status,
                message,
            ) = (
                self.executor.get_hardware_error_state()
            )

            raise RuntimeError(
                message
                or
                "Failed to read motor positions during reset."
            )

        self.initial_motor_positions = np.asarray(
            positions,
            dtype=np.float32,
        ).copy()


    # ========================================================
    # Build observation
    # ========================================================

    def get_observation_result(
        self,
    ) -> ObservationResult:

        # ----------------------------------------------------
        # Cube state
        # ----------------------------------------------------

        cube_state = (
            self.cube_tracker.get_state()
        )

        if not cube_state.detected:
            return ObservationResult(
                observation=None,
                hardware_error=False,
                error_message="Cube not detected",
            )

        # ----------------------------------------------------
        # Motor positions
        # ----------------------------------------------------

        motor_positions = (
            self.executor.read_positions()
        )

        if motor_positions is None:

            (
                hardware_error,
                error_ids,
                error_status,
                message,
            ) = (
                self.executor.get_hardware_error_state()
            )

            return ObservationResult(
                observation=None,

                hardware_error=hardware_error,

                hardware_error_ids=error_ids,

                hardware_error_status=error_status,

                error_message=(
                    message
                    or
                    "Failed to read motor positions"
                ),
            )

        # ----------------------------------------------------
        # Motor currents
        # ----------------------------------------------------

        motor_currents = (
            self.executor.read_currents()
        )

        if motor_currents is None:

            (
                hardware_error,
                error_ids,
                error_status,
                message,
            ) = (
                self.executor.get_hardware_error_state()
            )

            return ObservationResult(
                observation=None,

                hardware_error=hardware_error,

                hardware_error_ids=error_ids,

                hardware_error_status=error_status,

                error_message=(
                    message
                    or
                    "Failed to read motor currents"
                ),
            )

        # ----------------------------------------------------
        # Initial motor positions
        # ----------------------------------------------------

        if self.initial_motor_positions is None:

            self.initial_motor_positions = (
                np.asarray(
                    motor_positions,
                    dtype=np.float32,
                ).copy()
            )

        motor_positions = np.asarray(
            motor_positions,
            dtype=np.float32,
        )

        motor_currents = np.asarray(
            motor_currents,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Normalize cube position
        # ----------------------------------------------------

        cube_x_norm = (
            cube_state.x - self.x_min
        ) / (
            self.x_max - self.x_min
        )

        cube_y_norm = (
            cube_state.y - self.y_min
        ) / (
            self.y_max - self.y_min
        )

        # Clamp only for the PPO observation.
        #
        # The physical cube_x/cube_y remain untouched and
        # are used by the occupancy grid / bounds checking.

        cube_x_norm = float(
            np.clip(
                cube_x_norm,
                0.0,
                1.0,
            )
        )

        cube_y_norm = float(
            np.clip(
                cube_y_norm,
                0.0,
                1.0,
            )
        )

        # Build observation
        observation = Observation(

            cube_x=cube_state.x,

            cube_y=cube_state.y,

            cube_yaw=cube_state.yaw,

            motor_positions=motor_positions,

            motor_currents=motor_currents,

            initial_motor_positions=(
                self.initial_motor_positions.copy()
            ),

            cube_x_norm=cube_x_norm,

            cube_y_norm=cube_y_norm,
        )

        return ObservationResult(
            observation=observation,
            hardware_error=False,
        )
