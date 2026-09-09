"""
Observation builder for the reset policy environment.
20-dimensional observation space:
  [0-2]:   Cube position (normalized x, y, yaw)
  [3-4]:   Goal position (normalized x, y)
  [5-8]:   Motor position deltas (normalized)
  [9-12]:  Motor currents (normalized)
  [13-15]: Tension metrics (horizontal, vertical, total)
  [16-19]: Target errors (target - actual, 4 dims)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import time
import sys
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parent.parent))

from perception.cube_tracker import CubeTracker
from control.dynamixel_executor import DynamixelExecutor
from config import config

# Normalization constants (aligned with config)
CURRENT_LIMIT = config.dynamixel.max_encoder_travel / 5.71  # ~1750mA
MAX_POSITION_DELTA = config.safety.max_position  # 8000 ticks
TENSION_LIMIT = config.safety.max_pair_current * 2  # 1600mA


@dataclass
class Observation:
    """Single observation of the system state."""
    # Physical cube state (meters, radians)
    cube_x: float
    cube_y: float
    cube_yaw: float
    
    # Motor state
    motor_positions: np.ndarray
    motor_currents: np.ndarray
    initial_motor_positions: np.ndarray
    
    # Normalized cube position
    cube_x_norm: float
    cube_y_norm: float

    # Normalized goal position
    x_goal_norm: float  # Renamed for clarity
    y_goal_norm: float
    
    # Target error (target - actual position, 4 dims)
    target_error: np.ndarray
    
    # Tension metrics (mA)
    horizontal_tension: float = 0.0
    vertical_tension: float = 0.0
    total_tension: float = 0.0
    
    def as_numpy(self) -> np.ndarray:
        """Convert observation to 20-dim numpy array."""
        # Normalize yaw to [-1, 1]
        cube_yaw_norm = np.clip(self.cube_yaw / np.pi, -1.0, 1.0)
        
        # Normalize position deltas
        position_delta = self.motor_positions - self.initial_motor_positions
        position_delta_norm = np.clip(position_delta / MAX_POSITION_DELTA, -1.0, 1.0)
        
        # Normalize target error
        target_error_norm = np.clip(self.target_error / 4000.0, -1.0, 1.0)
        
        # Normalize currents
        current_norm = np.clip(self.motor_currents / CURRENT_LIMIT, -1.0, 1.0)
        
        # Normalize tensions
        horizontal_tension_norm = np.clip(self.horizontal_tension / TENSION_LIMIT, 0.0, 1.0)
        vertical_tension_norm = np.clip(self.vertical_tension / TENSION_LIMIT, 0.0, 1.0)
        total_tension_norm = np.clip(self.total_tension / TENSION_LIMIT, 0.0, 1.0)
        
        obs = np.concatenate([
            np.array([self.cube_x_norm, self.cube_y_norm, cube_yaw_norm], dtype=np.float32),
            np.array([self.x_goal_norm, self.y_goal_norm], dtype=np.float32),
            position_delta_norm.astype(np.float32),
            current_norm.astype(np.float32),
            np.array([horizontal_tension_norm, vertical_tension_norm, total_tension_norm], dtype=np.float32),
            target_error_norm.astype(np.float32),
        ])
        
        return obs.astype(np.float32)


@dataclass
class ObservationResult:
    """Result of observation attempt."""
    observation: Observation | None
    hardware_error: bool = False
    hardware_error_ids: list = None
    hardware_error_status: dict = None
    error_message: str = ""
    
    def __post_init__(self):
        if self.hardware_error_ids is None:
            self.hardware_error_ids = []
        if self.hardware_error_status is None:
            self.hardware_error_status = {}


class ObservationBuilder:
    """Builds observations from system state."""
    
    def __init__(
        self,
        cube_tracker: CubeTracker,
        executor: DynamixelExecutor,
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
        self.x_goal = None  # Current goal (meters)
        self.y_goal = None
    
    def set_goal(self, x_goal: float, y_goal: float):
        """Set the current goal position."""
        self.x_goal = x_goal
        self.y_goal = y_goal
    
    def reset(self, max_retries: int = 5, retry_delay: float = 0.2):
        """Reset initial motor positions with retries."""
        print("Waiting for motors to settle before reading positions...")
        time.sleep(1)
        
        positions = None
        for attempt in range(max_retries):
            positions = self.executor.read_positions()
            if positions is not None:
                break
            print(f"Retry {attempt+1}/{max_retries}: Failed to read positions, waiting...")
            time.sleep(retry_delay)
        
        if positions is None:
            raise RuntimeError("Failed to read motor positions during reset.")
        
        self.initial_motor_positions = np.asarray(positions, dtype=np.float32).copy()
    
    def get_observation_result(self) -> ObservationResult:
        """Get current observation with error handling."""
        # Get cube state with triple reading and majority vote
        cube_states = []
        for _ in range(3):
            cube_state = self.cube_tracker.get_state()
            if cube_state.detected:
                cube_states.append(cube_state)
            time.sleep(0.05)
        
        if len(cube_states) == 0:
            return ObservationResult(
                observation=None,
                error_message="Cube not detected"
            )
        
        # Majority vote for cube position
        x_values = [round(s.x, 3) for s in cube_states]
        y_values = [round(s.y, 3) for s in cube_states]
        
        x_counter = Counter(x_values)
        y_counter = Counter(y_values)
        
        majority_x = x_counter.most_common(1)[0][0]
        majority_y = y_counter.most_common(1)[0][0]
        
        best_state = min(cube_states, key=lambda s: abs(s.x - majority_x) + abs(s.y - majority_y))
        cube_state = best_state
        
        # Get motor positions
        motor_positions = self.executor.read_positions()
        if motor_positions is None:
            return self._create_error_result("Failed to read motor positions")
        
        # Get motor currents
        motor_currents = self.executor.read_currents()
        if motor_currents is None:
            return self._create_error_result("Failed to read motor currents")
        
        # Initialize if needed
        if self.initial_motor_positions is None:
            self.initial_motor_positions = np.asarray(motor_positions, dtype=np.float32).copy()
        
        # Convert to numpy arrays
        motor_positions = np.asarray(motor_positions, dtype=np.float32)
        motor_currents = np.asarray(motor_currents, dtype=np.float32)
        
        # Calculate tension
        h_idx = config.get_motor_indices(self.executor.motor_ids)['horizontal']
        v_idx = config.get_motor_indices(self.executor.motor_ids)['vertical']
        
        horizontal_tension = abs(motor_currents[h_idx[0]]) + abs(motor_currents[h_idx[1]])
        vertical_tension = abs(motor_currents[v_idx[0]]) + abs(motor_currents[v_idx[1]])
        total_tension = horizontal_tension + vertical_tension
        
        # Normalize cube position (unbounded)
        cube_x_norm = (cube_state.x - self.x_min) / (self.x_max - self.x_min)
        cube_y_norm = (cube_state.y - self.y_min) / (self.y_max - self.y_min)
        
        # Normalize goal position
        if self.x_goal is not None and self.y_goal is not None:
            x_goal_norm = (self.x_goal - self.x_min) / (self.x_max - self.x_min)
            y_goal_norm = (self.y_goal - self.y_min) / (self.y_max - self.y_min)
        else:
            # Default to center if no goal set
            x_goal_norm = 0.5
            y_goal_norm = 0.5
        
        # Target error
        targets = np.array([self.executor.targets.get(m, 0) for m in self.executor.motor_ids])
        target_error = targets - motor_positions
        
        observation = Observation(
            cube_x=cube_state.x,
            cube_y=cube_state.y,
            cube_yaw=cube_state.yaw,
            motor_positions=motor_positions,
            motor_currents=motor_currents,
            initial_motor_positions=self.initial_motor_positions.copy(),
            cube_x_norm=cube_x_norm,
            cube_y_norm=cube_y_norm,
            x_goal_norm=x_goal_norm,
            y_goal_norm=y_goal_norm,
            horizontal_tension=float(horizontal_tension),
            vertical_tension=float(vertical_tension),
            total_tension=float(total_tension),
            target_error=target_error,
        )
        
        return ObservationResult(observation=observation)
    
    def _create_error_result(self, message: str) -> ObservationResult:
        """Create error result with hardware error state."""
        hardware_error, error_ids, error_status, _ = self.executor.get_hardware_error_state()
        return ObservationResult(
            observation=None,
            hardware_error=hardware_error,
            hardware_error_ids=error_ids,
            hardware_error_status=error_status,
            error_message=message,
        )