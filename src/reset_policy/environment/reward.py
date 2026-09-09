"""
Reward function for goal-conditioned reset policy.

All reward components are in [0, 1] scale:
- Distance: +1 at goal, decays to 0 far away (inverse of euclidean distance)
- Current change: 0 penalty for no change, up to -0.1 for max change
- Hardware error: -1.0 (terminal, but still allowed as only negative)
- Tension: 0 to -0.3 penalty for high tension

Total reward is sum of components. Distance reward drives behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RewardBreakdown:
    """Breakdown of reward components."""
    total: float
    distance_reward: float
    current_change_penalty: float
    hardware_error_penalty: float
    tension_penalty: float = 0.0


class RewardFunction:
    """Computes rewards for goal-conditioned policy."""
    
    def __init__(
        self,
        # Distance reward
        distance_scale: float = 0.1,  # Controls decay rate (larger = decays faster)
        
        # Current change
        current_change_weight: float = 0.1,
        current_limit: float = 1750.0,
        
        # Hardware error
        hardware_error_penalty: float = 1.0,
        
        # Tension
        tension_penalty_weight: float = 0.3,
        tension_threshold: float = 500.0,
        tension_max: float = 1500.0,
    ):
        # Distance
        self.distance_scale = distance_scale
        
        # Current change
        self.current_change_weight = current_change_weight
        self.current_limit = current_limit
        
        # Hardware error
        self.hardware_error_penalty = hardware_error_penalty
        
        # Tension
        self.tension_penalty_weight = tension_penalty_weight
        self.tension_threshold = tension_threshold
        self.tension_max = tension_max
        
        # State
        self.prev_currents = None
    
    def reset(self):
        """Reset internal state (call at episode start)."""
        self.prev_currents = None
    
    def distance_reward(self, cube_x: float, cube_y: float, 
                        goal_x: float, goal_y: float) -> float:
        """
        Reward based on distance to goal: 0 to +1.
        
        Uses inverse of euclidean distance:
        reward = 1 / (1 + distance_scale * distance)
        
        At goal (distance=0): reward = 1.0
        Far away: reward approaches 0
        """
        distance = np.sqrt((cube_x - goal_x)**2 + (cube_y - goal_y)**2)
        return 1.0 / (1.0 + self.distance_scale * distance)
    
    def current_change_penalty(self, motor_currents: Sequence[float]) -> float:
        """
        Penalty for sudden current changes (0 to -0.1).
        """
        motor_currents = np.asarray(motor_currents, dtype=np.float32)
        
        if self.prev_currents is None:
            self.prev_currents = motor_currents.copy()
            return 0.0
        
        changes = np.abs(motor_currents - self.prev_currents)
        max_change = np.max(changes)
        normalized_change = np.clip(max_change / self.current_limit, 0, 1)
        
        self.prev_currents = motor_currents.copy()
        
        return -self.current_change_weight * normalized_change
    
    def hardware_error_penalty_value(self, hardware_error: bool) -> float:
        """Maximum penalty for hardware error."""
        return -self.hardware_error_penalty if hardware_error else 0.0
    
    def tension_penalty(self, motor_currents: Sequence[float]) -> float:
        """
        Penalty for high tension (0 to -0.3).
        """
        horizontal = abs(float(motor_currents[0])) + abs(float(motor_currents[1]))
        vertical = abs(float(motor_currents[2])) + abs(float(motor_currents[3]))
        max_tension = max(horizontal, vertical)
        
        if max_tension <= self.tension_threshold:
            return 0.0
        
        normalized = np.clip(
            (max_tension - self.tension_threshold) / (self.tension_max - self.tension_threshold),
            0, 1
        )
        
        return -self.tension_penalty_weight * normalized
    
    def compute(
        self,
        cube_x: float,
        cube_y: float,
        goal_x: float,
        goal_y: float,
        motor_currents: Sequence[float],
        hardware_error: bool = False,
    ) -> RewardBreakdown:
        """
        Compute total reward.
        
        Distance reward: 0 to +1 (primary driver)
        Current change: 0 to -0.1
        Hardware error: -1.0 (terminal only)
        Tension: 0 to -0.3
        """
        distance = self.distance_reward(cube_x, cube_y, goal_x, goal_y)
        change_penalty = self.current_change_penalty(motor_currents)
        hardware_penalty = self.hardware_error_penalty_value(hardware_error)
        tension = self.tension_penalty(motor_currents)
        
        total = (
            distance +
            change_penalty +
            hardware_penalty +
            tension
        )
        
        return RewardBreakdown(
            total=total,
            distance_reward=distance,
            current_change_penalty=change_penalty,
            hardware_error_penalty=hardware_penalty,
            tension_penalty=tension,
        )