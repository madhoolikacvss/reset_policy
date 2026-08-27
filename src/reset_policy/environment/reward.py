"""
Reward function for the reset policy.

All reward components are normalized to [-1, 1] scale:
- Coverage: +1 for new cell, 0 for revisited cell
- Current: +0.5 for safe, -0.5 for moderate, -1.0 for high
- Current change: 0 to -1 penalty for sudden changes
- Hardware error: -1.0 (maximum penalty)
- Tension: 0 to -1 penalty for high tension

Total reward is sum of components, typically in [-4, 2.5] range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from reset_policy.config import config


@dataclass
class RewardBreakdown:
    """Breakdown of reward components."""
    total: float
    coverage_reward: float
    current_reward: float
    current_change_penalty: float
    hardware_error_penalty: float
    tension_penalty: float = 0.0


class RewardFunction:
    """Computes rewards with consistent [-1, 1] scale components."""
    
    def __init__(
        self,
        # Coverage
        new_cell_reward: float = 1.0,      # +1 for new cell
        revisit_reward: float = 0.0,       # 0 for revisited cell
        
        # Current thresholds (mA)
        safe_current_threshold: float = 500.0,
        high_current_threshold: float = 1800.0,
        
        # Current rewards (normalized)
        safe_current_reward: float = 0.5,      # Positive but small
        moderate_current_penalty: float = 0.3,  # Small penalty
        high_current_penalty: float = 1.0,     # Maximum penalty
        
        # Hardware error
        hardware_error_penalty: float = 1.0,   # Maximum penalty
        
        # Current change
        current_change_weight: float = 0.5,    # Weight for change penalty
        current_limit: float = 1750.0,         # Normalization constant
        
        # Tension
        tension_penalty_weight: float = 0.5,   # Weight for tension penalty
        tension_threshold: float = 400.0,      # No penalty below this (mA)
        tension_max: float = 1500.0,           # Max penalty at this (mA)
    ):
        # Coverage
        self.new_cell_reward = new_cell_reward
        self.revisit_reward = revisit_reward
        
        # Current
        self.safe_current_threshold = safe_current_threshold
        self.high_current_threshold = high_current_threshold
        self.safe_current_reward = safe_current_reward
        self.moderate_current_penalty = moderate_current_penalty
        self.high_current_penalty = high_current_penalty
        
        # Hardware error
        self.hardware_error_penalty = hardware_error_penalty
        
        # Current change
        self.current_change_weight = current_change_weight
        self.current_limit = current_limit
        
        # Tension
        self.tension_penalty_weight = tension_penalty_weight
        self.tension_threshold = tension_threshold
        self.tension_max = tension_max
        
        # State
        self.prev_currents = None
    
    def reset(self):
        """Reset internal state (call at episode start)."""
        self.prev_currents = None
    
    def coverage_reward(self, visitation_count: int) -> float:
        """
        Reward for visiting cells.
        
        New cell: +1.0
        Revisited cell: 0.0
        
        This simplifies the reward - just explore new areas.
        """
        if visitation_count <= 0:
            return self.new_cell_reward
        return self.revisit_reward
    
    def current_reward(self, motor_currents: Sequence[float]) -> float:
        """
        Reward/penalty based on maximum motor current.
        
        Safe (< 500 mA): +0.5
        Moderate (500-1800 mA): -0.3
        High (> 1800 mA): -1.0
        """
        max_current = max(abs(float(c)) for c in motor_currents)
        
        if max_current < self.safe_current_threshold:
            return self.safe_current_reward
        elif max_current < self.high_current_threshold:
            return -self.moderate_current_penalty
        else:
            return -self.high_current_penalty
    
    def current_change_penalty(self, motor_currents: Sequence[float]) -> float:
        """
        Penalty for sudden current changes (0 to -1).
        
        Uses exponential scaling to heavily penalize large changes.
        """
        motor_currents = np.asarray(motor_currents, dtype=np.float32)
        
        if self.prev_currents is None:
            self.prev_currents = motor_currents.copy()
            return 0.0
        
        # Calculate max change
        changes = np.abs(motor_currents - self.prev_currents)
        max_change = np.max(changes)
        
        # Normalize to [0, 1]
        normalized_change = np.clip(max_change / self.current_limit, 0, 1)
        
        # Update previous currents
        self.prev_currents = motor_currents.copy()
        
        # Penalty (0 for no change, -weight for max change)
        return -self.current_change_weight * normalized_change
    
    def hardware_error_penalty_value(self, hardware_error: bool) -> float:
        """Maximum penalty for hardware error."""
        return -self.hardware_error_penalty if hardware_error else 0.0
    
    def tension_penalty(self, motor_currents: Sequence[float]) -> float:
        """
        Penalty for high tension (0 to -1).
        
        Tension = sum of absolute currents in each pair.
        High tension means motors are fighting each other.
        """
        # Calculate tensions
        horizontal = abs(float(motor_currents[0])) + abs(float(motor_currents[1]))
        vertical = abs(float(motor_currents[2])) + abs(float(motor_currents[3]))
        max_tension = max(horizontal, vertical)
        
        # No penalty below threshold
        if max_tension <= self.tension_threshold:
            return 0.0
        
        # Linear scaling from 0 at threshold to -weight at max
        normalized = np.clip(
            (max_tension - self.tension_threshold) / (self.tension_max - self.tension_threshold),
            0, 1
        )
        
        return -self.tension_penalty_weight * normalized
    
    def compute(
        self,
        visitation_count: int,
        motor_currents: Sequence[float],
        hardware_error: bool = False,
    ) -> RewardBreakdown:
        """
        Compute total reward.
        
        All components in [-1, 1]:
        - Coverage: +1 (new) or 0 (revisit)
        - Current: +0.5 (safe) to -1 (high)
        - Current change: 0 to -0.5
        - Hardware error: -1
        - Tension: 0 to -0.5
        
        Total range: approximately [-3, 2]
        """
        coverage = self.coverage_reward(visitation_count)
        current = self.current_reward(motor_currents)
        change_penalty = self.current_change_penalty(motor_currents)
        hardware_penalty = self.hardware_error_penalty_value(hardware_error)
        tension = self.tension_penalty(motor_currents)
        
        total = (
            coverage +
            current +
            change_penalty +
            hardware_penalty +
            tension
        )
        
        return RewardBreakdown(
            total=total,
            coverage_reward=coverage,
            current_reward=current,
            current_change_penalty=change_penalty,
            hardware_error_penalty=hardware_penalty,
            tension_penalty=tension,
        )