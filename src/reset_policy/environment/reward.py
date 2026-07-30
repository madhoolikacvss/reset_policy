"""
Current reward consists of:
    + Coverage reward
    - Motor current penalty
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class RewardBreakdown:
    total: float
    coverage_reward: float
    current_penalty: float


class RewardFunction:

    def __init__(
        self,
        coverage_weight: float = 1.0,
        current_weight: float = 0.001,
        current_change_weight: float = 0.01
    ):

        self.coverage_weight = coverage_weight
        self.current_weight = current_weight
        self.current_change_weight = current_change_weight 
        self.prev_currents = None

    def coverage_reward(
        self,
        visitation_count: int,
    ) -> float:
        """
        Reward visiting cells with low visitation count.

        First visit:
            1.0

        Second visit:
            0.5

        Third visit:
            0.33
        """

        if visitation_count <= 0:
            return 1.0

        return self.coverage_weight / visitation_count

    # reward.py - Modified current_penalty

    def current_change_penalty(self, motor_currents: Sequence[float]) -> float:
        """Penalize sudden changes in current (indicates binding/taut)."""
        if self.prev_currents is None:
            self.prev_currents = motor_currents
            return 0.0
        
        # Calculate change in current for each motor
        changes = [abs(c - prev) for c, prev in zip(motor_currents, self.prev_currents)]
        max_change = max(changes)
        
        self.prev_currents = motor_currents
        
        # Penalize large changes (squared for aggressiveness)
        return self.current_change_weight * (max_change ** 2)

    def current_penalty(self, motor_currents: Sequence[float]) -> float:
        """
        Penalize large motor currents.
        Using max current is more sensitive to taut strings than sum.
        """

        for id, m in zip([16, 17, 18, 19],motor_currents):
            print("current for id ", id, "is ", m)
        #individual motor overload
        max_current = max(abs(i) for i in motor_currents)
        
        # Square the penalty to make it more aggressive for high currents
        # This creates a much larger penalty when current is high
        return self.current_weight * (max_current ** 2)
    def compute(self, visitation_count: int, motor_currents: Sequence[float]) -> RewardBreakdown:
        coverage = self.coverage_reward(visitation_count)
        current_penalty = self.current_penalty(motor_currents)
        change_penalty = self.current_change_penalty(motor_currents)  # NEW
        
        total = coverage - current_penalty - change_penalty
        
        return RewardBreakdown(
            total=total,
            coverage_reward=coverage,
            current_penalty=current_penalty,
        )