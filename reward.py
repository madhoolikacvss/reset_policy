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
    ):

        self.coverage_weight = coverage_weight
        self.current_weight = current_weight

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

        return self.coverage_weight / visitation_count

    def current_penalty(self,motor_currents: Sequence[float],) -> float:
        #Penalize large motor currents.
        """
        we could also have max(currents)
        or a weighted sum 0.5(max_current) + 0.5(avg of the rest)
        """
        total_current = sum(abs(i) for i in motor_currents)
        return self.current_weight * total_current

    def compute(self,visitation_count: int,motor_currents: Sequence[float], ) -> RewardBreakdown:

        coverage = self.coverage_reward(visitation_count,)

        current = self.current_penalty(motor_currents,)
        total = coverage - current

        return RewardBreakdown(
            total=total,
            coverage_reward=coverage,
            current_penalty=current,
        )