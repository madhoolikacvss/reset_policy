"""
Reward function for the reset policy.

Reward components:

1. Coverage reward
   - New cell: positive reward
   - Previously visited cell: smaller positive reward

2. Motor current reward/penalty
   - Current < 500 mA: Positive reward (safe)
   - 500 <= Current <= 1800 mA: Small negative reward (moderate)
   - Current > 1800 mA: Large negative reward (high)

3. Current-change penalty
   - Penalizes sudden changes in motor current.

4. Hardware error penalty
   - Hardware error detected: Very large negative reward

5. Tension penalty (NEW)
   - Penalizes high tension (motors fighting each other)
   - Tension = sum of absolute currents in a pair
   - High tension = motors pulling against each other

NOTE:
    Episode termination is intentionally NOT handled here.
    The environment should decide whether a hardware error
    terminates the episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RewardBreakdown:

    total: float

    coverage_reward: float

    current_reward: float

    current_change_penalty: float

    hardware_error_penalty: float

    tension_penalty: float = 0.0  # <-- NEW


class RewardFunction:

    def __init__(
        self,

        # -------------------------
        # Coverage
        # -------------------------
        coverage_weight: float = 1.0,

        # -------------------------
        # Current thresholds
        # -------------------------
        safe_current_threshold: float = 500.0,
        moderate_current_threshold: float = 1800.0,

        # -------------------------
        # Current rewards
        # -------------------------
        safe_current_reward: float = 0.1,
        moderate_current_penalty: float = 0.5,
        high_current_penalty: float = 5.0,

        # -------------------------
        # Hardware error
        # -------------------------
        hardware_error_penalty: float = 50.0,

        # -------------------------
        # Current change
        # -------------------------
        current_change_weight: float = 0.1,

        # Dynamixel current limit
        current_limit: float = 1750.0,

        # -------------------------
        # Tension penalty (NEW)
        # -------------------------
        tension_penalty_weight: float = 0.5,    # How much to penalize tension
        tension_threshold: float = 400.0,       # mA - below this, no penalty
        tension_max: float = 1500.0,            # mA - max penalty at this level
    ):

        self.coverage_weight = coverage_weight

        self.safe_current_threshold = safe_current_threshold
        self.moderate_current_threshold = moderate_current_threshold

        self.safe_current_reward = safe_current_reward
        self.moderate_current_penalty = moderate_current_penalty
        self.high_current_penalty = high_current_penalty

        self.hardware_error_penalty = hardware_error_penalty

        self.current_change_weight = current_change_weight
        self.current_limit = current_limit

        # NEW: Tension penalty params
        self.tension_penalty_weight = tension_penalty_weight
        self.tension_threshold = tension_threshold
        self.tension_max = tension_max

        self.prev_currents = None


    # =========================================================
    # Coverage reward
    # =========================================================

    def coverage_reward(
        self,
        visitation_count: int,
    ) -> float:
        """
        Reward visiting cells with low visitation count.

        First visit: +1.0
        Second visit: +0.5
        Third visit: +0.33
        etc.
        """
        if visitation_count <= 0:
            return self.coverage_weight

        return self.coverage_weight / visitation_count


    # =========================================================
    # Current reward
    # =========================================================

    def current_reward(
        self,
        motor_currents: Sequence[float],
    ) -> float:
        """
        Compute reward/penalty based on maximum motor current.

        Thresholds:
            max current < 500 mA   -> +0.1
            500 mA <= max current <= 1800 mA -> -0.5
            max current > 1800 mA  -> -5.0
        """
        max_current = max(abs(float(current)) for current in motor_currents)

        if max_current < self.safe_current_threshold:
            return self.safe_current_reward

        elif max_current <= self.moderate_current_threshold:
            return -self.moderate_current_penalty

        else:
            return -self.high_current_penalty


    # =========================================================
    # Current change penalty
    # =========================================================

    def current_change_penalty(
        self,
        motor_currents: Sequence[float],
    ) -> float:
        """
        Penalize sudden changes in motor current.
        """
        motor_currents = np.asarray(motor_currents, dtype=np.float32)

        if self.prev_currents is None:
            self.prev_currents = motor_currents.copy()
            return 0.0

        changes = np.abs(motor_currents - self.prev_currents)
        max_change = np.max(changes)

        normalized_change = max_change / self.current_limit

        self.prev_currents = motor_currents.copy()

        return self.current_change_weight * (normalized_change ** 2)


    # =========================================================
    # Hardware error penalty
    # =========================================================

    def hardware_error_penalty_value(
        self,
        hardware_error: bool,
    ) -> float:
        """Apply a very large penalty when a hardware error is detected."""
        if hardware_error:
            return -self.hardware_error_penalty
        return 0.0


    # =========================================================
    # Tension penalty (NEW)
    # =========================================================

    def tension_penalty(
        self,
        motor_currents: Sequence[float],
    ) -> float:
        """
        Compute penalty based on tension (motors fighting each other).

        Tension is the sum of absolute currents in each pair:
            - Horizontal pair: motors 16, 17 (indices 0, 1)
            - Vertical pair: motors 18, 19 (indices 2, 3)

        High tension = motors fighting each other = bad.
        Penalty scales from 0 at tension_threshold to max at tension_max.

        This teaches the policy to coordinate motors so they don't fight.
        """
        # Calculate tension for each pair
        horizontal_tension = abs(float(motor_currents[0])) + abs(float(motor_currents[1]))
        vertical_tension = abs(float(motor_currents[2])) + abs(float(motor_currents[3]))

        # Use the max tension of the two pairs (most restrictive)
        max_pair_tension = max(horizontal_tension, vertical_tension)

        # No penalty if below threshold
        if max_pair_tension <= self.tension_threshold:
            return 0.0

        # Scale penalty from 0 at threshold to 1 at tension_max
        penalty_scale = min(
            1.0,
            (max_pair_tension - self.tension_threshold) / (self.tension_max - self.tension_threshold)
        )

        return -self.tension_penalty_weight * penalty_scale


    # =========================================================
    # Total reward
    # =========================================================

    def compute(
        self,
        visitation_count: int,
        motor_currents: Sequence[float],
        hardware_error: bool = False,
    ) -> RewardBreakdown:
        """
        Compute the complete reward.

        Total reward:
            coverage
            + current reward
            - current-change penalty
            + hardware-error penalty
            - tension penalty (NEW)
        """

        coverage = self.coverage_reward(visitation_count)

        current = self.current_reward(motor_currents)

        change_penalty = self.current_change_penalty(motor_currents)

        hardware_penalty = self.hardware_error_penalty_value(hardware_error)

        # NEW: Tension penalty
        tension_penalty = self.tension_penalty(motor_currents)

        total = (
            coverage
            + current
            - change_penalty
            + hardware_penalty
            + tension_penalty  # <-- tension_penalty is already negative
        )

        return RewardBreakdown(
            total=total,
            coverage_reward=coverage,
            current_reward=current,
            current_change_penalty=change_penalty,
            hardware_error_penalty=hardware_penalty,
            tension_penalty=tension_penalty,  # <-- NEW
        )