"""
Reward function for the reset policy.

Reward components:

1. Coverage reward
   - New cell: positive reward
   - Previously visited cell: smaller positive reward

2. Motor current reward/penalty
   - Current < 500 mA:
       Positive reward (safe)
   - 500 <= Current <= 1800 mA:
       Small negative reward (moderate)
   - Current > 1800 mA:
       Large negative reward (high)

3. Current-change penalty
   - Penalizes sudden changes in motor current.

4. Hardware error penalty
   - Hardware error detected:
       Very large negative reward

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
    ):

        self.coverage_weight = coverage_weight

        self.safe_current_threshold = (
            safe_current_threshold
        )

        self.moderate_current_threshold = (
            moderate_current_threshold
        )

        self.safe_current_reward = (
            safe_current_reward
        )

        self.moderate_current_penalty = (
            moderate_current_penalty
        )

        self.high_current_penalty = (
            high_current_penalty
        )

        self.hardware_error_penalty = (
            hardware_error_penalty
        )

        self.current_change_weight = (
            current_change_weight
        )

        self.current_limit = current_limit

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

        First visit:
            +1.0

        Second visit:
            +0.5

        Third visit:
            +0.33

        etc.
        """

        if visitation_count <= 0:

            return self.coverage_weight

        return (
            self.coverage_weight
            / visitation_count
        )


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

            max current < 500 mA
                -> +0.1

            500 mA <= max current <= 1800 mA
                -> -0.5

            max current > 1800 mA
                -> -5.0

        We use the maximum absolute current because one
        overloaded motor is enough to indicate a problematic
        state.
        """

        max_current = max(
            abs(float(current))
            for current in motor_currents
        )

        # -------------------------
        # Safe
        # -------------------------

        if max_current < self.safe_current_threshold:

            return self.safe_current_reward

        # -------------------------
        # Moderate
        # -------------------------

        elif (
            max_current
            <= self.moderate_current_threshold
        ):

            return -self.moderate_current_penalty

        # -------------------------
        # High
        # -------------------------

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

        Current changes are normalized by the Dynamixel
        current limit before squaring so that the penalty
        remains on a reasonable scale.

        Example:

            current changes by 175 mA

            normalized change =
                175 / 1750 = 0.1

            penalty =
                weight * 0.1^2
        """

        motor_currents = np.asarray(
            motor_currents,
            dtype=np.float32,
        )

        # First observation has no previous current
        if self.prev_currents is None:

            self.prev_currents = (
                motor_currents.copy()
            )

            return 0.0

        changes = np.abs(
            motor_currents
            - self.prev_currents
        )

        max_change = np.max(changes)

        # Normalize current change
        normalized_change = (
            max_change
            / self.current_limit
        )

        # Update previous currents
        self.prev_currents = (
            motor_currents.copy()
        )

        return (
            self.current_change_weight
            * (normalized_change ** 2)
        )


    # =========================================================
    # Hardware error penalty
    # =========================================================

    def hardware_error_penalty_value(
        self,
        hardware_error: bool,
    ) -> float:

        """
        Apply a very large penalty when a Dynamixel
        hardware error is detected.

        Episode termination is NOT handled here.

        Args:
            hardware_error:
                True if any motor reports a hardware error.

        Returns:
            0.0 if no error
            -hardware_error_penalty if error detected
        """

        if hardware_error:

            return -self.hardware_error_penalty

        return 0.0


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
        """

        coverage = self.coverage_reward(
            visitation_count
        )

        current = self.current_reward(
            motor_currents
        )

        change_penalty = (
            self.current_change_penalty(
                motor_currents
            )
        )

        hardware_penalty = (
            self.hardware_error_penalty_value(
                hardware_error
            )
        )

        total = (
            coverage
            + current
            - change_penalty
            + hardware_penalty
        )

        return RewardBreakdown(
            total=total,
            coverage_reward=coverage,
            current_reward=current,
            current_change_penalty=change_penalty,
            hardware_error_penalty=hardware_penalty,
        )