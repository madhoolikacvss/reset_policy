"""
Standalone reward reconstruction for the cube-string system.

Purpose:
    Reconstruct the exact reward the environment computed, given the per-step
    logged data. Used by the offline data loader and by HER relabeling.

The env's reward (from environment.py, step()):

    # 1. RewardFunction.compute
    total = distance_reward + current_change_penalty
          + hardware_error_penalty + tension_penalty
    # 2. Goal bonus
    if distance_to_goal < 0.005:
        total += 1.0
    # 3. Clip to [0, 1]
    total = clip(total, 0, 1)
    # 4. Add safety penalty (can be negative)
    total += safety_penalty

This module replicates the exact same computation given:
    - cube position (achieved goal)
    - goal position (desired goal)
    - raw motor currents in mA (physical units)
    - previous step's raw currents (for the change penalty)
    - safety penalty (from the log)
    - hardware_error flag (from episodes.csv)

All constants MUST match RewardFunction in reset_policy/environment/reward.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# ---------------- Constants (must match RewardFunction) ----------------

DISTANCE_SCALE = 10.0
CURRENT_CHANGE_WEIGHT = 0.1
CURRENT_LIMIT = 1750.0      # matches RewardFunction.current_limit
TENSION_PENALTY_WEIGHT = 0.3
TENSION_THRESHOLD = 500.0   # mA
TENSION_MAX = 1500.0        # mA

HARDWARE_ERROR_PENALTY = 1.0
GOAL_SUCCESS_THRESHOLD = 0.005   # meters


# ---------------- Reconstruction output ----------------

@dataclass
class RewardReconstruction:
    """All reward components for a single transition."""
    total: float
    distance_reward: float
    current_change_penalty: float
    tension_penalty: float
    hardware_error_penalty: float
    goal_bonus: float
    safety_penalty: float
    distance_to_goal: float


# ---------------- Individual components ----------------

def distance_reward(
    cube_x: float, cube_y: float,
    goal_x: float, goal_y: float,
    distance_scale: float = DISTANCE_SCALE,
) -> float:
    """exp(-distance_scale * ||cube - goal||). Range: (0, 1]."""
    distance = float(np.sqrt((cube_x - goal_x) ** 2 + (cube_y - goal_y) ** 2))
    return float(np.exp(-distance_scale * distance))


def current_change_penalty(
    motor_currents: np.ndarray,
    prev_currents: Optional[np.ndarray],
    current_change_weight: float = CURRENT_CHANGE_WEIGHT,
    current_limit: float = CURRENT_LIMIT,
) -> float:
    """
    Penalty for sudden current changes. Range: [-current_change_weight, 0].

    If prev_currents is None (first step of episode), returns 0 (matches
    RewardFunction's behavior when self.prev_currents is None).
    """
    currents = np.asarray(motor_currents, dtype=np.float32)
    if prev_currents is None:
        return 0.0

    prev = np.asarray(prev_currents, dtype=np.float32)
    max_change = float(np.max(np.abs(currents - prev)))
    normalized = float(np.clip(max_change / current_limit, 0.0, 1.0))
    return -current_change_weight * normalized


def tension_penalty(
    motor_currents: np.ndarray,
    tension_penalty_weight: float = TENSION_PENALTY_WEIGHT,
    tension_threshold: float = TENSION_THRESHOLD,
    tension_max: float = TENSION_MAX,
) -> float:
    """
    Penalty for high tension. Range: [-tension_penalty_weight, 0].

    Tension is defined as the sum of paired motor currents:
        horizontal = |currents[0]| + |currents[1]|   (motors 16, 17)
        vertical   = |currents[2]| + |currents[3]|   (motors 18, 19)
        max_tension = max(horizontal, vertical)
    """
    currents = np.asarray(motor_currents, dtype=np.float32)
    horizontal = abs(float(currents[0])) + abs(float(currents[1]))
    vertical = abs(float(currents[2])) + abs(float(currents[3]))
    max_tension = max(horizontal, vertical)

    if max_tension <= tension_threshold:
        return 0.0

    normalized = float(np.clip(
        (max_tension - tension_threshold) / (tension_max - tension_threshold),
        0.0, 1.0,
    ))
    return -tension_penalty_weight * normalized


def hardware_error_penalty(hardware_error: bool) -> float:
    """-HARDWARE_ERROR_PENALTY if hardware_error, else 0."""
    return -HARDWARE_ERROR_PENALTY if hardware_error else 0.0


def goal_bonus(cube_x: float, cube_y: float,
               goal_x: float, goal_y: float,
               threshold: float = GOAL_SUCCESS_THRESHOLD) -> float:
    """1.0 if the cube is within `threshold` of the goal, else 0.0."""
    distance = float(np.sqrt((cube_x - goal_x) ** 2 + (cube_y - goal_y) ** 2))
    return 1.0 if distance < threshold else 0.0


# ---------------- Full reconstruction ----------------

def reconstruct_reward(
    cube_x: float, cube_y: float,
    goal_x: float, goal_y: float,
    motor_currents: np.ndarray,
    prev_motor_currents: Optional[np.ndarray],
    safety_penalty: float,
    hardware_error: bool = False,
) -> RewardReconstruction:
    """
    Reconstruct the exact reward the env computed for a single step.

    Matches environment.py's step() reward order:
        1. sum distance + change + tension + hw_err
        2. add goal bonus
        3. clip to [0, 1]
        4. add safety penalty

    Args:
        cube_x, cube_y:      achieved goal (physical, meters)
        goal_x, goal_y:      desired goal (physical, meters)
        motor_currents:      (4,) raw currents in mA
        prev_motor_currents: (4,) previous step's raw currents, or None
        safety_penalty:      scalar in [-1, 0], already computed by env
        hardware_error:      True only for the terminal step of a hw-error episode

    Returns:
        RewardReconstruction with all components and the final total.
    """
    dist_rew = distance_reward(cube_x, cube_y, goal_x, goal_y)
    change_pen = current_change_penalty(motor_currents, prev_motor_currents)
    tens_pen = tension_penalty(motor_currents)
    hw_pen = hardware_error_penalty(hardware_error)
    bonus = goal_bonus(cube_x, cube_y, goal_x, goal_y)

    # Sum the RewardFunction components + goal bonus
    partial = dist_rew + change_pen + tens_pen + hw_pen + bonus

    # Clip to [0, 1]
    clipped = float(np.clip(partial, 0.0, 1.0))

    # Add safety penalty (already in [-1, 0])
    total = clipped + float(safety_penalty)

    # Distance for reference
    distance = float(np.sqrt((cube_x - goal_x) ** 2 + (cube_y - goal_y) ** 2))

    return RewardReconstruction(
        total=total,
        distance_reward=dist_rew,
        current_change_penalty=change_pen,
        tension_penalty=tens_pen,
        hardware_error_penalty=hw_pen,
        goal_bonus=bonus,
        safety_penalty=float(safety_penalty),
        distance_to_goal=distance,
    )


# ---------------- Utilities for batch use ----------------

def unnormalize_currents(current_norm: np.ndarray,
                         current_limit: float = CURRENT_LIMIT) -> np.ndarray:
    """
    Invert the observation normalization applied to motor currents.

    In observation.py:
        current_norm = clip(motor_currents / CURRENT_LIMIT, -1, 1)

    So:
        motor_currents ≈ current_norm * CURRENT_LIMIT

    Note: the clip loses information if currents exceeded CURRENT_LIMIT.
    """
    return (np.asarray(current_norm, dtype=np.float32) * current_limit).astype(np.float32)