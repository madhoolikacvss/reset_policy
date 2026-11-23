"""
Hindsight Experience Replay (HER) wrapper around ReplayBuffer.

Usage:
    base = ReplayBuffer()
    her = HERBuffer(base, goal_dims=slice(3, 5), k=4, reward_fn=compute_reward)

    # During training: add transitions normally.
    her.add(state, action, executed_action, reward, next_state, done,
            achieved_goal, desired_goal)

    # At episode end: relabel and add k relabeled transitions per original.
    her.finalize_episode()

    # Sampling: works like the base buffer.
    batch = her.sample(batch_size, device)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from pathlib import Path
import sys
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)
from rl.replay_buffer import Batch, ReplayBuffer

# Match the env's reward constants exactly.
DISTANCE_SCALE = 10.0
GOAL_SUCCESS_THRESHOLD = 0.005


class HERBuffer:
    """
    HER wrapper around ReplayBuffer with full reward reconstruction.

    Args:
        base_buffer:        The underlying ReplayBuffer.
        goal_dims:          slice indicating which dims of `state` are the goal.
        k:                  Number of relabeled transitions per original.
        strategy:           "future" (uniform over t+1..T-1).
        normalize_goal:     Callable mapping physical goal -> normalized goal
                            (to write into the goal dims of the observation).
                            If None, achieved_goal is assumed already normalized.
        distance_scale:     Exponential decay for distance reward.
        goal_threshold:     Distance below which the goal is considered reached.
    """

    def __init__(
        self,
        base_buffer: ReplayBuffer,
        goal_dims: slice = slice(3, 5),
        k: int = 4,
        strategy: str = "future",
        normalize_goal: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        distance_scale: float = DISTANCE_SCALE,
        goal_threshold: float = GOAL_SUCCESS_THRESHOLD,
    ):
        self.base = base_buffer
        self.goal_dims = goal_dims
        self.k = k
        self.strategy = strategy
        self.normalize_goal = normalize_goal
        self.distance_scale = distance_scale
        self.goal_threshold = goal_threshold

        if strategy != "future":
            raise NotImplementedError(f"Strategy '{strategy}' not implemented.")

        # Episode-in-progress buffer (cleared on finalize).
        self._current_episode: list[dict] = []

    # ---------- Core API ----------

    def __len__(self) -> int:
        return len(self.base)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        executed_action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        distance_reward: float = 0.0,
        current_change_penalty: float = 0.0,
        hardware_error_penalty: float = 0.0,
        tension_penalty: float = 0.0,
        safety_filtering_penalty: float = 0.0,
    ) -> None:
        """
        Add one transition.

        The `reward` argument is the final env reward for this transition. It is
        stored in the base buffer as-is. The individual penalty components are
        also stored (in the episode-in-progress buffer) so HER can reconstruct
        the full reward when relabeling.
        """
        # 1) Original transition goes into the base buffer with the env's final reward.
        self.base.add(
            state=state,
            action=action,
            executed_action=executed_action,
            reward=reward,
            next_state=next_state,
            done=done,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
        )

        # 2) Store a copy in the current-episode buffer with all components.
        self._current_episode.append({
            "state": np.asarray(state, dtype=np.float32).copy(),
            "action": np.asarray(action, dtype=np.float32).copy(),
            "executed_action": np.asarray(executed_action, dtype=np.float32).copy(),
            "next_state": np.asarray(next_state, dtype=np.float32).copy(),
            "done": float(done),
            "achieved_goal": np.asarray(achieved_goal, dtype=np.float32).copy(),
            "desired_goal": np.asarray(desired_goal, dtype=np.float32).copy(),
            # Penalty components (goal-invariant — preserved across relabeling)
            "distance_reward": float(distance_reward),
            "current_change_penalty": float(current_change_penalty),
            "hardware_error_penalty": float(hardware_error_penalty),
            "tension_penalty": float(tension_penalty),
            "safety_filtering_penalty": float(safety_filtering_penalty),
        })

    def finalize_episode(self) -> int:
        """
        Relabel the just-finished episode with k future-goal substitutions per
        transition, and add the relabeled transitions to the base buffer.

        The relabeled reward is:
            new_distance = exp(-scale * ||achieved_t - g'||)
            new_bonus = 1.0 if ||achieved_t - g'|| < threshold else 0.0
            partial = new_distance + change_t + hw_t + tension_t + new_bonus
            new_reward = clip(partial, 0, 1) + safety_t

        Returns the number of relabeled transitions added.
        """
        T = len(self._current_episode)
        if T == 0:
            return 0

        added = 0
        for t in range(T):
            trans = self._current_episode[t]

            # Skip relabeling for the last transition (no future states).
            if t >= T - 1:
                continue

            # Sample k future goals, uniform over (t+1 .. T-1).
            future_indices = np.random.randint(low=t + 1, high=T, size=self.k)

            for j in future_indices:
                new_goal_physical = self._current_episode[j]["achieved_goal"]

                # Recompute distance-dependent components
                new_distance = float(np.linalg.norm(
                    trans["achieved_goal"] - new_goal_physical
                ))
                new_distance_reward = float(np.exp(-self.distance_scale * new_distance))
                new_goal_bonus = 1.0 if new_distance < self.goal_threshold else 0.0

                # Combine with goal-invariant penalties
                partial = (
                    new_distance_reward
                    + trans["current_change_penalty"]
                    + trans["hardware_error_penalty"]
                    + trans["tension_penalty"]
                    + new_goal_bonus
                )
                clipped = float(np.clip(partial, 0.0, 1.0))
                new_reward = clipped + trans["safety_filtering_penalty"]

                # Relabel observation (replace goal dims)
                new_state = self._relabel_state(trans["state"], new_goal_physical)
                new_next_state = self._relabel_state(trans["next_state"], new_goal_physical)

                self.base.add(
                    state=new_state,
                    action=trans["action"],
                    executed_action=trans["executed_action"],
                    reward=new_reward,
                    next_state=new_next_state,
                    done=trans["done"],
                    achieved_goal=trans["achieved_goal"],
                    desired_goal=new_goal_physical,
                )
                added += 1

        self._current_episode.clear()
        return added

    def _relabel_state(
        self, state: np.ndarray, new_goal_physical: np.ndarray
    ) -> np.ndarray:
        """Return a copy of `state` with the goal dims replaced by the new goal."""
        new_state = state.copy()
        goal_physical = np.asarray(new_goal_physical, dtype=np.float32).reshape(2)

        if self.normalize_goal is not None:
            goal_in_obs = self.normalize_goal(goal_physical).astype(np.float32)
        else:
            goal_in_obs = goal_physical.astype(np.float32)

        new_state[self.goal_dims] = goal_in_obs
        return new_state

    # ---------- Delegation to base ----------

    def sample(self, batch_size: int, device=None) -> Batch:
        return self.base.sample(batch_size, device=device)

    def save(self, path) -> None:
        self.base.save(path)

    @classmethod
    def load(cls, path, **her_kwargs) -> "HERBuffer":
        """Load base buffer from disk and wrap it. HER config is passed in."""
        base = ReplayBuffer.load(path)
        return cls(base_buffer=base, **her_kwargs)

    def stats(self) -> dict:
        return self.base.stats()

    def clear(self) -> None:
        self.base.clear()
        self._current_episode.clear()