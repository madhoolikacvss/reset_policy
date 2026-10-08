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

from replay_buffer import Batch, ReplayBuffer


class HERBuffer:
    """
    HER wrapper around ReplayBuffer.

    - Transitions are added normally.
    - At episode end, `finalize_episode()` relabels each transition with k
      future-achieved-goals and adds the relabeled transitions to the buffer.
    - Sampling delegates to the base buffer.

    Args:
        base_buffer:      The underlying ReplayBuffer.
        goal_dims:        slice indicating which dims of `state` are the goal.
        k:                Number of relabeled transitions per original.
        reward_fn:        Callable (achieved_goal, desired_goal) -> float.
                          Must match the env's reward function.
        strategy:         "future" (uniform over t+1..T-1).
        normalize_goal:   Optional callable mapping physical goal -> normalized
                          goal (to write into the goal dims of the observation).
                          If None, the achieved_goal is assumed to already be
                          in the same space as the goal dims of the observation.
    """

    def __init__(
        self,
        base_buffer: ReplayBuffer,
        goal_dims: slice = slice(3, 5),
        k: int = 4,
        reward_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        strategy: str = "future",
        normalize_goal: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.base = base_buffer
        self.goal_dims = goal_dims
        self.k = k
        self.reward_fn = reward_fn
        self.strategy = strategy
        self.normalize_goal = normalize_goal

        if reward_fn is None:
            raise ValueError("reward_fn must be provided (used for relabeling).")
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
    ) -> None:
        """
        Add a transition to BOTH the base buffer and the current episode.

        The base buffer gets the original transition immediately (so sampling
        can proceed even before episode finalization). The current-episode
        list gets it so we can relabel it at the end.
        """
        # Add the original transition to the base buffer.
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

        # Store a copy for later relabeling.
        self._current_episode.append({
            "state": np.asarray(state, dtype=np.float32).copy(),
            "action": np.asarray(action, dtype=np.float32).copy(),
            "executed_action": np.asarray(executed_action, dtype=np.float32).copy(),
            "next_state": np.asarray(next_state, dtype=np.float32).copy(),
            "done": float(done),
            "achieved_goal": np.asarray(achieved_goal, dtype=np.float32).copy(),
            "desired_goal": np.asarray(desired_goal, dtype=np.float32).copy(),
        })

    def finalize_episode(self) -> int:
        T = len(self._current_episode)
        if T == 0:
            return 0

        added = 0
        for t in range(T):
            trans = self._current_episode[t]

            # Skip relabeling for the last transition (no future states available).
            if t >= T - 1:
                continue

            # Sample k future goals, uniform over (t+1 .. T-1).
            future_indices = np.random.randint(low=t + 1, high=T, size=self.k)

            for j in future_indices:
                new_goal_physical = self._current_episode[j]["achieved_goal"]

                # Reward at transition t = reward_fn(achieved_goal_t+1, new_goal).
                # The achieved_goal stored with transition t corresponds to s_{t+1}.
                new_reward = self.reward_fn(trans["achieved_goal"], new_goal_physical)

                # Relabel the state and next_state with the new goal.
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

    def _relabel_state(self, state: np.ndarray, new_goal_physical: np.ndarray) -> np.ndarray:
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