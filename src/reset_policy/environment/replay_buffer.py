"""
Replay buffer for SAC + HER training on the cube-string system.

Stores transitions:
    (state, action, executed_action, reward, next_state, done,
     achieved_goal, desired_goal)

- state:            (20,) float32  — observation at time t
- action:           (4,)  float32  — raw action from the policy (before safety filter)
- executed_action:  (4,)  float32  — actual action sent to motors (after safety filter)
- reward:           scalar float32
- next_state:       (20,) float32  — observation at time t+1
- done:             scalar float32 (1.0 if terminal, 0.0 otherwise)
- achieved_goal:    (2,)  float32  — cube position at time t+1 (physical units, meters)
- desired_goal:     (2,)  float32  — goal position for this transition (physical units, meters)

The buffer has UNLIMITED capacity for now. Add a `max_size` argument later
if memory becomes a concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class Batch:
    """A sampled minibatch of transitions, as torch tensors."""
    states: torch.Tensor
    actions: torch.Tensor
    executed_actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    achieved_goals: torch.Tensor
    desired_goals: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        """Move all tensors to a device."""
        return Batch(
            states=self.states.to(device),
            actions=self.actions.to(device),
            executed_actions=self.executed_actions.to(device),
            rewards=self.rewards.to(device),
            next_states=self.next_states.to(device),
            dones=self.dones.to(device),
            achieved_goals=self.achieved_goals.to(device),
            desired_goals=self.desired_goals.to(device),
        )

    def __len__(self) -> int:
        return self.states.shape[0]


class ReplayBuffer:
    """Simple unlimited-capacity replay buffer with both raw and executed actions."""

    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 4,
        goal_dim: int = 2,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

        # Preallocate as lists of numpy arrays; convert to stacked arrays on sample.
        # This is faster for append-heavy workloads than a preallocated ring buffer
        # when capacity is unlimited.
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._executed_actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._next_states: list[np.ndarray] = []
        self._dones: list[float] = []
        self._achieved_goals: list[np.ndarray] = []
        self._desired_goals: list[np.ndarray] = []

        # Cached stacked arrays for fast sampling. Invalidated on add.
        self._cache: Optional[dict] = None

    # ---------- Core API ----------

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
        """Add one transition."""
        state = np.asarray(state, dtype=np.float32).reshape(self.state_dim)
        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        executed_action = np.asarray(executed_action, dtype=np.float32).reshape(self.action_dim)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(self.state_dim)
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32).reshape(self.goal_dim)
        desired_goal = np.asarray(desired_goal, dtype=np.float32).reshape(self.goal_dim)

        self._states.append(state)
        self._actions.append(action)
        self._executed_actions.append(executed_action)
        self._rewards.append(float(reward))
        self._next_states.append(next_state)
        self._dones.append(float(done))
        self._achieved_goals.append(achieved_goal)
        self._desired_goals.append(desired_goal)

        self._cache = None  # invalidate

    def add_episode(self, episode: dict) -> int:
        """
        Add all transitions from one episode in a single call.

        `episode` must be a dict with keys:
            states:            (T, 20)
            actions:           (T, 4)
            executed_actions:  (T, 4)
            rewards:           (T,)
            next_states:       (T, 20)
            dones:             (T,)
            achieved_goals:    (T, 2)
            desired_goals:     (T, 2)

        Returns the number of transitions added.
        """
        T = len(episode["rewards"])
        for t in range(T):
            self.add(
                state=episode["states"][t],
                action=episode["actions"][t],
                executed_action=episode["executed_actions"][t],
                reward=episode["rewards"][t],
                next_state=episode["next_states"][t],
                done=episode["dones"][t],
                achieved_goal=episode["achieved_goals"][t],
                desired_goal=episode["desired_goals"][t],
            )
        return T

    def __len__(self) -> int:
        return len(self._states)

    def sample(self, batch_size: int, device: torch.device = None) -> Batch:
        """
        Sample a uniform random minibatch.

        Returns a `Batch` of torch tensors on `device` (or CPU if None).
        """
        if len(self) == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        cache = self._get_cache()
        n = len(self)

        # With replacement if batch_size > n, otherwise without replacement.
        replace = batch_size > n
        idx = np.random.choice(n, size=batch_size, replace=replace)

        batch = Batch(
            states=torch.from_numpy(cache["states"][idx].copy()),
            actions=torch.from_numpy(cache["actions"][idx].copy()),
            executed_actions=torch.from_numpy(cache["executed_actions"][idx].copy()),
            rewards=torch.from_numpy(cache["rewards"][idx].copy()),
            next_states=torch.from_numpy(cache["next_states"][idx].copy()),
            dones=torch.from_numpy(cache["dones"][idx].copy()),
            achieved_goals=torch.from_numpy(cache["achieved_goals"][idx].copy()),
            desired_goals=torch.from_numpy(cache["desired_goals"][idx].copy()),
        )

        if device is not None:
            batch = batch.to(device)

        return batch

    # ---------- Cache ----------

    def _get_cache(self) -> dict:
        """Build and cache stacked numpy arrays. Invalidated on add()."""
        if self._cache is None:
            self._cache = {
                "states": np.stack(self._states, axis=0),
                "actions": np.stack(self._actions, axis=0),
                "executed_actions": np.stack(self._executed_actions, axis=0),
                "rewards": np.asarray(self._rewards, dtype=np.float32),
                "next_states": np.stack(self._next_states, axis=0),
                "dones": np.asarray(self._dones, dtype=np.float32),
                "achieved_goals": np.stack(self._achieved_goals, axis=0),
                "desired_goals": np.stack(self._desired_goals, axis=0),
            }
        return self._cache

    # ---------- Persistence ----------

    def save(self, path: str | Path) -> None:
        """Save the buffer to a .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = self._get_cache()
        np.savez_compressed(
            path,
            states=cache["states"],
            actions=cache["actions"],
            executed_actions=cache["executed_actions"],
            rewards=cache["rewards"],
            next_states=cache["next_states"],
            dones=cache["dones"],
            achieved_goals=cache["achieved_goals"],
            desired_goals=cache["desired_goals"],
            state_dim=np.array([self.state_dim]),
            action_dim=np.array([self.action_dim]),
            goal_dim=np.array([self.goal_dim]),
        )
        print(f"Saved {len(self)} transitions to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        """Load the buffer from a .npz file."""
        path = Path(path)
        data = np.load(path)

        buf = cls(
            state_dim=int(data["state_dim"][0]),
            action_dim=int(data["action_dim"][0]),
            goal_dim=int(data["goal_dim"][0]),
        )

        states = data["states"]
        actions = data["actions"]
        executed_actions = data["executed_actions"]
        rewards = data["rewards"]
        next_states = data["next_states"]
        dones = data["dones"]
        achieved_goals = data["achieved_goals"]
        desired_goals = data["desired_goals"]

        for t in range(states.shape[0]):
            buf.add(
                state=states[t],
                action=actions[t],
                executed_action=executed_actions[t],
                reward=float(rewards[t]),
                next_state=next_states[t],
                done=float(dones[t]),
                achieved_goal=achieved_goals[t],
                desired_goal=desired_goals[t],
            )

        print(f"Loaded {len(buf)} transitions from {path}")
        return buf

    # ---------- Diagnostics ----------

    def stats(self) -> dict:
        """Return summary statistics for logging."""
        if len(self) == 0:
            return {"size": 0}

        cache = self._get_cache()
        return {
            "size": len(self),
            "reward_mean": float(cache["rewards"].mean()),
            "reward_std": float(cache["rewards"].std()),
            "reward_min": float(cache["rewards"].min()),
            "reward_max": float(cache["rewards"].max()),
            "done_fraction": float(cache["dones"].mean()),
            "action_abs_mean": float(np.abs(cache["actions"]).mean()),
            "executed_action_abs_mean": float(np.abs(cache["executed_actions"]).mean()),
            "action_filter_diff": float(
                np.abs(cache["actions"] - cache["executed_actions"]).mean()
            ),
        }

    def clear(self) -> None:
        """Remove all transitions."""
        self._states.clear()
        self._actions.clear()
        self._executed_actions.clear()
        self._rewards.clear()
        self._next_states.clear()
        self._dones.clear()
        self._achieved_goals.clear()
        self._desired_goals.clear()
        self._cache = None