"""
Soft Actor-Critic (SAC) for continuous control.

Implements:
- Twin Q-networks (Q1, Q2) with target networks
- Squashed Gaussian policy (TanhNormal)
- Automatic entropy tuning (learnable temperature α)
- Delayed policy updates (actor updates every `policy_delay` critic steps)
- Soft target updates (Polyak averaging with tau)

Reference: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy
Deep Reinforcement Learning with a Stochastic Actor" (2018) and
"Soft Actor-Critic Algorithms and Applications" (2018).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, TransformedDistribution


# ---------------- Networks ----------------

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SquashedGaussianMLP(nn.Module):
    """
    Actor: MLP producing mean and log_std of a Gaussian, then squashed by tanh.
    Output is in (-1, 1). Log-prob includes the tanh Jacobian correction.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        layers = []
        prev = obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.mean_head = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Return (action, log_prob, mean_action, mean, log_std).
        - action:     squashed action in (-1, 1)
        - log_prob:   log π(action | obs) with tanh correction, summed over action dims
        - mean_action: tanh(mean), for deterministic evaluation
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        normal = Normal(mean, std)
        dist = TransformedDistribution(normal, [TanhTransform(cache_size=1)])

        if deterministic:
            action = torch.tanh(mean)
            # Log-prob is not needed for deterministic evaluation
            log_prob = None
        else:
            action = dist.rsample()  # reparameterized sample
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, torch.tanh(mean), mean, log_std


class QNetwork(nn.Module):
    """Critic: MLP taking (obs, action) and producing a scalar Q-value."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        layers = []
        prev = obs_dim + action_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------- Config ----------------

@dataclass
class SACConfig:
    obs_dim: int = 20
    action_dim: int = 4
    hidden_dim: int = 256
    n_layers: int = 2

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2

    # Target entropy for automatic α tuning
    # Standard heuristic: -action_dim
    target_entropy: Optional[float] = None

    # Gradient clipping (None to disable)
    max_grad_norm: Optional[float] = 1.0


# ---------------- Agent ----------------

class SACAgent:
    """SAC agent with twin critics, automatic α tuning, and target networks."""

    def __init__(self, cfg: SACConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        # Actor
        self.actor = SquashedGaussianMLP(
            cfg.obs_dim, cfg.action_dim,
            cfg.hidden_dim, cfg.n_layers,
        ).to(device)

        # Twin critics
        self.q1 = QNetwork(
            cfg.obs_dim, cfg.action_dim,
            cfg.hidden_dim, cfg.n_layers,
        ).to(device)
        self.q2 = QNetwork(
            cfg.obs_dim, cfg.action_dim,
            cfg.hidden_dim, cfg.n_layers,
        ).to(device)

        # Target critics (frozen copies, soft-updated)
        self.q1_target = QNetwork(
            cfg.obs_dim, cfg.action_dim,
            cfg.hidden_dim, cfg.n_layers,
        ).to(device)
        self.q2_target = QNetwork(
            cfg.obs_dim, cfg.action_dim,
            cfg.hidden_dim, cfg.n_layers,
        ).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Freeze target params
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # Learnable temperature α = exp(log_alpha)
        if cfg.target_entropy is None:
            self.target_entropy = -float(cfg.action_dim)
        else:
            self.target_entropy = float(cfg.target_entropy)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()

        # Optimizers
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.lr_actor
        )
        self.q1_opt = torch.optim.Adam(
            self.q1.parameters(), lr=cfg.lr_critic
        )
        self.q2_opt = torch.optim.Adam(
            self.q2.parameters(), lr=cfg.lr_critic
        )
        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha], lr=cfg.lr_alpha
        )

        # Counters for delayed policy updates
        self._update_step = 0

    # ---------------- Action selection ----------------

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select one action for one observation. Returns numpy (action_dim,)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _, mean_action, _, _ = self.actor.sample(obs_t, deterministic=deterministic)
        if deterministic:
            a = mean_action
        else:
            a = action
        return a.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def select_actions_batch(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select actions for a batch of observations."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action, _, mean_action, _, _ = self.actor.sample(obs_t, deterministic=deterministic)
        a = mean_action if deterministic else action
        return a.cpu().numpy()

    # ---------------- Update ----------------

    def update(self, batch) -> dict:
        """
        One SAC update step on a minibatch.

        `batch` is a `replay_buffer.Batch` (already on device).
        Uses batch.actions (raw actions) for the critic update.
        """
        obs = batch.states
        action = batch.actions
        reward = batch.rewards
        next_obs = batch.next_states
        done = batch.dones

        # ---------------- Critic update ----------------
        with torch.no_grad():
            next_action, next_log_prob, _, _, _ = self.actor.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_action)
            q2_next = self.q2_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next)
            target_q = reward + self.cfg.gamma * (1.0 - done) * (q_next - self.alpha * next_log_prob)

        q1_pred = self.q1(obs, action)
        q2_pred = self.q2(obs, action)

        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        if self.cfg.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q1.parameters(), self.cfg.max_grad_norm)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        if self.cfg.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q2.parameters(), self.cfg.max_grad_norm)
        self.q2_opt.step()

        self._update_step += 1

        # ---------------- Actor + α update (delayed) ----------------
        actor_loss_val = float("nan")
        alpha_loss_val = float("nan")
        alpha_val = float(self.alpha.item())

        if self._update_step % self.cfg.policy_delay == 0:
            # Actor
            new_action, log_prob, _, _, _ = self.actor.sample(obs)
            q1_new = self.q1(obs, new_action)
            q2_new = self.q2(obs, new_action)
            q_new = torch.min(q1_new, q2_new)

            actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            if self.cfg.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.actor_opt.step()

            # Temperature (α)
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # Update cached alpha
            self.alpha = self.log_alpha.exp()

            actor_loss_val = float(actor_loss.item())
            alpha_loss_val = float(alpha_loss.item())
            alpha_val = float(self.alpha.item())

        # ---------------- Soft target updates (every step) ----------------
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": actor_loss_val,
            "alpha_loss": alpha_loss_val,
            "alpha": alpha_val,
            "target_q_mean": float(target_q.mean().item()),
            "q1_mean": float(q1_pred.mean().item()),
            "log_prob_mean": float(next_log_prob.mean().item()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = self.cfg.tau
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    # ---------------- Persistence ----------------

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "update_step": self._update_step,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.alpha = self.log_alpha.exp()
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.q1_opt.load_state_dict(ckpt["q1_opt"])
        self.q2_opt.load_state_dict(ckpt["q2_opt"])
        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        self._update_step = int(ckpt.get("update_step", 0))