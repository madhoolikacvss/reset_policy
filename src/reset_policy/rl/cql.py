"""
Conservative Q-Learning (CQL) for offline RL.
Reference: Kumar et al., "Conservative Q-Learning for Offline Reinforcement
Learning" (2020).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlitb import lib
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)
from sac import SquashedGaussianMLP, QNetwork, LOG_STD_MIN, LOG_STD_MAX


@dataclass
class CQLConfig:
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

    # CQL-specific
    # α_cql controls the strength of the conservative penalty.
    # Typical range: 0.1 – 10.0. Start at 1.0 and tune.
    cql_alpha: float = 1.0
    # Number of action samples from the current policy for the penalty.
    n_action_samples: int = 10
    # Temperature for the "soft-max" over sampled actions (used in the
    # log-sum-exp version of the CQL penalty). Standard: 1.0 for continuous.
    cql_temp: float = 1.0
    # Lagged α_cql: automatically adjust so that the CQL penalty stays in a
    # reasonable range. Optional, disable by setting lagrange=False.
    lagrange: bool = False
    # Target for lagged α_cql (used when lagrange=True).
    lagrange_target: float = 5.0

    # Entropy temp
    target_entropy: float = -4.0

    max_grad_norm: float = 1.0


class CQLAgent:
    """
    Offline CQL agent. Reuses SAC's networks. Trains from a fixed buffer.

    Usage:
        agent = CQLAgent(cfg, device)
        for step in range(num_steps):
            batch = buffer.sample(batch_size, device)
            losses = agent.update(batch)
    """

    def __init__(self, cfg: CQLConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        # Networks (same as SAC)
        self.actor = SquashedGaussianMLP(
            cfg.obs_dim, cfg.action_dim, cfg.hidden_dim, cfg.n_layers
        ).to(device)
        self.q1 = QNetwork(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim, cfg.n_layers).to(device)
        self.q2 = QNetwork(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim, cfg.n_layers).to(device)

        self.q1_target = QNetwork(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim, cfg.n_layers).to(device)
        self.q2_target = QNetwork(cfg.obs_dim, cfg.action_dim, cfg.hidden_dim, cfg.n_layers).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # Temperature
        self.target_entropy = float(cfg.target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()

        # CQL α
        if cfg.lagrange:
            # Log-space for positivity
            self.log_cql_alpha = torch.tensor(
                np.log(cfg.cql_alpha), requires_grad=True, device=device
            )
            self.cql_alpha = self.log_cql_alpha.exp()
            self.cql_alpha_opt = torch.optim.Adam(
                [self.log_cql_alpha], lr=cfg.lr_alpha
            )
        else:
            self.cql_alpha = torch.tensor(cfg.cql_alpha, device=device)

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.lr_critic)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.lr_critic)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)

        self._update_step = 0

    # ---------------- Inference ----------------

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _, mean_action, _, _ = self.actor.sample(obs_t, deterministic=deterministic)
        a = mean_action if deterministic else action
        return a.squeeze(0).cpu().numpy()

    # ---------------- Update ----------------

    def update(self, batch) -> dict:
        obs = batch.states
        action = batch.actions
        reward = batch.rewards
        next_obs = batch.next_states
        done = batch.dones

        # ---------------- Target Q ----------------
        with torch.no_grad():
            next_action, next_log_prob, _, _, _ = self.actor.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_action)
            q2_next = self.q2_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next)
            target_q = reward + self.cfg.gamma * (1.0 - done) * (q_next - self.alpha * next_log_prob)

        # ---------------- Current Q ----------------
        q1_pred = self.q1(obs, action)
        q2_pred = self.q2(obs, action)

        # Bellman loss
        q1_bellman = F.mse_loss(q1_pred, target_q)
        q2_bellman = F.mse_loss(q2_pred, target_q)

        # ---------------- CQL penalty ----------------
        # Sample actions from the current policy
        with torch.no_grad():
            # Shape: (N_samples, batch, action_dim)
            random_actions = torch.stack([
                torch.empty_like(action).uniform_(-1.0, 1.0)
                for _ in range(self.cfg.n_action_samples)
            ], dim=0)
            # Also sample from the current policy
            n_pi = self.cfg.n_action_samples
            pi_actions, pi_log_probs, _, _, _ = self.actor.sample(obs)
            # pi_actions shape: (batch, action_dim)
            # Repeat for n_pi samples
            pi_actions = pi_actions.unsqueeze(0)  # (1, batch, action_dim)
            # Broadcast to n_pi
            pi_actions = pi_actions.expand(n_pi, -1, -1)  # (n_pi, batch, action_dim)

            # Combine random + policy samples
            # Use only policy samples for the standard CQL loss
            # (Standard implementation uses policy samples + random samples)
            all_actions = torch.cat([random_actions, pi_actions], dim=0)  # (2*n_pi, batch, action_dim)
            n_samples = all_actions.shape[0]

            # Expand obs to match
            obs_expanded = obs.unsqueeze(0).expand(n_samples, -1, -1)  # (n_samples, batch, obs_dim)

            # Flatten for Q forward
            obs_flat = obs_expanded.reshape(-1, obs.shape[-1])
            actions_flat = all_actions.reshape(-1, action.shape[-1])

            q1_samples = self.q1(obs_flat, actions_flat).reshape(n_samples, -1)
            q2_samples = self.q2(obs_flat, actions_flat).reshape(n_samples, -1)
            q_samples = torch.min(q1_samples, q2_samples)  # (n_samples, batch)

            # Log-sum-exp over samples (soft-max) for the "push-down" term
            # log(1/N Σ exp(Q/τ)) * τ
            q_logsumexp = torch.logsumexp(
                q_samples / self.cfg.cql_temp, dim=0
            ) * self.cfg.cql_temp  # (batch,)

            # First term: soft-max over sampled actions
            cql_q1_loss = q_logsumexp.mean() - q1_pred.mean()
            cql_q2_loss = q_logsumexp.mean() - q2_pred.mean()

        # Total critic losses
        q1_loss = q1_bellman + self.cql_alpha * cql_q1_loss
        q2_loss = q2_bellman + self.cql_alpha * cql_q2_loss

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

        # ---------------- Actor + α updates (delayed) ----------------
        actor_loss_val = float("nan")
        alpha_loss_val = float("nan")
        cql_alpha_val = float(self.cql_alpha.item()) if not self.cfg.lagrange else float(self.cql_alpha.item())

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

            # Temperature α
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp()

            # Lagged CQL α (optional)
            if self.cfg.lagrange:
                cql_alpha_loss = -(self.log_cql_alpha * (
                    cql_q1_loss.detach() - self.cfg.lagrange_target
                )).mean()
                self.cql_alpha_opt.zero_grad()
                cql_alpha_loss.backward()
                self.cql_alpha_opt.step()
                self.cql_alpha = self.log_cql_alpha.exp()
                cql_alpha_val = float(self.cql_alpha.item())

            actor_loss_val = float(actor_loss.item())
            alpha_loss_val = float(alpha_loss.item())

        # Soft target update
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "q1_bellman": float(q1_bellman.item()),
            "q2_bellman": float(q2_bellman.item()),
            "cql_q1_loss": float(cql_q1_loss.item()),
            "cql_q2_loss": float(cql_q2_loss.item()),
            "actor_loss": actor_loss_val,
            "alpha_loss": alpha_loss_val,
            "alpha": float(self.alpha.item()),
            "cql_alpha": cql_alpha_val,
            "q1_mean": float(q1_pred.mean().item()),
            "q1_max": float(q1_pred.max().item()),
            "q1_min": float(q1_pred.min().item()),
            "target_q_mean": float(target_q.mean().item()),
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
            "cql_alpha": float(self.cql_alpha.item()),
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