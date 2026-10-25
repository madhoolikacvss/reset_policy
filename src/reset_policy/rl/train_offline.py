"""
Offline CQL training from the pre-populated replay buffer.

Usage:
    python -m reset_policy.scripts.train_offline_cql \
        --buffer /path/to/buffer_from_logs.npz \
        --updates 100000 \
        --batch-size 256 \
        --cql-alpha 1.0
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from replay_buffer import ReplayBuffer
from cql import CQLAgent, CQLConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, required=True,
                        help="Path to the pre-populated .npz buffer.")
    parser.add_argument("--updates", type=int, default=100_000,
                        help="Total gradient updates to perform.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cql-alpha", type=float, default=1.0)
    parser.add_argument("--lr-actor", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--target-entropy", type=float, default=-2.0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--log-dir", type=str,
                        default="/home/madhoolika/workspace/reset_policy/src/logs/cql")
    parser.add_argument("--ckpt-dir", type=str,
                        default="/home/madhoolika/workspace/reset_policy/src/logs/cql/checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    log_dir = Path(args.log_dir)
    ckpt_dir = Path(args.ckpt_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Load buffer ---
    print(f"Loading buffer from {args.buffer}")
    buf = ReplayBuffer.load(args.buffer)
    print(f"Buffer size: {len(buf)}")
    print(f"Buffer stats: {buf.stats()}")

    # --- Build agent ---
    cfg = CQLConfig(
        obs_dim=20, action_dim=4,
        hidden_dim=256, n_layers=2,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=0.99, tau=0.005, policy_delay=2,
        cql_alpha=args.cql_alpha,
        n_action_samples=10,
        cql_temp=1.0,
        lagrange=False,
        target_entropy=args.target_entropy,
        max_grad_norm=1.0,
    )
    agent = CQLAgent(cfg, device=device)

    start_step = 0
    if args.resume_from is not None and Path(args.resume_from).exists():
        agent.load(args.resume_from)
        print(f"Resumed agent from {args.resume_from}")

    # --- Metrics log ---
    metrics_file = log_dir / "cql_train_metrics.csv"
    if not metrics_file.exists():
        with open(metrics_file, "w", newline="") as f:
            csv.writer(f).writerow([
                "step", "q1_loss", "q2_loss",
                "q1_bellman", "q2_bellman",
                "cql_q1_loss", "cql_q2_loss",
                "actor_loss", "alpha", "cql_alpha",
                "q1_mean", "q1_max", "q1_min", "target_q_mean",
            ])

    # --- Training loop ---
    print(f"\nStarting CQL training: {args.updates} updates")
    print(f"  batch_size={args.batch_size}, cql_alpha={args.cql_alpha}")
    print(f"  lr_actor={args.lr_actor}, lr_critic={args.lr_critic}")
    print(f"  target_entropy={args.target_entropy}")

    t0 = time.time()
    losses_recent = []

    for step in range(start_step, start_step + args.updates):
        batch = buf.sample(args.batch_size, device=device)
        loss_dict = agent.update(batch)

        losses_recent.append(loss_dict)
        if len(losses_recent) > 100:
            losses_recent.pop(0)

        # Log
        if (step + 1) % args.log_every == 0:
            with open(metrics_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    step + 1,
                    loss_dict["q1_loss"], loss_dict["q2_loss"],
                    loss_dict["q1_bellman"], loss_dict["q2_bellman"],
                    loss_dict["cql_q1_loss"], loss_dict["cql_q2_loss"],
                    loss_dict["actor_loss"],
                    loss_dict["alpha"], loss_dict["cql_alpha"],
                    loss_dict["q1_mean"], loss_dict["q1_max"], loss_dict["q1_min"],
                    loss_dict["target_q_mean"],
                ])

            elapsed = time.time() - t0
            # Rolling averages
            q1_mean = float(np.mean([d["q1_loss"] for d in losses_recent]))
            q2_mean = float(np.mean([d["q2_loss"] for d in losses_recent]))
            print(
                f"step {step+1:7d} | "
                f"q1 {q1_mean:9.4f} | "
                f"q2 {q2_mean:9.4f} | "
                f"α {loss_dict['alpha']:.4f} | "
                f"cql_α {loss_dict['cql_alpha']:.4f} | "
                f"Q̄ {loss_dict['q1_mean']:8.3f} | "
                f"Qmax {loss_dict['q1_max']:8.3f} | "
                f"elapsed {elapsed:.1f}s"
            )

        # Checkpoint
        if (step + 1) % args.ckpt_every == 0:
            ckpt_path = ckpt_dir / f"cql_ckpt_{step+1}.pt"
            agent.save(str(ckpt_path))
            print(f"  >> saved checkpoint to {ckpt_path}")

    # Final save
    agent.save(str(ckpt_dir / "cql_final.pt"))
    print(f"\nDone. Final checkpoint: {ckpt_dir / 'cql_final.pt'}")


if __name__ == "__main__":
    main()