"""
SAC + HER training loop for the cube-string reset policy.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from reset_policy.config import config
from reset_policy.rl.replay_buffer import ReplayBuffer
from reset_policy.rl.her import HERBuffer
from reset_policy.rl.sac import SACAgent, SACConfig
from reset_policy.logging.sac_logger import SACLogger
from reset_policy.logging.video_recorder import (
    EpisodeVideoRecorder,
    RECORD_VIDEO,
)


# ---------------- Goal normalization helper for HER ----------------

def _normalize_goal(goal_physical) -> np.ndarray:
    """Physical goal (meters) -> normalized goal (for HER observation relabeling)."""
    X_MIN = config.environment.x_min
    X_MAX = config.environment.x_max
    Y_MIN = config.environment.y_min
    Y_MAX = config.environment.y_max
    gx, gy = float(goal_physical[0]), float(goal_physical[1])
    return np.array([
        (gx - X_MIN) / (X_MAX - X_MIN),
        (gy - Y_MIN) / (Y_MAX - Y_MIN),
    ], dtype=np.float32)


# ---------------- Build helpers ----------------

def _build_agent(device: torch.device) -> SACAgent:
    """Build SAC agent with the same hyperparameters as the offline script."""
    cfg = SACConfig(
        obs_dim=20, action_dim=4,
        hidden_dim=256, n_layers=2,
        lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
        gamma=0.99, tau=0.005, policy_delay=2,
        target_entropy=-4.0,
        max_grad_norm=1.0,
    )
    return SACAgent(cfg, device=device)


def _build_her_buffer() -> HERBuffer:
    """Build the HER-wrapped replay buffer, warm-starting if configured."""
    base = ReplayBuffer(state_dim=20, action_dim=4, goal_dim=2)

    buffer_init = getattr(config.training, "buffer_init", None)
    if buffer_init is not None and Path(buffer_init).exists():
        print(f"Warm-starting buffer from {buffer_init}")
        base = ReplayBuffer.load(buffer_init)
        print(f"Loaded {len(base)} transitions")

    her_k = getattr(config.training, "her_k", 4)

    return HERBuffer(
        base_buffer=base,
        goal_dims=slice(3, 5),
        k=her_k,
        strategy="future",
        normalize_goal=_normalize_goal,
    )


def _resume_agent(agent: SACAgent, resume_from: str, device: torch.device) -> int:
    """Load agent weights from a checkpoint. Returns start_global_step."""
    ckpt = torch.load(resume_from, map_location=device)

    agent.actor.load_state_dict(ckpt["agent"]["actor"])
    agent.q1.load_state_dict(ckpt["agent"]["q1"])
    agent.q2.load_state_dict(ckpt["agent"]["q2"])
    agent.q1_target.load_state_dict(ckpt["agent"]["q1_target"])
    agent.q2_target.load_state_dict(ckpt["agent"]["q2_target"])
    agent.log_alpha.data.copy_(ckpt["agent"]["log_alpha"].to(device))
    agent.alpha = agent.log_alpha.exp()
    agent.actor_opt.load_state_dict(ckpt["agent"]["actor_opt"])
    agent.q1_opt.load_state_dict(ckpt["agent"]["q1_opt"])
    agent.q2_opt.load_state_dict(ckpt["agent"]["q2_opt"])
    agent.alpha_opt.load_state_dict(ckpt["agent"]["alpha_opt"])
    agent._update_step = int(ckpt["agent"].get("update_step", 0))

    return int(ckpt.get("global_step", 0)), int(ckpt.get("episode", -1)) + 1


# ---------------- Main training loop ----------------

def train(env, config=config):
    """Train SAC + HER policy. Mirrors PPO's train(env, config)."""

    # --- Device ---
    device = torch.device(config.training.device)
    print(f"Training on {device}")

    # --- Log / checkpoint directories ---
    logs_root = Path("/home/madhoolika/workspace/reset_policy/src/logs")
    log_dir = logs_root / "sac"
    ckpt_dir = Path(config.checkpoint_dir) / "sac"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to: {log_dir}")
    print(f"Checkpoints: {ckpt_dir}")

    # --- Build agent ---
    agent = _build_agent(device)

    # --- Resume agent if requested ---
    start_episode = 0
    start_global_step = 0

    resume_from = getattr(config.training, "resume_from", None)
    if resume_from is not None and Path(resume_from).exists():
        print(f"\nResuming from checkpoint: {resume_from}")
        start_global_step, start_episode = _resume_agent(agent, resume_from, device)
        print(f"  episode={start_episode}, global_step={start_global_step}")

    # --- Build buffer ---
    her_buffer = _build_her_buffer()

    # --- Build logger ---
    logger = SACLogger(log_dir)

    # --- Training hyperparameters ---
    total_episodes = config.training.episodes
    batch_size = 256
    warmup_steps = 5000
    updates_per_step = 1
    log_every = 10
    save_every = config.training.save_every
    render_every = config.training.render_every

    # --- Video recorder (same as PPO) ---
    video_recorder = None
    if RECORD_VIDEO:
        try:
            video_recorder = EpisodeVideoRecorder()
            print(f"[VIDEO] Recorder initialized (serial={video_recorder.serial})")
        except Exception as e:
            print(f"[VIDEO] WARNING: Could not initialize video recorder: {e}")
            video_recorder = None

    # --- Running history ---
    global_step = start_global_step
    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    ep_safety_history: list[int] = []
    q1_losses: list[float] = []
    q2_losses: list[float] = []
    actor_losses: list[float] = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for episode in range(start_episode, start_episode + total_episodes):
        ep_start = time.time()

        # Per-episode accumulators
        ep_reward = 0.0
        ep_steps = 0
        ep_safety_count = 0
        ep_safety_penalty = 0.0
        ep_safety_by_reason: dict = {}
        ep_distance_reward = 0.0
        ep_current_change_penalty = 0.0
        ep_hardware_error_penalty = 0.0
        ep_tension_penalty = 0.0
        ep_max_current = 0.0
        ep_hardware_error_ids: set = set()
        hardware_error = False
        termination_reason = "unknown"
        step_info: dict = {}
        terminated = False
        truncated = False

        # --- Start episode logging ---
        logger.start_episode(episode + 1)
        should_record = (
            video_recorder is not None
            and (episode + 1) % save_every == 0
        )
        if should_record:
            video_recorder.start_episode(episode + 1)

        # --- Update renderer for the new episode ---
        if hasattr(env, "renderer") and env.renderer is not None:
            env.renderer.update_episode(episode + 1)

        # --- Reset env ---
        try:
            obs, reset_info = env.reset(options={"episode_num": episode + 1})
        except RuntimeError as e:
            print(f"\n!! env.reset failed: {e}")
            print("!! sleeping 10s, then retrying")
            time.sleep(10.0)
            logger.close_motor_log()
            continue

        goal_xy = np.asarray(
            reset_info.get("goal_position", env.goal_position),
            dtype=np.float32,
        )

        # --- Rollout ---
        done = False
        while not done:
            ep_steps += 1
            global_step += 1

            obs_before = obs

            # Action selection
            if global_step <= warmup_steps:
                action = np.random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
            else:
                action = agent.select_action(obs.astype(np.float32), deterministic=False)

            # Step env
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)

            # --- Step logging ---
            logger.log_step(
                step_num=ep_steps,
                action_count=env.executor.action_count if hasattr(env, "executor") else 0,
                obs_before=obs_before,
                action_raw=action,
                step_info=step_info,
            )

            # --- Video frame ---
            if should_record:
                video_recorder.capture_frame()

            # --- Add to HER buffer ---
            her_buffer.add(
                state=obs,
                action=action,
                executed_action=step_info["executed_actions"],
                reward=float(reward),
                next_state=next_obs,
                done=float(terminated),
                achieved_goal=np.asarray(step_info["cube_position"], dtype=np.float32),
                desired_goal=np.asarray(step_info["goal_position"], dtype=np.float32),
                distance_reward=float(step_info.get("distance_reward", 0.0)),
                current_change_penalty=float(step_info.get("current_change_penalty", 0.0)),
                hardware_error_penalty=float(step_info.get("hardware_error_penalty", 0.0)),
                tension_penalty=float(step_info.get("tension_penalty", 0.0)),
                safety_filtering_penalty=float(step_info.get("safety_filtering_penalty", 0.0)),
            )

            # --- Accumulate episode stats ---
            ep_reward += float(reward)
            ep_distance_reward += float(step_info.get("distance_reward", 0.0))
            ep_current_change_penalty += float(step_info.get("current_change_penalty", 0.0))
            ep_hardware_error_penalty += float(step_info.get("hardware_error_penalty", 0.0))
            ep_tension_penalty += float(step_info.get("tension_penalty", 0.0))
            ep_max_current = max(ep_max_current, float(step_info.get("max_current", 0.0)))

            if step_info.get("action_modified", False):
                ep_safety_count += 1
                p = float(step_info.get("safety_filtering_penalty", 0.0))
                ep_safety_penalty += p
                reason = step_info.get("safety_reason", "unknown")
                ep_safety_by_reason[reason] = ep_safety_by_reason.get(reason, 0.0) + p

            if step_info.get("hardware_error", False):
                hardware_error = True
                for mid in step_info.get("hardware_error_ids", []):
                    ep_hardware_error_ids.add(int(mid))

            obs = next_obs

            # --- SAC update ---
            if global_step > warmup_steps and len(her_buffer) >= batch_size:
                for _ in range(updates_per_step):
                    batch = her_buffer.sample(batch_size, device=device)
                    loss_dict = agent.update(batch)

                    q1_losses.append(loss_dict["q1_loss"])
                    q2_losses.append(loss_dict["q2_loss"])
                    if not np.isnan(loss_dict["actor_loss"]):
                        actor_losses.append(loss_dict["actor_loss"])

                    if global_step % 50 == 0:
                        logger.log_metric({
                            "global_step": global_step,
                            "q1_loss": loss_dict["q1_loss"],
                            "q2_loss": loss_dict["q2_loss"],
                            "actor_loss": loss_dict["actor_loss"],
                            "alpha": loss_dict["alpha"],
                            "target_q_mean": loss_dict["target_q_mean"],
                        })

            # --- Early exit on hardware error ---
            if step_info.get("hardware_error", False):
                termination_reason = "hardware_error"
                break

        # --- Episode finalization ---
        n_relabeled = her_buffer.finalize_episode()

        ep_returns.append(ep_reward)
        ep_lengths.append(ep_steps)
        ep_safety_history.append(ep_safety_count)
        ep_time = time.time() - ep_start

        final_distance = step_info.get("distance_to_goal", None)
        goal_reached = bool(step_info.get("goal_reached", False))

        if should_record:
            video_recorder.end_episode()

        # --- Episode log ---
        logger.log_episode({
            "episode": episode,
            "steps": ep_steps,
            "total_reward": ep_reward,
            "termination_reason": step_info.get("termination_reason", termination_reason),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "goal_x": float(goal_xy[0]),
            "goal_y": float(goal_xy[1]),
            "final_distance_to_goal": float(final_distance) if final_distance is not None else float("nan"),
            "goal_reached": goal_reached,
            "buffer_size": len(her_buffer),
            "alpha": float(agent.alpha.item()),
            "q1_loss_mean": float(np.mean(q1_losses[-100:])) if q1_losses else float("nan"),
            "q2_loss_mean": float(np.mean(q2_losses[-100:])) if q2_losses else float("nan"),
            "actor_loss_mean": float(np.mean(actor_losses[-100:])) if actor_losses else float("nan"),
            "safety_interventions": ep_safety_count,
            "safety_penalty_sum": ep_safety_penalty,
            "safety_penalty_by_reason": ep_safety_by_reason,
            "distance_reward_sum": ep_distance_reward,
            "current_change_penalty_sum": ep_current_change_penalty,
            "hardware_error_penalty_sum": ep_hardware_error_penalty,
            "tension_penalty_sum": ep_tension_penalty,
            "max_current": ep_max_current,
            "hardware_error": hardware_error,
            "hardware_error_ids": ",".join(map(str, sorted(ep_hardware_error_ids))),
            "wall_time_s": ep_time,
        })

        logger.close_motor_log()

        # --- Render every render_every episodes ---
        if (episode + 1) % render_every == 0:
            if hasattr(env, "renderer") and env.renderer is not None:
                if env.last_valid_observation is not None:
                    cube_x_cm = (env.last_valid_observation.cube_x - env.grid.x_min) * 100
                    cube_y_cm = (env.last_valid_observation.cube_y - env.grid.y_min) * 100

                    goals_cm = [
                        ((gx - env.grid.x_min) * 100, (gy - env.grid.y_min) * 100)
                        for gx, gy in config.environment.fixed_goals
                    ]
                    current_goal_idx = episode % len(config.environment.fixed_goals)

                    env.renderer.render(
                        cube_x_cm=cube_x_cm,
                        cube_y_cm=cube_y_cm,
                        occupancy_grid=env.grid.as_numpy(),
                        coverage=env.grid.coverage(),
                        goals=goals_cm,
                        current_goal_idx=current_goal_idx,
                    )

        # --- Console log ---
        if (episode + 1) % log_every == 0 or episode == start_episode:
            recent_ret = float(np.mean(ep_returns[-20:]))
            recent_len = float(np.mean(ep_lengths[-20:]))
            recent_safe = float(np.mean(ep_safety_history[-20:]))
            dist_str = f"{final_distance:.3f}" if final_distance is not None else "  nan"
            print(
                f"ep {episode:5d} | "
                f"ret(last20) {recent_ret:8.2f} | "
                f"len {recent_len:6.1f} | "
                f"safety {recent_safe:5.1f} | "
                f"α {agent.alpha.item():.3f} | "
                f"buffer {len(her_buffer):7d} | "
                f"relabeled {n_relabeled} | "
                f"dist {dist_str} | "
                f"reason {step_info.get('termination_reason', 'unknown')} | "
                f"t {ep_time:.1f}s"
            )

        # --- Checkpoint ---
        if (episode + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"sac_ckpt_{episode + 1}.pt"
            torch.save({
                "agent": {
                    "actor": agent.actor.state_dict(),
                    "q1": agent.q1.state_dict(),
                    "q2": agent.q2.state_dict(),
                    "q1_target": agent.q1_target.state_dict(),
                    "q2_target": agent.q2_target.state_dict(),
                    "log_alpha": agent.log_alpha.detach().cpu(),
                    "actor_opt": agent.actor_opt.state_dict(),
                    "q1_opt": agent.q1_opt.state_dict(),
                    "q2_opt": agent.q2_opt.state_dict(),
                    "alpha_opt": agent.alpha_opt.state_dict(),
                    "update_step": agent._update_step,
                },
                "global_step": global_step,
                "episode": episode,
            }, ckpt_path)

            her_buffer.save(ckpt_dir / f"buffer_ckpt_{episode + 1}.npz")
            print(f"  >> saved checkpoint to {ckpt_path}")

        # --- Hardware error: stop cleanly ---
        if hardware_error:
            print("\nHardware error triggered.")
            print("Cool down motors and restart the training script.")
            break

    # ------------------------------------------------------------------
    # End of training
    # ------------------------------------------------------------------
    if video_recorder is not None:
        video_recorder.close()
        print("[VIDEO] Video recorder closed")

    final_agent_path = ckpt_dir / "sac_final.pt"
    torch.save({
        "agent": {
            "actor": agent.actor.state_dict(),
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
            "q1_target": agent.q1_target.state_dict(),
            "q2_target": agent.q2_target.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
            "actor_opt": agent.actor_opt.state_dict(),
            "q1_opt": agent.q1_opt.state_dict(),
            "q2_opt": agent.q2_opt.state_dict(),
            "alpha_opt": agent.alpha_opt.state_dict(),
            "update_step": agent._update_step,
        },
        "global_step": global_step,
        "episode": episode,
    }, final_agent_path)
    print(f"Saved final checkpoint: {final_agent_path}")

    final_buffer_path = ckpt_dir / "buffer_final.npz"
    her_buffer.save(final_buffer_path)
    print(f"Saved final buffer: {final_buffer_path}")

    logger.close()
    print("Training complete.")

    return agent