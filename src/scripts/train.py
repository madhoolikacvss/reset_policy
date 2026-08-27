"""
Training loop for PPO on Reset Policy environment.
"""

import numpy as np
import torch
from pathlib import Path
import sys
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)
from reset_policy.rl.actor_critic import ActorCritic
from reset_policy.rl.rollout_buffer import RolloutBuffer
from reset_policy.rl.ppo import PPO
from reset_policy.config import config
from reset_policy.logging.logger import logger


def train(env, config=config):
    """Train PPO policy."""
    
    actor_critic = ActorCritic().to(config.training.device)
    rollout_buffer = RolloutBuffer()
    ppo = PPO(actor_critic)
    
    print(f"Training on {config.training.device}")
    print(f"Logging to: {logger.log_dir}")
    
    steps_since_update = 0
    start_episode = 0
    
    # Resume from checkpoint
    if config.training.resume_from and Path(config.training.resume_from).exists():
        print(f"\nResuming from checkpoint: {config.training.resume_from}")
        checkpoint = torch.load(config.training.resume_from, map_location=config.training.device)
        actor_critic.load_state_dict(checkpoint['model_state_dict'])
        ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint.get('episode', 0) + 1
        steps_since_update = checkpoint.get('steps_since_update', 0)
        print(f"Resuming from episode {start_episode}\n")
    
    for episode in range(start_episode, config.training.episodes):
        # Start episode logging
        logger.start_episode(episode + 1)
        
        state, _ = env.reset()
        episode_reward = 0.0
        terminated = False
        truncated = False
        steps = 0
        info = {}
        
        # Episode stats
        episode_stats = {
            'coverage_reward': 0.0,
            'current_reward': 0.0,
            'current_change_penalty': 0.0,
            'hardware_error_penalty': 0.0,
            'tension_penalty': 0.0,
            'safety_penalty': 0.0,
            'max_current': 0.0,
            'safety_interventions': 0,
            'hardware_error_ids': set(),
        }
        
        # Update renderer for new episode
        if hasattr(env, 'renderer') and env.renderer is not None:
            env.renderer.update_episode(episode + 1)
        
        # Rollout
        while not (terminated or truncated):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=config.training.device)
            
            with torch.no_grad():
                action, log_prob, value = actor_critic.act(state_tensor)
            
            action_np = action.cpu().numpy().astype(np.float32)
            next_state, reward, terminated, truncated, info = env.step(action_np)
            
            # Log motor data for this step
            motor_data = {
                'currents': info.get('motor_currents', [None]*4),
                'voltages': info.get('motor_voltages', [None]*4),
                'temperatures': info.get('motor_temperatures', [None]*4),
            }
            logger.log_step(
                step_num=steps + 1,
                action_count=env.executor.action_count if hasattr(env, 'executor') else 0,
                motor_data=motor_data,
            )
            
            rollout_buffer.add(
                state=state_tensor,
                action=action.detach(),
                reward=reward,
                done=(terminated or truncated),
                value=value.detach(),
                log_prob=log_prob.detach(),
                bootstrap_value=None,
            )
            
            episode_reward += float(reward)
            steps += 1
            steps_since_update += 1
            
            # Update stats
            episode_stats['coverage_reward'] += float(info.get("coverage_reward", 0.0))
            episode_stats['current_reward'] += float(info.get("current_reward", 0.0))
            episode_stats['current_change_penalty'] += float(info.get("current_change_penalty", 0.0))
            episode_stats['hardware_error_penalty'] += float(info.get("hardware_error_penalty", 0.0))
            episode_stats['tension_penalty'] += float(info.get("tension_penalty", 0.0))
            episode_stats['safety_penalty'] += float(info.get("safety_penalty", 0.0))
            episode_stats['max_current'] = max(episode_stats['max_current'], float(info.get("max_current", 0.0)))
            
            if info.get("action_modified", False):
                episode_stats['safety_interventions'] += 1
            
            if info.get("hardware_error", False):
                for motor_id in info.get("hardware_error_ids", []):
                    episode_stats['hardware_error_ids'].add(int(motor_id))
            
            state = next_state
        
        # After episode ends, handle bootstrapping for truncated episodes
        if truncated and not terminated:
            with torch.no_grad():
                _, _, last_value = actor_critic.act(state_tensor)
            rollout_buffer.update_last_bootstrap_value(last_value)
        
        # Log episode summary
        logger.log_episode(
            episode_num=episode + 1,
            episode_reward=episode_reward,
            info=info,
            steps=steps,
            terminated=terminated,
            truncated=truncated,
            episode_stats=episode_stats,
        )
        
        # Close motor log for this episode
        logger.close_motor_log()
        
        # PPO update
        buffer_size = len(rollout_buffer)
        if (steps_since_update >= config.training.min_steps_before_update or 
            buffer_size > config.training.max_buffer_size):
            losses = ppo.update(rollout_buffer)
            steps_since_update = 0
            
            # Log training metrics
            logger.log_training_metrics(episode + 1, losses, buffer_size)
            
            print(f"Episode {episode + 1:4d} | UPDATE | Steps: {steps:3d} | "
                  f"Buffer: {buffer_size} | Actor: {losses['actor_loss']:.4f} | "
                  f"Critic: {losses['critic_loss']:.4f} | Entropy: {losses['entropy']:.4f}")
        else:
            print(f"Episode {episode + 1:4d} | SKIP UPDATE | Steps: {steps:3d} | "
                  f"Buffer: {buffer_size} | Reason: {info.get('termination_reason', 'unknown')}")
        
        # Console output
        print(f"Episode {episode + 1:4d} | Reward: {episode_reward:8.3f} | "
              f"Coverage: {info.get('coverage', 0.0):.2f} | Steps: {steps:3d} | "
              f"Max current: {episode_stats['max_current']:.1f}mA | "
              f"Safety: {episode_stats['safety_interventions']} | "
              f"Reason: {info.get('termination_reason', 'unknown')}")
        
        # Save render
        if hasattr(env, 'renderer') and env.renderer is not None:
            env.renderer.save_final_render(
                env.grid.as_numpy(),
                env.grid.coverage(),
            )
        
        # Checkpoint
        if (episode + 1) % config.training.save_every == 0:
            checkpoint_path = config.checkpoint_dir / f"ppo_checkpoint_{episode + 1}.pth"
            torch.save({
                "episode": episode,
                "model_state_dict": actor_critic.state_dict(),
                "optimizer_state_dict": ppo.optimizer.state_dict(),
                "steps_since_update": steps_since_update,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Final checkpoint
    final_path = config.checkpoint_dir / "ppo_final.pth"
    torch.save({
        "episode": config.training.episodes - 1,
        "model_state_dict": actor_critic.state_dict(),
        "optimizer_state_dict": ppo.optimizer.state_dict(),
        "steps_since_update": steps_since_update,
    }, final_path)
    print(f"Saved final checkpoint: {final_path}")
    print("Training complete.")
    
    # Close logger
    logger.close()
    
    return actor_critic