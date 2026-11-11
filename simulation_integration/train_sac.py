"""Online SAC training with the original network/update and CSV logger."""
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from config import ROOT, SIMULATOR
from environment import ResetPolicyEnv
from rl.sac import SACAgent, SACConfig
from rl.replay_buffer import ReplayBuffer
from rl.her import HERBuffer
from sac_logger import SACLogger

def train(cfg, run_name):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)
    run = ROOT / 'runs' / run_name
    run.mkdir(parents=True, exist_ok=False)
    ckpts = run / 'checkpoints' / 'sac'
    ckpts.mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps(asdict(cfg), indent=2))
    source_params = SIMULATOR / 'params.yaml'
    source_sim_core = SIMULATOR / 'sim_core.py'
    params_bytes = source_params.read_bytes()
    (run / 'params.yaml').write_bytes(params_bytes)
    provenance = {
        'simulator': 'v4_noelastic_unchanged',
        'simulator_directory': str(SIMULATOR.resolve()),
        'source_params': str(source_params.resolve()),
        'params_sha256': hashlib.sha256(params_bytes).hexdigest(),
        'sim_core_sha256': hashlib.sha256(source_sim_core.read_bytes()).hexdigest(),
    }
    (run / 'simulator_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance), flush=True)
    env = ResetPolicyEnv(cfg, run / 'params.yaml')
    agent = SACAgent(SACConfig(obs_dim=4, action_dim=4), torch.device(cfg.device))
    base = ReplayBuffer(state_dim=4)
    her = HERBuffer(base, cfg)
    logger = SACLogger(run / 'src' / 'logs' / 'sac')
    global_step = 0
    timings = []
    run_start = time.perf_counter()
    try:
        for episode in range(1, cfg.episodes+1):
            start = time.perf_counter()
            logger.start_episode(episode)
            obs, _ = env.reset(options={'episode_num': episode})
            total_reward = distance_sum = sim_time = update_time = 0.
            losses = []
            for step in range(1, cfg.max_steps+1):
                global_step += 1
                action = (np.random.uniform(-1, 1, 4).astype(np.float32) if global_step <= cfg.warmup_steps
                          else agent.select_action(obs))
                tick = time.perf_counter()
                next_obs, reward, terminated, truncated, info = env.step(action)
                sim_time += time.perf_counter()-tick
                # Preserve exact legacy CSV columns; unavailable hardware observations
                # are blank/NaN, never policy inputs. Yaw is fixed at zero in this model.
                legacy_obs = np.full(20, np.nan)
                legacy_obs[:2], legacy_obs[2], legacy_obs[3:5] = obs[:2], 0., obs[2:4]
                logger.log_step(step, global_step, legacy_obs, action, info)
                her.add(state=obs, action=action, executed_action=action, reward=reward,
                        next_state=next_obs, done=float(terminated),
                        achieved_goal=info['cube_position'], desired_goal=info['goal_position'])
                total_reward += reward
                distance_sum += info['distance_reward']
                obs = next_obs
                if global_step > cfg.warmup_steps and len(her) >= cfg.batch_size:
                    tick = time.perf_counter()
                    metrics = agent.update(her.sample(cfg.batch_size, agent.device))
                    update_time += time.perf_counter()-tick
                    if not all(np.isfinite(metrics[k]) for k in ('q1_loss', 'q2_loss', 'alpha', 'target_q_mean')):
                        raise FloatingPointError(f'Nonfinite SAC update: {metrics}')
                    losses.append(metrics)
                    if global_step % 50 == 0:
                        logger.log_metric({'global_step': global_step, **metrics})
                if terminated or truncated:
                    break
            relabeled = her.finalize_episode()
            # Save the same model/optimizer payload as the original training script.
            agent_path = ckpts / f'sac_ckpt_{episode}.pt'
            agent.save(str(agent_path))
            payload = torch.load(agent_path, weights_only=False)
            torch.save({'agent': payload, 'global_step': global_step, 'episode': episode-1,
                        'sac_config': asdict(agent.cfg), 'training_config': asdict(cfg)}, agent_path)
            base.save(ckpts / f'buffer_ckpt_{episode}.npz')
            elapsed = time.perf_counter()-start
            def mean_loss(key):
                values = [x[key] for x in losses if np.isfinite(x[key])]
                return float(np.mean(values)) if values else float('nan')
            logger.log_episode(dict(episode=episode, steps=step, total_reward=total_reward,
                termination_reason=info['termination_reason'], terminated=terminated, truncated=truncated,
                goal_x=env.goal_position[0], goal_y=env.goal_position[1],
                final_distance_to_goal=info['distance_to_goal'], goal_reached=terminated,
                buffer_size=len(her), alpha=agent.alpha.item(),
                q1_loss_mean=mean_loss('q1_loss'), q2_loss_mean=mean_loss('q2_loss'),
                actor_loss_mean=mean_loss('actor_loss'), safety_interventions=0, safety_penalty_sum=0.,
                distance_reward_sum=distance_sum, hardware_error=False, hardware_error_ids='',
                max_current=float('nan'), wall_time_s=elapsed))
            logger.close_motor_log()
            timing = dict(episode=episode, steps=step, wall_time_s=elapsed, physics_time_s=sim_time,
                          update_time_s=update_time, her_added=relabeled,
                          off_board_rows=env.sim.off_board_rows, success=terminated,
                          final_distance_m=info['distance_to_goal'])
            timings.append(timing)
            with (run / 'timings.jsonl').open('a') as f:
                f.write(json.dumps(timing)+'\n')
            print(f'Episode {episode:02}/{cfg.episodes}: steps={step}, return={total_reward:.2f}, '
                  f'distance={info["distance_to_goal"]:.4f}m, success={terminated}, '
                  f'time={elapsed:.2f}s, updates={agent._update_step}', flush=True)
        import shutil
        shutil.copyfile(agent_path, ckpts / 'sac_final.pt')
        shutil.copyfile(ckpts / f'buffer_ckpt_{cfg.episodes}.npz', ckpts / 'buffer_final.npz')
        summary = dict(episodes=len(timings), steps=global_step, updates=agent._update_step,
                       wall_time_s=time.perf_counter()-run_start,
                       successes=sum(t['success'] for t in timings),
                       off_board_rows=sum(t['off_board_rows'] for t in timings), buffer_size=len(her))
        (run / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        logger.close()
    return run
