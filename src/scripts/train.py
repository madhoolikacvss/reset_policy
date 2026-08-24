
"""
Flow:

    Environment
    ActorCritic
    PPO
    Train

Training diagnostics:

    training_logs/
        episodes.csv
        occupancy/
            episode_0001.npy
            episode_0002.npy
            ...
"""

import csv
import numpy as np
import torch
import sys

from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from reset_policy.rl.actor_critic import ActorCritic
from reset_policy.rl.rollout_buffer import RolloutBuffer
from reset_policy.rl.ppo import PPO


# ============================================================
# Paths
# ============================================================

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

training_log_dir = Path("training_logs")
occupancy_dir = training_log_dir / "occupancy"

training_log_dir.mkdir(exist_ok=True)
occupancy_dir.mkdir(exist_ok=True)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Episode CSV
# ============================================================

LOG_FIELDS = [
    "episode",
    "episode_reward",
    "coverage",
    "steps",
    "termination_reason",
    "terminated",
    "truncated",

    # Reward components
    "coverage_reward_sum",
    "current_reward_sum",
    "current_change_penalty_sum",
    "hardware_error_penalty_sum",

    # Current diagnostics
    "max_current",

    # Hardware diagnostics
    "hardware_error",
    "hardware_error_ids",
]


def initialize_training_log():

    log_file = (
        training_log_dir / "episodes.csv"
    )

    if not log_file.exists():

        with open(
            log_file,
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=LOG_FIELDS,
            )

            writer.writeheader()

    return log_file


def append_episode_log(
    log_file,
    episode,
    episode_reward,
    info,
    steps,
    terminated,
    truncated,
    coverage_reward_sum,
    current_reward_sum,
    current_change_penalty_sum,
    hardware_error_penalty_sum,
    max_current,
    hardware_error,
    hardware_error_ids,
):

    row = {

        "episode":
            episode,

        "episode_reward":
            episode_reward,

        "coverage":
            info.get("coverage", np.nan),

        "steps":
            steps,

        "termination_reason":
            info.get(
                "termination_reason",
                "",
            ),

        "terminated":
            terminated,

        "truncated":
            truncated,

        "coverage_reward_sum":
            coverage_reward_sum,

        "current_reward_sum":
            current_reward_sum,

        "current_change_penalty_sum":
            current_change_penalty_sum,

        "hardware_error_penalty_sum":
            hardware_error_penalty_sum,

        "max_current":
            max_current,

        "hardware_error":
            hardware_error,

        "hardware_error_ids":
            ",".join(
                map(
                    str,
                    hardware_error_ids,
                )
            )
            if hardware_error_ids
            else "",
    }

    with open(
        log_file,
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=LOG_FIELDS,
        )

        writer.writerow(row)


# ============================================================
# Training
# ============================================================

def train(
    env,
    episodes=1000,
    save_every=50,
    resume_from=None,
):

    steps_since_update = 0
    MIN_STEPS_BEFORE_UPDATE = 50  # Minimum steps before PPO update

    actor_critic = ActorCritic().to(DEVICE)
    rollout_buffer = RolloutBuffer()
    ppo = PPO(actor_critic)

    log_file = initialize_training_log()

    print(f"Training on {DEVICE}")
    print(f"Training logs: {training_log_dir.resolve()}")

    # Track total steps for updating
    total_steps = 0

    if resume_from is not None and Path(resume_from).exists():
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT: {resume_from}")
        print(f"{'='*60}")
        
        checkpoint = torch.load(resume_from, map_location=DEVICE)
        
        # Load model weights
        actor_critic.load_state_dict(checkpoint['model_state_dict'])
        ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Get starting episode
        start_episode = checkpoint.get('episode', 0) + 1
        steps_since_update = checkpoint.get('steps_since_update', 0)
        
        print(f"  Resuming from episode {start_episode}")
        print(f"  Steps since last update: {steps_since_update}")
        print(f"{'='*60}\n")
    else:
        if resume_from is not None:
            print(f"WARNING: Checkpoint not found: {resume_from}")
        print("Starting fresh training...")

    # ========================================================
    # Episodes
    # ========================================================

    for episode in range(episodes):
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0.0
        terminated = False
        truncated = False

        # Episode diagnostics
        coverage_reward_sum = 0.0
        current_reward_sum = 0.0
        current_change_penalty_sum = 0.0
        hardware_error_penalty_sum = 0.0
        max_current_seen = 0.0
        hardware_error_seen = False
        hardware_error_ids_seen = set()
        steps = 0
        info = {}
        
        # Safety intervention tracking for this episode
        safety_interventions_this_episode = 0

        # ====================================================
        # Rollout
        # ====================================================

        while not (terminated or truncated):
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=DEVICE,
            )

            with torch.no_grad():
                action, log_prob, value = actor_critic.act(state_tensor)

            action_np = action.cpu().numpy().astype(np.float32)

            next_state, reward, terminated, truncated, info = env.step(action_np)

            # ------------------------------------------------
            # PPO buffer
            # ------------------------------------------------

            rollout_buffer.add(
                state=state_tensor,
                action=action.detach(),
                reward=reward,
                done=(terminated or truncated),
                value=value.detach(),
                log_prob=log_prob.detach(),
            )

            # ------------------------------------------------
            # Episode diagnostics
            # ------------------------------------------------

            episode_reward += float(reward)
            steps += 1
            total_steps += 1
            steps_since_update += 1

            coverage_reward_sum += float(info.get("coverage_reward", 0.0))
            current_reward_sum += float(info.get("current_reward", 0.0))
            current_change_penalty_sum += float(info.get("current_change_penalty", 0.0))
            hardware_error_penalty_sum += float(info.get("hardware_error_penalty", 0.0))
            max_current_seen = max(max_current_seen, float(info.get("max_current", 0.0)))
            
            # Track safety interventions
            if info.get("action_modified", False):
                safety_interventions_this_episode += 1

            # Hardware errors
            if info.get("hardware_error", False):
                hardware_error_seen = True
                for motor_id in info.get("hardware_error_ids", []):
                    hardware_error_ids_seen.add(int(motor_id))

            state = next_state

        # ====================================================
        # Episode ended - decide whether to update
        # ====================================================

        # Save occupancy grid
        occupancy = np.asarray(env.grid.as_numpy()).copy()
        occupancy_file = occupancy_dir / f"episode_{episode + 1:04d}.npy"
        np.save(occupancy_file, occupancy)

        # Save episode CSV
        append_episode_log(
            log_file=log_file,
            episode=episode + 1,
            episode_reward=episode_reward,
            info=info,
            steps=steps,
            terminated=terminated,
            truncated=truncated,
            coverage_reward_sum=coverage_reward_sum,
            current_reward_sum=current_reward_sum,
            current_change_penalty_sum=current_change_penalty_sum,
            hardware_error_penalty_sum=hardware_error_penalty_sum,
            max_current=max_current_seen,
            hardware_error=hardware_error_seen,
            hardware_error_ids=sorted(hardware_error_ids_seen),
        )

        # ====================================================
        # PPO Update (if we have enough steps)
        # ====================================================

        # IMPORTANT: Reset steps_since_update when we update
        if steps_since_update >= MIN_STEPS_BEFORE_UPDATE:
            losses = ppo.update(rollout_buffer)
            steps_since_update = 0
            
            # Print update info
            print(
                f"Episode {episode + 1:4d} | "
                f"UPDATE | "
                f"Steps: {steps:3d} | "
                f"Buffer: {len(rollout_buffer)} | "
                f"Actor: {losses['actor_loss']:.4f} | "
                f"Critic: {losses['critic_loss']:.4f} | "
                f"Entropy: {losses['entropy']:.4f}"
            )
        else:
            # Keep buffer for next episode
            # If the buffer is getting too large, warn
            buffer_size = len(rollout_buffer.states)
            if buffer_size > 500:
                print(f"WARNING: Buffer size {buffer_size} without update!")
                # Force update if buffer is too large
                losses = ppo.update(rollout_buffer)
                steps_since_update = 0
            else:
                losses = None  # No update happened
                print(
                    f"Episode {episode + 1:4d} | "
                    f"SKIP UPDATE | "
                    f"Steps: {steps:3d} | "
                    f"Buffer: {len(rollout_buffer.states)} | "
                    f"Reason: {info.get('termination_reason', 'unknown')}"
                )

        # ====================================================
        # Safety Filter Stats
        # ====================================================

        if hasattr(env, 'safety_filter') and env.safety_filter is not None:
            stats = env.safety_filter.get_stats()
            if safety_interventions_this_episode > 0:
                print(
                    f"  Safety interventions: {safety_interventions_this_episode} "
                    f"(total: {stats['total_interventions']})"
                )
                if stats.get('intervention_reasons'):
                    reasons = stats['intervention_reasons']
                    print(f"  Reasons: {reasons}")

        # ====================================================
        # Console output (basic episode summary)
        # ====================================================

        print(
            f"Episode {episode + 1:4d} | "
            f"Reward: {episode_reward:8.3f} | "
            f"Coverage: {info.get('coverage', 0.0):.2f} | "
            f"Steps: {steps:3d} | "
            f"Max current: {max_current_seen:.1f}mA | "
            f"Reason: {info.get('termination_reason', 'unknown')}"
        )

        # ====================================================
        # Checkpoint
        # ====================================================

        if (episode + 1) % save_every == 0:
            filename = checkpoint_dir / f"ppo_checkpoint_{episode + 1}.pth"
            torch.save(
                {
                    "episode": episode,
                    "model_state_dict": actor_critic.state_dict(),
                    "optimizer_state_dict": ppo.optimizer.state_dict(),
                    "steps_since_update": steps_since_update,
                },
                filename,
            )
            print(f"Saved checkpoint: {filename}")

        if (episode + 1) % save_every == 0 or episode == 0:
            # Save final state of the episode
            if hasattr(env, 'renderer') and env.renderer is not None:
                # Get final occupancy grid and coverage
                occupancy = env.grid.as_numpy()
                coverage = env.grid.coverage()
                
                # Save final render
                env.renderer.save_final_render(occupancy, coverage)
                print(f"Saved final render for episode {episode + 1}")

    # ========================================================
    # Final checkpoint
    # ========================================================

    final_filename = checkpoint_dir / "ppo_final.pth"
    torch.save(
        {
            "episode": episode,
            "model_state_dict": actor_critic.state_dict(),
            "optimizer_state_dict": ppo.optimizer.state_dict(),
            "steps_since_update": steps_since_update,
        },
        final_filename,
    )
    print(f"Saved final checkpoint: {final_filename}")
    print("Training complete.")


    return actor_critic