"""
Flow:

    Environment
    ActorCritic
    PPO
    Train
"""

import numpy as np
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from reset_policy.rl.actor_critic import ActorCritic
from reset_policy.rl.rollout_buffer import RolloutBuffer
from reset_policy.rl.ppo import PPO

from pathlib import Path

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def train(
    env,
    episodes=1000,
    save_every=50,
):

    actor_critic = ActorCritic().to(DEVICE)

    rollout_buffer = RolloutBuffer()

    ppo = PPO(actor_critic)

    print(f"Training on {DEVICE}")

    for episode in range(episodes):

        state, _ = env.reset()
        print(state)

        episode_reward = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=DEVICE,
            )

            with torch.no_grad():

                action, log_prob, value = actor_critic.act(
                    state_tensor
                )

            action_np = (
                action.cpu()
                .numpy()
                .astype(np.float32)
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action_np)

            rollout_buffer.add(
                state=state_tensor,
                action=action.detach(),
                reward=reward,
                done=(terminated or truncated),
                value=value.detach(),
                log_prob=log_prob.detach(),
            )

            state = next_state

            episode_reward += reward

        losses = ppo.update(
            rollout_buffer
        )

        print(
            f"Episode {episode+1:4d} | "
            f"Reward: {episode_reward:8.3f} | "
            f"Coverage: {info['coverage']:.2f} | "
            f"Reason: {info['termination_reason']}"
        )

        print(
            f"Actor: {losses['actor_loss']:.4f}   "
            f"Critic: {losses['critic_loss']:.4f}   "
            f"Entropy: {losses['entropy']:.4f}"
        )

        if (episode + 1) % save_every == 0:

            filename = (
                f"checkpoint_dir/ppo_checkpoint_{episode+1}.pth"
            )

            if (episode + 1) % save_every == 0:

                filename = checkpoint_dir / f"ppo_checkpoint_{episode+1}.pth"

                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": actor_critic.state_dict(),
                        "optimizer_state_dict": ppo.optimizer.state_dict(),
                    },
                    filename,
                )

                print(
                    f"Saved checkpoint: {filename}"
                )

    final_filename = checkpoint_dir / "ppo_final.pth"

    torch.save(
        {
            "episode": episode,
            "model_state_dict": actor_critic.state_dict(),
            "optimizer_state_dict": ppo.optimizer.state_dict(),
        },
        final_filename,
    )

    print(f"Saved final checkpoint: {final_filename}")

    print("Training complete.")

    return actor_critic