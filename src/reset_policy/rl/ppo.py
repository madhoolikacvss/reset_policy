import torch
import torch.nn.functional as F
import torch.optim as optim


class PPO:

    def __init__(
        self,
        actor_critic,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        epochs=10,
    ):

        self.actor_critic = actor_critic

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.clip_epsilon = clip_epsilon

        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.epochs = epochs

        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=learning_rate,
        )
        
        # Get the device from the actor_critic
        self.device = next(self.actor_critic.parameters()).device

    # Compute Returns + Advantages (GAE)
    def compute_returns_and_advantages(self, rewards, dones, values):
        # Ensure all tensors are on the same device
        device = values.device
        
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        next_value = torch.tensor(0.0, device=device)  # Move next_value to same device
        
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = (rewards[t] + self.gamma * next_value * mask - values[t])
            gae = (delta + self.gamma * self.gae_lambda * mask * gae)
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values

        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)
        print(torch.isnan(advantages).any())
        print(torch.min(advantages))
        print(torch.max(advantages))

        return returns, advantages

    # PPO Update
    def update(self, rollout_buffer):
        (
            states,
            actions,
            rewards,
            dones,
            values,
            old_log_probs,
        ) = rollout_buffer.get()

        # Move all tensors from rollout_buffer to the same device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        values = values.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        returns, advantages = (
            self.compute_returns_and_advantages(
                rewards,
                dones,
                values,
            )
        )
        print("returns")
        print(torch.min(returns))
        print(torch.max(returns))
        print(torch.isnan(returns).any())

        print("values")
        print(torch.min(values))
        print(torch.max(values))
        print(torch.isnan(values).any())

        for _ in range(self.epochs):
            (new_log_probs, entropy, predicted_values) = self.actor_critic.evaluate(
                states, 
                actions,
            )

            print("predicted_values")
            print(predicted_values)

            print("returns")
            print(returns)

            print("critic loss")
            print(F.mse_loss(predicted_values, returns))

            ratios = torch.exp(new_log_probs - old_log_probs)

            surrogate1 = ratios * advantages

            surrogate2 = torch.clamp(
                ratios,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon,
            ) * advantages

            actor_loss = -torch.min(
                surrogate1,
                surrogate2,
            ).mean()

            critic_loss = F.mse_loss(predicted_values, returns)
            entropy_bonus = entropy.mean()
            loss = (actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_bonus)
            
            self.optimizer.zero_grad()
            loss.backward()
            for name, p in self.actor_critic.named_parameters():
                if p.grad is None:
                    continue

                print(
                    name,
                    "grad max:",
                    p.grad.abs().max().item(),
                    "nan:",
                    torch.isnan(p.grad).any().item(),
                    "inf:",
                    torch.isinf(p.grad).any().item(),
                )

            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                0.5,
            )
            
            self.optimizer.step()

        rollout_buffer.clear()

        return {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_bonus.item(),
        }