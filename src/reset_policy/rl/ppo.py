"""
PPO (Proximal Policy Optimization) implementation with proper bootstrapping.

Key features:
- GAE (Generalized Advantage Estimation)
- Value clipping for stable training
- Proper bootstrapping for truncated episodes
- Gradient clipping
"""

import torch
import torch.nn.functional as F
import torch.optim as optim


class PPO:
    """PPO algorithm with GAE and bootstrapping."""
    
    def __init__(
        self,
        actor_critic,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        epochs=10,
        value_clip=True,
    ):
        self.actor_critic = actor_critic
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.value_clip = value_clip
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=learning_rate,
        )
        
        # Get device from model
        self.device = next(self.actor_critic.parameters()).device
    
    def compute_returns_and_advantages(
        self, 
        rewards, 
        dones, 
        values, 
        bootstrap_values=None
    ):
        """
        Compute returns and advantages using GAE.
        
        Args:
            rewards: Tensor of rewards [T]
            dones: Tensor of done flags [T] (1 if terminal, 0 if not)
            values: Tensor of value estimates [T]
            bootstrap_values: Tensor of bootstrap values [T] or None
                - For terminal states: bootstrap value should be 0
                - For truncated states: bootstrap value is V(s_next)
                - If None: assumes all episodes end terminally
        
        Returns:
            returns: Tensor of returns [T]
            advantages: Tensor of normalized advantages [T]
        """
        device = values.device
        T = len(rewards)
        
        advantages = torch.zeros_like(rewards, device=device)
        returns = torch.zeros_like(rewards, device=device)
        
        gae = 0.0
        next_value = 0.0  # Default for terminal state
        
        # Process timesteps in reverse order
        for t in reversed(range(T)):
            if t == T - 1:
                # Last timestep - use bootstrap value if available
                if bootstrap_values is not None and not dones[t]:
                    # Episode was truncated - use bootstrap value
                    next_value = bootstrap_values[t]
                else:
                    # Episode terminated - no future value
                    next_value = 0.0
            else:
                # Not last timestep
                if dones[t]:
                    # Terminal state - no future value
                    next_value = 0.0
                else:
                    # Non-terminal - use value of next state
                    next_value = values[t + 1]
            
            # Compute TD error
            delta = rewards[t] + self.gamma * next_value - values[t]
            
            # Compute GAE
            if dones[t]:
                gae = delta  # Reset GAE at terminal states
            else:
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages[t] = gae
        
        # Compute returns (advantages + values)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        
        return returns, advantages
    
    def update(self, rollout_buffer):
        """
        Perform PPO update using data from rollout buffer.
        
        Args:
            rollout_buffer: Buffer containing states, actions, rewards, dones, values, 
                           log_probs, and optionally bootstrap_values
        
        Returns:
            Dictionary with loss statistics
        """
        # Get data from buffer
        buffer_data = rollout_buffer.get()
        
        # Handle different buffer formats (with or without bootstrap_values)
        if len(buffer_data) == 6:
            # No bootstrap values
            states, actions, rewards, dones, values, old_log_probs = buffer_data
            bootstrap_values = None
        elif len(buffer_data) == 7:
            # With bootstrap values
            states, actions, rewards, dones, values, old_log_probs, bootstrap_values = buffer_data
        else:
            raise ValueError(f"Unexpected buffer data length: {len(buffer_data)}")
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        values = values.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        
        if bootstrap_values is not None:
            bootstrap_values = bootstrap_values.to(self.device)
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(
            rewards, dones, values, bootstrap_values
        )
        
        # Track losses for logging
        epoch_losses = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'total_loss': [],
        }
        
        # PPO update epochs
        for epoch in range(self.epochs):
            # Evaluate current policy
            new_log_probs, entropy, predicted_values = self.actor_critic.evaluate(
                states, actions
            )
            
            # Compute policy ratio
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate loss (policy loss with clipping)
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(
                ratios,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon,
            ) * advantages
            
            # Actor loss (min of surrogate for conservative updates)
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            
            # Critic loss (value function loss with optional clipping)
            if self.value_clip:
                # Clip value predictions to prevent large updates
                value_pred_clipped = values + torch.clamp(
                    predicted_values - values,
                    -self.clip_epsilon,
                    self.clip_epsilon,
                )
                value_loss_original = F.mse_loss(predicted_values, returns)
                value_loss_clipped = F.mse_loss(value_pred_clipped, returns)
                critic_loss = torch.max(value_loss_original, value_loss_clipped)
            else:
                critic_loss = F.mse_loss(predicted_values, returns)
            
            # Entropy bonus (encourage exploration)
            entropy_bonus = entropy.mean()
            
            # Total loss
            loss = (
                actor_loss +
                self.value_coef * critic_loss -
                self.entropy_coef * entropy_bonus
            )
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.max_grad_norm,
            )
            
            self.optimizer.step()
            
            # Track losses
            epoch_losses['actor_loss'].append(actor_loss.item())
            epoch_losses['critic_loss'].append(critic_loss.item())
            epoch_losses['entropy'].append(entropy_bonus.item())
            epoch_losses['total_loss'].append(loss.item())
        
        # Clear buffer
        rollout_buffer.clear()
        
        # Return average losses across epochs
        return {
            "loss": sum(epoch_losses['total_loss']) / len(epoch_losses['total_loss']),
            "actor_loss": sum(epoch_losses['actor_loss']) / len(epoch_losses['actor_loss']),
            "critic_loss": sum(epoch_losses['critic_loss']) / len(epoch_losses['critic_loss']),
            "entropy": sum(epoch_losses['entropy']) / len(epoch_losses['entropy']),
        }