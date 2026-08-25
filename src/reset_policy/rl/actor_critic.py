import torch
import torch.nn as nn
from torch.distributions import Normal, TanhTransform, TransformedDistribution


class ActorCritic(nn.Module):

    def __init__(
        self,
        observation_dim=14,
        action_dim=4,
        hidden_dim=128,
    ):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor - outputs mean in [-1, 1]
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # Learnable standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, observation):
        features = self.shared(observation)
        mean = self.actor(features)
        value = self.critic(features)
        return mean, value

    def act(self, observation):
        """
        Sample an action.
        
        Returns:
            action: sampled action in [-1, 1]
            log_prob: log probability of the action
            value: state value estimate
        """
        mean, value = self.forward(observation)
        std = self.log_std.exp()

        # Use TanhNormal to keep actions in [-1, 1]
        normal = Normal(mean, std)
        distribution = TransformedDistribution(normal, TanhTransform())
        action = distribution.rsample()
        log_prob = distribution.log_prob(action).sum(-1)

        return action, log_prob, value.squeeze(-1)

    def evaluate(self, observations, actions):
        """
        Used during PPO update.
        Computes log_probs, entropy, and values for given actions.
        """
        mean, values = self.forward(observations)
        std = self.log_std.exp()

        # Use the SAME distribution as act()
        normal = Normal(mean, std)
        distribution = TransformedDistribution(normal, TanhTransform())

        # Log probabilities from the transformed distribution
        log_probs = distribution.log_prob(actions).sum(-1)

        # ============================================================
        # CRITICAL: Use entropy from the BASE Normal distribution
        # NOT from the transformed distribution
        # ============================================================
        # TransformedDistribution with TanhTransform doesn't have
        # closed-form entropy, but Normal does.
        # This is an approximation that works well for PPO.
        entropy = normal.entropy().sum(-1)

        return log_probs, entropy, values.squeeze(-1)