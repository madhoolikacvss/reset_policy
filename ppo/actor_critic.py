import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):

    def __init__(
        self,
        observation_dim=11,
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

        # Actor
        self.actor = nn.Linear(
            hidden_dim,
            action_dim,
        )

        # Learnable standard deviation
        self.log_std = nn.Parameter(
            torch.zeros(action_dim)
        )

        # Critic
        self.critic = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(self, observation):

        features = self.shared(observation)

        mean = self.actor(features)

        value = self.critic(features)

        return mean, value

    def act(self, observation):
        """
        Sample an action.

        Returns:
            action
            log probability
            value estimate
        """

        mean, value = self.forward(observation)

        std = self.log_std.exp()

        distribution = Normal(mean, std)

        action = distribution.sample()
        action = torch.clamp(action, -1.0, 1.0)

        log_prob = distribution.log_prob(action).sum(-1)

        return (action,log_prob,value.squeeze(-1),)

    def evaluate(self, observations, actions):
        """
        Used during PPO update.
        """

        mean, values = self.forward(observations)

        std = self.log_std.exp()

        distribution = Normal(mean, std)

        log_probs = distribution.log_prob(actions).sum(-1)

        entropy = distribution.entropy().sum(-1)

        return (log_probs,entropy,values.squeeze(-1),)