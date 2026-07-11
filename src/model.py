"""
Soft Actor-Critic (SAC) model definitions for actor and critic networks.
"""
from typing import Tuple

import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.nn.utils import spectral_norm

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-7


class Actor(nn.Module):
    """Actor network for Soft Actor-Critic (SAC)."""
    def __init__(
        self,
        state_dim: int, # oq eh esse?
        action_dim: int, # não precisa mexer, o MuJuCo já define por padrao pelo env
        hidden_dim: int, # psor falou pra deixar em 88
        action_scale: torch.Tensor = torch.tensor(1.0),
        action_bias: torch.Tensor = torch.tensor(0.0),
    ):
        """Initialize the Actor network."""
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        with torch.no_grad():
            self.mean_head.weight.data.copy_(self.mean_head.weight.data * 0.01)
            self.log_std_head.weight.data.copy_(self.log_std_head.weight.data * 0.01)

        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Actor network."""
        x = self.body(state)

        mean = self.mean_head(x)

        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = log_std.exp()

        return mean, std

    def sample(self, state: torch.Tensor) -> Tuple:
        """Sample an action from the policy given the state."""
        mean, std = self.forward(state)

        dist = Normal(mean, std)

        x_t = dist.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias

        log_prob = dist.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1. - y_t.pow(2)) + EPS)
        log_prob = log_prob.sum(1, keepdim=True)

        action_mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, action_mean


class Critic(nn.Module):
    """Critic network for Soft Actor-Critic (SAC)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        """Initialize the Critic network."""
        super().__init__()

        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Critic network."""
        x = torch.cat([state, action], dim=1)

        q1 = self.q1(x)
        q2 = self.q2(x)

        return q1, q2
