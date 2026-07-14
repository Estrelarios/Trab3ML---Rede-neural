"""
Soft Actor-Critic (SAC) agent implementation.
"""
from pathlib import Path

import torch
from torch import optim
from torch.nn import functional as F
import numpy as np
from gymnasium.spaces import Space

from buffer import ReplayBuffer
from model import Actor, Critic


class SAC:
    """Soft Actor-Critic (SAC) agent implementation."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        capacity: int = 1_000_000,
        batch_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.055,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        action_space: Space | None = None,
        device: torch.device = torch.device("cpu"),
        activation: str = "tanh",
    ):
        """Initialize the SAC agent."""
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.device = device

        self.memory = ReplayBuffer(capacity, state_dim, action_dim, device)

        if action_space is None:
            action_scale = torch.tensor(1., device=device)
            bias_scale = torch.tensor(0., device=device)
        else:
            action_scale = torch.tensor(
                (action_space.high - action_space.low) / 2.,
                device=device
            )
            bias_scale = torch.tensor(
                (action_space.high + action_space.low) / 2.,
                device=device
            )

        self.actor = Actor(
            state_dim, action_dim, hidden_dim, action_scale, bias_scale, activation=activation
        ).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.alpha = alpha
        self.auto_tune = auto_tune_alpha

        if self.auto_tune:
            self.target_entropy = -torch.prod(
                torch.tensor(action_space.shape, device=device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.log_alpha_optim = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action given the current state."""
        tensor_state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            _, _, action = self.actor.sample(tensor_state)
        else:
            action, _, _ = self.actor.sample(tensor_state)

        return action.detach()

    def learn(self):
        """Update the agent's networks based on a batch of experiences."""
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            q1_next_target, q2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_q_next_target

        q1, q2 = self.critic(state_batch, action_batch)
        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)
        q_loss = q1_loss + q2_loss

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.actor.sample(state_batch)
        q1_pi, q2_pi = self.critic(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        if self.auto_tune:
            with torch.no_grad():
                _, log_pi, _ = self.actor.sample(state_batch)
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.log_alpha_optim.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save_model(self, filepath: Path) -> None:
        """Save the model to the specified filepath."""
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "actor_state_dict": self.actor.state_dict(),
            }
            torch.save(checkpoint, filepath)
            print(f"Model saved successfully to {filepath}")
        except IOError as e:
            print(f"I/O error({e.errno}) while saving model at {filepath}: {e.strerror}")

    def load_model(self, filepath: Path) -> None:
        """Load the model from the specified filepath."""
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return
        try:
            checkpoint = torch.load(
                filepath, map_location=self.device
            )
            actor_state = checkpoint.get("actor_state_dict")

            self.actor.load_state_dict(actor_state, strict=False)
            print(f"Model loaded successfully from {filepath}")
        except FileNotFoundError:
            print(f"File not found: {filepath}")
