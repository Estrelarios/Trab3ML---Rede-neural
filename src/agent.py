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
        tau: float = 0.005,
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
        self.activation = activation.lower()

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
            state_dim, action_dim, hidden_dim, action_scale, bias_scale, activation=self.activation
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
            # alpha inicial = valor do config (antes: log_alpha=0 -> alpha=1.0)
            self.log_alpha = torch.full(
                (1,), float(np.log(alpha)), requires_grad=True, device=device
            )
            self.log_alpha_optim = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action given the current state."""
        tensor_state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor.sample(tensor_state)
            else:
                action, _, _ = self.actor.sample(tensor_state)

        return action

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
                "activation": self.activation,
            }
            torch.save(checkpoint, filepath)
            print(f"Model saved successfully to {filepath}")
        except IOError as e:
            print(f"I/O error({e.errno}) while saving model at {filepath}: {e.strerror}")

    def save_checkpoint(self, filepath: Path) -> None:
        """Save the full training state (actor, critics, optimizers, alpha).

        O replay buffer não é salvo (seria grande); ao retomar, ele recomeça vazio.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "activation": self.activation,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optim_state_dict": self.actor_optim.state_dict(),
            "critic_optim_state_dict": self.critic_optim.state_dict(),
            "alpha": self.alpha,
        }
        if self.auto_tune:
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
            checkpoint["log_alpha_optim_state_dict"] = self.log_alpha_optim.state_dict()
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved successfully to {filepath}")

    def load_checkpoint(self, filepath: Path) -> None:
        """Restore the full training state saved by `save_checkpoint`."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self._check_activation(checkpoint)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optim.load_state_dict(checkpoint["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(checkpoint["critic_optim_state_dict"])
        self.alpha = checkpoint.get("alpha", self.alpha)

        if self.auto_tune and "log_alpha" in checkpoint:
            with torch.no_grad():
                self.log_alpha.copy_(checkpoint["log_alpha"].to(self.device))
            self.log_alpha_optim.load_state_dict(checkpoint["log_alpha_optim_state_dict"])
        print(f"Checkpoint loaded successfully from {filepath}")

    def _check_activation(self, checkpoint: dict) -> None:
        """Fail loudly if the checkpoint was trained with another activation."""
        saved = checkpoint.get("activation")
        if saved is not None and saved.lower() != self.activation:
            raise ValueError(
                f"Ativação do checkpoint ('{saved}') difere da do agente "
                f"('{self.activation}'). Instancie o SAC com activation='{saved}'."
            )

    def load_model(self, filepath: Path) -> None:
        """Load the model from the specified filepath."""
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return
        try:
            checkpoint = torch.load(
                filepath, map_location=self.device
            )
            self._check_activation(checkpoint)
            actor_state = checkpoint.get("actor_state_dict")

            self.actor.load_state_dict(actor_state, strict=False)
            print(f"Model loaded successfully from {filepath}")
        except FileNotFoundError:
            print(f"File not found: {filepath}")
