"""
Replay buffer for storing and sampling experiences.
"""
from typing import Tuple

import torch


class ReplayBuffer:
    """A simple replay buffer."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: torch.device | None,
    ):
        self.device = device

        self.state = torch.zeros((capacity, state_dim))
        self.action = torch.zeros((capacity, action_dim))
        self.reward = torch.zeros((capacity, 1))
        self.next_state = torch.zeros((capacity, state_dim))
        self.done = torch.zeros((capacity, 1), dtype=torch.int64)

        self.ptr = 0
        self.size = 0
        self.capacity = capacity

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ):
        """Add a new experience to the replay buffer."""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences from the replay buffer."""
        batch_idx = torch.randint(0, self.size, size=(batch_size,))

        state = self.state[batch_idx].to(self.device)
        action = self.action[batch_idx].to(self.device)
        reward = self.reward[batch_idx].to(self.device)
        next_state = self.next_state[batch_idx].to(self.device)
        done = self.done[batch_idx].to(self.device)

        return (state, action, reward, next_state, done)
