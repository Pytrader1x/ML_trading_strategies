"""
RL Model Definitions for EMA BB V2

Neural network architectures for policy and value functions.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class MLP(nn.Module):
    """Simple MLP for actor-critic."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        output_dim: int,
        activation: str = "relu"
    ):
        super().__init__()

        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
        }
        act_fn = activations.get(activation, nn.ReLU)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO/A2C."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [256, 256],
        activation: str = "relu"
    ):
        super().__init__()

        # Shared feature extractor
        self.shared = MLP(obs_dim, hidden_sizes[:-1], hidden_sizes[-1], activation)

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_sizes[-1], action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (value)
        self.critic = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action from policy."""
        action_probs, value = self.forward(obs)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

        return action, action_probs, value


class LSTMActorCritic(nn.Module):
    """LSTM-based Actor-Critic for sequential decision making."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Linear(hidden_size, 1)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ):
        # obs shape: (batch, seq_len, obs_dim)
        if hidden is None:
            hidden = self.init_hidden(obs.size(0), obs.device)

        lstm_out, new_hidden = self.lstm(obs, hidden)

        # Use last output
        features = lstm_out[:, -1, :]

        action_probs = self.actor(features)
        value = self.critic(features)

        return action_probs, value, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)
