"""
Actor-Critic Network for PPO Exit Policy.

Architecture designed for stable RL training:
- Orthogonal initialization
- Layer normalization
- Separate actor/critic heads with appropriate output scaling
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from typing import Tuple, Optional, List

try:
    from .config import PPOConfig
except ImportError:
    from config import PPOConfig


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Architecture:
        State -> Shared Encoder -> Actor Head -> Action Logits
                                -> Critic Head -> Value

    The shared encoder provides feature extraction, while separate
    heads allow for different learning dynamics between policy and value.
    """

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim

        # Build shared encoder
        encoder_layers = []
        in_dim = config.state_dim
        for hidden_dim in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_layer_norm:
                encoder_layers.append(nn.LayerNorm(hidden_dim))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        encoder_out_dim = config.hidden_dims[-1]

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_out_dim // 2, config.action_dim),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_out_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_out_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialization for stable RL training.

        - Main layers: gain = sqrt(2) (ReLU optimal)
        - Actor output: gain = 0.01 (encourages exploration)
        - Critic output: gain = 1.0 (accurate value estimates)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Smaller init for actor output (exploration)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0.0)

        # Standard init for critic output
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: Tensor of shape (batch_size, state_dim)

        Returns:
            action_logits: (batch_size, action_dim)
            value: (batch_size, 1)
        """
        features = self.encoder(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: (batch_size, state_dim)
            deterministic: If True, return argmax action

        Returns:
            action: (batch_size,) action indices
            log_prob: (batch_size,) log probability of actions
            value: (batch_size,) value estimates
        """
        action_logits, value = self.forward(state)

        dist = Categorical(logits=action_logits)

        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions.

        Used during PPO update to compute policy gradient.

        Args:
            state: (batch_size, state_dim)
            action: (batch_size,) action indices

        Returns:
            log_prob: (batch_size,) log probability of actions
            entropy: (batch_size,) policy entropy
            value: (batch_size,) value estimates
        """
        action_logits, value = self.forward(state)

        dist = Categorical(logits=action_logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, entropy, value.squeeze(-1)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for states.

        Args:
            state: (batch_size, state_dim)

        Returns:
            value: (batch_size,)
        """
        features = self.encoder(state)
        value = self.critic(features)
        return value.squeeze(-1)

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities for visualization.

        Args:
            state: (batch_size, state_dim)

        Returns:
            probs: (batch_size, action_dim) probability distribution
        """
        action_logits, _ = self.forward(state)
        return torch.softmax(action_logits, dim=-1)


class RolloutBuffer:
    """
    Buffer for storing rollout data during PPO training.

    Stores transitions and computes returns/advantages using GAE.
    """

    def __init__(
        self,
        buffer_size: int,
        n_envs: int,
        state_dim: int,
        device: str = "cuda",
    ):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.state_dim = state_dim
        self.device = torch.device(device)

        # Pre-allocate tensors
        self.states = torch.zeros(buffer_size, n_envs, state_dim, device=self.device)
        self.actions = torch.zeros(buffer_size, n_envs, dtype=torch.long, device=self.device)
        self.rewards = torch.zeros(buffer_size, n_envs, device=self.device)
        self.dones = torch.zeros(buffer_size, n_envs, device=self.device)
        self.log_probs = torch.zeros(buffer_size, n_envs, device=self.device)
        self.values = torch.zeros(buffer_size, n_envs, device=self.device)

        self.advantages = torch.zeros(buffer_size, n_envs, device=self.device)
        self.returns = torch.zeros(buffer_size, n_envs, device=self.device)

        self.ptr = 0
        self.full = False

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ):
        """Add a transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done.float()
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ):
        """
        Compute returns and advantages using GAE.

        GAE formula:
            A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        """
        last_gae = torch.zeros(self.n_envs, device=self.device)

        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """
        Generate minibatches for PPO update.

        Yields dictionaries containing flattened tensors.
        """
        # Flatten envs dimension
        size = self.ptr * self.n_envs
        indices = torch.randperm(size, device=self.device)

        states = self.states[:self.ptr].view(-1, self.state_dim)
        actions = self.actions[:self.ptr].view(-1)
        log_probs = self.log_probs[:self.ptr].view(-1)
        returns = self.returns[:self.ptr].view(-1)
        advantages = self.advantages[:self.ptr].view(-1)
        values = self.values[:self.ptr].view(-1)

        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_indices = indices[start:end]

            yield {
                'states': states[batch_indices],
                'actions': actions[batch_indices],
                'log_probs': log_probs[batch_indices],
                'returns': returns[batch_indices],
                'advantages': advantages[batch_indices],
                'values': values[batch_indices],
            }

    def reset(self):
        """Reset buffer for new rollout."""
        self.ptr = 0
        self.full = False
