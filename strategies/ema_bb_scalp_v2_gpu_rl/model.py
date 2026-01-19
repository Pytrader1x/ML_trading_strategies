"""
Neural Network Models for Meta-Labeling + RL

This module contains:
1. ActorCritic: PPO policy network for the RL agent
2. TransformerMetaLabeler: Optional transformer for meta-labeling

All models are optimized for GPU training with:
- Orthogonal initialization
- LayerNorm for stability
- torch.compile compatibility
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

from .config import RLConfig, MetaLabelerConfig


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Architecture:
    - Shared feature extractor (MLP with LayerNorm)
    - Actor head (policy logits)
    - Critic head (state value)

    Optimized for RTX 4090/5090 with torch.compile.
    """

    def __init__(
        self,
        obs_dim: int = 10,
        action_dim: int = 4,
        hidden_size: int = 256,
        num_layers: int = 2,
        use_layer_norm: bool = True
    ):
        """
        Initialize ActorCritic network.

        Args:
            obs_dim: Observation dimension (default 10)
            action_dim: Number of discrete actions (default 4)
            hidden_size: Hidden layer size
            num_layers: Number of hidden layers
            use_layer_norm: Whether to use LayerNorm
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Build shared feature extractor
        layers = []
        in_dim = obs_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size

        self.shared = nn.Sequential(*layers)

        # Actor head (policy - outputs logits for each action)
        self.actor = nn.Linear(hidden_size, action_dim)

        # Critic head (value function - outputs scalar value estimate)
        self.critic = nn.Linear(hidden_size, 1)

        # Initialize weights using orthogonal initialization
        self._init_weights()

    def _init_weights(self):
        """Apply orthogonal initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Special initialization for output heads
        nn.init.orthogonal_(self.actor.weight, gain=0.01)  # Small for policy
        nn.init.orthogonal_(self.critic.weight, gain=1.0)  # Standard for value

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Observations (batch, obs_dim)

        Returns:
            logits: Action logits (batch, action_dim)
            value: State value estimate (batch,)
        """
        features = self.shared(obs)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        features = self.shared(obs)
        return self.critic(features)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.

        Args:
            obs: Observations (batch, obs_dim)
            action: If provided, compute log_prob for this action
            deterministic: If True, take argmax action (for evaluation)

        Returns:
            action: Sampled or provided action (batch,)
            log_prob: Log probability of action (batch,)
            entropy: Policy entropy (batch,)
            value: State value estimate (batch,)
        """
        logits, value = self(obs)

        # Create categorical distribution
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


class TransformerMetaLabeler(nn.Module):
    """
    Transformer-based meta-labeler for sequence pattern recognition.

    Takes a sequence of market features and predicts P(success) for the
    current bar as a potential entry signal.

    This is an alternative to XGBoost that can capture temporal patterns.
    """

    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        seq_len: int = 20,
        dropout: float = 0.1
    ):
        """
        Initialize Transformer meta-labeler.

        Args:
            input_dim: Number of input features per timestep
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            seq_len: Sequence length (lookback window)
            dropout: Dropout rate
        """
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequence (batch, seq_len, input_dim)

        Returns:
            probs: Probability of success (batch,)
        """
        # Project input to model dimension
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Transformer encoding
        x = self.transformer(x)

        # Take last timestep for classification
        x = x[:, -1, :]

        # Output probability
        return self.output_head(x).squeeze(-1)


class SimpleMLP(nn.Module):
    """
    Simple MLP for quick testing or baseline comparison.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 2
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_actor_critic(config: RLConfig, compile_model: bool = True) -> ActorCritic:
    """
    Factory function to create and optionally compile ActorCritic.

    Args:
        config: RL configuration
        compile_model: Whether to use torch.compile (recommended for GPU)

    Returns:
        Compiled ActorCritic model on the specified device
    """
    model = ActorCritic(
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        use_layer_norm=config.use_layer_norm
    )

    device = torch.device(config.device)
    model = model.to(device)

    if compile_model and config.use_torch_compile and config.device == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled with torch.compile (reduce-overhead mode)")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

    return model


def create_transformer_meta_labeler(config: MetaLabelerConfig) -> TransformerMetaLabeler:
    """
    Factory function to create TransformerMetaLabeler.
    """
    return TransformerMetaLabeler(
        input_dim=len(config.feature_columns),
        d_model=config.transformer_d_model,
        nhead=config.transformer_nhead,
        num_layers=config.transformer_num_layers,
        seq_len=config.transformer_seq_len,
        dropout=config.transformer_dropout
    )
