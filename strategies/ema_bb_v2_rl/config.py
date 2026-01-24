"""
Configuration for PPO training and reward shaping.

Contains all hyperparameters with mathematical justification.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class RewardConfig:
    """Reward function weights and parameters."""

    # Primary reward components
    w_realized: float = 1.0          # Weight on realized P&L at exit
    w_mtm: float = 0.1               # Weight on mark-to-market changes (dense signal)

    # Risk penalties
    risk_coef: float = 0.3           # Drawdown penalty coefficient
    dd_threshold: float = 0.1        # Drawdown penalty kicks in after this %

    # Time management
    time_coef: float = 0.005         # Time decay penalty coefficient
    time_sigmoid_center: float = 100 # Bars at which time penalty is 50%

    # Regret penalty (for missing optimal exit)
    regret_coef: float = 0.5         # Penalty for exiting below optimal

    # Action costs (prevent excessive adjustments)
    tighten_sl_cost: float = 0.001
    trail_sl_cost: float = 0.0005

    # Reward scaling
    reward_scale: float = 100.0      # Scale factor for stable gradients

    # ==========================================================================
    # ACTION GUARDS - Prevent wasteful immediate exits
    # ==========================================================================
    # Problem: Model learns to PARTIAL/EXIT at Bar 0 with 0 pips to avoid time penalty
    # Solution: Block exit actions unless conditions are met, penalize invalid attempts
    #
    # These guards force the model to HOLD until profitable before taking profit.
    # Invalid action attempts are converted to HOLD with a penalty.

    # Minimum normalized PnL required for PARTIAL_EXIT (0.001 ≈ 10 pips for typical FX)
    min_profit_for_partial: float = 0.001

    # Minimum normalized PnL required for EXIT (0 = always allowed, set higher to force holds)
    min_profit_for_exit: float = 0.0

    # Minimum bars held before PARTIAL is allowed (0 = allowed immediately)
    min_bars_for_partial: int = 1

    # Minimum bars held before EXIT is allowed (0 = allowed immediately)
    min_bars_for_exit: int = 0

    # Penalty for attempting invalid actions (converted to HOLD + this penalty)
    invalid_action_penalty: float = 0.01


@dataclass
class PPOConfig:
    """
    PPO training configuration.

    Hyperparameters are tuned for trading exit optimization:
    - Higher gamma (0.99) for long-term reward consideration
    - Conservative clip_epsilon (0.2) for stable updates
    - Entropy annealing for exploration -> exploitation
    """

    # Environment
    n_envs: int = 64                 # Parallel environments (GPU saturation)
    max_episode_length: int = 200    # Max bars per trade

    # State/Action dimensions
    state_dim: int = 25              # Position + Market + Entry + History
    action_dim: int = 5              # HOLD, EXIT, TIGHTEN, TRAIL, PARTIAL

    # PPO Core hyperparameters
    gamma: float = 0.99              # Discount factor
    gae_lambda: float = 0.95         # GAE lambda (variance-bias tradeoff)
    clip_epsilon: float = 0.2        # PPO clipping range
    target_kl: float = 0.015         # KL divergence threshold for early stopping

    # Training schedule
    learning_rate: float = 3e-4      # Initial learning rate
    lr_end: float = 1e-5             # Final learning rate
    lr_schedule: str = "linear"      # "linear", "cosine", or "constant"

    n_epochs: int = 10               # PPO epochs per rollout
    batch_size: int = 2048           # Minibatch size
    n_steps: int = 2048              # Steps per rollout (before update)
    total_timesteps: int = 10_000_000  # Total training steps

    # Loss weights
    value_coef: float = 0.5          # Value function loss weight
    max_grad_norm: float = 0.5       # Gradient clipping

    # Entropy annealing (exploration -> exploitation)
    entropy_coef_start: float = 0.05    # Initial entropy bonus
    entropy_coef_end: float = 0.001     # Final entropy bonus
    entropy_anneal_steps: int = 5_000_000  # Steps to anneal over

    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    use_layer_norm: bool = True
    dropout: float = 0.0             # No dropout in RL (hurts value estimation)

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_interval: int = 10           # Log every N updates
    eval_interval: int = 50          # Evaluate every N updates
    save_interval: int = 100         # Save checkpoint every N updates

    # W&B
    wandb_project: str = "rl-exit-optimizer"
    wandb_entity: Optional[str] = None

    # Reward config
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        """Validate configuration."""
        assert self.n_envs > 0, "n_envs must be positive"
        assert self.n_steps > 0, "n_steps must be positive"
        assert 0 < self.gamma <= 1, "gamma must be in (0, 1]"
        assert 0 < self.gae_lambda <= 1, "gae_lambda must be in (0, 1]"
        assert 0 < self.clip_epsilon < 1, "clip_epsilon must be in (0, 1)"

        # Ensure batch_size divides rollout size
        rollout_size = self.n_envs * self.n_steps
        if rollout_size % self.batch_size != 0:
            # Adjust batch size to divide evenly
            self.batch_size = min(self.batch_size, rollout_size)


class Actions:
    """Discrete action space for exit management."""

    HOLD = 0              # Maintain current position
    EXIT = 1              # Close position immediately
    TIGHTEN_SL = 2        # Move SL 25% closer to price
    TRAIL_BREAKEVEN = 3   # Move SL to entry if in profit
    PARTIAL_EXIT = 4      # Close 50% (simplified as EXIT for v1)

    NAMES = ['HOLD', 'EXIT', 'TIGHTEN_SL', 'TRAIL_BE', 'PARTIAL']

    @classmethod
    def to_name(cls, action: int) -> str:
        return cls.NAMES[action] if 0 <= action < len(cls.NAMES) else "UNKNOWN"
