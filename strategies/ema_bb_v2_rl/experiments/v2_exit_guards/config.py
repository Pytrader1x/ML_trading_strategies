"""
Configuration for v2_exit_guards experiment.

FROZEN SNAPSHOT - Do not modify after training begins.
Changes from v1_baseline:
1. EXIT guards enabled (min_profit=0.001, min_bars=2)
2. PARTIAL guards strengthened (min_profit=0.002, min_bars=2)
3. Reduced time penalty (0.002 vs 0.005) - less incentive for early exits
4. Increased regret coefficient (0.8 vs 0.5) - penalize missing optimal exit more
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

    # Time management - REDUCED from 0.005 to 0.002
    # Lower value = less incentive to exit early just to avoid time penalty
    time_coef: float = 0.002
    time_sigmoid_center: float = 100 # Bars at which time penalty is 50%

    # Regret penalty - INCREASED from 0.5 to 0.8
    # Higher value = more penalty for exiting before optimal
    regret_coef: float = 0.8

    # Action costs (prevent excessive adjustments)
    tighten_sl_cost: float = 0.001
    trail_sl_cost: float = 0.0005

    # Reward scaling
    reward_scale: float = 100.0      # Scale factor for stable gradients

    # ==========================================================================
    # ACTION GUARDS - ENABLED FOR V2
    # ==========================================================================
    # Problem solved: Model was exiting at Bar 0 with 0 pips to avoid time penalty
    # Solution: Block exits unless profit AND holding time requirements are met

    # EXIT guards - NOW ENABLED (was 0.0 and 0 in v1)
    min_profit_for_exit: float = 0.001    # ~10 pips minimum for EXIT
    min_bars_for_exit: int = 2            # Must hold at least 2 bars

    # PARTIAL guards - STRENGTHENED (was 0.001 and 1 in v1)
    min_profit_for_partial: float = 0.002  # ~20 pips minimum for PARTIAL
    min_bars_for_partial: int = 2          # Must hold at least 2 bars

    # Penalty for attempting blocked actions
    invalid_action_penalty: float = 0.02   # Higher penalty than v1 (was 0.01)


@dataclass
class PPOConfig:
    """PPO training configuration for v2_exit_guards."""

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

    # Entropy annealing - SLOWER DECAY for more exploration
    entropy_coef_start: float = 0.05    # Initial entropy bonus
    entropy_coef_end: float = 0.005     # Final entropy (higher than v1's 0.001)
    entropy_anneal_steps: int = 7_000_000  # Longer anneal (was 5M)

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
