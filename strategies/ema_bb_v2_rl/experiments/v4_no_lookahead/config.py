"""
Configuration for v4_no_lookahead experiment.

FROZEN SNAPSHOT - Do not modify after training begins.

Key Changes from v3:
1. ANTI-LOOKAHEAD: min_exit_bar=3 - block EXIT/PARTIAL until bar 3
2. ACTION MASKING: Invalid actions masked at distribution level
3. GPU OPTIMIZATION: 256 envs, 8192 batch for RTX 4090
4. LARGER NETWORK: [512, 256] hidden dims
5. MORE TRAINING: 15M timesteps (vs 10M in v3)

Design Goal: Fix look-ahead bias where v3 learned to exit at bar 0/1 to
capture "free" profits visible in the market_tensor.
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
    time_coef: float = 0.005
    time_sigmoid_center: float = 100 # Bars at which time penalty is 50%

    # ==========================================================================
    # COUNTERFACTUAL REWARDS - Keep from v3
    # ==========================================================================
    regret_coef: float = 0.4         # Opportunity cost for premature exits
    defensive_coef: float = 0.5      # Bonus for defensive exits
    counterfactual_window: int = 8   # Look 8 bars ahead (2 hours @ 15M)

    # Exit costs - keep from v3
    exit_cost: float = 0.0
    exit_bonus: float = 0.0
    hold_bonus: float = 0.0

    # ==========================================================================
    # ACTION DIVERSITY - Keep from v3
    # ==========================================================================
    min_action_prob: float = 0.02    # 2% minimum per action
    diversity_coef: float = 0.1      # Weight of diversity loss

    # Action costs
    tighten_sl_cost: float = 0.001
    trail_sl_cost: float = 0.0005

    # Reward scaling
    reward_scale: float = 100.0

    # ==========================================================================
    # ANTI-LOOKAHEAD PROTECTION - NEW in v4
    # ==========================================================================
    # Prevent exploitation of entry bar information by masking exit actions
    # until min_exit_bar is reached.
    #
    # Problem: v3 learned to exit at bar 0/1 with small profits that were
    # already visible in the market_tensor. In real trading, you can't know
    # the first bar's PnL at the exact moment of entry.
    #
    # Solution: Block EXIT (1) and PARTIAL_EXIT (4) until bar 3.
    # HOLD (0), TIGHTEN_SL (2), TRAIL_BE (3) allowed at any bar.
    #
    # 3 bars @ 15M = 45 minutes - enough for initial volatility to settle

    min_exit_bar: int = 3            # Minimum bars before EXIT/PARTIAL allowed
    use_action_masking: bool = True  # Mask at distribution level (set logits to -inf)


@dataclass
class PPOConfig:
    """PPO training configuration for v4_no_lookahead - optimized for RTX 4090."""

    # ==========================================================================
    # GPU OPTIMIZATION FOR RTX 4090
    # ==========================================================================
    # RTX 4090 specs:
    # - 24 GB VRAM
    # - 128 SMs (16384 CUDA cores)
    # - 1 TB/s memory bandwidth
    #
    # Increased from v3 to fully utilize GPU:

    n_envs: int = 256                # Up from 128 - more parallelism
    batch_size: int = 8192           # Up from 4096 - fill more VRAM
    n_steps: int = 2048              # Keep same - good rollout length
    total_timesteps: int = 15_000_000  # Up from 10M - more thorough training

    max_episode_length: int = 200    # Max bars per trade

    # State/Action dimensions
    state_dim: int = 25              # Position + Market + Entry + History
    action_dim: int = 5              # HOLD, EXIT, TIGHTEN, TRAIL, PARTIAL

    # PPO Core hyperparameters
    gamma: float = 0.99              # Discount factor
    gae_lambda: float = 0.95         # GAE lambda
    clip_epsilon: float = 0.2        # PPO clipping range
    target_kl: float = 0.015         # KL divergence threshold

    # Training schedule
    learning_rate: float = 3e-4      # Initial learning rate
    lr_end: float = 1e-5             # Final learning rate
    lr_schedule: str = "linear"

    n_epochs: int = 8                # Reduced from 10 (larger batches = fewer epochs)

    # Loss weights
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Entropy annealing
    entropy_coef_start: float = 0.05
    entropy_coef_end: float = 0.01
    entropy_anneal_steps: int = 7_000_000  # Extended for 15M training

    # Network architecture - LARGER for RTX 4090
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    use_layer_norm: bool = True
    dropout: float = 0.0

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100

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
        assert self.reward.min_exit_bar >= 0, "min_exit_bar must be non-negative"

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

    @classmethod
    def is_exit_action(cls, action: int) -> bool:
        """Check if action closes the position."""
        return action in [cls.EXIT, cls.PARTIAL_EXIT]
