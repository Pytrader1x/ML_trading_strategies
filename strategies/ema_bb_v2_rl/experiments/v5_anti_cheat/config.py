"""
Configuration for v5_anti_cheat experiment.

FROZEN SNAPSHOT - Do not modify after training begins.

Key Changes from v4:
1. EXIT/PARTIAL FROM BAR 1: min_exit_bar=1 - allow exits from t+1
2. TRAIL_BE FROM BAR 1: min_trail_bar=1 - block TRAIL_BE until bar 1 (t+1)
3. TRUE BREAKEVEN: 0 buffer - exit exactly at entry price, no free money
4. MIN PROFIT FOR TRAIL: 0.0002 (2 pips) - must have real profit before activating
5. ALL EARLY ACTIONS MASKED: EXIT, PARTIAL, TRAIL_BE all masked at bar 0

Design Goal: Prevent all forms of look-ahead exploitation including immediate
breakeven activation which was exploited in v4 testing (99.2% fake win rate).
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
    # COUNTERFACTUAL REWARDS - Keep from v3/v4
    # ==========================================================================
    regret_coef: float = 0.4         # Opportunity cost for premature exits
    defensive_coef: float = 0.5      # Bonus for defensive exits
    counterfactual_window: int = 8   # Look 8 bars ahead (2 hours @ 15M)

    # Exit costs - keep from v3/v4
    exit_cost: float = 0.0
    exit_bonus: float = 0.0
    hold_bonus: float = 0.0

    # ==========================================================================
    # ACTION DIVERSITY - Keep from v3/v4
    # ==========================================================================
    min_action_prob: float = 0.02    # 2% minimum per action
    diversity_coef: float = 0.1      # Weight of diversity loss

    # Action costs
    tighten_sl_cost: float = 0.001
    trail_sl_cost: float = 0.0005

    # Reward scaling
    reward_scale: float = 100.0

    # ==========================================================================
    # ANTI-LOOKAHEAD PROTECTION - Enhanced in v5
    # ==========================================================================
    # Prevent exploitation of entry bar information by masking exit actions
    # until min_exit_bar is reached AND TRAIL_BE until min_trail_bar.
    #
    # v4 Problem: TRAIL_BE at bar 0 allowed model to immediately lock in profits
    # v4 Problem: +0.25 pips buffer created "free money" (99.2% fake win rate)
    #
    # v5 Solution:
    # - Block EXIT/PARTIAL until bar 3 (keep from v4)
    # - Block TRAIL_BE until bar 1 (t+1 per user request)
    # - Use true breakeven (0 buffer, no free money)
    # - Require minimum profit for TRAIL_BE activation

    min_exit_bar: int = 1            # Minimum bars before EXIT/PARTIAL allowed (t+1)
    min_trail_bar: int = 1           # Minimum bars before TRAIL_BE allowed (t+1)
    use_action_masking: bool = True  # Mask at distribution level (set logits to -inf)

    # ==========================================================================
    # TRUE BREAKEVEN - NEW in v5
    # ==========================================================================
    # When TRAIL_BE is activated, position exits if PnL drops below this threshold.
    # v4 used +0.25 pips which was exploited. v5 uses 0 (true breakeven).
    #
    # Setting to 0 means:
    # - Exit exactly at entry price (PnL = 0)
    # - No "free money" buffer
    # - More realistic behavior

    breakeven_buffer_pct: float = 0.0  # True breakeven, no buffer (v4 had +0.25 pips)

    # ==========================================================================
    # MINIMUM PROFIT FOR TRAIL_BE - NEW in v5
    # ==========================================================================
    # Only allow TRAIL_BE activation if unrealized PnL exceeds this threshold.
    # Prevents activating breakeven on tiny profits.
    #
    # 0.0002 = 0.02% = ~2 pips for EUR/USD
    # Model must have meaningful profit before locking in breakeven.

    min_profit_for_trail: float = 0.0002  # 0.02% = ~2 pips minimum profit to activate TRAIL_BE


@dataclass
class PPOConfig:
    """PPO training configuration for v5_anti_cheat - optimized for RTX 4090."""

    # ==========================================================================
    # GPU OPTIMIZATION FOR RTX 4090 (same as v4)
    # ==========================================================================
    n_envs: int = 256                # 256 parallel environments
    batch_size: int = 8192           # Large batch for GPU utilization
    n_steps: int = 2048              # Rollout length
    total_timesteps: int = 15_000_000  # 15M timesteps

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

    n_epochs: int = 8

    # Loss weights
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Entropy annealing
    entropy_coef_start: float = 0.05
    entropy_coef_end: float = 0.01
    entropy_anneal_steps: int = 7_000_000

    # Network architecture
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
        assert self.reward.min_trail_bar >= 0, "min_trail_bar must be non-negative"
        assert self.reward.min_trail_bar <= self.reward.min_exit_bar, \
            "min_trail_bar should be <= min_exit_bar (TRAIL_BE is less exploitable)"

        # Ensure batch_size divides rollout size
        rollout_size = self.n_envs * self.n_steps
        if rollout_size % self.batch_size != 0:
            self.batch_size = min(self.batch_size, rollout_size)


class Actions:
    """Discrete action space for exit management."""

    HOLD = 0              # Maintain current position
    EXIT = 1              # Close position immediately
    TIGHTEN_SL = 2        # Move SL 25% closer to price
    TRAIL_BREAKEVEN = 3   # Move SL to entry if in profit (delayed until min_trail_bar)
    PARTIAL_EXIT = 4      # Close 50% (simplified as EXIT for v1)

    NAMES = ['HOLD', 'EXIT', 'TIGHTEN_SL', 'TRAIL_BE', 'PARTIAL']

    @classmethod
    def to_name(cls, action: int) -> str:
        return cls.NAMES[action] if 0 <= action < len(cls.NAMES) else "UNKNOWN"

    @classmethod
    def is_exit_action(cls, action: int) -> bool:
        """Check if action closes the position."""
        return action in [cls.EXIT, cls.PARTIAL_EXIT]
