"""
Configuration for v6_anti_gaming experiment.

FROZEN SNAPSHOT - Do not modify after training begins.

V6 ANTI-GAMING: Fix V5's "breakeven farming" gaming behavior.

V5 Problem: Agent exits 71.6% of trades at breakeven, never lets winners run (0.2% profit exits)

V6 Solution:
1. ASYMMETRIC BREAKEVEN REWARDS: Reward recovering losers to scratch, penalize giving back winners
2. MFE DECAY PENALTY: Penalize giving back too much of the maximum favorable excursion
3. TIGHTEN COOLDOWN: Prevent TIGHTEN_SL spam with cooldown between uses
4. MODERATE INCREASES: regret_coef, min_exit_bar, min_trail_bar, min_profit_for_trail

Design Goal: Force the model to let winners run instead of farming breakeven exits.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class RewardConfig:
    """Reward function weights and parameters."""

    # ==========================================================================
    # PRIMARY REWARD COMPONENTS (unchanged from V5)
    # ==========================================================================
    w_realized: float = 1.0          # Weight on realized P&L at exit
    w_mtm: float = 0.1               # Weight on mark-to-market changes (dense signal)

    # Risk penalties
    risk_coef: float = 0.3           # Drawdown penalty coefficient
    dd_threshold: float = 0.1        # Drawdown penalty kicks in after this %

    # Time management
    time_coef: float = 0.005
    time_sigmoid_center: float = 100 # Bars at which time penalty is 50%

    # ==========================================================================
    # COUNTERFACTUAL REWARDS (unchanged from V5)
    # ==========================================================================
    defensive_coef: float = 0.5      # Bonus for defensive exits
    counterfactual_window: int = 8   # Look 8 bars ahead (2 hours @ 15M)

    # Exit costs - keep from v3/v4/v5
    exit_cost: float = 0.0
    exit_bonus: float = 0.0
    hold_bonus: float = 0.0

    # ==========================================================================
    # ACTION DIVERSITY (unchanged from V5)
    # ==========================================================================
    min_action_prob: float = 0.02    # 2% minimum per action
    diversity_coef: float = 0.1      # Weight of diversity loss

    # Action costs - keep cheap so defensive actions aren't discouraged
    tighten_sl_cost: float = 0.001
    trail_sl_cost: float = 0.0005

    # Reward scaling
    reward_scale: float = 100.0

    # ==========================================================================
    # ANTI-LOOKAHEAD PROTECTION (unchanged from V5)
    # ==========================================================================
    use_action_masking: bool = True  # Mask at distribution level (set logits to -inf)
    breakeven_buffer_pct: float = 0.0  # True breakeven, no buffer

    # ==========================================================================
    # V6: MODERATE INCREASES FROM V5
    # ==========================================================================
    # These are slightly increased to discourage early/premature actions
    regret_coef: float = 0.5         # Was 0.4 in V5 - opportunity cost for premature exits
    min_exit_bar: int = 2            # Was 1 in V5 - minimum bars before EXIT/PARTIAL allowed
    min_trail_bar: int = 2           # Was 1 in V5 - minimum bars before TRAIL_BE allowed
    min_profit_for_trail: float = 0.0005  # Was 0.0002 in V5 (~5 pips vs 2 pips)

    # ==========================================================================
    # V6 NEW: ASYMMETRIC BREAKEVEN REWARDS
    # ==========================================================================
    # The core fix for V5's breakeven farming:
    # - REWARD exiting at scratch if we were in a losing position (good recovery)
    # - PENALIZE exiting at scratch if we were winning (gave back profits)
    #
    # This creates asymmetric incentives that discourage farming breakeven exits
    # while still rewarding legitimate loss recovery.

    recovery_bonus: float = 0.2      # Bonus for recovering a loser to scratch
    giveback_penalty: float = 0.3    # Penalty for giving back a winner to scratch
    breakeven_band: float = 0.001    # +/- 0.1% considered "breakeven" exit

    # ==========================================================================
    # V6 NEW: MFE DECAY PENALTY
    # ==========================================================================
    # Penalize giving back too much of the maximum favorable excursion (MFE).
    # This encourages the model to exit while still holding meaningful profit
    # rather than letting winners reverse all the way back to breakeven.

    mfe_decay_coef: float = 0.4      # Penalty coefficient for MFE giveback
    min_mfe_for_decay: float = 0.001 # Only apply if MFE > 10 pips (0.1%)

    # ==========================================================================
    # V6 NEW: TIGHTEN COOLDOWN
    # ==========================================================================
    # Prevent TIGHTEN_SL spam by requiring a cooldown between uses.
    # This forces the model to be more deliberate about stop loss tightening.

    tighten_cooldown: int = 2        # Bars between TIGHTEN_SL allowed


@dataclass
class PPOConfig:
    """PPO training configuration for v6_anti_gaming - optimized for RTX 4090."""

    # ==========================================================================
    # GPU OPTIMIZATION FOR RTX 4090 (same as v4/v5)
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
        assert self.reward.tighten_cooldown >= 0, "tighten_cooldown must be non-negative"
        assert self.reward.recovery_bonus >= 0, "recovery_bonus must be non-negative"
        assert self.reward.giveback_penalty >= 0, "giveback_penalty must be non-negative"
        assert self.reward.mfe_decay_coef >= 0, "mfe_decay_coef must be non-negative"

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
