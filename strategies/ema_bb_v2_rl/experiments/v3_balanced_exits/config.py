"""
Configuration for v3_balanced_exits experiment.

FROZEN SNAPSHOT - Do not modify after training begins.

Key Changes from v2:
1. REDUCED regret_coef (0.4 vs 2.0) - gentle learning, not punishment
2. ASYMMETRIC coefficients - defensive_coef (0.5) > regret_coef (0.4)
3. BOUNDED counterfactual window (8 bars vs 20)
4. RESTORED time penalty (0.005 vs 0.002)
5. NO hold bonus - remove free reward
6. NO exit bonus - removed after testing showed EXIT collapse
7. Diversity loss - force minimum 2% action probability
8. Higher entropy end (0.01) - maintain exploration
9. Defensive bonus only when in profit - prevents gaming

Design Goal: Prevent policy collapse to 100% HOLD while keeping counterfactual learning.
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

    # Time management - RESTORED from v1 (was 0.002 in v2)
    # Higher value = more incentive to take action
    time_coef: float = 0.005
    time_sigmoid_center: float = 100 # Bars at which time penalty is 50%

    # ==========================================================================
    # COUNTERFACTUAL REWARDS - BALANCED for v3
    # ==========================================================================
    # Key insight from v2 failure: aggressive coefficients caused HOLD collapse
    # v3 uses asymmetric coefficients to bias toward defensive exits
    #
    # When model exits:
    #   1. Look at what happens AFTER the exit (next N bars) - BOUNDED to 8
    #   2. If price drops below exit_pnl -> defensive bonus (good exit)
    #   3. If price rises above exit_pnl -> opportunity cost (bad exit)
    #
    # Asymmetry: defensive_coef > regret_coef
    # This makes defensive exits more rewarding than premature exits are punishing

    # Opportunity cost: penalize missing future gains
    # Increased from 0.3 to 0.4 to reduce premature exits
    regret_coef: float = 0.4

    # Defensive bonus: reward avoiding future losses
    # REDUCED but kept HIGHER than regret_coef (asymmetric)
    # Ratio 0.5:0.3 = 1.67x bias toward defensive exits
    defensive_coef: float = 0.5

    # How far ahead to look for counterfactual (in bars)
    # REDUCED from 20 to 8 - near-term horizon is achievable
    # 8 bars @ 15M = 2 hours = reasonable defensive window
    counterfactual_window: int = 8

    # Exit cost - REMOVED (was 0.0001)
    # v2's tiny exit cost still penalized all exits
    exit_cost: float = 0.0

    # Exit bonus - REMOVED after testing showed EXIT collapse
    # The defensive bonus is sufficient incentive for good exits
    exit_bonus: float = 0.0

    # Hold bonus - REMOVED (was 0.005 in v2)
    # v2 gave free reward for HOLD, making it dominant
    hold_bonus: float = 0.0

    # ==========================================================================
    # ACTION DIVERSITY - NEW in v3
    # ==========================================================================
    # Force minimum probability for each action to prevent collapse
    min_action_prob: float = 0.02    # 2% minimum per action
    diversity_coef: float = 0.1      # Weight of diversity loss

    # Action costs (prevent excessive adjustments)
    tighten_sl_cost: float = 0.001
    trail_sl_cost: float = 0.0005

    # Reward scaling
    reward_scale: float = 100.0      # Scale factor for stable gradients


@dataclass
class PPOConfig:
    """PPO training configuration for v3_balanced_exits."""

    # Environment
    n_envs: int = 128                # Parallel environments (optimized for GPU)
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
    batch_size: int = 4096           # Minibatch size
    n_steps: int = 2048              # Steps per rollout (before update)
    total_timesteps: int = 10_000_000  # Total training steps

    # Loss weights
    value_coef: float = 0.5          # Value function loss weight
    max_grad_norm: float = 0.5       # Gradient clipping

    # Entropy annealing - HIGHER END for more exploration
    # v2 had 0.005 end, v3 keeps 0.01 to maintain action diversity
    entropy_coef_start: float = 0.05    # Initial entropy bonus
    entropy_coef_end: float = 0.01      # Final entropy (HIGHER than v2's 0.005)
    entropy_anneal_steps: int = 5_000_000  # Faster anneal than v2 (was 7M)

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
