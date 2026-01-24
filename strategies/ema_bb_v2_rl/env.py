"""
Vectorized GPU Environment for RL Exit Optimization.

This environment simulates trade management after entry.
Key design principles:
1. All operations use PyTorch tensors (GPU-accelerated)
2. Pre-computed features eliminate Python bottlenecks
3. Supports 64+ parallel environments for sample efficiency
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

try:
    from .config import PPOConfig, RewardConfig, Actions
except ImportError:
    from config import PPOConfig, RewardConfig, Actions


@dataclass
class TradeEpisode:
    """Pre-computed trade episode for efficient training."""
    entry_bar_idx: int
    entry_price: float
    direction: int  # 1 = long, -1 = short
    entry_atr: float
    entry_adx: float
    entry_rsi: float
    entry_bb_width: float
    entry_ema_diff: float
    ml_confidence: float
    classical_pnl: float
    classical_exit_reason: str
    market_tensor: torch.Tensor  # (max_bars, n_features)
    valid_mask: torch.Tensor     # (max_bars,)
    optimal_bar: int
    optimal_pnl: float


class EpisodeDataset:
    """
    GPU-optimized dataset of pre-computed trade episodes.

    All episodes are stacked into contiguous tensors for efficient
    batched random access during training.
    """

    def __init__(self, episodes: list, device: str = "cuda"):
        self.device = torch.device(device)
        self.n_episodes = len(episodes)

        if self.n_episodes == 0:
            raise ValueError("No episodes provided")

        max_bars = episodes[0].market_tensor.shape[0]
        n_features = episodes[0].market_tensor.shape[1]

        # Stack all episodes
        self.market_tensors = torch.stack(
            [e.market_tensor for e in episodes]
        ).to(self.device)  # (N, max_bars, n_features)

        self.valid_masks = torch.stack(
            [e.valid_mask for e in episodes]
        ).to(self.device)  # (N, max_bars)

        # Entry context features
        self.entry_features = torch.tensor([
            [e.entry_atr, e.entry_adx / 100.0, e.entry_rsi / 100.0,
             e.entry_bb_width, e.entry_ema_diff, e.ml_confidence]
            for e in episodes
        ], dtype=torch.float32, device=self.device)  # (N, 6)

        self.directions = torch.tensor(
            [e.direction for e in episodes],
            dtype=torch.float32, device=self.device
        )  # (N,)

        self.entry_prices = torch.tensor(
            [e.entry_price for e in episodes],
            dtype=torch.float32, device=self.device
        )  # (N,)

        self.entry_atrs = torch.tensor(
            [e.entry_atr for e in episodes],
            dtype=torch.float32, device=self.device
        )  # (N,)

        self.optimal_bars = torch.tensor(
            [e.optimal_bar for e in episodes],
            dtype=torch.long, device=self.device
        )  # (N,)

        self.optimal_pnls = torch.tensor(
            [e.optimal_pnl for e in episodes],
            dtype=torch.float32, device=self.device
        )  # (N,)

        self.classical_pnls = torch.tensor(
            [e.classical_pnl for e in episodes],
            dtype=torch.float32, device=self.device
        )  # (N,)

        self.max_bars = max_bars
        self.n_features = n_features

    def __len__(self):
        return self.n_episodes


class VectorizedExitEnv:
    """
    Vectorized environment for learning exit strategies.

    Each environment instance represents one trade from entry to exit.
    Multiple environments run in parallel for GPU efficiency.

    State Space (25 dimensions):
    - Position features (5): bars_held_norm, unrealized_pnl, mfe, mae, sl_distance
    - Market features (10): from pre-computed market_tensor
    - Entry context (5): atr_norm, adx, rsi, bb_width, ema_diff
    - Action history (5): last 5 actions (normalized)

    Action Space (5 discrete):
    0: HOLD - maintain position
    1: EXIT - close position
    2: TIGHTEN_SL - tighten stop loss
    3: TRAIL_BREAKEVEN - move SL to entry
    4: PARTIAL_EXIT - partial close (treated as EXIT for v1)
    """

    def __init__(
        self,
        dataset: EpisodeDataset,
        config: PPOConfig,
    ):
        self.dataset = dataset
        self.config = config
        self.reward_config = config.reward
        self.n_envs = config.n_envs
        self.max_bars = dataset.max_bars
        self.device = torch.device(config.device)

        # State tracking
        self.episode_indices = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self.current_bar = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self.position_open = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)

        # Dynamic SL tracking (in ATR units)
        self.current_sl_atr = torch.full((self.n_envs,), 1.1, device=self.device)

        # Action history
        self.action_history = torch.zeros(self.n_envs, 5, device=self.device)

        # Track max favorable/adverse excursion
        self.max_favorable = torch.zeros(self.n_envs, device=self.device)
        self.max_adverse = torch.zeros(self.n_envs, device=self.device)

        # Running reward statistics for normalization
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4

        # Episode tracking
        self.episode_returns = torch.zeros(self.n_envs, device=self.device)
        self.episode_lengths = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)

    def reset(self, episode_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reset all environments to new random episodes.

        Returns:
            Initial state tensor of shape (n_envs, state_dim)
        """
        if episode_indices is None:
            episode_indices = torch.randint(
                0, len(self.dataset), (self.n_envs,), device=self.device
            )

        self.episode_indices = episode_indices
        self.current_bar = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        self.position_open = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        self.current_sl_atr = torch.full((self.n_envs,), 1.1, device=self.device)
        self.action_history = torch.zeros(self.n_envs, 5, device=self.device)
        self.max_favorable = torch.zeros(self.n_envs, device=self.device)
        self.max_adverse = torch.zeros(self.n_envs, device=self.device)
        self.episode_returns = torch.zeros(self.n_envs, device=self.device)
        self.episode_lengths = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)

        return self._get_state()

    def _get_state(self) -> torch.Tensor:
        """
        Construct state tensor for all environments.

        Returns:
            State tensor of shape (n_envs, 25)
        """
        # Clamp bar index
        bar_idx = self.current_bar.clamp(max=self.max_bars - 1)

        # Gather market features: market_tensors[episode_idx, bar_idx, :]
        market_features = self.dataset.market_tensors[
            self.episode_indices, bar_idx
        ]  # (n_envs, n_features)

        # Entry context (first 5 features)
        entry_ctx = self.dataset.entry_features[self.episode_indices, :5]  # (n_envs, 5)

        # Position features
        bars_held_norm = self.current_bar.float() / self.max_bars

        # Current PnL is in market_tensor column 0 (normalized)
        unrealized_pnl = market_features[:, 0]

        # Update MFE/MAE
        self.max_favorable = torch.maximum(self.max_favorable, unrealized_pnl)
        self.max_adverse = torch.minimum(self.max_adverse, unrealized_pnl)

        position_features = torch.stack([
            bars_held_norm,
            unrealized_pnl,
            self.max_favorable,
            self.max_adverse,
            self.current_sl_atr / 2.0,  # Normalize SL distance
        ], dim=1)  # (n_envs, 5)

        # Truncate market features to 10 dims
        market_feats = market_features[:, :10]  # (n_envs, 10)

        # Concatenate all
        state = torch.cat([
            position_features,      # 5
            market_feats,           # 10
            entry_ctx,              # 5
            self.action_history,    # 5
        ], dim=1)  # (n_envs, 25)

        return state

    def step(
        self,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Execute actions in all environments.

        Args:
            actions: (n_envs,) tensor of action indices

        Returns:
            next_state: (n_envs, state_dim)
            rewards: (n_envs,)
            dones: (n_envs,) boolean
            info: dict with episode statistics
        """
        # Update action history
        self.action_history = torch.roll(self.action_history, -1, dims=1)
        self.action_history[:, -1] = actions.float() / 4.0

        # Get current market state
        bar_idx = self.current_bar.clamp(max=self.max_bars - 1)
        current_market = self.dataset.market_tensors[self.episode_indices, bar_idx]
        unrealized_pnl = current_market[:, 0]

        # Initialize outputs
        rewards = torch.zeros(self.n_envs, device=self.device)
        dones = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)

        # === Action Processing with Guards ===
        cfg = self.reward_config

        # EXIT action (1) - with optional guards
        exit_mask = (actions == Actions.EXIT) & self.position_open
        if cfg.min_profit_for_exit > 0 or cfg.min_bars_for_exit > 0:
            # Apply guards: must meet profit AND bars requirements
            exit_allowed = (unrealized_pnl >= cfg.min_profit_for_exit) & \
                          (self.current_bar >= cfg.min_bars_for_exit)
            valid_exit = exit_mask & exit_allowed
            invalid_exit = exit_mask & ~exit_allowed

            if valid_exit.any():
                rewards[valid_exit] = unrealized_pnl[valid_exit] * cfg.reward_scale
                dones[valid_exit] = True
                self.position_open[valid_exit] = False

            if invalid_exit.any():
                # Convert to HOLD + penalty
                rewards[invalid_exit] -= cfg.invalid_action_penalty
        else:
            # No guards - allow all exits
            if exit_mask.any():
                rewards[exit_mask] = unrealized_pnl[exit_mask] * cfg.reward_scale
                dones[exit_mask] = True
                self.position_open[exit_mask] = False

        # TIGHTEN_SL action (2) - no guards needed
        tighten_mask = (actions == Actions.TIGHTEN_SL) & self.position_open
        if tighten_mask.any():
            self.current_sl_atr[tighten_mask] *= 0.75
            rewards[tighten_mask] -= cfg.tighten_sl_cost

        # TRAIL_BREAKEVEN action (3) - already has profit guard
        trail_mask = (actions == Actions.TRAIL_BREAKEVEN) & self.position_open
        in_profit = unrealized_pnl > 0
        trail_and_profit = trail_mask & in_profit
        if trail_and_profit.any():
            self.current_sl_atr[trail_and_profit] = 0.1
            rewards[trail_and_profit] -= cfg.trail_sl_cost

        # PARTIAL_EXIT (4) - WITH GUARDS to prevent wasteful immediate exits
        # This is the main fix for the "Bar 0, 0.0 pips" problem
        partial_mask = (actions == Actions.PARTIAL_EXIT) & self.position_open

        # Guards: must have minimum profit AND minimum bars held
        partial_allowed = (unrealized_pnl >= cfg.min_profit_for_partial) & \
                         (self.current_bar >= cfg.min_bars_for_partial)

        valid_partial = partial_mask & partial_allowed
        invalid_partial = partial_mask & ~partial_allowed

        if valid_partial.any():
            # Allow the partial exit
            rewards[valid_partial] = unrealized_pnl[valid_partial] * cfg.reward_scale
            dones[valid_partial] = True
            self.position_open[valid_partial] = False

        if invalid_partial.any():
            # Block the partial - convert to HOLD with penalty
            # This teaches the model that attempting PARTIAL when not profitable is bad
            rewards[invalid_partial] -= cfg.invalid_action_penalty

        # === Advance Time ===
        self.current_bar += 1
        self.episode_lengths += 1

        # === Check SL Hit ===
        next_bar = self.current_bar.clamp(max=self.max_bars - 1)
        next_market = self.dataset.market_tensors[self.episode_indices, next_bar]

        # MAE is in column 2 (or compute from column 0)
        next_pnl = next_market[:, 0]

        # SL hit if adverse move exceeds SL distance
        entry_atrs = self.dataset.entry_atrs[self.episode_indices]
        entry_prices = self.dataset.entry_prices[self.episode_indices]

        # Convert SL distance to normalized PnL units
        sl_pnl_threshold = -self.current_sl_atr * entry_atrs / entry_prices
        sl_hit = (next_pnl < sl_pnl_threshold) & self.position_open & ~dones

        if sl_hit.any():
            # Exit at SL
            rewards[sl_hit] = sl_pnl_threshold[sl_hit] * self.reward_config.reward_scale
            dones[sl_hit] = True
            self.position_open[sl_hit] = False

        # === Check Episode End ===
        valid_mask = self.dataset.valid_masks[self.episode_indices, next_bar]
        episode_end = ~valid_mask & self.position_open & ~dones

        if episode_end.any():
            # Force exit at end
            rewards[episode_end] = next_pnl[episode_end] * self.reward_config.reward_scale
            dones[episode_end] = True
            self.position_open[episode_end] = False

        # === Apply Reward Shaping ===
        rewards = self._shape_reward(rewards, actions, unrealized_pnl, dones)

        # Track episode returns
        self.episode_returns += rewards

        # Build info dict
        info = {
            'episode_returns': [],
            'episode_lengths': [],
        }

        # Collect completed episode stats
        if dones.any():
            completed_returns = self.episode_returns[dones].cpu().numpy().tolist()
            completed_lengths = self.episode_lengths[dones].cpu().numpy().tolist()
            info['episode_returns'] = completed_returns
            info['episode_lengths'] = completed_lengths

            # Auto-reset completed environments
            reset_indices = dones.nonzero(as_tuple=True)[0]
            new_episodes = torch.randint(0, len(self.dataset), (len(reset_indices),), device=self.device)
            self.episode_indices[reset_indices] = new_episodes
            self.current_bar[reset_indices] = 0
            self.position_open[reset_indices] = True
            self.current_sl_atr[reset_indices] = 1.1
            self.action_history[reset_indices] = 0
            self.max_favorable[reset_indices] = 0
            self.max_adverse[reset_indices] = 0
            self.episode_returns[reset_indices] = 0
            self.episode_lengths[reset_indices] = 0

        next_state = self._get_state()

        return next_state, rewards, dones, info

    def _shape_reward(
        self,
        base_reward: torch.Tensor,
        actions: torch.Tensor,
        unrealized_pnl: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply reward shaping components.

        R = R_base + R_sharpe + R_regret + R_time
        """
        rewards = base_reward.clone()
        cfg = self.reward_config

        # === Differential Sharpe (for non-terminal states) ===
        if self.reward_count > 100:
            sharpe_bonus = (unrealized_pnl - self.reward_mean) / (np.sqrt(self.reward_var) + 1e-8)
            rewards += cfg.w_mtm * sharpe_bonus * (~dones).float()

        # === Regret Penalty (for exits) ===
        if dones.any():
            optimal_pnls = self.dataset.optimal_pnls[self.episode_indices[dones]]
            realized_pnls = unrealized_pnl[dones]
            regret = (optimal_pnls - realized_pnls).clamp(min=0)
            rewards[dones] -= cfg.regret_coef * regret

            # Update reward statistics
            new_returns = base_reward[dones]
            n = len(new_returns)
            self.reward_count += n
            delta = new_returns.mean().item() - self.reward_mean
            self.reward_mean += delta * n / self.reward_count
            delta2 = new_returns.mean().item() - self.reward_mean
            self.reward_var += (delta * delta2)

        # === Time Decay (for holding) ===
        bars = self.current_bar.float()
        time_penalty = 1.0 / (1.0 + torch.exp(-0.05 * (bars - cfg.time_sigmoid_center)))
        rewards -= cfg.time_coef * time_penalty * (~dones).float()

        # === Drawdown Penalty ===
        in_drawdown = (self.max_favorable - unrealized_pnl) > cfg.dd_threshold
        dd_penalty = cfg.risk_coef * (self.max_favorable - unrealized_pnl) * in_drawdown.float()
        rewards -= dd_penalty * (~dones).float()

        return rewards
