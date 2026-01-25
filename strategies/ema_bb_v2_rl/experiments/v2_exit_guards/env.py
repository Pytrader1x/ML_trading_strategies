"""
Vectorized GPU Environment for v2_exit_guards experiment.

KEY INNOVATION: Counterfactual rewards instead of hard guards.

When the model exits, we look at what happens AFTER:
- If price drops below exit_pnl → defensive bonus (good exit, avoided loss)
- If price rises above exit_pnl → opportunity cost (bad exit, missed gain)

This naturally teaches the model WHEN to exit without blocking actions.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Import from THIS experiment's frozen config
EXPERIMENT_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENT_DIR))
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
    """GPU-optimized dataset of pre-computed trade episodes."""

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
        ).to(self.device)

        self.valid_masks = torch.stack(
            [e.valid_mask for e in episodes]
        ).to(self.device)

        self.entry_features = torch.tensor([
            [e.entry_atr, e.entry_adx / 100.0, e.entry_rsi / 100.0,
             e.entry_bb_width, e.entry_ema_diff, e.ml_confidence]
            for e in episodes
        ], dtype=torch.float32, device=self.device)

        self.directions = torch.tensor(
            [e.direction for e in episodes],
            dtype=torch.float32, device=self.device
        )

        self.entry_prices = torch.tensor(
            [e.entry_price for e in episodes],
            dtype=torch.float32, device=self.device
        )

        self.entry_atrs = torch.tensor(
            [e.entry_atr for e in episodes],
            dtype=torch.float32, device=self.device
        )

        self.optimal_bars = torch.tensor(
            [e.optimal_bar for e in episodes],
            dtype=torch.long, device=self.device
        )

        self.optimal_pnls = torch.tensor(
            [e.optimal_pnl for e in episodes],
            dtype=torch.float32, device=self.device
        )

        self.classical_pnls = torch.tensor(
            [e.classical_pnl for e in episodes],
            dtype=torch.float32, device=self.device
        )

        self.max_bars = max_bars
        self.n_features = n_features

    def __len__(self):
        return self.n_episodes


class VectorizedExitEnv:
    """
    Vectorized environment for v2_exit_guards with counterfactual rewards.

    KEY DIFFERENCE from v1:
    - NO hard guards on EXIT or PARTIAL
    - Counterfactual rewards teach when exits are good/bad
    - Tiny exit cost discourages truly wasteful exits
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

        # Dynamic SL tracking
        self.current_sl_atr = torch.full((self.n_envs,), 1.1, device=self.device)

        # Action history
        self.action_history = torch.zeros(self.n_envs, 5, device=self.device)

        # Track max favorable/adverse excursion
        self.max_favorable = torch.zeros(self.n_envs, device=self.device)
        self.max_adverse = torch.zeros(self.n_envs, device=self.device)

        # Running reward statistics
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 1e-4

        # Episode tracking
        self.episode_returns = torch.zeros(self.n_envs, device=self.device)
        self.episode_lengths = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)

        # Stats for monitoring counterfactual rewards
        self.total_exits = 0
        self.defensive_exits = 0  # Exits where we avoided future loss
        self.premature_exits = 0  # Exits where we missed future gain
        self.total_defensive_bonus = 0.0
        self.total_opportunity_cost = 0.0

    def reset(self, episode_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset all environments to new random episodes."""
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
        """Construct state tensor for all environments."""
        bar_idx = self.current_bar.clamp(max=self.max_bars - 1)

        market_features = self.dataset.market_tensors[
            self.episode_indices, bar_idx
        ]

        entry_ctx = self.dataset.entry_features[self.episode_indices, :5]

        bars_held_norm = self.current_bar.float() / self.max_bars
        unrealized_pnl = market_features[:, 0]

        self.max_favorable = torch.maximum(self.max_favorable, unrealized_pnl)
        self.max_adverse = torch.minimum(self.max_adverse, unrealized_pnl)

        position_features = torch.stack([
            bars_held_norm,
            unrealized_pnl,
            self.max_favorable,
            self.max_adverse,
            self.current_sl_atr / 2.0,
        ], dim=1)

        market_feats = market_features[:, :10]

        state = torch.cat([
            position_features,
            market_feats,
            entry_ctx,
            self.action_history,
        ], dim=1)

        return state

    def _compute_counterfactual_reward(
        self,
        exit_pnl: torch.Tensor,
        episode_indices: torch.Tensor,
        current_bar: torch.Tensor,
        exit_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute counterfactual rewards for exits.

        For each exit:
        1. Look at next N bars (counterfactual_lookforward)
        2. Compute defensive bonus if we avoided future loss
        3. Compute opportunity cost if we missed future gain
        4. Apply tiny exit cost

        Returns:
            Additional reward adjustment for exiting environments
        """
        cfg = self.reward_config
        lookforward = cfg.counterfactual_lookforward

        # Initialize reward adjustment
        counterfactual_reward = torch.zeros(self.n_envs, device=self.device)

        if not exit_mask.any():
            return counterfactual_reward

        # Get indices of exiting environments
        exit_indices = exit_mask.nonzero(as_tuple=True)[0]

        for idx in exit_indices:
            ep_idx = episode_indices[idx].item()
            bar = current_bar[idx].item()
            pnl = exit_pnl[idx].item()

            # Get future PnLs (what would have happened if we held)
            end_bar = min(bar + lookforward, self.max_bars)

            if end_bar > bar:
                future_pnls = self.dataset.market_tensors[ep_idx, bar:end_bar, 0]

                if len(future_pnls) > 0:
                    future_worst = future_pnls.min().item()
                    future_best = future_pnls.max().item()

                    # Defensive bonus: reward if we avoided a future drawdown
                    avoided_loss = max(0, pnl - future_worst)
                    defensive_bonus = avoided_loss * cfg.defensive_coef

                    # Opportunity cost: penalize if we missed future gains
                    missed_gain = max(0, future_best - pnl)
                    opportunity_cost = missed_gain * cfg.regret_coef

                    # Net counterfactual adjustment
                    counterfactual_reward[idx] = defensive_bonus - opportunity_cost

                    # Track statistics
                    self.total_exits += 1
                    if avoided_loss > 0:
                        self.defensive_exits += 1
                    if missed_gain > 0:
                        self.premature_exits += 1
                    self.total_defensive_bonus += defensive_bonus
                    self.total_opportunity_cost += opportunity_cost

            # Apply tiny exit cost to all exits
            counterfactual_reward[idx] -= cfg.exit_cost

        return counterfactual_reward

    def step(
        self,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Execute actions with counterfactual rewards (no hard guards)."""

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

        cfg = self.reward_config

        # ================================================================
        # EXIT action (1) - NO GUARDS, uses counterfactual rewards
        # ================================================================
        exit_mask = (actions == Actions.EXIT) & self.position_open

        if exit_mask.any():
            # Base reward: realized PnL
            rewards[exit_mask] = unrealized_pnl[exit_mask] * cfg.reward_scale

            # Counterfactual adjustment (defensive bonus - opportunity cost - exit cost)
            cf_reward = self._compute_counterfactual_reward(
                unrealized_pnl, self.episode_indices, self.current_bar, exit_mask
            )
            rewards += cf_reward

            dones[exit_mask] = True
            self.position_open[exit_mask] = False

        # ================================================================
        # TIGHTEN_SL action (2)
        # ================================================================
        tighten_mask = (actions == Actions.TIGHTEN_SL) & self.position_open
        if tighten_mask.any():
            self.current_sl_atr[tighten_mask] *= 0.75
            rewards[tighten_mask] -= cfg.tighten_sl_cost

        # ================================================================
        # TRAIL_BREAKEVEN action (3)
        # ================================================================
        trail_mask = (actions == Actions.TRAIL_BREAKEVEN) & self.position_open
        in_profit = unrealized_pnl > 0
        trail_and_profit = trail_mask & in_profit
        if trail_and_profit.any():
            self.current_sl_atr[trail_and_profit] = 0.1
            rewards[trail_and_profit] -= cfg.trail_sl_cost

        # ================================================================
        # PARTIAL_EXIT (4) - NO GUARDS, uses counterfactual rewards
        # ================================================================
        partial_mask = (actions == Actions.PARTIAL_EXIT) & self.position_open

        if partial_mask.any():
            # Base reward: realized PnL
            rewards[partial_mask] = unrealized_pnl[partial_mask] * cfg.reward_scale

            # Counterfactual adjustment
            cf_reward = self._compute_counterfactual_reward(
                unrealized_pnl, self.episode_indices, self.current_bar, partial_mask
            )
            rewards += cf_reward

            dones[partial_mask] = True
            self.position_open[partial_mask] = False

        # Advance Time
        self.current_bar += 1
        self.episode_lengths += 1

        # Check SL Hit
        next_bar = self.current_bar.clamp(max=self.max_bars - 1)
        next_market = self.dataset.market_tensors[self.episode_indices, next_bar]
        next_pnl = next_market[:, 0]

        entry_atrs = self.dataset.entry_atrs[self.episode_indices]
        entry_prices = self.dataset.entry_prices[self.episode_indices]

        sl_pnl_threshold = -self.current_sl_atr * entry_atrs / entry_prices
        sl_hit = (next_pnl < sl_pnl_threshold) & self.position_open & ~dones

        if sl_hit.any():
            rewards[sl_hit] = sl_pnl_threshold[sl_hit] * cfg.reward_scale
            dones[sl_hit] = True
            self.position_open[sl_hit] = False

        # Check Episode End
        valid_mask = self.dataset.valid_masks[self.episode_indices, next_bar]
        episode_end = ~valid_mask & self.position_open & ~dones

        if episode_end.any():
            rewards[episode_end] = next_pnl[episode_end] * cfg.reward_scale
            dones[episode_end] = True
            self.position_open[episode_end] = False

        # Apply standard reward shaping (time penalty, drawdown penalty)
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
        """Apply standard reward shaping (time penalty, drawdown penalty)."""
        rewards = base_reward.clone()
        cfg = self.reward_config

        # Differential Sharpe (for non-terminal states)
        if self.reward_count > 100:
            sharpe_bonus = (unrealized_pnl - self.reward_mean) / (np.sqrt(self.reward_var) + 1e-8)
            rewards += cfg.w_mtm * sharpe_bonus * (~dones).float()

        # Update reward statistics on exits
        if dones.any():
            new_returns = base_reward[dones]
            n = len(new_returns)
            self.reward_count += n
            delta = new_returns.mean().item() - self.reward_mean
            self.reward_mean += delta * n / self.reward_count
            delta2 = new_returns.mean().item() - self.reward_mean
            self.reward_var += (delta * delta2)

        # Time Decay (REDUCED in v2: 0.002 vs 0.005)
        bars = self.current_bar.float()
        time_penalty = 1.0 / (1.0 + torch.exp(-0.05 * (bars - cfg.time_sigmoid_center)))
        rewards -= cfg.time_coef * time_penalty * (~dones).float()

        # Drawdown Penalty
        in_drawdown = (self.max_favorable - unrealized_pnl) > cfg.dd_threshold
        dd_penalty = cfg.risk_coef * (self.max_favorable - unrealized_pnl) * in_drawdown.float()
        rewards -= dd_penalty * (~dones).float()

        return rewards

    def get_counterfactual_stats(self) -> Dict[str, float]:
        """Get statistics on counterfactual rewards."""
        if self.total_exits == 0:
            return {
                'defensive_rate': 0.0,
                'premature_rate': 0.0,
                'avg_defensive_bonus': 0.0,
                'avg_opportunity_cost': 0.0,
                'total_exits': 0,
            }

        return {
            'defensive_rate': self.defensive_exits / self.total_exits,
            'premature_rate': self.premature_exits / self.total_exits,
            'avg_defensive_bonus': self.total_defensive_bonus / self.total_exits,
            'avg_opportunity_cost': self.total_opportunity_cost / self.total_exits,
            'total_exits': self.total_exits,
        }
