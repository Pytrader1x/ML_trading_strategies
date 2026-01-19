"""
Layer 3a: VectorizedMarket - GPU-Native Trading Environment

This module implements a batched trading environment that runs entirely on GPU.
Key design principles:
1. ALL tensors reside on GPU - zero CPU transfers during stepping
2. Branchless logic using boolean masks (no if/else)
3. Pre-allocated tensors - no dynamic memory allocation
4. Parallel simulation of 4096+ environments simultaneously

State Space (10 features):
- 7 market features (ADX, RSI, ATR_pct, BB_Width, EMA_diff, dist_fast, dist_slow)
- Meta-label probability
- Current position (-1, 0, 1)
- Unrealized P&L (normalized by ATR)

Action Space (discrete, 4 actions):
- 0: Hold (no action)
- 1: Long (enter/maintain long position)
- 2: Short (enter/maintain short position)
- 3: Close (close current position)
"""

import torch
from typing import Dict, Tuple, Optional

from .config import RLConfig


class VectorizedMarket:
    """
    Batched trading environment running entirely on GPU.

    This environment simulates thousands of trading sessions in parallel.
    Each environment has its own position in the historical data, allowing
    for de-correlated learning across different market regimes.
    """

    def __init__(
        self,
        features: torch.Tensor,      # (T, F) all market features
        prices: torch.Tensor,        # (T, 4) OHLC prices
        meta_labels: torch.Tensor,   # (T,) meta-label probabilities
        base_signals: torch.Tensor,  # (T,) candidate entry signals
        atr: torch.Tensor,           # (T,) ATR for reward scaling
        config: RLConfig,
    ):
        """
        Initialize the vectorized environment.

        Args:
            features: Market features tensor (T, num_features)
            prices: OHLC price tensor (T, 4)
            meta_labels: Meta-label scores (T,)
            base_signals: Candidate signals (T,)
            atr: ATR values for normalization (T,)
            config: RL configuration
        """
        self.device = torch.device(config.device)
        self.num_envs = config.num_envs
        self.episode_length = config.episode_length
        self.config = config

        # Store data on GPU (already should be there from data_factory)
        self.features = features.to(self.device)
        self.prices = prices.to(self.device)
        self.meta_labels = meta_labels.to(self.device)
        self.base_signals = base_signals.to(self.device)
        self.atr = atr.to(self.device)

        # Data dimensions
        self.T = features.shape[0]  # Total timesteps
        self.F = features.shape[1]  # Feature dimension
        self.obs_dim = self.F + 3   # features + meta_label + position + unrealized_pnl

        # Validate data
        assert self.T > self.episode_length + 100, \
            f"Data too short: {self.T} bars, need at least {self.episode_length + 100}"

        # Pre-allocate all state tensors
        self._allocate_state_tensors()

        print(f"VectorizedMarket initialized:")
        print(f"  Data: {self.T:,} timesteps, {self.F} features")
        print(f"  Envs: {self.num_envs:,} parallel environments")
        print(f"  Episode: {self.episode_length} steps")
        print(f"  Device: {self.device}")

    def _allocate_state_tensors(self):
        """Pre-allocate all state tensors to avoid dynamic allocation."""
        N = self.num_envs

        # Current indices into data (one per environment)
        self.indices = torch.zeros(N, dtype=torch.long, device=self.device)

        # Episode step counter
        self.steps = torch.zeros(N, dtype=torch.long, device=self.device)

        # Trading state
        # Position: -1 (short), 0 (flat), 1 (long)
        self.positions = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.entry_prices = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.entry_indices = torch.zeros(N, dtype=torch.long, device=self.device)

        # Cumulative P&L tracking
        self.episode_pnl = torch.zeros(N, dtype=torch.float32, device=self.device)
        self.total_trades = torch.zeros(N, dtype=torch.long, device=self.device)

        # Pre-allocated observation buffer
        self.obs_buffer = torch.zeros(N, self.obs_dim, dtype=torch.float32, device=self.device)

    def reset(
        self,
        env_mask: Optional[torch.Tensor] = None,
        deterministic_start: Optional[int] = None
    ) -> torch.Tensor:
        """
        Reset environments.

        Args:
            env_mask: If provided, only reset environments where mask is True
            deterministic_start: If provided, all envs start at this index (for eval)

        Returns:
            observations: (num_envs, obs_dim)
        """
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        num_reset = env_mask.sum().item()

        # Random or deterministic starting indices
        max_start = self.T - self.episode_length - 10
        if deterministic_start is not None:
            new_indices = torch.full((num_reset,), deterministic_start, dtype=torch.long, device=self.device)
        else:
            new_indices = torch.randint(0, max_start, (num_reset,), device=self.device)

        # Apply resets using mask (branchless)
        self.indices[env_mask] = new_indices
        self.steps[env_mask] = 0
        self.positions[env_mask] = 0.0
        self.entry_prices[env_mask] = 0.0
        self.entry_indices[env_mask] = 0
        self.episode_pnl[env_mask] = 0.0
        self.total_trades[env_mask] = 0

        return self._get_observations()

    def _get_observations(self) -> torch.Tensor:
        """
        Build observation tensor for all environments (branchless).

        Observation = [market_features(7), meta_label(1), position(1), unrealized_pnl(1)]
        """
        # Gather market features at current indices
        # Advanced indexing: features[indices] gives (num_envs, F)
        batch_features = self.features[self.indices]

        # Meta-label at current position
        batch_meta = self.meta_labels[self.indices].unsqueeze(-1)

        # Position encoding (normalized to [-1, 1])
        position_enc = self.positions.unsqueeze(-1)

        # Unrealized P&L (branchless calculation)
        current_close = self.prices[self.indices, 3]  # Close price
        price_diff = current_close - self.entry_prices

        # P&L depends on direction: long gains from price up, short gains from price down
        unrealized_pnl_raw = price_diff * self.positions

        # Normalize by ATR (with safety epsilon)
        current_atr = self.atr[self.indices]
        unrealized_pnl_norm = unrealized_pnl_raw / (current_atr + 1e-8)

        # Clamp to reasonable range
        unrealized_pnl_norm = torch.clamp(unrealized_pnl_norm, -10, 10).unsqueeze(-1)

        # Concatenate all observations
        self.obs_buffer[:, :self.F] = batch_features
        self.obs_buffer[:, self.F] = batch_meta.squeeze(-1)
        self.obs_buffer[:, self.F + 1] = position_enc.squeeze(-1)
        self.obs_buffer[:, self.F + 2] = unrealized_pnl_norm.squeeze(-1)

        return self.obs_buffer.clone()

    def step(
        self,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Execute actions across all environments simultaneously.

        Actions:
        - 0: Hold (do nothing)
        - 1: Long (enter long or maintain)
        - 2: Short (enter short or maintain)
        - 3: Close (close current position)

        All logic is branchless using boolean masks for GPU efficiency.

        Returns:
            observations: (num_envs, obs_dim)
            rewards: (num_envs,)
            dones: (num_envs,) boolean
            infos: dict with additional data
        """
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Current market data
        current_close = self.prices[self.indices, 3]
        current_high = self.prices[self.indices, 1]
        current_low = self.prices[self.indices, 2]
        current_atr = self.atr[self.indices]
        current_meta = self.meta_labels[self.indices]
        current_signal = self.base_signals[self.indices]

        # =====================================================================
        # CLOSE POSITION LOGIC
        # Close on:
        # 1. Action = 3 (explicit close)
        # 2. Action = 1 (long) while in short position (reverse)
        # 3. Action = 2 (short) while in long position (reverse)
        # =====================================================================
        close_mask = (
            (actions == 3) |
            ((actions == 1) & (self.positions < 0)) |
            ((actions == 2) & (self.positions > 0))
        )
        close_mask = close_mask & (self.positions != 0)

        # Calculate realized P&L for closing positions
        close_pnl_raw = (current_close - self.entry_prices) * self.positions
        close_pnl_norm = close_pnl_raw / (current_atr + 1e-8)

        # Apply close rewards (only where close_mask is True)
        rewards = torch.where(close_mask, close_pnl_norm, rewards)

        # Reset position state where closed
        self.positions = torch.where(close_mask, torch.zeros_like(self.positions), self.positions)
        self.entry_prices = torch.where(close_mask, torch.zeros_like(self.entry_prices), self.entry_prices)

        # =====================================================================
        # OPEN POSITION LOGIC
        # =====================================================================
        # Long entry: action = 1 AND currently flat
        long_entry_mask = (actions == 1) & (self.positions == 0)

        # Short entry: action = 2 AND currently flat
        short_entry_mask = (actions == 2) & (self.positions == 0)

        entry_mask = long_entry_mask | short_entry_mask

        # Update positions
        self.positions = torch.where(long_entry_mask, torch.ones_like(self.positions), self.positions)
        self.positions = torch.where(short_entry_mask, -torch.ones_like(self.positions), self.positions)

        # Record entry prices
        self.entry_prices = torch.where(entry_mask, current_close, self.entry_prices)
        self.entry_indices = torch.where(entry_mask, self.indices, self.entry_indices)

        # Increment trade counter
        self.total_trades = torch.where(entry_mask, self.total_trades + 1, self.total_trades)

        # =====================================================================
        # REWARD SHAPING: META-LABEL ALIGNMENT
        # =====================================================================
        # Calculate signal alignment: does our position match the candidate signal?
        signal_alignment = self.positions * current_signal  # -1, 0, or 1

        # Bonus for entering when meta-label is high AND signal aligns
        good_entry_bonus = (
            entry_mask.float() *
            (current_meta > 0.5).float() *
            (signal_alignment > 0).float() *
            self.config.meta_label_reward_scale
        )
        rewards = rewards + good_entry_bonus

        # Penalty for entering when meta-label is low (bad trade prediction)
        bad_entry_penalty = (
            entry_mask.float() *
            (current_meta < 0.5).float() *
            self.config.bad_entry_penalty
        )
        rewards = rewards - bad_entry_penalty

        # Small holding cost to encourage decisive actions
        holding_cost = (
            (self.positions != 0).float() *
            self.config.holding_cost
        )
        rewards = rewards - holding_cost

        # =====================================================================
        # ADVANCE TIME
        # =====================================================================
        self.indices = self.indices + 1
        self.steps = self.steps + 1

        # Track cumulative episode P&L
        self.episode_pnl = self.episode_pnl + rewards

        # =====================================================================
        # EPISODE TERMINATION
        # =====================================================================
        # Episode ends after episode_length steps
        dones = self.steps >= self.episode_length

        # Force close any open positions at episode end
        force_close_mask = dones & (self.positions != 0)
        force_close_pnl = (current_close - self.entry_prices) * self.positions
        force_close_pnl_norm = force_close_pnl / (current_atr + 1e-8)
        rewards = torch.where(force_close_mask, rewards + force_close_pnl_norm, rewards)

        # Check for data boundary (safety)
        data_boundary = self.indices >= (self.T - 1)
        dones = dones | data_boundary

        # Get next observations
        observations = self._get_observations()

        # Build info dict
        infos = {
            "episode_pnl": self.episode_pnl.clone(),
            "positions": self.positions.clone(),
            "total_trades": self.total_trades.clone(),
            "steps": self.steps.clone(),
        }

        return observations, rewards, dones, infos

    def get_env_info(self) -> Dict:
        """Return environment specifications."""
        return {
            "obs_dim": self.obs_dim,
            "action_dim": 4,
            "num_envs": self.num_envs,
            "episode_length": self.episode_length,
            "data_length": self.T,
            "device": str(self.device),
        }


class VectorizedMarketWrapper:
    """
    Wrapper that handles auto-reset and provides a gym-like interface.
    """

    def __init__(self, env: VectorizedMarket):
        self.env = env

    def reset(self) -> torch.Tensor:
        """Reset all environments."""
        return self.env.reset()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Step with auto-reset for done environments."""
        obs, rewards, dones, infos = self.env.step(actions)

        # Auto-reset done environments
        if dones.any():
            obs = self.env.reset(env_mask=dones)

        return obs, rewards, dones, infos
