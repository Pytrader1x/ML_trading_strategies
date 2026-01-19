"""
KNN State-Space Strategy - True Implementation

Non-parametric conditional return estimator using k-nearest neighbors in state space.

Core Concept:
- Define market state as normalized feature vector (volatility, momentum, trend)
- Find k nearest historical states using Euclidean distance
- Estimate conditional forward return distribution from neighbors
- Trade only when distribution shows consistent, large directional bias

Key Filters:
1. Magnitude: |mean_return| >= threshold
2. Consistency: P(R > 0) >= p_min (or <= 1-p_min for shorts)
3. Confidence: |mean_return| / std_return >= c

Uses MPS acceleration on Mac for fast distance computation.
"""

import pandas as pd
import numpy as np
import torch
from backtest_engine import Strategy


def get_device():
    """Get best available device (MPS for Mac, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class KNNStateSpace(Strategy):
    """
    KNN State-Space Strategy.

    Trades when current market state is similar to historical states
    that produced consistent directional returns.
    """

    def init(self):
        """Initialize indicators and pre-compute state matrix."""
        # State vector parameters
        self.vol_period = self.params.get('vol_period', 20)
        self.mom_short = self.params.get('mom_short', 5)
        self.mom_long = self.params.get('mom_long', 20)
        self.trend_period = self.params.get('trend_period', 50)
        self.rsi_period = self.params.get('rsi_period', 14)

        # Normalization window
        self.norm_window = self.params.get('norm_window', 200)

        # KNN parameters
        self.k = self.params.get('k', 50)  # Neighbors
        self.lookback = self.params.get('lookback', 3000)  # History window
        self.horizon = self.params.get('horizon', 8)  # Forward horizon

        # Signal filters - balanced
        self.mag_thresh = self.params.get('mag_thresh', 0.002)  # Expected move threshold
        self.consistency_thresh = self.params.get('consistency_thresh', 0.65)  # Directional agreement
        self.confidence_thresh = self.params.get('confidence_thresh', 0.3)  # Signal-to-noise
        self.density_thresh = self.params.get('density_thresh', 5.0)  # Density (disabled effectively)

        # Risk parameters
        self.atr_period = self.params.get('atr_period', 14)
        self.sl_atr = self.params.get('sl_atr', 2.0)
        self.tp_atr = self.params.get('tp_atr', 3.0)
        self.base_size = self.params.get('base_size', 1_000_000)

        # Cooldown
        self.cooldown = self.params.get('cooldown', 3)
        self.max_hold = self.params.get('max_hold', 20)
        self._last_exit = -100
        self._entry_bar = None

        # Device for MPS/CUDA acceleration
        self.device = get_device()
        print(f"KNN State-Space initialized on {self.device}")

        # Calculate raw features
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']

        # 1. Volatility: rolling std of log returns
        log_ret = np.log(close / close.shift(1))
        self.volatility = log_ret.rolling(self.vol_period).std()

        # 2. Momentum short
        self.mom_s = np.log(close / close.shift(self.mom_short))

        # 3. Momentum long
        self.mom_l = np.log(close / close.shift(self.mom_long))

        # 4. Trend: distance from SMA as % of price
        sma = close.rolling(self.trend_period).mean()
        self.trend = (close - sma) / sma

        # 5. RSI
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        self.rsi = (100 - 100 / (1 + rs)) / 100 - 0.5  # Center around 0

        # ATR for stops
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.atr = tr.rolling(self.atr_period).mean()

        # Z-score normalize each feature over rolling window
        self.vol_z = self._rolling_zscore(self.volatility)
        self.mom_s_z = self._rolling_zscore(self.mom_s)
        self.mom_l_z = self._rolling_zscore(self.mom_l)
        self.trend_z = self._rolling_zscore(self.trend)
        self.rsi_z = self._rolling_zscore(self.rsi)

        # Pre-compute forward returns
        self.fwd_returns = np.log(close.shift(-self.horizon) / close)

        # Build state matrix (N x d) - simplified to 3 key features
        # Volatility + Momentum + Trend captures most market state information
        self.state_matrix = np.column_stack([
            self.vol_z.values,
            self.mom_l_z.values,  # Longer momentum is more stable
            self.trend_z.values,
        ])

        # Warmup period
        self.warmup = max(self.norm_window, self.trend_period, self.lookback) + 50

        print(f"State matrix shape: {self.state_matrix.shape}")
        print(f"Features: volatility, momentum_long, trend (3D)")
        print(f"k={self.k}, lookback={self.lookback}, horizon={self.horizon}")

    def _rolling_zscore(self, series: pd.Series) -> pd.Series:
        """Compute rolling z-score for regime invariance."""
        mean = series.rolling(self.norm_window).mean()
        std = series.rolling(self.norm_window).std()
        return (series - mean) / (std + 1e-10)

    def _find_knn_signal(self, i: int) -> tuple:
        """
        Find k nearest neighbors and compute weighted conditional return stats.

        Returns: (mean_return, std_return, consistency, density, n_valid)
        """
        # Current state vector
        current_state = self.state_matrix[i]

        if not np.isfinite(current_state).all():
            return 0, 0, 0.5, 999, 0

        # Historical window: from (i - lookback) to (i - horizon - 1)
        start_idx = max(self.warmup, i - self.lookback)
        end_idx = i - self.horizon - 1

        if end_idx <= start_idx:
            return 0, 0, 0.5, 999, 0

        # Get historical states and their forward returns
        hist_states = self.state_matrix[start_idx:end_idx]
        hist_returns = self.fwd_returns.iloc[start_idx:end_idx].values

        # Filter out invalid states
        valid_mask = np.isfinite(hist_states).all(axis=1) & np.isfinite(hist_returns)
        hist_states = hist_states[valid_mask]
        hist_returns = hist_returns[valid_mask]

        if len(hist_states) < self.k:
            return 0, 0, 0.5, 999, 0

        # Compute distances using MPS/CUDA acceleration
        with torch.no_grad():
            current_t = torch.tensor(current_state, dtype=torch.float32, device=self.device)
            hist_t = torch.tensor(hist_states, dtype=torch.float32, device=self.device)

            # Euclidean distance
            diff = hist_t - current_t
            distances = torch.sqrt((diff ** 2).sum(dim=1))

            # Get k nearest indices and distances
            top_distances, indices = torch.topk(distances, self.k, largest=False)
            indices = indices.cpu().numpy()
            top_distances = top_distances.cpu().numpy()

        # Density metric: median distance to k neighbors (smaller = denser)
        density = np.median(top_distances)

        # Get returns of k nearest neighbors
        neighbor_returns = hist_returns[indices]
        neighbor_distances = top_distances

        # Distance-weighted mean (closer neighbors weighted more)
        # Use inverse distance weighting with small epsilon
        weights = 1.0 / (neighbor_distances + 0.1)
        weights = weights / weights.sum()

        mean_ret = np.sum(weights * neighbor_returns)
        std_ret = np.sqrt(np.sum(weights * (neighbor_returns - mean_ret) ** 2)) + 1e-10

        # Directional consistency (unweighted for robustness)
        # For contrarian: we want consistency in the direction we're fading
        if mean_ret > 0:
            consistency = np.mean(neighbor_returns > 0)  # Most went up (we'll short)
        else:
            consistency = np.mean(neighbor_returns < 0)  # Most went down (we'll buy)

        return mean_ret, std_ret, consistency, density, len(neighbor_returns)

    def next(self, i: int, record: pd.Series):
        """Process each bar."""
        if i < self.warmup:
            return

        close = record['Close']
        atr = self.atr.iloc[i]

        if pd.isna(atr) or atr <= 0:
            return

        # Manage active trade
        if self.broker.active_trade:
            if self._entry_bar is not None and (i - self._entry_bar) >= self.max_hold:
                self.broker.close(i, close, reason="MaxHold")
                self._reset(i)
            return

        # Cooldown
        if i - self._last_exit < self.cooldown:
            return

        # Get KNN signal
        mean_ret, std_ret, consistency, density, n_valid = self._find_knn_signal(i)

        if n_valid < self.k:
            return

        # Filter 1: Density (only trade when in historically dense region)
        if density > self.density_thresh:
            return

        # Filter 2: Magnitude
        if abs(mean_ret) < self.mag_thresh:
            return

        # Filter 3: Consistency
        if consistency < self.consistency_thresh:
            return

        # Filter 4: Confidence (signal to noise)
        confidence = abs(mean_ret) / std_ret
        if confidence < self.confidence_thresh:
            return

        # Position sizing: inverse volatility
        vol = self.volatility.iloc[i]
        if pd.isna(vol) or vol <= 0:
            vol = 0.01
        size = self.base_size * (0.01 / vol)  # Normalize to 1% baseline vol
        size = min(size, self.base_size * 3)  # Cap at 3x base

        # Calculate stops
        sl_dist = atr * self.sl_atr
        tp_dist = atr * self.tp_atr

        # Execute trade - CONTRARIAN: fade the historical signal
        # If neighbors went up, we expect mean-reversion down
        if mean_ret < 0:  # Neighbors lost -> expect bounce
            sl = close - sl_dist
            tp = close + tp_dist
            self.broker.buy(i, close, size=size, sl=sl, tp=tp)
        else:  # Neighbors won -> expect pullback
            sl = close + sl_dist
            tp = close - tp_dist
            self.broker.sell(i, close, size=size, sl=sl, tp=tp)

        self._entry_bar = i

    def _reset(self, i: int):
        """Reset state after trade exit."""
        self._entry_bar = None
        self._last_exit = i

    def get_indicators(self):
        """Return indicators for plotting."""
        return [
            {'name': 'Vol Z', 'data': self.vol_z, 'color': '#ff6600', 'width': 1, 'panel': 1},
            {'name': 'Mom S Z', 'data': self.mom_s_z, 'color': '#00ff00', 'width': 1, 'panel': 2},
            {'name': 'Trend Z', 'data': self.trend_z, 'color': '#0066ff', 'width': 1, 'panel': 3},
        ]
