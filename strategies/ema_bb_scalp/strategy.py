"""
EMA Bollinger Band Scalp Strategy (ADX Filtered)

A trend-following mean-reversion scalper that:
- Uses EMA crossovers (30/50) to establish directional bias
- Enters on Bollinger Band touches (mean reversion within trend)
- Filters with ADX (20-50) for optimal trend strength
- Uses ATR-based stops with 1.5:1 reward/risk ratio
- Limits risk with daily caps and cooldown periods

Optimal timeframe: 15M (also works on 1H, 4H with reduced performance)
"""

import pandas as pd
import numpy as np
from backtest_engine import Strategy


# --- Indicator Functions ---

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """
    Calculate Bollinger Bands.

    Returns:
        tuple: (upper_band, lower_band, middle_band)
    """
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, lower, sma


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = data['High']
    low = data['Low']
    close = data['Close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index using Wilder's smoothing.

    ADX measures trend strength (not direction):
    - ADX < 20: Weak/no trend (choppy)
    - 20-50: Medium trend (optimal for trading)
    - ADX > 50: Strong trend (potential exhaustion)
    """
    high = data['High']
    low = data['Low']

    # Directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    # +DM and -DM
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=data.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=data.index
    )

    # True Range for ADX calculation
    prev_close = data['Close'].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing (alpha = 1/period)
    alpha = 1 / period
    atr_smooth = true_range.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_smooth)

    # DX and ADX
    sum_di = plus_di + minus_di
    sum_di = sum_di.replace(0, 1)  # Avoid division by zero
    dx = (abs(plus_di - minus_di) / sum_di) * 100
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx


class EMABBScalpADXStrategy(Strategy):
    """
    EMA Bollinger Band Scalp Strategy with ADX Filter.

    Logic:
    1. Trend Bias: EMA(30) vs EMA(50)
       - Long bias: EMA(30) > EMA(50)
       - Short bias: EMA(30) < EMA(50)
    2. Confirmation: Trend must persist for N consecutive bars
    3. Filter: ADX between 20-50 (medium trend strength)
    4. Entry:
       - Long: Close <= Lower Bollinger Band
       - Short: Close >= Upper Bollinger Band
    5. Exit:
       - SL: ATR * sl_mult
       - TP: SL distance * tp_ratio
    6. Risk Controls:
       - Max trades per day
       - Cooldown period between trades
    """

    def init(self):
        """Initialize indicators and strategy state."""
        # Get parameters with defaults
        self.ema_fast_period = self.params.get('ema_fast', 30)
        self.ema_slow_period = self.params.get('ema_slow', 50)
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_std = self.params.get('bb_std', 2.0)
        self.trend_bars = self.params.get('trend_bars', 6)
        self.adx_period = self.params.get('adx_period', 14)
        self.adx_min = self.params.get('adx_min', 20)
        self.adx_max = self.params.get('adx_max', 50)
        self.atr_period = self.params.get('atr_period', 14)
        self.sl_mult = self.params.get('sl_mult', 1.1)
        self.tp_ratio = self.params.get('tp_ratio', 1.5)
        self.position_size = self.params.get('position_size', 2_000_000)
        self.max_daily_trades = self.params.get('max_daily_trades', 10)
        self.cooldown_bars = self.params.get('cooldown_bars', 4)

        # Calculate indicators
        close = self.data['Close']

        self.ema_fast = calculate_ema(close, self.ema_fast_period)
        self.ema_slow = calculate_ema(close, self.ema_slow_period)
        self.bb_upper, self.bb_lower, self.bb_mid = calculate_bollinger_bands(
            close, self.bb_period, self.bb_std
        )
        self.atr = calculate_atr(self.data, self.atr_period)
        self.adx = calculate_adx(self.data, self.adx_period)

        # Calculate warmup period
        self.warmup = max(
            self.ema_slow_period,
            self.bb_period,
            self.adx_period,
            self.atr_period,
            self.trend_bars
        )

        # State tracking
        self._last_exit_bar = -999  # Bar index of last trade exit
        self._daily_trades = 0      # Trades opened today
        self._current_day = None    # Current trading day
        self._was_in_trade = False  # Track trade state for cooldown

    def next(self, i: int, record: pd.Series):
        """
        Process each bar and generate trading signals.

        Args:
            i: Current bar index
            record: Current bar data as pd.Series
        """
        # Track trade exits for cooldown
        if self._was_in_trade and not self.broker.active_trade:
            self._last_exit_bar = i
            self._was_in_trade = False

        if self.broker.active_trade:
            self._was_in_trade = True
            return  # Already in a trade

        # Warmup check
        if i < self.warmup:
            return

        # Daily trade limit reset
        current_time = self.data.index[i]
        current_day = current_time.date()

        if current_day != self._current_day:
            self._current_day = current_day
            self._daily_trades = 0

        if self._daily_trades >= self.max_daily_trades:
            return

        # Cooldown check
        if (i - self._last_exit_bar) < self.cooldown_bars:
            return

        # Trend confirmation: Check last N bars
        start_idx = i - self.trend_bars + 1
        if start_idx < 0:
            return

        ema_fast_slice = self.ema_fast.iloc[start_idx:i + 1]
        ema_slow_slice = self.ema_slow.iloc[start_idx:i + 1]

        if len(ema_fast_slice) < self.trend_bars:
            return

        is_uptrend = (ema_fast_slice > ema_slow_slice).all()
        is_downtrend = (ema_fast_slice < ema_slow_slice).all()

        if not (is_uptrend or is_downtrend):
            return

        # ADX filter
        adx_val = self.adx.iloc[i]
        if not (self.adx_min <= adx_val <= self.adx_max):
            return

        # Entry logic
        close = record['Close']
        atr = self.atr.iloc[i]

        if pd.isna(atr) or atr <= 0:
            return

        sl_dist = atr * self.sl_mult
        tp_dist = sl_dist * self.tp_ratio

        if is_uptrend and close <= self.bb_lower.iloc[i]:
            # Long entry: Close touched/below lower BB in uptrend
            sl = close - sl_dist
            tp = close + tp_dist

            self.broker.buy(i, close, size=self.position_size, sl=sl, tp=tp)
            self._daily_trades += 1
            self._was_in_trade = True

        elif is_downtrend and close >= self.bb_upper.iloc[i]:
            # Short entry: Close touched/above upper BB in downtrend
            sl = close + sl_dist
            tp = close - tp_dist

            self.broker.sell(i, close, size=self.position_size, sl=sl, tp=tp)
            self._daily_trades += 1
            self._was_in_trade = True

    def get_indicators(self):
        """Return indicators for plotting on HTML report."""
        return [
            {'name': 'EMA 30', 'data': self.ema_fast, 'color': '#00bfff', 'width': 1.5},
            {'name': 'EMA 50', 'data': self.ema_slow, 'color': '#ffa500', 'width': 1.5},
            {'name': 'BB Upper', 'data': self.bb_upper, 'color': '#888888', 'dash': 'dash', 'width': 1},
            {'name': 'BB Lower', 'data': self.bb_lower, 'color': '#888888', 'dash': 'dash', 'width': 1},
        ]
