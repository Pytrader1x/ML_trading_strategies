"""
Market Bias Bounce Strategy (Original)

A trend-following pullback strategy that trades bounces from the mean (midline)
in the direction of the Market Bias indicator.

Core Logic:
- Uses Heikin Ashi-based Market Bias for trend direction
- Enters on pullbacks to the midline that respect band boundaries
- Exits on bias flip or via ATR-based trailing stop

Optimal timeframe: 15M, 1H
"""

import pandas as pd
import numpy as np
from backtest_engine import Strategy

try:
    from technical_indicators_custom import TIC
    HAS_TIC = True
except ImportError:
    HAS_TIC = False


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
    return true_range.rolling(window=period).mean()


def calculate_market_bias_manual(data: pd.DataFrame, ha_len: int = 300, ha_len2: int = 30):
    """
    Calculate Market Bias manually if TIC not available.

    This is a simplified implementation - use TIC.add_market_bias() for production.
    Creates ceiling (h2), floor (l2), and mean (ha_avg) levels.
    """
    close = data['Close']
    high = data['High']
    low = data['Low']

    # Simplified: Use rolling high/low channels as proxy
    h2 = high.rolling(window=ha_len).max()
    l2 = low.rolling(window=ha_len).min()
    ha_avg = (h2 + l2) / 2

    # Bias: +1 if close > mean, -1 if close < mean
    bias = pd.Series(np.where(close > ha_avg, 1, -1), index=data.index)

    return h2, l2, ha_avg, bias


class MarketBiasBounce(Strategy):
    """
    Market Bias Bounce Strategy.

    Entry Logic:
    - Long (Bias=1): Price enters from above ceiling, touches midline,
      respects floor, then breaks back above ceiling
    - Short (Bias=-1): Price enters from below floor, touches midline,
      respects ceiling, then breaks back below floor

    Exit Logic:
    - Bias flip exit (immediate close on direction change)
    - Chandelier ATR trailing stop (3.0 ATR multiplier)
    """

    def init(self):
        """Initialize indicators and state."""
        # Parameters
        self.ha_len = self.params.get('ha_len', 300)
        self.ha_len2 = self.params.get('ha_len2', 30)
        self.atr_period = self.params.get('atr_period', 14)
        self.atr_mult = self.params.get('atr_mult', 3.0)
        self.position_size = self.params.get('position_size', 1_000_000)

        # Calculate indicators
        if HAS_TIC:
            # Use TIC library for proper Market Bias
            self.data = TIC.add_market_bias(self.data, ha_len=self.ha_len, ha_len2=self.ha_len2)
            self.ceiling = self.data['MB_h2']
            self.floor = self.data['MB_l2']
            self.midline = self.data['MB_ha_avg']
            self.bias = self.data['MB_Bias']
        else:
            # Fallback to manual calculation
            self.ceiling, self.floor, self.midline, self.bias = calculate_market_bias_manual(
                self.data, self.ha_len, self.ha_len2
            )

        self.atr = calculate_atr(self.data, self.atr_period)

        # Calculate warmup period
        self.warmup = max(self.ha_len, self.ha_len2, self.atr_period) + 10

        # State tracking for trailing stop
        self._peak_high = None
        self._peak_low = None
        self._entry_bias = None

    def next(self, i: int, record: pd.Series):
        """Process each bar."""
        # Warmup check
        if i < self.warmup:
            return

        # Get current values
        close = record['Close']
        high = record['High']
        low = record['Low']

        ceiling = self.ceiling.iloc[i]
        floor = self.floor.iloc[i]
        mid = self.midline.iloc[i]
        bias = self.bias.iloc[i]
        atr = self.atr.iloc[i]

        # Skip if any NaN
        if pd.isna(ceiling) or pd.isna(floor) or pd.isna(mid) or pd.isna(atr):
            return

        # Previous bar values
        prev_close = self.data['Close'].iloc[i-1]
        prev_ceiling = self.ceiling.iloc[i-1]
        prev_floor = self.floor.iloc[i-1]

        # Active trade management
        if self.broker.active_trade:
            trade = self.broker.active_trade
            direction = trade['direction']

            # Bias flip exit - immediate close
            if (direction == 1 and bias == -1) or (direction == -1 and bias == 1):
                self.broker.close(i, close, reason="BiasFlip")
                self._peak_high = None
                self._peak_low = None
                return

            # Update trailing stop (Chandelier style)
            if direction == 1:  # Long
                if self._peak_high is None or high > self._peak_high:
                    self._peak_high = high
                tsl = self._peak_high - (atr * self.atr_mult)
                if low <= tsl:
                    self.broker.close(i, tsl, reason="TSL")
                    self._peak_high = None
                    return
            else:  # Short
                if self._peak_low is None or low < self._peak_low:
                    self._peak_low = low
                tsl = self._peak_low + (atr * self.atr_mult)
                if high >= tsl:
                    self.broker.close(i, tsl, reason="TSL")
                    self._peak_low = None
                    return
            return

        # Entry logic - no active trade
        pip_buffer = 0.0002  # 2 pips

        # Long entry (Bias = 1)
        if bias == 1:
            # Check pullback pattern:
            # 1. Price entered from above ceiling
            # 2. Price touched midline (mid within current bar's range)
            # 3. Price respected floor (low >= floor)
            # 4. Close breaks above ceiling

            entered_from_above = prev_close > prev_ceiling
            touched_mid = low <= mid <= high
            respected_floor = low >= floor
            breaks_ceiling = close > ceiling

            if entered_from_above and touched_mid and respected_floor and breaks_ceiling:
                sl = floor - pip_buffer
                risk = close - sl
                tp = close + (2.0 * risk)  # 2R target

                if risk > 0:
                    self.broker.buy(i, close, size=self.position_size, sl=sl, tp=tp)
                    self._peak_high = high
                    self._entry_bias = bias

        # Short entry (Bias = -1)
        elif bias == -1:
            # Check pullback pattern:
            # 1. Price entered from below floor
            # 2. Price touched midline
            # 3. Price respected ceiling (high <= ceiling)
            # 4. Close breaks below floor

            entered_from_below = prev_close < prev_floor
            touched_mid = low <= mid <= high
            respected_ceiling = high <= ceiling
            breaks_floor = close < floor

            if entered_from_below and touched_mid and respected_ceiling and breaks_floor:
                sl = ceiling + pip_buffer
                risk = sl - close
                tp = close - (2.0 * risk)  # 2R target

                if risk > 0:
                    self.broker.sell(i, close, size=self.position_size, sl=sl, tp=tp)
                    self._peak_low = low
                    self._entry_bias = bias

    def get_indicators(self):
        """Return indicators for plotting."""
        return [
            {'name': 'MB Ceiling', 'data': self.ceiling, 'color': '#ff6b6b', 'width': 1, 'dash': 'dash'},
            {'name': 'MB Floor', 'data': self.floor, 'color': '#4ecdc4', 'width': 1, 'dash': 'dash'},
            {'name': 'MB Midline', 'data': self.midline, 'color': '#ffe66d', 'width': 1.5},
        ]
