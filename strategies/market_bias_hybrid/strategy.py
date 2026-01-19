"""
Market Bias Hybrid Strategy

Combines:
1. Trend Pullback: Bounces off mean in direction of Bias (original)
2. Range Fade: Commented out but available - fades bands when ADX is low

Uses intelligent mean-based stop loss and Chandelier trailing stop.
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
    high, low, close = data['High'], data['Low'], data['Close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low = data['High'], data['Low']
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=data.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=data.index)
    prev_close = data['Close'].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    alpha = 1 / period
    atr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_smooth)
    sum_di = plus_di + minus_di
    sum_di = sum_di.replace(0, 1)
    dx = (abs(plus_di - minus_di) / sum_di) * 100
    return dx.ewm(alpha=alpha, adjust=False).mean()


def calculate_market_bias_manual(data: pd.DataFrame, ha_len: int = 300, ha_len2: int = 30):
    close, high, low = data['Close'], data['High'], data['Low']
    h2 = high.rolling(window=ha_len).max()
    l2 = low.rolling(window=ha_len).min()
    ha_avg = (h2 + l2) / 2
    bias = pd.Series(np.where(close > ha_avg, 1, -1), index=data.index)
    return h2, l2, ha_avg, bias


class MarketBiasHybrid(Strategy):
    """
    Hybrid Strategy combining trend pullback with intelligent exits.

    Entry Logic:
    - Trend Pullback: Price enters from outside bands, touches mean,
      respects opposite band, then breaks back above/below ceiling/floor

    Exit Logic:
    - Bias flip exit
    - Chandelier-style ATR trailing stop (2.0 ATR)
    """

    def init(self):
        self.ha_len = self.params.get('ha_len', 300)
        self.ha_len2 = self.params.get('ha_len2', 30)
        self.atr_period = self.params.get('atr_period', 14)
        self.sl_atr_mult = self.params.get('sl_atr_mult', 1.5)
        self.tsl_atr_mult = self.params.get('tsl_atr_mult', 2.0)
        self.adx_range_threshold = self.params.get('adx_range_threshold', 25)
        self.position_size = self.params.get('position_size', 1_000_000)

        if HAS_TIC:
            self.data = TIC.add_market_bias(self.data, ha_len=self.ha_len, ha_len2=self.ha_len2)
            self.ceiling = self.data['MB_h2']
            self.floor = self.data['MB_l2']
            self.midline = self.data['MB_ha_avg']
            self.bias = self.data['MB_Bias']
        else:
            self.ceiling, self.floor, self.midline, self.bias = calculate_market_bias_manual(
                self.data, self.ha_len, self.ha_len2)

        self.atr = calculate_atr(self.data, self.atr_period)
        self.adx = calculate_adx(self.data, self.atr_period)
        self.warmup = max(self.ha_len, self.ha_len2, self.atr_period) + 10

        # State tracking
        self._bull_pullback_touch = False
        self._bear_pullback_touch = False
        self._last_trade_entry_time = None
        self._trade_peak = None

    def next(self, i: int, record: pd.Series):
        if i < self.warmup:
            return

        close, high, low = record['Close'], record['High'], record['Low']
        ceiling = self.ceiling.iloc[i]
        floor = self.floor.iloc[i]
        mid = self.midline.iloc[i]
        bias = self.bias.iloc[i]
        atr = self.atr.iloc[i]

        if pd.isna(ceiling) or pd.isna(atr):
            return

        prev_bias = self.bias.iloc[i-1]
        prev_close = self.data['Close'].iloc[i-1]

        # Reset touch flags on bias change
        if bias != prev_bias:
            self._bull_pullback_touch = False
            self._bear_pullback_touch = False

        # Active trade management
        if self.broker.active_trade:
            trade = self.broker.active_trade
            direction = trade['direction']

            # Bias flip exit
            if (direction == 1 and bias == -1) or (direction == -1 and bias == 1):
                self.broker.close(i, close, reason="BiasFlip")
                self._trade_peak = None
                return

            # Chandelier trailing stop
            entry_time = trade.get('entry_time') or trade.get('EntryTime')
            if entry_time != self._last_trade_entry_time:
                self._last_trade_entry_time = entry_time
                self._trade_peak = high if direction == 1 else low

            if direction == 1:  # Long
                if self._trade_peak is None:
                    self._trade_peak = high
                self._trade_peak = max(self._trade_peak, high)
                tsl = self._trade_peak - (atr * self.tsl_atr_mult)
                if low <= tsl:
                    self.broker.close(i, tsl, reason="TSL")
                    self._trade_peak = None
                    return
            else:  # Short
                if self._trade_peak is None:
                    self._trade_peak = low
                self._trade_peak = min(self._trade_peak, low)
                tsl = self._trade_peak + (atr * self.tsl_atr_mult)
                if high >= tsl:
                    self.broker.close(i, tsl, reason="TSL")
                    self._trade_peak = None
                    return
            return

        # Entry logic - Trend Pullback
        pip_buffer = 0.0002

        if bias == 1:
            entered_from_above = prev_close > self.ceiling.iloc[i-1]
            touched_mid = low <= mid <= high
            respected_floor = low >= floor

            if (entered_from_above or self._bull_pullback_touch) and touched_mid and respected_floor:
                self._bull_pullback_touch = True
            if low < floor:
                self._bull_pullback_touch = False

            if self._bull_pullback_touch and close > ceiling:
                # Intelligent mean-based SL
                mean_based_sl = mid - (atr * self.sl_atr_mult)
                sl = max(mean_based_sl, floor) - pip_buffer
                risk = close - sl
                if risk < pip_buffer * 2:
                    risk = pip_buffer * 2
                tp = close + (2.0 * risk)

                if risk > 0:
                    self.broker.buy(i, close, size=self.position_size, sl=sl, tp=tp)
                    self._bull_pullback_touch = False

        elif bias == -1:
            entered_from_below = prev_close < self.floor.iloc[i-1]
            touched_mid = low <= mid <= high
            respected_ceiling = high <= ceiling

            if (entered_from_below or self._bear_pullback_touch) and touched_mid and respected_ceiling:
                self._bear_pullback_touch = True
            if high > ceiling:
                self._bear_pullback_touch = False

            if self._bear_pullback_touch and close < floor:
                mean_based_sl = mid + (atr * self.sl_atr_mult)
                sl = min(mean_based_sl, ceiling) + pip_buffer
                risk = sl - close
                if risk < pip_buffer * 2:
                    risk = pip_buffer * 2
                tp = close - (2.0 * risk)

                if risk > 0:
                    self.broker.sell(i, close, size=self.position_size, sl=sl, tp=tp)
                    self._bear_pullback_touch = False

    def get_indicators(self):
        return [
            {'name': 'MB Ceiling', 'data': self.ceiling, 'color': '#ff6b6b', 'dash': 'dash'},
            {'name': 'MB Floor', 'data': self.floor, 'color': '#4ecdc4', 'dash': 'dash'},
            {'name': 'MB Midline', 'data': self.midline, 'color': '#ffe66d', 'width': 1.5},
        ]
