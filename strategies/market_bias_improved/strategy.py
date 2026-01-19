"""
Market Bias Improved Exits Strategy

Enhanced Market Bias with:
- Tighter mean-based SL calculations
- Dynamic position sizing based on ADX
- Tighter trailing stop (2.0 ATR vs 3.0)
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


class MarketBiasImproved(Strategy):
    """
    Market Bias Improved Exits Strategy.

    Key improvements:
    - Mean-based SL calculation (tighter than floor/ceiling)
    - Dynamic sizing: 2M in low-ADX (range), 1M in trending
    - Tighter TSL (2.0 ATR vs 3.0)
    """

    def init(self):
        self.ha_len = self.params.get('ha_len', 300)
        self.ha_len2 = self.params.get('ha_len2', 30)
        self.atr_period = self.params.get('atr_period', 14)
        self.adx_period = self.params.get('adx_period', 14)
        self.sl_atr_mult = self.params.get('sl_atr_mult', 1.5)
        self.tsl_atr_mult = self.params.get('tsl_atr_mult', 2.0)
        self.base_size = self.params.get('base_size', 1_000_000)
        self.range_size = self.params.get('range_size', 2_000_000)
        self.adx_threshold = self.params.get('adx_threshold', 20)

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
        self.adx = calculate_adx(self.data, self.adx_period)
        self.warmup = max(self.ha_len, self.atr_period, self.adx_period) + 10
        self._peak_high = None
        self._peak_low = None

    def next(self, i: int, record: pd.Series):
        if i < self.warmup:
            return

        close, high, low = record['Close'], record['High'], record['Low']
        ceiling = self.ceiling.iloc[i]
        floor = self.floor.iloc[i]
        mid = self.midline.iloc[i]
        bias = self.bias.iloc[i]
        atr = self.atr.iloc[i]
        adx = self.adx.iloc[i]

        if pd.isna(ceiling) or pd.isna(atr):
            return

        prev_close = self.data['Close'].iloc[i-1]
        prev_ceiling = self.ceiling.iloc[i-1]
        prev_floor = self.floor.iloc[i-1]

        if self.broker.active_trade:
            trade = self.broker.active_trade
            direction = trade['direction']

            # Bias flip exit
            if (direction == 1 and bias == -1) or (direction == -1 and bias == 1):
                self.broker.close(i, close, reason="BiasFlip")
                self._peak_high = self._peak_low = None
                return

            # Tighter TSL (2.0 ATR)
            if direction == 1:
                if self._peak_high is None or high > self._peak_high:
                    self._peak_high = high
                tsl = self._peak_high - (atr * self.tsl_atr_mult)
                if low <= tsl:
                    self.broker.close(i, tsl, reason="TSL")
                    self._peak_high = None
                    return
            else:
                if self._peak_low is None or low < self._peak_low:
                    self._peak_low = low
                tsl = self._peak_low + (atr * self.tsl_atr_mult)
                if high >= tsl:
                    self.broker.close(i, tsl, reason="TSL")
                    self._peak_low = None
                    return
            return

        # Dynamic position sizing
        size = self.range_size if adx < self.adx_threshold else self.base_size
        pip_buffer = 0.0002

        # Long entry
        if bias == 1:
            entered_from_above = prev_close > prev_ceiling
            touched_mid = low <= mid <= high
            respected_floor = low >= floor
            breaks_ceiling = close > ceiling

            if entered_from_above and touched_mid and respected_floor and breaks_ceiling:
                # Mean-based SL, constrained by floor
                sl = max(mid - (atr * self.sl_atr_mult), floor) - pip_buffer
                risk = close - sl
                tp = close + (2.0 * risk)
                if risk > 0:
                    self.broker.buy(i, close, size=size, sl=sl, tp=tp)
                    self._peak_high = high

        # Short entry
        elif bias == -1:
            entered_from_below = prev_close < prev_floor
            touched_mid = low <= mid <= high
            respected_ceiling = high <= ceiling
            breaks_floor = close < floor

            if entered_from_below and touched_mid and respected_ceiling and breaks_floor:
                sl = min(mid + (atr * self.sl_atr_mult), ceiling) + pip_buffer
                risk = sl - close
                tp = close - (2.0 * risk)
                if risk > 0:
                    self.broker.sell(i, close, size=size, sl=sl, tp=tp)
                    self._peak_low = low

    def get_indicators(self):
        return [
            {'name': 'MB Ceiling', 'data': self.ceiling, 'color': '#ff6b6b', 'dash': 'dash'},
            {'name': 'MB Floor', 'data': self.floor, 'color': '#4ecdc4', 'dash': 'dash'},
            {'name': 'MB Midline', 'data': self.midline, 'color': '#ffe66d', 'width': 1.5},
        ]
