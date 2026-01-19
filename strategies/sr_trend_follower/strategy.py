"""
SR Trend Follower Strategy

Trades pullbacks to SR levels in direction of Market Bias trend:
- Trend filter: Market Bias direction
- Entry: Pullback to SR levels with RSI confirmation
- Exit: ATR-based SL/TP
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


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    close = data['Close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_market_bias_manual(data: pd.DataFrame, ha_len: int = 300, ha_len2: int = 30):
    close, high, low = data['Close'], data['High'], data['Low']
    h2 = high.rolling(window=ha_len).max()
    l2 = low.rolling(window=ha_len).min()
    ha_avg = (h2 + l2) / 2
    bias = pd.Series(np.where(close > ha_avg, 1, -1), index=data.index)
    return h2, l2, ha_avg, bias


def calculate_sr_levels(data: pd.DataFrame, lookback: int = 50):
    """Calculate simple support/resistance levels."""
    high = data['High']
    low = data['Low']
    resistance = high.rolling(window=lookback).max()
    support = low.rolling(window=lookback).min()
    return resistance, support


class SRTrendFollower(Strategy):
    """
    SR Trend Follower - Pullbacks to SR levels in trend direction.

    Entry Logic:
    - Long (Bias=1): Price near support, RSI not overbought (< 45)
    - Short (Bias=-1): Price near resistance, RSI not oversold (> 55)

    Exit: ATR-based SL/TP with minimum R:R check
    """

    def init(self):
        self.level_type = self.params.get('level_type', 'Med')
        self.rsi_period = self.params.get('rsi_period', 14)
        self.rsi_long_max = self.params.get('rsi_long_max', 45)
        self.rsi_short_min = self.params.get('rsi_short_min', 55)
        self.sl_atr_mult = self.params.get('sl_atr_mult', 2.0)
        self.tp_atr_mult = self.params.get('tp_atr_mult', 4.0)
        self.min_rr = self.params.get('min_rr', 1.5)
        self.position_size = self.params.get('position_size', 2_000_000)
        self.ha_len = self.params.get('ha_len', 300)
        self.sr_lookback = self.params.get('sr_lookback', 50)

        # SR Levels
        res_col = f"SR_{self.level_type}_Resistance"
        sup_col = f"SR_{self.level_type}_Support"

        if HAS_TIC and res_col in self.data.columns:
            self.resistance = self.data[res_col]
            self.support = self.data[sup_col]
        else:
            self.resistance, self.support = calculate_sr_levels(self.data, self.sr_lookback)

        # Market Bias
        if HAS_TIC and 'MB_Bias' in self.data.columns:
            self.bias = self.data['MB_Bias']
        else:
            _, _, _, self.bias = calculate_market_bias_manual(self.data, self.ha_len)

        # RSI
        if 'RSI' in self.data.columns:
            self.rsi = self.data['RSI']
        else:
            self.rsi = calculate_rsi(self.data, self.rsi_period)

        # ATR
        if 'ATR' in self.data.columns:
            self.atr = self.data['ATR']
        else:
            self.atr = calculate_atr(self.data)

        self.warmup = max(self.ha_len, self.sr_lookback, self.rsi_period) + 10

    def next(self, i: int, record: pd.Series):
        if i < self.warmup:
            return
        if self.broker.active_trade:
            return

        close, high, low = record['Close'], record['High'], record['Low']
        atr = self.atr.iloc[i]
        res = self.resistance.iloc[i]
        sup = self.support.iloc[i]
        bias = self.bias.iloc[i]
        rsi = self.rsi.iloc[i]

        if pd.isna(atr) or pd.isna(res) or pd.isna(sup) or pd.isna(bias):
            return

        # LONG: Bullish Trend + Dip to Support + RSI filter
        if bias == 1:
            is_near_sup = abs(low - sup) <= (0.5 * atr) or low < sup
            rsi_ok = rsi < self.rsi_long_max

            if is_near_sup and rsi_ok:
                sl_dist = self.sl_atr_mult * atr
                tp_dist = self.tp_atr_mult * atr
                potential_tp = close + tp_dist
                actual_reward = tp_dist

                if actual_reward >= (sl_dist * self.min_rr):
                    self.broker.buy(i, close, size=self.position_size,
                                   sl=close - sl_dist, tp=potential_tp)

        # SHORT: Bearish Trend + Rally to Resistance + RSI filter
        elif bias == -1:
            is_near_res = abs(high - res) <= (0.5 * atr) or high > res
            rsi_ok = rsi > self.rsi_short_min

            if is_near_res and rsi_ok:
                sl_dist = self.sl_atr_mult * atr
                tp_dist = self.tp_atr_mult * atr
                potential_tp = close - tp_dist
                actual_reward = tp_dist

                if actual_reward >= (sl_dist * self.min_rr):
                    self.broker.sell(i, close, size=self.position_size,
                                    sl=close + sl_dist, tp=potential_tp)

    def get_indicators(self):
        return [
            {'name': 'Resistance', 'data': self.resistance, 'color': '#ff6b6b', 'dash': 'dash'},
            {'name': 'Support', 'data': self.support, 'color': '#4ecdc4', 'dash': 'dash'},
        ]
