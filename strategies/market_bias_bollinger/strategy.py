"""
Market Bias Bollinger Strategy

Combines Market Bias (trend direction) with Bollinger Band pullbacks for entry timing.
Uses ADX filter to ensure sufficient trend strength.

Entry: BB hook rejection in bias direction
Exit: Fixed TP/SL based on ATR
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


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma + (std_dev * std), sma - (std_dev * std), sma


def calculate_market_bias_manual(data: pd.DataFrame, ha_len: int = 300):
    close, high, low = data['Close'], data['High'], data['Low']
    h2 = high.rolling(window=ha_len).max()
    l2 = low.rolling(window=ha_len).min()
    ha_avg = (h2 + l2) / 2
    bias = pd.Series(np.where(close > ha_avg, 1, -1), index=data.index)
    return bias


class MarketBiasBollinger(Strategy):
    """
    Market Bias Bollinger Strategy.

    Entry: Price pierces BB in bias direction, then hooks back inside.
    Exit: Fixed ATR-based TP/SL (1.5:3.0 ratio = 2R target)
    """

    def init(self):
        self.ha_len = self.params.get('ha_len', 300)
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_dev = self.params.get('bb_dev', 2.0)
        self.adx_period = self.params.get('adx_period', 14)
        self.adx_min = self.params.get('adx_min', 20)
        self.atr_period = self.params.get('atr_period', 14)
        self.sl_atr_mult = self.params.get('sl_atr_mult', 1.5)
        self.tp_atr_mult = self.params.get('tp_atr_mult', 3.0)
        self.position_size = self.params.get('position_size', 1_000_000)

        if HAS_TIC:
            self.data = TIC.add_market_bias(self.data, ha_len=self.ha_len, ha_len2=30)
            self.bias = self.data['MB_Bias']
        else:
            self.bias = calculate_market_bias_manual(self.data, self.ha_len)

        self.bb_upper, self.bb_lower, self.bb_mid = calculate_bollinger_bands(
            self.data['Close'], self.bb_period, self.bb_dev)
        self.atr = calculate_atr(self.data, self.atr_period)
        self.adx = calculate_adx(self.data, self.adx_period)
        self.warmup = max(self.ha_len, self.bb_period, self.adx_period, self.atr_period) + 10

    def next(self, i: int, record: pd.Series):
        if i < self.warmup:
            return

        if self.broker.active_trade:
            return

        close, high, low = record['Close'], record['High'], record['Low']
        bias = self.bias.iloc[i]
        adx = self.adx.iloc[i]
        atr = self.atr.iloc[i]
        bb_upper = self.bb_upper.iloc[i]
        bb_lower = self.bb_lower.iloc[i]

        if pd.isna(adx) or pd.isna(atr) or adx < self.adx_min:
            return

        # Long: Bias=1, low pierces lower BB, close hooks back inside
        if bias == 1 and low < bb_lower and close > bb_lower:
            sl = close - (atr * self.sl_atr_mult)
            tp = close + (atr * self.tp_atr_mult)
            self.broker.buy(i, close, size=self.position_size, sl=sl, tp=tp)

        # Short: Bias=-1, high pierces upper BB, close hooks back inside
        elif bias == -1 and high > bb_upper and close < bb_upper:
            sl = close + (atr * self.sl_atr_mult)
            tp = close - (atr * self.tp_atr_mult)
            self.broker.sell(i, close, size=self.position_size, sl=sl, tp=tp)

    def get_indicators(self):
        return [
            {'name': 'BB Upper', 'data': self.bb_upper, 'color': '#888888', 'dash': 'dash'},
            {'name': 'BB Lower', 'data': self.bb_lower, 'color': '#888888', 'dash': 'dash'},
            {'name': 'BB Mid', 'data': self.bb_mid, 'color': '#ffa500', 'width': 1},
        ]
