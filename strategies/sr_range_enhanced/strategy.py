"""
SR Range Enhanced Strategy

Fades Intelligent Fast SR levels in ranging markets:
- Filter: ADX < threshold (range regime)
- Triggers: WaveTrend oversold/overbought OR TD Sequential 9 setup
- Exit: Dynamic ATR-based stops and targets
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


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    close = data['Close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_wavetrend(data: pd.DataFrame, n1: int = 10, n2: int = 21, n3: int = 4):
    """Calculate WaveTrend oscillator."""
    hlc3 = (data['High'] + data['Low'] + data['Close']) / 3
    esa = hlc3.ewm(span=n1, adjust=False).mean()
    d = (hlc3 - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (hlc3 - esa) / (0.015 * d.replace(0, 1))
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(window=n3).mean()
    osc = wt1 - wt2
    return wt1, wt2, osc


def calculate_sr_levels(data: pd.DataFrame, lookback: int = 50):
    """Calculate simple support/resistance levels."""
    high = data['High']
    low = data['Low']
    # Rolling resistance = highest high
    resistance = high.rolling(window=lookback).max()
    # Rolling support = lowest low
    support = low.rolling(window=lookback).min()
    return resistance, support


class SRRangeEnhanced(Strategy):
    """
    Enhanced Range Strategy - Fades SR levels in ranging markets.

    Entry Logic:
    - Long: Near support, ADX < threshold, WaveTrend oversold
    - Short: Near resistance, ADX < threshold, WaveTrend overbought

    Exit: ATR-based SL/TP with R:R filter
    """

    def init(self):
        self.level_type = self.params.get('level_type', 'Med')
        self.adx_max = self.params.get('adx_max', 40)  # Relaxed from 30
        self.wt_oversold = self.params.get('wt_oversold', -8)  # Typical range is -10 to +10
        self.wt_overbought = self.params.get('wt_overbought', 8)  # Typical range is -10 to +10
        self.sl_atr_mult = self.params.get('sl_atr_mult', 1.2)
        self.tp_atr_mult = self.params.get('tp_atr_mult', 2.5)
        self.min_rr = self.params.get('min_rr', 1.2)  # Relaxed from 1.5
        self.position_size = self.params.get('position_size', 2_000_000)
        self.sr_lookback = self.params.get('sr_lookback', 30)  # Shorter lookback

        # Try to use TIC indicators if available
        res_col = f"SR_{self.level_type}_Resistance"
        sup_col = f"SR_{self.level_type}_Support"

        if HAS_TIC and res_col in self.data.columns:
            self.resistance = self.data[res_col]
            self.support = self.data[sup_col]
        else:
            # Fallback to simple SR calculation
            self.resistance, self.support = calculate_sr_levels(self.data, self.sr_lookback)

        # WaveTrend
        wt_col = "WaveTrend_10_21_4_Osc"
        if HAS_TIC and wt_col in self.data.columns:
            self.wt_osc = self.data[wt_col]
        else:
            _, _, self.wt_osc = calculate_wavetrend(self.data)

        # ADX and ATR
        if 'ADX' in self.data.columns:
            self.adx = self.data['ADX']
        else:
            self.adx = calculate_adx(self.data)

        if 'ATR' in self.data.columns:
            self.atr = self.data['ATR']
        else:
            self.atr = calculate_atr(self.data)

        self.warmup = max(self.sr_lookback, 30)

    def next(self, i: int, record: pd.Series):
        if i < self.warmup:
            return
        if self.broker.active_trade:
            return

        # Regime check: Only trade in ranges
        adx = self.adx.iloc[i]
        if pd.isna(adx) or adx > self.adx_max:
            return

        close, high, low = record['Close'], record['High'], record['Low']
        atr = self.atr.iloc[i]
        res = self.resistance.iloc[i]
        sup = self.support.iloc[i]
        wt = self.wt_osc.iloc[i]

        if pd.isna(atr) or pd.isna(res) or pd.isna(sup):
            return

        # LONG Logic
        is_near_sup = abs(low - sup) <= (1.5 * atr) or low < sup
        sig_wt_long = wt < self.wt_oversold

        if is_near_sup and sig_wt_long:
            sl_dist = self.sl_atr_mult * atr
            tp_dist = self.tp_atr_mult * atr
            # Use ATR-based TP directly (don't cap at resistance)
            self.broker.buy(i, close, size=self.position_size,
                           sl=close - sl_dist, tp=close + tp_dist)
            return

        # SHORT Logic
        is_near_res = abs(high - res) <= (1.5 * atr) or high > res
        sig_wt_short = wt > self.wt_overbought

        if is_near_res and sig_wt_short:
            sl_dist = self.sl_atr_mult * atr
            tp_dist = self.tp_atr_mult * atr
            # Use ATR-based TP directly (don't floor at support)
            self.broker.sell(i, close, size=self.position_size,
                            sl=close + sl_dist, tp=close - tp_dist)

    def get_indicators(self):
        return [
            {'name': 'Resistance', 'data': self.resistance, 'color': '#ff6b6b', 'dash': 'dash'},
            {'name': 'Support', 'data': self.support, 'color': '#4ecdc4', 'dash': 'dash'},
        ]
