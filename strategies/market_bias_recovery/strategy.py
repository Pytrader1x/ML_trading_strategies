"""
Market Bias Recovery Strategy

Trades quick recovery from mean excursions:
- Entry: Price dips to wrong side of mean, then recovers within 5 bars
- Exit: Bias flip or ATR trailing stop

The theory is that quick mean excursions that recover are continuation signals.
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


def calculate_market_bias_manual(data: pd.DataFrame, ha_len: int = 300, ha_len2: int = 30):
    close, high, low = data['Close'], data['High'], data['Low']
    h2 = high.rolling(window=ha_len).max()
    l2 = low.rolling(window=ha_len).min()
    ha_avg = (h2 + l2) / 2
    bias = pd.Series(np.where(close > ha_avg, 1, -1), index=data.index)
    return h2, l2, ha_avg, bias


class MarketBiasRecovery(Strategy):
    """
    Bias Recovery Strategy.

    Entry Logic:
    - Long (Bias=1): Close crosses back above mean after being below for 1-5 bars
    - Short (Bias=-1): Close crosses back below mean after being above for 1-5 bars

    Exit Logic:
    - Bias flip exit
    - ATR trailing stop (3.0 ATR default)
    """

    def init(self):
        self.ha_len = self.params.get('ha_len', 300)
        self.ha_len2 = self.params.get('ha_len2', 30)
        self.atr_period = self.params.get('atr_period', 14)
        self.atr_mult = self.params.get('atr_mult', 3.0)
        self.max_recovery_bars = self.params.get('max_recovery_bars', 5)
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
        self.warmup = max(self.ha_len, self.ha_len2, self.atr_period) + 10

    def next(self, i: int, record: pd.Series):
        if i < self.warmup:
            return

        close, high, low = record['Close'], record['High'], record['Low']
        mid = self.midline.iloc[i]
        bias = self.bias.iloc[i]
        atr = self.atr.iloc[i]

        if pd.isna(mid) or pd.isna(atr):
            return

        # Exit logic
        if self.broker.active_trade:
            trade = self.broker.active_trade
            direction = trade['direction']

            # Bias flip exit
            if (direction == 1 and bias == -1) or (direction == -1 and bias == 1):
                self.broker.close(i, close, reason="BiasFlip")
                return

            # TSL update - modifying SL in trade
            current_sl = trade.get('sl')
            if direction == 1:  # Long
                new_sl = close - (atr * self.atr_mult)
                if current_sl is None or new_sl > current_sl:
                    trade['sl'] = new_sl
            else:  # Short
                new_sl = close + (atr * self.atr_mult)
                if current_sl is None or new_sl < current_sl:
                    trade['sl'] = new_sl
            return

        # Entry logic - Recovery Pattern
        prev_close = self.data['Close'].iloc[i-1]
        prev_mid = self.midline.iloc[i-1]

        if bias == 1:
            # Long entry: Current close > mean (back on right side)
            if close > mid:
                # Check previous bar to confirm crossover
                if prev_close < prev_mid:
                    # Just crossed up - count bars below
                    bars_below = 0
                    for k in range(1, self.max_recovery_bars + 2):
                        idx = i - k
                        if idx < 0:
                            break
                        if self.data['Close'].iloc[idx] < self.midline.iloc[idx]:
                            bars_below += 1
                        else:
                            break

                    # Valid recovery: 1 to max_recovery_bars bars below
                    if 1 <= bars_below <= self.max_recovery_bars:
                        sl = close - (atr * self.atr_mult)
                        # No TP - let TSL/Bias handle exits
                        self.broker.buy(i, close, size=self.position_size, sl=sl)

        elif bias == -1:
            # Short entry: Current close < mean (back on right side)
            if close < mid:
                # Check previous bar to confirm crossover
                if prev_close > prev_mid:
                    # Just crossed down - count bars above
                    bars_above = 0
                    for k in range(1, self.max_recovery_bars + 2):
                        idx = i - k
                        if idx < 0:
                            break
                        if self.data['Close'].iloc[idx] > self.midline.iloc[idx]:
                            bars_above += 1
                        else:
                            break

                    # Valid recovery: 1 to max_recovery_bars bars above
                    if 1 <= bars_above <= self.max_recovery_bars:
                        sl = close + (atr * self.atr_mult)
                        self.broker.sell(i, close, size=self.position_size, sl=sl)

    def get_indicators(self):
        return [
            {'name': 'MB Ceiling', 'data': self.ceiling, 'color': '#ff6b6b', 'dash': 'dash'},
            {'name': 'MB Floor', 'data': self.floor, 'color': '#4ecdc4', 'dash': 'dash'},
            {'name': 'MB Midline', 'data': self.midline, 'color': '#ffe66d', 'width': 1.5},
        ]
