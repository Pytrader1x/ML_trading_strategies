"""
EMA BB Scalp V2 Strategy (ML Enhanced)

Core EMA + BB mean reversion with XGBoost ML filter for entry quality.

Improvements over V1:
1. XGBoost filter: Scores entries based on volatility/momentum features
2. Dynamic position sizing based on ML confidence
3. RSI + BB Width regime checks
"""

import pandas as pd
import numpy as np
from pathlib import Path
from backtest_engine import Strategy

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(series, period=20, dev=2.0):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * dev)
    lower = sma - (std * dev)
    return upper, lower, sma


def calculate_atr(data, period=14):
    high, low, close = data['High'], data['Low'], data['Close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_adx(data, period=14):
    high, low = data['High'], data['Low']
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=data.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=data.index)
    tr = calculate_atr(data, 1)
    atr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_smooth)
    sum_di = plus_di + minus_di
    sum_di = sum_di.replace(0, 1)
    dx = (abs(plus_di - minus_di) / sum_di) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean()


class EMABBScalpV2Strategy(Strategy):
    """
    EMA BB Scalp V2 - XGBoost ML Enhanced.

    Entry Logic:
    - EMA trend confirmation (fast > slow for longs)
    - BB band touch (close <= lower for longs)
    - ADX filter (20-50 range)
    - ML probability > threshold (if enabled)

    Exit: Fixed ATR-based SL/TP
    """

    def init(self):
        # Parameters
        self.ema_fast_len = self.params.get('ema_fast', 30)
        self.ema_slow_len = self.params.get('ema_slow', 50)
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_dev = self.params.get('bb_dev', 2.0)
        self.trend_conf_bars = self.params.get('trend_conf_bars', 3)
        self.adx_period = self.params.get('adx_period', 14)
        self.adx_min = self.params.get('adx_min', 20)
        self.adx_max = self.params.get('adx_max', 50)
        self.atr_period = self.params.get('atr_period', 14)
        self.sl_coef = self.params.get('sl_coef', 1.1)
        self.tp_ratio = self.params.get('tp_ratio', 1.5)
        self.max_trades_per_day = self.params.get('max_trades_per_day', 10)
        self.cooldown_bars = self.params.get('cooldown_bars', 1)
        self.use_adx_filter = self.params.get('use_adx_filter', True)

        # ML params
        self.use_ml = self.params.get('use_ml', True) and HAS_XGB
        self.threshold = self.params.get('threshold', 0.52)
        self.high_conf_threshold = self.params.get('high_conf_threshold', 0.7)
        self.base_size = self.params.get('base_size', 2_000_000)
        self.high_conf_size = self.params.get('high_conf_size', 5_000_000)

        # Calculate indicators
        close = self.data['Close']
        self.ema_fast = calculate_ema(close, self.ema_fast_len)
        self.ema_slow = calculate_ema(close, self.ema_slow_len)
        self.bb_upper, self.bb_lower, self.bb_mid = calculate_bollinger_bands(close, self.bb_period, self.bb_dev)
        self.atr = calculate_atr(self.data, self.atr_period)
        self.adx = calculate_adx(self.data, self.adx_period)
        self.rsi = calculate_rsi(close, 14)

        # Features for ML
        self.bb_width = (self.bb_upper - self.bb_lower) / self.bb_mid.replace(0, 1)
        self.ema_diff = (self.ema_fast - self.ema_slow) / close.replace(0, 1)

        # Load ML model
        self.model = None
        if self.use_ml:
            model_path = self.params.get('model_path')
            if model_path is None:
                model_path = Path(__file__).parent / "models" / "model.json"
                if not model_path.exists():
                    model_path = Path(__file__).parent / "model.json"

            if model_path.exists():
                try:
                    self.model = xgb.Booster()
                    self.model.load_model(str(model_path))
                except Exception as e:
                    print(f"Warning: Could not load ML model: {e}")
                    self.model = None

        if self.model is None:
            self.use_ml = False

        # State
        self.was_in_trade = False
        self.last_trade_exit_bar = -999
        self.daily_trades = 0
        self.current_day = None
        self.warmup = max(self.ema_slow_len, self.bb_period, self.adx_period, self.trend_conf_bars)

    def get_features(self, i, record):
        close = record['Close']
        adx = self.adx.iloc[i]
        rsi = self.rsi.iloc[i]
        atr = self.atr.iloc[i]
        bb_width = self.bb_width.iloc[i]
        ema_diff = self.ema_diff.iloc[i]
        dist_fast = (close - self.ema_fast.iloc[i]) / close
        dist_slow = (close - self.ema_slow.iloc[i]) / close
        atr_pct = atr / close
        return np.array([[adx, rsi, atr_pct, bb_width, ema_diff, dist_fast, dist_slow]])

    def next(self, i: int, record: pd.Series):
        # Trade exit tracking
        if self.was_in_trade and not self.broker.active_trade:
            self.last_trade_exit_bar = i
            self.was_in_trade = False

        if self.broker.active_trade:
            self.was_in_trade = True
            return

        if i < self.warmup:
            return

        # Daily trade limit
        current_day = self.data.index[i].date()
        if current_day != self.current_day:
            self.current_day = current_day
            self.daily_trades = 0

        if self.daily_trades >= self.max_trades_per_day:
            return

        # Cooldown
        if i - self.last_trade_exit_bar < self.cooldown_bars:
            return

        # Trend confirmation
        ema_fast_slice = self.ema_fast.iloc[i - self.trend_conf_bars + 1:i + 1]
        ema_slow_slice = self.ema_slow.iloc[i - self.trend_conf_bars + 1:i + 1]

        if len(ema_fast_slice) < self.trend_conf_bars:
            return

        is_uptrend = (ema_fast_slice > ema_slow_slice).all()
        is_downtrend = (ema_fast_slice < ema_slow_slice).all()

        if not (is_uptrend or is_downtrend):
            return

        # ADX filter
        if self.use_adx_filter:
            adx_val = self.adx.iloc[i]
            if not (self.adx_min <= adx_val <= self.adx_max):
                return

        # Entry triggers
        close = record['Close']
        atr = self.atr.iloc[i]
        signal = 0

        if is_uptrend and close <= self.bb_lower.iloc[i]:
            signal = 1
        elif is_downtrend and close >= self.bb_upper.iloc[i]:
            signal = -1

        if signal == 0:
            return

        # ML filter
        prob = 0.5
        if self.use_ml and self.model:
            features = self.get_features(i, record)
            dmatrix = xgb.DMatrix(features, feature_names=['adx', 'rsi', 'atr_pct', 'bb_width', 'ema_diff', 'dist_fast', 'dist_slow'])
            prob = self.model.predict(dmatrix)[0]
            if prob < self.threshold:
                return

        # Position sizing
        size = self.high_conf_size if prob >= self.high_conf_threshold else self.base_size

        # Execute trade
        sl_dist = atr * self.sl_coef
        tp_dist = sl_dist * self.tp_ratio

        if signal == 1:
            sl = close - sl_dist
            tp = close + tp_dist
            self.broker.buy(i, close, size=size, sl=sl, tp=tp)
        else:
            sl = close + sl_dist
            tp = close - tp_dist
            self.broker.sell(i, close, size=size, sl=sl, tp=tp)

        self.daily_trades += 1
        self.was_in_trade = True

    def get_indicators(self):
        return [
            {'name': 'EMA Fast', 'data': self.ema_fast, 'color': '#00ff00', 'width': 1},
            {'name': 'EMA Slow', 'data': self.ema_slow, 'color': '#ff6600', 'width': 1},
            {'name': 'BB Upper', 'data': self.bb_upper, 'color': '#888888', 'dash': 'dash'},
            {'name': 'BB Lower', 'data': self.bb_lower, 'color': '#888888', 'dash': 'dash'},
        ]
