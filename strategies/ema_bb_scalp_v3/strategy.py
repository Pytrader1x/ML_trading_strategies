"""
EMA BB Scalp V3 Strategy (Stat + ML Hybrid)

Core EMA + BB with robust statistics and exit management.

Enhancements:
- Robust z-score filter (median + MAD)
- Volatility regime filter
- Edge score combining extremity, trend strength, regime
- Trailing stop + breakeven logic
- Optional XGBoost ML filter
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
    return upper, lower, sma, std


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


class EMABBScalpV3Strategy(Strategy):
    """
    EMA BB Scalp V3 - Robust Statistics + Exit Management.

    Entry Logic:
    - EMA trend confirmation
    - BB touch + robust z-score filter
    - ADX range + volatility regime filter
    - Edge score threshold
    - Optional ML filter

    Exit Logic:
    - Fixed TP/SL
    - Trailing stop (after activation)
    - Breakeven move protection
    - Max hold time exit
    """

    def init(self):
        # Core params
        self.ema_fast_len = self.params.get('ema_fast', 30)
        self.ema_slow_len = self.params.get('ema_slow', 50)
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_dev = self.params.get('bb_dev', 2.0)
        self.trend_conf_bars = self.params.get('trend_conf_bars', 3)
        self.adx_period = self.params.get('adx_period', 14)
        self.adx_min = self.params.get('adx_min', 18)
        self.adx_max = self.params.get('adx_max', 45)
        self.atr_period = self.params.get('atr_period', 14)
        self.sl_atr_mult = self.params.get('sl_atr_mult', 1.1)
        self.tp_ratio = self.params.get('tp_ratio', 1.6)
        self.max_trades_per_day = self.params.get('max_trades_per_day', 5)
        self.cooldown_bars = self.params.get('cooldown_bars', 2)

        # Robust filter params
        self.robust_window = self.params.get('robust_window', 200)
        self.robust_z_thresh = self.params.get('robust_z_thresh', 1.6)
        self.vol_window = self.params.get('vol_window', 200)
        self.vol_z_max = self.params.get('vol_z_max', 1.5)

        # Edge score params
        self.edge_threshold = self.params.get('edge_threshold', 0.9)
        self.edge_z_weight = self.params.get('edge_z_weight', 0.6)
        self.edge_trend_weight = self.params.get('edge_trend_weight', 0.25)
        self.edge_adx_weight = self.params.get('edge_adx_weight', 0.15)
        self.edge_vol_weight = self.params.get('edge_vol_weight', 0.2)

        # RSI filter
        self.rsi_low = self.params.get('rsi_low', 45)
        self.rsi_high = self.params.get('rsi_high', 55)

        # Position sizing
        self.base_size = self.params.get('base_size', 1_000_000)
        self.min_size = self.params.get('min_size', 200_000)
        self.max_size = self.params.get('max_size', 5_000_000)
        self.target_atr_pct = self.params.get('target_atr_pct', 0.002)
        self.vol_scale_min = self.params.get('vol_scale_min', 0.5)
        self.vol_scale_max = self.params.get('vol_scale_max', 2.0)

        # Exit management
        self.use_tsl = self.params.get('use_tsl', True)
        self.tsl_atr_mult = self.params.get('tsl_atr_mult', 1.3)
        self.tsl_activation_atr = self.params.get('tsl_activation_atr', 0.8)
        self.use_breakeven = self.params.get('use_breakeven', True)
        self.breakeven_atr = self.params.get('breakeven_atr', 0.6)
        self.breakeven_buffer_pips = self.params.get('breakeven_buffer_pips', 1.0)
        self.max_hold_bars = self.params.get('max_hold_bars', 24)

        # ML params
        self.use_ml = self.params.get('use_ml', True) and HAS_XGB
        self.threshold = self.params.get('threshold', 0.54)
        self.high_conf_threshold = self.params.get('high_conf_threshold', 0.7)

        # Calculate indicators
        close = self.data['Close']
        self.ema_fast = calculate_ema(close, self.ema_fast_len)
        self.ema_slow = calculate_ema(close, self.ema_slow_len)
        self.bb_upper, self.bb_lower, self.bb_mid, self.bb_std = calculate_bollinger_bands(
            close, self.bb_period, self.bb_dev)
        self.atr = calculate_atr(self.data, self.atr_period)
        self.adx = calculate_adx(self.data, self.adx_period)
        self.rsi = calculate_rsi(close, 14)

        # Derived features
        self.bb_width = (self.bb_upper - self.bb_lower) / self.bb_mid.replace(0, np.nan)
        self.atr_pct = self.atr / close.replace(0, np.nan)
        self.ema_diff = (self.ema_fast - self.ema_slow) / close.replace(0, np.nan)

        # Z-scores
        self.bb_width_z = (self.bb_width - self.bb_width.rolling(self.vol_window).mean()) / self.bb_width.rolling(self.vol_window).std()
        self.atr_pct_z = (self.atr_pct - self.atr_pct.rolling(self.vol_window).mean()) / self.atr_pct.rolling(self.vol_window).std()
        self.ema_diff_z = (self.ema_diff - self.ema_diff.rolling(self.vol_window).mean()) / self.ema_diff.rolling(self.vol_window).std()

        # Robust z-score (median/MAD)
        dev = close - self.bb_mid
        median = dev.rolling(self.robust_window).median()
        mad = (dev - median).abs().rolling(self.robust_window).median()
        self.robust_z = 0.6745 * (dev - median) / (mad + 1e-9)

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
                except:
                    self.model = None

        if self.model is None:
            self.use_ml = False

        # State
        self.last_trade_exit_bar = -999
        self.daily_trades = 0
        self.current_day = None
        self.was_in_trade = False
        self.last_trade_entry_time = None
        self.trade_peak = None

        self.warmup = max(self.ema_slow_len, self.bb_period, self.adx_period,
                         self.trend_conf_bars, self.robust_window, self.vol_window)

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

    def _edge_score(self, i, direction):
        robust_z = self.robust_z.iloc[i]
        bb_width_z = self.bb_width_z.iloc[i]
        atr_pct_z = self.atr_pct_z.iloc[i]
        ema_diff_z = self.ema_diff_z.iloc[i]
        adx_val = self.adx.iloc[i]

        trend_strength = max(0.0, ema_diff_z * direction)
        adx_norm = 0.0
        if self.adx_max > self.adx_min:
            adx_norm = (adx_val - self.adx_min) / (self.adx_max - self.adx_min)
            adx_norm = float(np.clip(adx_norm, 0.0, 1.0))

        vol_penalty = abs(bb_width_z) + abs(atr_pct_z)

        score = (
            abs(robust_z) * self.edge_z_weight
            + trend_strength * self.edge_trend_weight
            + adx_norm * self.edge_adx_weight
            - vol_penalty * self.edge_vol_weight
        )
        return score

    def _position_size(self, i, edge_score, ml_prob):
        atr_pct = self.atr_pct.iloc[i]
        vol_scale = 1.0
        if atr_pct is not None and np.isfinite(atr_pct) and atr_pct > 0:
            vol_scale = self.target_atr_pct / atr_pct
        vol_scale = float(np.clip(vol_scale, self.vol_scale_min, self.vol_scale_max))

        score_scale = 1.0 + max(0.0, edge_score - self.edge_threshold) * 0.25
        ml_scale = 1.0
        if ml_prob is not None and ml_prob >= self.high_conf_threshold:
            ml_scale = 1.4

        size = self.base_size * vol_scale * score_scale * ml_scale
        return float(np.clip(size, self.min_size, self.max_size))

    def _manage_open_trade(self, i, record):
        trade = self.broker.active_trade
        if not trade:
            return False

        close, high, low = record['Close'], record['High'], record['Low']
        direction = trade['direction']
        entry_idx = trade['entry_idx']
        entry_price = trade['entry_price']
        atr = self.atr.iloc[i]

        if atr is None or not np.isfinite(atr) or atr <= 0:
            return True

        # Time exit
        if self.max_hold_bars and (i - entry_idx) >= self.max_hold_bars:
            self.broker.close(i, close, reason="Time")
            return True

        current_sl = trade.get('sl')
        entry_time = trade.get('entry_time')

        if entry_time != self.last_trade_entry_time:
            self.last_trade_entry_time = entry_time
            self.trade_peak = high if direction == 1 else low

        # Breakeven
        if self.use_breakeven:
            move = (close - entry_price) * direction
            if move >= self.breakeven_atr * atr:
                be_offset = self.breakeven_buffer_pips * 0.0001
                be_sl = entry_price + (be_offset * direction)
                if current_sl is None or (direction == 1 and be_sl > current_sl) or (direction == -1 and be_sl < current_sl):
                    trade['sl'] = be_sl
                    current_sl = be_sl

        # Trailing stop
        if self.use_tsl:
            if direction == 1:
                self.trade_peak = max(self.trade_peak, high)
                if (self.trade_peak - entry_price) >= self.tsl_activation_atr * atr:
                    new_sl = self.trade_peak - (atr * self.tsl_atr_mult)
                    if current_sl is None or new_sl > current_sl:
                        trade['sl'] = new_sl
            else:
                self.trade_peak = min(self.trade_peak, low)
                if (entry_price - self.trade_peak) >= self.tsl_activation_atr * atr:
                    new_sl = self.trade_peak + (atr * self.tsl_atr_mult)
                    if current_sl is None or new_sl < current_sl:
                        trade['sl'] = new_sl

        return True

    def next(self, i: int, record: pd.Series):
        if self.was_in_trade and not self.broker.active_trade:
            self.last_trade_exit_bar = i
            self.was_in_trade = False
            self.trade_peak = None
            self.last_trade_entry_time = None

        if self.broker.active_trade:
            self.was_in_trade = True
            self._manage_open_trade(i, record)
            return

        if i < self.warmup:
            return

        current_day = self.data.index[i].date()
        if current_day != self.current_day:
            self.current_day = current_day
            self.daily_trades = 0

        if self.daily_trades >= self.max_trades_per_day:
            return

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
        adx_val = self.adx.iloc[i]
        if not (self.adx_min <= adx_val <= self.adx_max):
            return

        # Volatility regime filter
        bb_width_z = self.bb_width_z.iloc[i]
        atr_pct_z = self.atr_pct_z.iloc[i]
        if not np.isfinite(bb_width_z) or not np.isfinite(atr_pct_z):
            return
        if abs(bb_width_z) > self.vol_z_max or abs(atr_pct_z) > self.vol_z_max:
            return

        # Entry signal
        close = record['Close']
        robust_z = self.robust_z.iloc[i]
        if not np.isfinite(robust_z):
            return
        rsi_val = self.rsi.iloc[i]
        signal = 0

        if is_uptrend and close <= self.bb_lower.iloc[i] and robust_z <= -self.robust_z_thresh and rsi_val <= self.rsi_low:
            signal = 1
        elif is_downtrend and close >= self.bb_upper.iloc[i] and robust_z >= self.robust_z_thresh and rsi_val >= self.rsi_high:
            signal = -1

        if signal == 0:
            return

        # Edge score
        edge_score = self._edge_score(i, signal)
        if edge_score < self.edge_threshold:
            return

        # ML filter
        ml_prob = None
        if self.use_ml and self.model:
            features = self.get_features(i, record)
            dmatrix = xgb.DMatrix(features, feature_names=['adx', 'rsi', 'atr_pct', 'bb_width', 'ema_diff', 'dist_fast', 'dist_slow'])
            ml_prob = float(self.model.predict(dmatrix)[0])
            if ml_prob < self.threshold:
                return

        # Position size
        atr = self.atr.iloc[i]
        if atr is None or not np.isfinite(atr) or atr <= 0:
            return

        sl_dist = atr * self.sl_atr_mult
        tp_dist = sl_dist * self.tp_ratio
        size = self._position_size(i, edge_score, ml_prob)

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
