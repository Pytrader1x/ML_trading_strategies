"""
EMA BB Scalp V4 Strategy (LSTM + XGBoost Ensemble)

Regime-aware hybrid ML scalping strategy combining:
- V1: Core EMA + BB mean reversion logic
- V2: XGBoost ML filtering + dynamic position sizing
- V3: Robust statistics + exit management
- V4: LSTM sequence patterns + regime detection

Note: This strategy can run without ML models (graceful degradation to rule-based logic).
For best performance, train models using tools/train_model.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from backtest_engine import Strategy

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


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


class EMABBScalpV4Strategy(Strategy):
    """
    EMA BB Scalp V4 - Regime-Aware Hybrid ML Strategy.

    Entry Logic:
    1. EMA Trend: Fast > Slow for longs (3-bar confirmation)
    2. BB Touch: Close <= BB_Lower (long) or >= BB_Upper (short)
    3. ADX Range: 18-45 (medium trend)
    4. Robust Z-Score: > 1.6 std from median
    5. RSI Filter: < 45 for longs, > 55 for shorts
    6. Regime Score: > min_regime_score (if regime detection enabled)
    7. ML Ensemble: > threshold (if models available)

    Exit Logic:
    - Fixed TP/SL (ATR-based)
    - Trailing stop (after 0.8*ATR profit)
    - Breakeven (after 0.6*ATR profit)
    - Time exit (max 24 bars)
    """

    def init(self):
        # Core params
        self.pair = self.params.get('pair', 'AUDUSD')
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

        # Regime params
        self.min_regime_score = self.params.get('min_regime_score', 0.5)
        self.regime_exit_threshold = self.params.get('regime_exit_threshold', 0.3)

        # ML params
        self.use_ml = self.params.get('use_ml', True) and (HAS_XGB or HAS_TORCH)
        self.threshold = self.params.get('threshold', 0.55)
        self.high_conf_threshold = self.params.get('high_conf_threshold', 0.70)
        self.ml_size_boost = self.params.get('ml_size_boost', 1.4)
        model_dir = self.params.get('model_dir') or str(Path(__file__).parent / "models")
        self.model_dir = Path(model_dir)

        # Calculate indicators
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']

        # EMAs
        self.ema_fast = calculate_ema(close, self.ema_fast_len)
        self.ema_slow = calculate_ema(close, self.ema_slow_len)

        # Bollinger Bands
        self.bb_upper, self.bb_lower, self.bb_mid, self.bb_std = calculate_bollinger_bands(
            close, self.bb_period, self.bb_dev)
        self.bb_width = (self.bb_upper - self.bb_lower) / self.bb_mid.replace(0, 1)

        # ATR
        self.atr = calculate_atr(self.data, self.atr_period)
        self.atr_pct = self.atr / close.replace(0, 1)

        # ADX
        self.adx = calculate_adx(self.data, self.adx_period)

        # RSI
        self.rsi = calculate_rsi(close, 14)

        # Derived features
        self.ema_diff = (self.ema_fast - self.ema_slow) / close.replace(0, 1)
        self.dist_fast = (close - self.ema_fast) / close.replace(0, 1)
        self.dist_slow = (close - self.ema_slow) / close.replace(0, 1)

        # Z-scores
        self.bb_width_z = (self.bb_width - self.bb_width.rolling(self.vol_window).mean()) / self.bb_width.rolling(self.vol_window).std()
        self.atr_pct_z = (self.atr_pct - self.atr_pct.rolling(self.vol_window).mean()) / self.atr_pct.rolling(self.vol_window).std()

        # Robust z-score (median/MAD)
        dev = close - self.bb_mid
        median = dev.rolling(self.robust_window).median()
        mad = (dev - median).abs().rolling(self.robust_window).median()
        self.robust_z = 0.6745 * (dev - median) / (mad + 1e-9)

        # Regime features
        self.vol_percentile = self.atr_pct.rolling(self.vol_window).apply(
            lambda x: (x < x.iloc[-1]).mean() if len(x) > 0 else 0.5, raw=False)
        self.adx_slope = self.adx.diff(5) / 5
        self.bb_squeeze = self.bb_width.rolling(20).apply(
            lambda x: 1.0 if len(x) > 0 and x.iloc[-1] <= x.min() * 1.1 else 0.0, raw=False)

        # Returns for LSTM
        self.close_returns = close.pct_change()

        # Load ML models
        self.xgb_model = None
        self.lstm_model = None
        self.device = None

        if self.use_ml:
            self._load_models()

        # State
        self.last_trade_exit_bar = -999
        self.daily_trades = 0
        self.current_day = None
        self.was_in_trade = False
        self.last_trade_entry_time = None
        self.trade_peak = None

        self.warmup = max(self.ema_slow_len, self.bb_period, self.adx_period,
                         self.trend_conf_bars, self.robust_window, self.vol_window)

    def _load_models(self):
        """Load LSTM and XGBoost models if available."""
        # XGBoost
        if HAS_XGB:
            xgb_path = self.model_dir / f"xgb_{self.pair}.json"
            if xgb_path.exists():
                try:
                    self.xgb_model = xgb.Booster()
                    self.xgb_model.load_model(str(xgb_path))
                except Exception as e:
                    print(f"Warning: XGBoost load failed: {e}")

        # LSTM
        if HAS_TORCH:
            lstm_path = self.model_dir / f"lstm_{self.pair}.pt"
            if lstm_path.exists():
                try:
                    # Detect device
                    if torch.backends.mps.is_available():
                        self.device = torch.device("mps")
                    elif torch.cuda.is_available():
                        self.device = torch.device("cuda")
                    else:
                        self.device = torch.device("cpu")

                    # Simple LSTM model definition for loading
                    class SimpleLSTM(torch.nn.Module):
                        def __init__(self, input_size=3, hidden_size=32, num_layers=2):
                            super().__init__()
                            self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,
                                                     batch_first=True, dropout=0.2)
                            self.fc = torch.nn.Linear(hidden_size, 1)
                            self.sigmoid = torch.nn.Sigmoid()

                        def forward(self, x):
                            out, _ = self.lstm(x)
                            out = self.fc(out[:, -1, :])
                            return self.sigmoid(out)

                    self.lstm_model = SimpleLSTM().to(self.device)
                    self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
                    self.lstm_model.eval()
                except Exception as e:
                    print(f"Warning: LSTM load failed: {e}")

        # Disable ML if no models loaded
        if self.xgb_model is None and self.lstm_model is None:
            self.use_ml = False

    def _get_point_features(self, i: int) -> np.ndarray:
        """Extract point-in-time features for XGBoost."""
        features = np.array([[
            self.adx.iloc[i],
            self.rsi.iloc[i],
            self.atr_pct.iloc[i],
            self.bb_width.iloc[i],
            self.ema_diff.iloc[i],
            self.dist_fast.iloc[i],
            self.dist_slow.iloc[i],
            0,  # market_bias placeholder
            self.vol_percentile.iloc[i] if pd.notna(self.vol_percentile.iloc[i]) else 0.5,
            self.adx_slope.iloc[i] if pd.notna(self.adx_slope.iloc[i]) else 0,
            self.bb_squeeze.iloc[i] if pd.notna(self.bb_squeeze.iloc[i]) else 0,
        ]])
        return np.nan_to_num(features, nan=0.0)

    def _get_sequence_features(self, i: int, seq_len: int = 20) -> Optional[np.ndarray]:
        """Extract sequence features for LSTM."""
        if i < seq_len or not HAS_TORCH:
            return None

        start_idx = i - seq_len + 1
        end_idx = i + 1

        returns = self.close_returns.iloc[start_idx:end_idx].values
        adx = self.adx.iloc[start_idx:end_idx].values
        rsi = self.rsi.iloc[start_idx:end_idx].values

        # Normalize
        returns = np.clip(returns, -0.05, 0.05) / 0.05
        adx = adx / 100.0
        rsi = rsi / 100.0

        sequence = np.stack([returns, adx, rsi], axis=1)
        sequence = np.nan_to_num(sequence, nan=0.0)

        return sequence.reshape(1, seq_len, 3)

    def _get_ml_probability(self, i: int) -> Optional[float]:
        """Get ensemble ML probability."""
        if not self.use_ml:
            return None

        X_point = self._get_point_features(i)

        lstm_prob = 0.5
        xgb_prob = 0.5

        # XGBoost prediction
        if self.xgb_model is not None and HAS_XGB:
            dmatrix = xgb.DMatrix(X_point)
            xgb_prob = self.xgb_model.predict(dmatrix)[0]

        # LSTM prediction
        if self.lstm_model is not None and HAS_TORCH:
            X_seq = self._get_sequence_features(i)
            if X_seq is not None:
                X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    lstm_prob = self.lstm_model(X_seq_tensor).cpu().numpy().flatten()[0]

        # Ensemble (40% LSTM, 60% XGBoost)
        if self.lstm_model is not None and self.xgb_model is not None:
            return 0.4 * lstm_prob + 0.6 * xgb_prob
        elif self.xgb_model is not None:
            return xgb_prob
        elif self.lstm_model is not None:
            return lstm_prob
        return None

    def _get_regime_score(self, i: int) -> float:
        """Calculate regime score based on current conditions."""
        adx_val = self.adx.iloc[i]
        vol_pct = self.vol_percentile.iloc[i] if pd.notna(self.vol_percentile.iloc[i]) else 0.5

        # ADX in sweet spot (25-35) gives higher score
        adx_score = 0.0
        if 20 <= adx_val <= 40:
            adx_score = 1.0 - abs(adx_val - 30) / 15

        # Vol percentile in middle range is better
        vol_score = 1.0 - abs(vol_pct - 0.5) * 2

        return 0.6 * adx_score + 0.4 * vol_score

    def _position_size(self, i: int, ml_prob: Optional[float]) -> float:
        """Calculate position size based on volatility and ML confidence."""
        atr_pct = self.atr_pct.iloc[i]
        vol_scale = 1.0
        if atr_pct is not None and np.isfinite(atr_pct) and atr_pct > 0:
            vol_scale = self.target_atr_pct / atr_pct
        vol_scale = float(np.clip(vol_scale, self.vol_scale_min, self.vol_scale_max))

        size = self.base_size * vol_scale

        # ML confidence boost
        if ml_prob is not None and ml_prob >= self.high_conf_threshold:
            size *= self.ml_size_boost

        return float(np.clip(size, self.min_size, self.max_size))

    def _manage_open_trade(self, i: int, record: pd.Series) -> bool:
        """Manage exit logic for open trade."""
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

        # Regime exit
        regime_score = self._get_regime_score(i)
        if regime_score < self.regime_exit_threshold:
            self.broker.close(i, close, reason="Regime")
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
        """Process next bar."""
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
        adx_val = self.adx.iloc[i]
        if not (self.adx_min <= adx_val <= self.adx_max):
            return

        # Regime check
        regime_score = self._get_regime_score(i)
        if regime_score < self.min_regime_score:
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

        # ML filter
        ml_prob = self._get_ml_probability(i)
        if self.use_ml and ml_prob is not None and ml_prob < self.threshold:
            return

        # Position size
        atr = self.atr.iloc[i]
        if atr is None or not np.isfinite(atr) or atr <= 0:
            return

        sl_dist = atr * self.sl_atr_mult
        tp_dist = sl_dist * self.tp_ratio
        size = self._position_size(i, ml_prob)

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
