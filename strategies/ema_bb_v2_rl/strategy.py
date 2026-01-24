"""
EMA BB Scalp V2 RL Strategy - PPO-based Exit Optimization.

This strategy extends the classical EMA BB Scalp V2 with learned exit policies.
Entry logic is preserved from the original, while exits are managed by a trained
PPO agent that learns to maximize risk-adjusted returns.

Key Differences from Classical:
- Entries: Identical (EMA + BB + ADX + XGBoost)
- Exits: RL policy replaces fixed SL/TP

Actions:
    0: HOLD - maintain position
    1: EXIT - close position
    2: TIGHTEN_SL - tighten stop loss
    3: TRAIL_BREAKEVEN - move SL to entry
    4: PARTIAL_EXIT - close 50%
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ENGINE_PATH = Path("/Users/williamsmith/Python_local_Mac/04_backtesting/production_backtest_engine/src")
if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))

from backtest_engine import Strategy

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

STRATEGY_DIR = Path(__file__).parent


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


class EMABBScalpV2RLStrategy(Strategy):
    """
    EMA BB Scalp V2 with RL Exit Management.

    Entry Logic (unchanged from classical):
    - EMA trend confirmation (fast > slow for longs)
    - BB band touch (close <= lower for longs)
    - ADX filter (20-50 range)
    - ML probability > threshold (if enabled)

    Exit Logic (RL-managed):
    - PPO policy decides: HOLD, EXIT, TIGHTEN_SL, TRAIL, PARTIAL
    - Fallback to classical SL/TP if RL model not available
    """

    def init(self):
        # === Classical Parameters ===
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

        # === Calculate Indicators ===
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

        # === Load XGBoost Model ===
        self.xgb_model = None
        if self.use_ml:
            model_path = self.params.get('xgb_model_path')
            if model_path is None:
                model_path = STRATEGY_DIR.parent / "ema_bb_scalp_v2" / "models" / "model.json"

            if model_path.exists():
                try:
                    self.xgb_model = xgb.Booster()
                    self.xgb_model.load_model(str(model_path))
                except Exception as e:
                    print(f"Warning: Could not load XGBoost model: {e}")
                    self.xgb_model = None

        if self.xgb_model is None:
            self.use_ml = False

        # === Load RL Exit Policy ===
        self.use_rl_exit = self.params.get('use_rl_exit', True)
        self.rl_policy = None
        self.device = 'cpu'  # Use CPU for inference during backtest

        if self.use_rl_exit:
            rl_model_path = self.params.get('rl_model_path')
            if rl_model_path is None:
                rl_model_path = STRATEGY_DIR / "models" / "exit_policy_final.pt"

            if rl_model_path.exists():
                try:
                    from model import ActorCritic
                    from config import PPOConfig

                    checkpoint = torch.load(rl_model_path, map_location=self.device, weights_only=False)
                    config = checkpoint.get('config', PPOConfig())
                    self.rl_policy = ActorCritic(config)
                    self.rl_policy.load_state_dict(checkpoint['model_state_dict'])
                    self.rl_policy.eval()
                    print(f"Loaded RL exit policy from {rl_model_path}")
                except Exception as e:
                    print(f"Warning: Could not load RL model: {e}")
                    self.rl_policy = None

        if self.rl_policy is None:
            self.use_rl_exit = False
            print("RL exit disabled, using classical SL/TP")

        # === State Tracking ===
        self.was_in_trade = False
        self.last_trade_exit_bar = -999
        self.daily_trades = 0
        self.current_day = None
        self.warmup = max(self.ema_slow_len, self.bb_period, self.adx_period, self.trend_conf_bars)

        # RL state
        self.trade_entry_bar = None
        self.trade_entry_price = None
        self.trade_direction = None
        self.trade_entry_atr = None
        self.action_history = [0.0] * 5
        self.current_sl_atr = 1.1
        self.max_favorable = 0.0
        self.max_adverse = 0.0

    def get_features(self, i, record):
        """Get ML features for entry filtering."""
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

    def _build_rl_state(self, i: int, record: pd.Series) -> torch.Tensor:
        """Build RL state vector (25 dimensions)."""
        if self.trade_entry_bar is None:
            return None

        close = record['Close']
        bars_held = i - self.trade_entry_bar
        max_bars = 200

        # Normalized PnL
        pnl = (close - self.trade_entry_price) / self.trade_entry_price * self.trade_direction

        # Update MFE/MAE
        self.max_favorable = max(self.max_favorable, pnl)
        self.max_adverse = min(self.max_adverse, pnl)

        # Position features (5)
        bars_held_norm = bars_held / max_bars
        sl_dist_norm = self.current_sl_atr / 2.0

        # Market features (10)
        atr_norm = self.atr.iloc[i] / close
        adx_norm = self.adx.iloc[i] / 100.0
        rsi_norm = self.rsi.iloc[i] / 100.0
        bb_pos = (close - self.bb_lower.iloc[i]) / (self.bb_upper.iloc[i] - self.bb_lower.iloc[i] + 1e-10)
        ema_spread = (self.ema_fast.iloc[i] - self.ema_slow.iloc[i]) / close * 100

        # Entry context (5)
        entry_atr_norm = self.trade_entry_atr / self.trade_entry_price
        entry_adx = self.adx.iloc[self.trade_entry_bar] / 100.0
        entry_rsi = self.rsi.iloc[self.trade_entry_bar] / 100.0
        entry_bb_width = self.bb_width.iloc[self.trade_entry_bar]
        entry_ema_diff = self.ema_diff.iloc[self.trade_entry_bar] * 100

        state = torch.tensor([
            # Position (5)
            bars_held_norm,
            pnl,
            self.max_favorable,
            self.max_adverse,
            sl_dist_norm,
            # Market (10)
            pnl,
            self.max_favorable,
            self.max_adverse,
            atr_norm,
            adx_norm,
            rsi_norm,
            bb_pos,
            ema_spread,
            0.0,  # Return placeholder
            bars_held_norm,  # Time
            # Entry context (5)
            entry_atr_norm,
            entry_adx,
            entry_rsi,
            entry_bb_width,
            entry_ema_diff,
            # Action history (5)
            *self.action_history
        ], dtype=torch.float32).unsqueeze(0)

        return state

    def _apply_rl_action(self, action: int, i: int, record: pd.Series):
        """Apply RL policy action."""
        close = record['Close']

        if action == 1:  # EXIT
            self.broker.close(i, close, reason="RL_EXIT")
            self._reset_trade_state()

        elif action == 2:  # TIGHTEN_SL
            self.current_sl_atr *= 0.75
            self._update_sl(i, record)

        elif action == 3:  # TRAIL_BREAKEVEN
            pnl = (close - self.trade_entry_price) * self.trade_direction
            if pnl > 0:
                self.current_sl_atr = 0.1
                self._update_sl(i, record)

        elif action == 4:  # PARTIAL_EXIT (treat as full exit for v1)
            self.broker.close(i, close, reason="RL_PARTIAL")
            self._reset_trade_state()

        # Update action history
        self.action_history = self.action_history[1:] + [action / 4.0]

    def _update_sl(self, i: int, record: pd.Series):
        """Update stop loss based on current SL multiplier."""
        if not self.broker.active_trade:
            return

        close = record['Close']
        atr = self.atr.iloc[i]

        if self.trade_direction == 1:  # Long
            new_sl = close - atr * self.current_sl_atr
            self.broker.active_trade['sl'] = max(self.broker.active_trade['sl'], new_sl)
        else:  # Short
            new_sl = close + atr * self.current_sl_atr
            self.broker.active_trade['sl'] = min(self.broker.active_trade['sl'], new_sl)

    def _reset_trade_state(self):
        """Reset RL trade tracking state."""
        self.trade_entry_bar = None
        self.trade_entry_price = None
        self.trade_direction = None
        self.trade_entry_atr = None
        self.action_history = [0.0] * 5
        self.current_sl_atr = 1.1
        self.max_favorable = 0.0
        self.max_adverse = 0.0

    def next(self, i: int, record: pd.Series):
        # === RL Exit Management ===
        if self.broker.active_trade and self.use_rl_exit:
            state = self._build_rl_state(i, record)
            if state is not None:
                with torch.no_grad():
                    action, _, _ = self.rl_policy.get_action(state, deterministic=True)
                    action = action.item()

                self._apply_rl_action(action, i, record)

                if not self.broker.active_trade:
                    return

        # === Trade Exit Tracking ===
        if self.was_in_trade and not self.broker.active_trade:
            self.last_trade_exit_bar = i
            self.was_in_trade = False
            self._reset_trade_state()

        if self.broker.active_trade:
            self.was_in_trade = True
            return

        if i < self.warmup:
            return

        # === Daily Trade Limit ===
        current_day = self.data.index[i].date()
        if current_day != self.current_day:
            self.current_day = current_day
            self.daily_trades = 0

        if self.daily_trades >= self.max_trades_per_day:
            return

        # === Cooldown ===
        if i - self.last_trade_exit_bar < self.cooldown_bars:
            return

        # === Trend Confirmation ===
        ema_fast_slice = self.ema_fast.iloc[i - self.trend_conf_bars + 1:i + 1]
        ema_slow_slice = self.ema_slow.iloc[i - self.trend_conf_bars + 1:i + 1]

        if len(ema_fast_slice) < self.trend_conf_bars:
            return

        is_uptrend = (ema_fast_slice > ema_slow_slice).all()
        is_downtrend = (ema_fast_slice < ema_slow_slice).all()

        if not (is_uptrend or is_downtrend):
            return

        # === ADX Filter ===
        if self.use_adx_filter:
            adx_val = self.adx.iloc[i]
            if not (self.adx_min <= adx_val <= self.adx_max):
                return

        # === Entry Triggers ===
        close = record['Close']
        atr = self.atr.iloc[i]
        signal = 0

        if is_uptrend and close <= self.bb_lower.iloc[i]:
            signal = 1
        elif is_downtrend and close >= self.bb_upper.iloc[i]:
            signal = -1

        if signal == 0:
            return

        # === ML Filter ===
        prob = 0.5
        if self.use_ml and self.xgb_model:
            features = self.get_features(i, record)
            dmatrix = xgb.DMatrix(features, feature_names=['adx', 'rsi', 'atr_pct', 'bb_width', 'ema_diff', 'dist_fast', 'dist_slow'])
            prob = self.xgb_model.predict(dmatrix)[0]
            if prob < self.threshold:
                return

        # === Position Sizing ===
        size = self.high_conf_size if prob >= self.high_conf_threshold else self.base_size

        # === Execute Trade ===
        sl_dist = atr * self.sl_coef
        tp_dist = sl_dist * self.tp_ratio

        if signal == 1:
            sl = close - sl_dist
            tp = close + tp_dist
            self.broker.buy(i, close, size=size, sl=sl, tp=tp)
            self.trade_direction = 1
        else:
            sl = close + sl_dist
            tp = close - tp_dist
            self.broker.sell(i, close, size=size, sl=sl, tp=tp)
            self.trade_direction = -1

        # Initialize RL state
        self.trade_entry_bar = i
        self.trade_entry_price = close
        self.trade_entry_atr = atr
        self.action_history = [0.0] * 5
        self.current_sl_atr = self.sl_coef
        self.max_favorable = 0.0
        self.max_adverse = 0.0

        self.daily_trades += 1
        self.was_in_trade = True

    def get_indicators(self):
        return [
            {'name': 'EMA Fast', 'data': self.ema_fast, 'color': '#00ff00', 'width': 1},
            {'name': 'EMA Slow', 'data': self.ema_slow, 'color': '#ff6600', 'width': 1},
            {'name': 'BB Upper', 'data': self.bb_upper, 'color': '#888888', 'dash': 'dash'},
            {'name': 'BB Lower', 'data': self.bb_lower, 'color': '#888888', 'dash': 'dash'},
        ]
