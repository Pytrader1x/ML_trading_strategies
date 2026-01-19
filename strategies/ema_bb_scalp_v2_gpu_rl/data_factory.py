"""
Layer 1: Data Factory - Vectorized Feature Generation & Triple Barrier Labeling

This module handles:
1. Loading and preprocessing OHLCV data
2. Vectorized calculation of all indicators (no loops)
3. Generation of candidate entry signals
4. Triple Barrier meta-labeling (forward-looking)
5. Conversion to GPU tensors

IMPORTANT: Triple Barrier labeling is forward-looking and must ONLY be used
for training. Never use in live trading or proper backtesting.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional
from numba import njit, prange

from .config import StrategyConfig, TripleBarrierConfig, DataConfig


# =============================================================================
# Vectorized Indicator Calculations
# =============================================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Vectorized Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Vectorized Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Vectorized Bollinger Bands. Returns (upper, lower, mid)."""
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + (std * dev)
    lower = mid - (std * dev)
    return upper, lower, mid


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Vectorized Average True Range."""
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Vectorized Average Directional Index."""
    high, low = df['High'], df['Low']
    up = high - high.shift(1)
    down = low.shift(1) - low

    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)

    # True Range for smoothing
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_smooth)

    sum_di = plus_di + minus_di
    sum_di = sum_di.replace(0, 1)
    dx = (abs(plus_di - minus_di) / sum_di) * 100

    return dx.ewm(alpha=1/period, adjust=False).mean()


def calculate_all_indicators(
    df: pd.DataFrame,
    config: StrategyConfig
) -> pd.DataFrame:
    """
    Calculate all indicators in a vectorized manner.

    Returns DataFrame with original OHLCV plus:
    - ema_fast, ema_slow
    - bb_upper, bb_lower, bb_mid
    - atr, adx, rsi
    - Derived features for ML
    """
    df = df.copy()
    close = df['Close']

    # Core indicators
    df['ema_fast'] = calculate_ema(close, config.ema_fast)
    df['ema_slow'] = calculate_ema(close, config.ema_slow)
    df['bb_upper'], df['bb_lower'], df['bb_mid'] = calculate_bollinger_bands(
        close, config.bb_period, config.bb_dev
    )
    df['atr'] = calculate_atr(df, config.atr_period)
    df['adx'] = calculate_adx(df, config.adx_period)
    df['rsi'] = calculate_rsi(close, 14)

    # Derived features (normalized for ML)
    df['adx_norm'] = df['adx'] / 100.0  # ADX is 0-100
    df['rsi_norm'] = (df['rsi'] - 50) / 50.0  # Center around 0
    df['atr_pct'] = df['atr'] / close  # ATR as % of price
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'].replace(0, 1)
    df['ema_diff'] = (df['ema_fast'] - df['ema_slow']) / close
    df['dist_fast'] = (close - df['ema_fast']) / close
    df['dist_slow'] = (close - df['ema_slow']) / close

    return df


# =============================================================================
# Signal Generation
# =============================================================================

def generate_candidate_signals(
    df: pd.DataFrame,
    config: StrategyConfig
) -> pd.DataFrame:
    """
    Generate candidate entry signals based on EMA + BB logic.

    Long signal (1): Uptrend (EMA fast > slow) + price touches lower BB
    Short signal (-1): Downtrend (EMA fast < slow) + price touches upper BB
    No signal (0): Otherwise

    ADX filter is applied if enabled.
    """
    df = df.copy()

    # Trend confirmation: EMA fast vs slow for N consecutive bars
    trend_up = (df['ema_fast'] > df['ema_slow']).rolling(config.trend_conf_bars).min() == 1
    trend_down = (df['ema_fast'] < df['ema_slow']).rolling(config.trend_conf_bars).min() == 1

    # BB touches
    close = df['Close']
    touch_lower = close <= df['bb_lower']
    touch_upper = close >= df['bb_upper']

    # ADX filter
    if config.use_adx_filter:
        adx_valid = (df['adx'] >= config.adx_min) & (df['adx'] <= config.adx_max)
    else:
        adx_valid = pd.Series(True, index=df.index)

    # Generate signals
    df['base_signal'] = 0
    long_mask = trend_up & touch_lower & adx_valid
    short_mask = trend_down & touch_upper & adx_valid

    df.loc[long_mask, 'base_signal'] = 1
    df.loc[short_mask, 'base_signal'] = -1

    # Count signals for debugging
    n_long = long_mask.sum()
    n_short = short_mask.sum()
    print(f"Generated {n_long} long signals, {n_short} short signals")

    return df


# =============================================================================
# Triple Barrier Labeling (Numba-accelerated)
# =============================================================================

@njit(parallel=True)
def _triple_barrier_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    signal_indices: np.ndarray,
    directions: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    max_bars: int,
    use_high_low: bool,
    min_return: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated Triple Barrier labeling.

    Returns:
    - labels: 1 for win, 0 for loss
    - returns: actual return achieved
    """
    n_signals = len(signal_indices)
    labels = np.zeros(n_signals, dtype=np.float32)
    returns = np.zeros(n_signals, dtype=np.float32)

    for k in prange(n_signals):
        i = signal_indices[k]
        direction = directions[k]
        entry_price = close[i]
        current_atr = atr[i]

        if current_atr < 1e-10:
            continue

        tp_dist = current_atr * tp_mult
        sl_dist = current_atr * sl_mult

        if direction == 1:  # Long
            tp_level = entry_price + tp_dist
            sl_level = entry_price - sl_dist
        else:  # Short
            tp_level = entry_price - tp_dist
            sl_level = entry_price + sl_dist

        # Look forward
        hit_tp = False
        hit_sl = False
        final_price = entry_price

        for j in range(i + 1, min(i + max_bars + 1, len(close))):
            if use_high_low:
                bar_high = high[j]
                bar_low = low[j]
            else:
                bar_high = close[j]
                bar_low = close[j]

            final_price = close[j]

            if direction == 1:  # Long
                if bar_high >= tp_level:
                    hit_tp = True
                    break
                if bar_low <= sl_level:
                    hit_sl = True
                    break
            else:  # Short
                if bar_low <= tp_level:
                    hit_tp = True
                    break
                if bar_high >= sl_level:
                    hit_sl = True
                    break

        # Determine outcome
        actual_return = (final_price - entry_price) * direction / entry_price

        if hit_tp:
            labels[k] = 1.0
            returns[k] = tp_dist * direction / entry_price
        elif hit_sl:
            labels[k] = 0.0
            returns[k] = -sl_dist * direction / entry_price
        else:
            # Time expired - use actual return
            labels[k] = 1.0 if actual_return > min_return else 0.0
            returns[k] = actual_return

    return labels, returns


def apply_triple_barrier(
    df: pd.DataFrame,
    config: TripleBarrierConfig
) -> pd.DataFrame:
    """
    Apply Triple Barrier labeling to all candidate signals.

    This is FORWARD-LOOKING and must only be used for training.
    """
    df = df.copy()

    # Get signal indices and directions
    signal_mask = df['base_signal'] != 0
    signal_indices = np.where(signal_mask)[0].astype(np.int64)
    directions = df.loc[signal_mask, 'base_signal'].values.astype(np.float64)

    # Get price arrays
    close = df['Close'].values.astype(np.float64)
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    atr = df['atr'].values.astype(np.float64)

    print(f"Applying Triple Barrier to {len(signal_indices)} signals...")

    # Run Numba-accelerated labeling
    labels, returns = _triple_barrier_numba(
        close, high, low, atr,
        signal_indices, directions,
        config.profit_take_atr_mult,
        config.stop_loss_atr_mult,
        config.max_holding_bars,
        config.use_high_low,
        config.min_return_threshold
    )

    # Store results
    df['meta_label'] = 0.0
    df['meta_return'] = 0.0
    df.loc[signal_mask, 'meta_label'] = labels
    df.loc[signal_mask, 'meta_return'] = returns

    # Stats
    win_rate = labels.mean() * 100 if len(labels) > 0 else 0
    avg_return = returns.mean() * 100 if len(returns) > 0 else 0
    print(f"Triple Barrier Results: Win Rate={win_rate:.1f}%, Avg Return={avg_return:.4f}%")

    return df


# =============================================================================
# GPU Tensor Preparation
# =============================================================================

def prepare_gpu_tensors(
    df: pd.DataFrame,
    device: str = "cuda",
    normalize: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Convert preprocessed DataFrame to GPU tensors.

    Returns dict with:
    - features: (N, num_features) market features
    - prices: (N, 4) OHLC prices
    - meta_labels: (N,) win/loss labels
    - base_signals: (N,) candidate signals (-1, 0, 1)
    - atr: (N,) ATR values for reward scaling
    """
    # Feature columns for the RL agent
    feature_cols = ['adx_norm', 'rsi_norm', 'atr_pct', 'bb_width', 'ema_diff', 'dist_fast', 'dist_slow']

    # Extract and validate
    features = df[feature_cols].values.astype(np.float32)
    prices = df[['Open', 'High', 'Low', 'Close']].values.astype(np.float32)
    meta_labels = df['meta_label'].values.astype(np.float32)
    base_signals = df['base_signal'].values.astype(np.float32)
    atr = df['atr'].values.astype(np.float32)

    # Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize features if requested
    if normalize:
        # Clip outliers and standardize
        for i in range(features.shape[1]):
            col = features[:, i]
            mean = np.nanmean(col)
            std = np.nanstd(col) + 1e-8
            col = (col - mean) / std
            col = np.clip(col, -5, 5)  # Clip extreme values
            features[:, i] = col

    # Convert to tensors on specified device
    tensors = {
        "features": torch.tensor(features, dtype=torch.float32, device=device),
        "prices": torch.tensor(prices, dtype=torch.float32, device=device),
        "meta_labels": torch.tensor(meta_labels, dtype=torch.float32, device=device),
        "base_signals": torch.tensor(base_signals, dtype=torch.float32, device=device),
        "atr": torch.tensor(atr, dtype=torch.float32, device=device),
    }

    print(f"Prepared GPU tensors on {device}:")
    for name, t in tensors.items():
        print(f"  {name}: {t.shape}, dtype={t.dtype}")

    return tensors


# =============================================================================
# Data Loading Pipeline
# =============================================================================

def load_and_prepare_data(
    data_path: Path,
    timeframe: str,
    strategy_config: StrategyConfig,
    tb_config: TripleBarrierConfig,
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Full data loading and preparation pipeline.

    1. Load CSV data
    2. Resample to timeframe
    3. Calculate indicators
    4. Generate signals
    5. Apply Triple Barrier labeling
    """
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Find datetime column
    datetime_col = None
    for col in ['DateTime', 'Date', 'Time', 'time', 'datetime', 'date']:
        if col in df.columns:
            datetime_col = col
            break

    if datetime_col is None:
        raise ValueError("No datetime column found in data")

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)
    df.index.name = 'Time'

    # Filter date range
    df = df.loc[start_date:]
    if end_date:
        df = df.loc[:end_date]

    # Resample to target timeframe
    tf_map = {'15M': '15min', '1H': '1h', '4H': '4h', '1D': '1d'}
    tf_pandas = tf_map.get(timeframe.upper(), timeframe.lower())

    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    if 'Volume' in df.columns:
        agg_dict['Volume'] = 'sum'

    df = df.resample(tf_pandas).agg(agg_dict).dropna()
    print(f"Loaded {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    # Calculate all indicators
    print("Calculating indicators...")
    df = calculate_all_indicators(df, strategy_config)

    # Generate candidate signals
    print("Generating candidate signals...")
    df = generate_candidate_signals(df, strategy_config)

    # Apply Triple Barrier labeling
    print("Applying Triple Barrier labeling...")
    df = apply_triple_barrier(df, tb_config)

    # Drop warmup period
    warmup = strategy_config.warmup_bars
    df = df.iloc[warmup:].reset_index(drop=True)

    print(f"Final dataset: {len(df):,} bars")
    return df


def create_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train/val/test sets.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df
