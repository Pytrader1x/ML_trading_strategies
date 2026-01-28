#!/usr/bin/env python3
"""
Episode Extraction for RL Training.

Extracts trade episodes from classical strategy backtest results.
Each episode is a pre-computed tensor containing market data from
entry to maximum hold duration.

Usage:
    python extract_episodes.py -i AUDUSD -t 15M
    python extract_episodes.py -i AUDUSD -t 15M --max-bars 200
"""

import sys
import argparse
from pathlib import Path
import pickle
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from dataclasses import dataclass

# Paths - relative to script location for portability
STRATEGY_DIR = Path(__file__).parent
DATA_DIR = STRATEGY_DIR / "data"

# Default data files for 2005-2025 dataset
DEFAULT_PRICE_FILE = DATA_DIR / "AUDUSD_15M.parquet"
DEFAULT_TRADES_FILE = DATA_DIR / "trades_train_2005_2021.csv"


@dataclass
class TradeEpisode:
    """Pre-computed trade episode."""
    entry_bar_idx: int
    entry_price: float
    direction: int
    entry_atr: float
    entry_adx: float
    entry_rsi: float
    entry_bb_width: float
    entry_ema_diff: float
    ml_confidence: float
    classical_pnl: float
    classical_exit_reason: str
    market_tensor: torch.Tensor
    valid_mask: torch.Tensor
    optimal_bar: int
    optimal_pnl: float


def calculate_ema(series: np.ndarray, period: int) -> np.ndarray:
    """Fast EMA calculation."""
    alpha = 2.0 / (period + 1)
    result = np.zeros_like(series)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def calculate_rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI calculation."""
    delta = np.diff(series, prepend=series[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    alpha = 1.0 / period
    avg_gain = np.zeros_like(series)
    avg_loss = np.zeros_like(series)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]

    for i in range(1, len(series)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR calculation."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        )
    )

    # Use SMA for simplicity
    atr = np.convolve(tr, np.ones(period) / period, mode='full')[:len(tr)]
    atr[:period] = atr[period]
    return atr


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ADX calculation."""
    up = np.diff(high, prepend=high[0])
    down = np.diff(low, prepend=low[0]) * -1

    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)

    tr = calculate_atr(high, low, close, 1)
    alpha = 1.0 / period

    # Smooth DM
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    tr_smooth = np.zeros_like(tr)

    plus_dm_smooth[0] = plus_dm[0]
    minus_dm_smooth[0] = minus_dm[0]
    tr_smooth[0] = tr[0]

    for i in range(1, len(plus_dm)):
        plus_dm_smooth[i] = alpha * plus_dm[i] + (1 - alpha) * plus_dm_smooth[i - 1]
        minus_dm_smooth[i] = alpha * minus_dm[i] + (1 - alpha) * minus_dm_smooth[i - 1]
        tr_smooth[i] = alpha * tr[i] + (1 - alpha) * tr_smooth[i - 1]

    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)

    sum_di = plus_di + minus_di
    sum_di = np.where(sum_di == 0, 1, sum_di)
    dx = np.abs(plus_di - minus_di) / sum_di * 100

    # Smooth DX to get ADX
    adx = np.zeros_like(dx)
    adx[0] = dx[0]
    for i in range(1, len(dx)):
        adx[i] = alpha * dx[i] + (1 - alpha) * adx[i - 1]

    return adx


def calculate_bollinger(series: np.ndarray, period: int = 20, dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands calculation."""
    sma = np.convolve(series, np.ones(period) / period, mode='full')[:len(series)]
    sma[:period] = sma[period]

    std = np.zeros_like(series)
    for i in range(period, len(series)):
        std[i] = np.std(series[i - period + 1:i + 1])
    std[:period] = std[period]

    upper = sma + dev * std
    lower = sma - dev * std

    return upper, lower, sma


def load_price_data(instrument: str, timeframe: str, start_date: str = "2005-01-01") -> pd.DataFrame:
    """Load price data from parquet file."""
    # Try parquet first, then CSV
    parquet_file = DATA_DIR / f"{instrument.upper()}_{timeframe.upper()}.parquet"
    csv_file = DATA_DIR / f"{instrument.upper()}_{timeframe.upper()}.csv"

    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
    elif csv_file.exists():
        df = pd.read_csv(csv_file)
        # Find datetime column
        for col in ['DateTime', 'Date', 'Time', 'time', 'datetime', 'date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                df.index.name = 'Time'
                break
    else:
        raise FileNotFoundError(f"Data file not found: {parquet_file} or {csv_file}")

    df = df.loc[start_date:]
    return df


def load_trades(instrument: str, timeframe: str) -> pd.DataFrame:
    """Load trades from data directory."""
    # Look for trades file in data directory
    trades_file = DATA_DIR / f"trades_{instrument.lower()}_{timeframe.lower()}.csv"
    if not trades_file.exists():
        # Try default training file
        trades_file = DEFAULT_TRADES_FILE
    if not trades_file.exists():
        raise FileNotFoundError(f"Trades file not found: {trades_file}")

    df = pd.read_csv(trades_file)

    # Parse datetime columns
    for col in ['Entry Time', 'Exit Time', 'entry_time', 'exit_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df


def extract_episodes(
    price_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    max_bars: int = 200,
    n_features: int = 10,
) -> List[TradeEpisode]:
    """
    Extract trade episodes from price data and trades.

    Features per bar (10):
        0: Normalized PnL: (close - entry) / entry * direction
        1: Max favorable excursion (up to this bar)
        2: Max adverse excursion (up to this bar)
        3: ATR / price (volatility)
        4: ADX / 100 (trend strength)
        5: RSI / 100 (momentum)
        6: BB position: (close - lower) / (upper - lower)
        7: EMA spread: (fast - slow) / close
        8: Bar return: (close - prev_close) / prev_close
        9: Bars since entry (normalized)
    """
    print("Computing indicators...")

    # Extract numpy arrays
    close = price_df['Close'].values
    high = price_df['High'].values
    low = price_df['Low'].values

    # Pre-compute all indicators
    ema_fast = calculate_ema(close, 30)
    ema_slow = calculate_ema(close, 50)
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)

    # Derived features
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    ema_diff = (ema_fast - ema_slow) / (close + 1e-10)
    bb_pos = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    returns = np.diff(close, prepend=close[0]) / (np.roll(close, 1) + 1e-10)
    returns[0] = 0

    print(f"Processing {len(trades_df)} trades...")

    episodes = []
    skipped = 0

    for _, trade in trades_df.iterrows():
        # Find entry bar index - try multiple column names
        entry_idx = trade.get('entry_idx')
        if entry_idx is not None:
            idx = int(entry_idx)
        else:
            entry_time = trade.get('EntryTime', trade.get('Entry Time', trade.get('entry_time')))
            if entry_time is None:
                skipped += 1
                continue

            # Find closest bar
            try:
                idx = price_df.index.get_indexer([entry_time], method='nearest')[0]
            except Exception:
                skipped += 1
                continue

        if idx < 50 or idx + max_bars >= len(price_df):
            skipped += 1
            continue

        # Entry info
        entry_price = trade.get('entry_price', close[idx])
        if isinstance(entry_price, str):
            entry_price = float(entry_price)

        direction_val = trade.get('direction', 'Long')
        direction = 1 if direction_val in ['Long', 'long', 1, '1'] else -1

        # Entry indicators
        entry_atr = atr[idx]
        entry_adx = adx[idx]
        entry_rsi = rsi[idx]
        entry_bb_width = bb_width[idx]
        entry_ema_diff = ema_diff[idx]

        # Build market tensor
        market_tensor = torch.zeros(max_bars, n_features, dtype=torch.float32)
        valid_mask = torch.zeros(max_bars, dtype=torch.bool)

        max_favorable_so_far = 0.0
        max_adverse_so_far = 0.0
        optimal_bar = 0
        optimal_pnl = 0.0

        for j in range(max_bars):
            bar_idx = idx + j
            if bar_idx >= len(close):
                break

            valid_mask[j] = True

            # Normalized PnL
            pnl = (close[bar_idx] - entry_price) / entry_price * direction

            # Track MFE/MAE
            if pnl > max_favorable_so_far:
                max_favorable_so_far = pnl
                optimal_bar = j
                optimal_pnl = pnl
            if pnl < max_adverse_so_far:
                max_adverse_so_far = pnl

            market_tensor[j] = torch.tensor([
                pnl,                                    # 0: Normalized PnL
                max_favorable_so_far,                   # 1: MFE
                max_adverse_so_far,                     # 2: MAE
                atr[bar_idx] / close[bar_idx],          # 3: ATR %
                adx[bar_idx] / 100.0,                   # 4: ADX
                rsi[bar_idx] / 100.0,                   # 5: RSI
                bb_pos[bar_idx],                        # 6: BB position
                ema_diff[bar_idx] * 100,                # 7: EMA spread
                returns[bar_idx],                       # 8: Return
                j / max_bars,                           # 9: Time normalized
            ])

        # Classical exit info - use pnl_pct if available, otherwise compute from PnL
        classical_pnl_pct = trade.get('pnl_pct', None)
        if classical_pnl_pct is not None:
            classical_pnl = float(classical_pnl_pct) if not isinstance(classical_pnl_pct, (int, float)) else classical_pnl_pct
        else:
            classical_pnl = trade.get('PnL', trade.get('pnl', 0))
            if isinstance(classical_pnl, str):
                classical_pnl = float(classical_pnl.replace(',', ''))
            # Normalize: if large absolute value, it's in dollars not percentage
            if abs(classical_pnl) > 100:
                classical_pnl = classical_pnl / (entry_price * 1_000_000) * direction

        classical_exit_reason = str(trade.get('exit_reason', trade.get('Exit Reason', 'unknown')))

        episode = TradeEpisode(
            entry_bar_idx=idx,
            entry_price=entry_price,
            direction=direction,
            entry_atr=entry_atr,
            entry_adx=entry_adx,
            entry_rsi=entry_rsi,
            entry_bb_width=entry_bb_width,
            entry_ema_diff=entry_ema_diff,
            ml_confidence=trade.get('ml_confidence', 0.55),
            classical_pnl=classical_pnl,
            classical_exit_reason=classical_exit_reason,
            market_tensor=market_tensor,
            valid_mask=valid_mask,
            optimal_bar=optimal_bar,
            optimal_pnl=optimal_pnl,
        )
        episodes.append(episode)

    print(f"Extracted {len(episodes)} episodes (skipped {skipped})")

    return episodes


def analyze_episodes(episodes: List[TradeEpisode]):
    """Print episode statistics."""
    if not episodes:
        print("No episodes to analyze")
        return

    classical_pnls = [e.classical_pnl for e in episodes]
    optimal_pnls = [e.optimal_pnl for e in episodes]
    optimal_bars = [e.optimal_bar for e in episodes]
    directions = [e.direction for e in episodes]

    print("\n" + "=" * 60)
    print(" EPISODE STATISTICS")
    print("=" * 60)
    print(f"Total episodes: {len(episodes)}")
    print(f"Long trades: {sum(1 for d in directions if d == 1)}")
    print(f"Short trades: {sum(1 for d in directions if d == -1)}")
    print()
    print("Classical Strategy Performance:")
    print(f"  Mean PnL: {np.mean(classical_pnls):.4f}")
    print(f"  Std PnL: {np.std(classical_pnls):.4f}")
    print(f"  Win Rate: {sum(1 for p in classical_pnls if p > 0) / len(classical_pnls) * 100:.1f}%")
    print()
    print("Optimal Exit (Oracle):")
    print(f"  Mean PnL: {np.mean(optimal_pnls):.4f}")
    print(f"  Std PnL: {np.std(optimal_pnls):.4f}")
    print(f"  Win Rate: {sum(1 for p in optimal_pnls if p > 0) / len(optimal_pnls) * 100:.1f}%")
    print(f"  Mean Optimal Bar: {np.mean(optimal_bars):.1f}")
    print()
    print(f"Improvement Potential: {(np.mean(optimal_pnls) - np.mean(classical_pnls)) * 100:.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Extract RL Training Episodes")
    parser.add_argument('-i', '--instrument', default='AUDUSD', help='Currency pair')
    parser.add_argument('-t', '--timeframe', default='15M', help='Timeframe')
    parser.add_argument('--max-bars', type=int, default=200, help='Max bars per episode')
    parser.add_argument('--start', type=str, default='2005-01-01', help='Start date')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--trades-file', type=str, default=None, help='Direct path to trades CSV')
    parser.add_argument('--price-file', type=str, default=None, help='Direct path to price CSV')
    args = parser.parse_args()

    print(f"\nExtracting episodes for {args.instrument} {args.timeframe}")
    print("=" * 60)

    # Load data
    print("Loading price data...")
    if args.price_file:
        price_file = Path(args.price_file)
        if price_file.suffix == '.parquet':
            df = pd.read_parquet(price_file)
        else:
            df = pd.read_csv(price_file)
            for col in ['DateTime', 'Date', 'Time', 'time', 'datetime', 'date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    df.index.name = 'Time'
                    break
        price_df = df.loc[args.start:]
    else:
        price_df = load_price_data(args.instrument, args.timeframe, args.start)
    print(f"Loaded {len(price_df):,} bars ({price_df.index[0]} to {price_df.index[-1]})")

    print("\nLoading trades...")
    if args.trades_file:
        trades_df = pd.read_csv(args.trades_file, parse_dates=['entry_time', 'exit_time'])
    else:
        trades_df = load_trades(args.instrument, args.timeframe)
    print(f"Loaded {len(trades_df)} trades")

    # Extract episodes
    print("\nExtracting episodes...")
    episodes = extract_episodes(price_df, trades_df, max_bars=args.max_bars)

    # Analyze
    analyze_episodes(episodes)

    # Save
    output_dir = STRATEGY_DIR / "data"
    output_dir.mkdir(exist_ok=True)

    output_file = args.output or output_dir / f"episodes_{args.instrument}_{args.timeframe}.pkl"
    output_file = Path(output_file)

    print(f"\nSaving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'episodes': episodes,
            'instrument': args.instrument,
            'timeframe': args.timeframe,
            'max_bars': args.max_bars,
            'n_episodes': len(episodes),
        }, f)

    print("Done!")


if __name__ == "__main__":
    main()
