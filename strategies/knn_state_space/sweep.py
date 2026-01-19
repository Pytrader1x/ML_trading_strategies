#!/usr/bin/env python3
"""Parameter optimization for Contrarian KNN State-Space strategy."""

import sys
from pathlib import Path

ENGINE_PATH = Path("/Users/williamsmith/Python_local_Mac/production_backtest_engine/src")
sys.path.insert(0, str(ENGINE_PATH))
sys.path.insert(0, str(Path(__file__).parent))

from backtest_engine import Backtester, MonteCarloEngine
from strategy import KNNStateSpace
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("/Users/williamsmith/Python_local_Mac/strategy_research/data")


def load_data(instrument: str, timeframe: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{instrument}_MASTER.csv")
    for col in ['DateTime', 'Date', 'Time', 'time', 'datetime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
            break
    df = df.loc["2010-01-01":]
    tf_map = {'15M': '15min', '1H': '1h', '4H': '4h'}
    return df.resample(tf_map[timeframe]).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()


def test_config(params: dict, df: pd.DataFrame, n_sims: int = 150) -> dict:
    """Run single backtest."""
    bt = Backtester(KNNStateSpace, df, cash=1_000_000, commission=1.0, spread=0.0)
    bt.strategy.params = params
    stats = bt.run(full_report=False)
    trades_df = stats.get('trades_df', pd.DataFrame())

    if len(trades_df) < 30:
        return None

    mc = MonteCarloEngine(trades_df)
    mc_result = mc.run(n_sims=n_sims)

    return {
        'params': params,
        'trades': len(trades_df),
        'sharpe': mc_result.metrics.sharpe_50,
        'return_pct': stats.get('return_pct', 0),
        'win_rate': stats.get('win_rate', 0),
        'max_dd': mc_result.metrics.max_dd_50 * 100,
        'pf': stats.get('profit_factor', 0)
    }


def main():
    print("Loading GBPUSD 1H data...")
    df = load_data("GBPUSD", "1H")
    print(f"Loaded {len(df):,} bars")
    print("Testing Contrarian KNN parameters...\n")

    results = []

    # Key parameter combinations to test
    configs = [
        # Baseline
        ({}, "Baseline"),

        # k variations
        ({'k': 30}, "k=30"),
        ({'k': 75}, "k=75"),
        ({'k': 100}, "k=100"),

        # Horizon variations
        ({'horizon': 5}, "H=5"),
        ({'horizon': 10}, "H=10"),
        ({'horizon': 15}, "H=15"),

        # Consistency variations
        ({'consistency_thresh': 0.60}, "Cons=60%"),
        ({'consistency_thresh': 0.70}, "Cons=70%"),
        ({'consistency_thresh': 0.75}, "Cons=75%"),

        # Lookback variations
        ({'lookback': 2000}, "LB=2000"),
        ({'lookback': 4000}, "LB=4000"),

        # Risk params
        ({'sl_atr': 1.5, 'tp_atr': 2.0}, "SL1.5/TP2.0"),
        ({'sl_atr': 2.5, 'tp_atr': 4.0}, "SL2.5/TP4.0"),
        ({'sl_atr': 1.5, 'tp_atr': 3.0}, "SL1.5/TP3.0"),

        # Combined optimized
        ({'k': 75, 'horizon': 10, 'consistency_thresh': 0.70}, "Combo1"),
        ({'k': 50, 'horizon': 8, 'sl_atr': 1.5, 'tp_atr': 2.5}, "Combo2"),
        ({'k': 75, 'horizon': 10, 'lookback': 4000, 'consistency_thresh': 0.68}, "Combo3"),
        ({'k': 100, 'horizon': 12, 'consistency_thresh': 0.65, 'sl_atr': 1.5, 'tp_atr': 2.5}, "Combo4"),
    ]

    print(f"{'Config':<20} {'Trades':>7} {'Sharpe':>8} {'Return%':>10} {'WinRate':>8} {'MaxDD%':>8}")
    print("-" * 70)

    for params, label in configs:
        result = test_config(params, df)
        if result and np.isfinite(result['sharpe']):
            results.append((label, result))
            print(f"{label:<20} {result['trades']:>7} {result['sharpe']:>8.3f} "
                  f"{result['return_pct']:>9.2f}% {result['win_rate']:>7.1f}% "
                  f"{result['max_dd']:>7.2f}%")
        else:
            print(f"{label:<20} {'N/A':>7} {'N/A':>8} {'N/A':>10} {'N/A':>8} {'N/A':>8}")

    if results:
        best_label, best = max(results, key=lambda x: x[1]['sharpe'])
        print("\n" + "=" * 70)
        print(f"BEST: {best_label}")
        print(f"Sharpe: {best['sharpe']:.3f}, Return: {best['return_pct']:.2f}%, "
              f"Trades: {best['trades']}, Win Rate: {best['win_rate']:.1f}%")
        print(f"Params: {best['params']}")


if __name__ == "__main__":
    main()
