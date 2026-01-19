#!/usr/bin/env python3
"""
SR Trend Follower Strategy Runner

Usage:
    python run.py -i AUDUSD -t 1H
    python run.py -i AUDUSD --all
"""

import sys
import argparse
from pathlib import Path

ENGINE_PATH = Path("/Users/williamsmith/Python_local_Mac/production_backtest_engine/src")
if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))

STRATEGY_DIR = Path(__file__).parent
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from backtest_engine import (
    Backtester, MonteCarloEngine, generate_html_report,
    generate_summary_report, generate_instrument_summary,
)
from strategy import SRTrendFollower

import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

STRATEGY_NAME = "SR Trend Follower"
DATA_DIR = Path("/Users/williamsmith/Python_local_Mac/strategy_research/data")
RESULTS_BASE = STRATEGY_DIR / "results"
DEFAULT_TIMEFRAMES = ["15M", "1H", "4H"]


def get_results_dir(instrument: str, timeframe: str) -> Path:
    results_dir = RESULTS_BASE / instrument.upper() / timeframe.upper()
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_data(instrument: str, timeframe: str, start_date: str = "2010-01-01") -> pd.DataFrame:
    data_file = DATA_DIR / f"{instrument.upper()}_MASTER.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)
    for col in ['DateTime', 'Date', 'Time', 'time', 'datetime', 'date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
            df.index.name = 'Time'
            break

    df = df.loc[start_date:]
    tf_map = {'15M': '15min', '1H': '1h', '4H': '4h', '1D': '1d'}
    tf_pandas = tf_map.get(timeframe.upper(), timeframe.lower())
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    if 'Volume' in df.columns:
        agg_dict['Volume'] = 'sum'
    df = df.resample(tf_pandas).agg(agg_dict).dropna()
    print(f"Loaded {len(df):,} bars ({df.index[0]} to {df.index[-1]})")
    return df


def run_single(instrument: str, timeframe: str, cash: float = 1_000_000,
               commission: float = 1.0, spread: float = 0.0, n_sims: int = 500,
               params: dict = None, start_date: str = "2010-01-01") -> dict:
    print("\n" + "=" * 60)
    print(f" {STRATEGY_NAME} - {instrument} {timeframe}")
    print("=" * 60)

    df = load_data(instrument, timeframe, start_date)
    output_dir = get_results_dir(instrument, timeframe)

    bt = Backtester(SRTrendFollower, df, cash=cash, commission=commission,
                    spread=spread, params=params or {})
    stats = bt.run(full_report=True)
    trades_df = stats.get('trades_df', pd.DataFrame())

    if trades_df.empty:
        print("No trades generated!")
        return {'stats': stats, 'mc_result': None}

    print(f"Generated {len(trades_df)} trades")
    mc_engine = MonteCarloEngine(trades_df)
    mc_result = mc_engine.run(n_sims=n_sims, label=f"{STRATEGY_NAME}_{timeframe}")

    trades_df.to_csv(output_dir / "trades.csv", index=False)
    indicators = bt.strategy.get_indicators()

    generate_html_report(
        backtest_results=stats, trades_df=trades_df, price_df=df,
        mc_result=mc_result, strategy_name=STRATEGY_NAME,
        instrument=instrument, timeframe=timeframe,
        output_path=str(output_dir / "backtest_report.html"),
        dark_theme=True, indicators=indicators, show_trade_lines=True
    )
    generate_summary_report(
        backtest_stats=stats, mc_result=mc_result, trades_df=trades_df,
        strategy_name=STRATEGY_NAME, instrument=instrument, timeframe=timeframe,
        output_path=str(output_dir / "backtest_summary.png"), dark_theme=True, dpi=200
    )

    stats_json = {k: v for k, v in stats.items() if k not in ['trades_df', 'equity_curve']}
    for key, value in stats_json.items():
        if hasattr(value, 'item'):
            stats_json[key] = value.item()
    with open(output_dir / "backtest_results.json", "w") as f:
        json.dump(stats_json, f, indent=2)

    mc_metrics = mc_result.metrics
    print(f"\nReturn: ${stats.get('total_return', 0):,.2f} ({stats.get('return_pct', 0):.2f}%)")
    print(f"Sharpe (MC): {mc_metrics.sharpe_50:.2f}")
    print(f"Max DD (MC): {mc_metrics.max_dd_50 * 100:.2f}%")
    print(f"Win Rate: {stats.get('win_rate', 0):.1f}%")

    return {'stats': stats, 'mc_result': mc_result, 'output_dir': output_dir, 'trades_df': trades_df}


def run_multi_timeframe(instrument: str, timeframes: list = None, **kwargs) -> dict:
    timeframes = timeframes or DEFAULT_TIMEFRAMES
    results = {}
    summary = []

    for tf in timeframes:
        try:
            result = run_single(instrument, tf, **kwargs)
            results[tf] = result
            if result['mc_result']:
                mc = result['mc_result'].metrics
                stats = result['stats']
                summary.append({
                    'TF': tf, 'Return': stats.get('total_return', 0),
                    'Return %': stats.get('return_pct', 0), 'Sharpe (MC)': mc.sharpe_50,
                    'DD (MC)': mc.max_dd_50 * 100, 'Win Rate': stats.get('win_rate', 0),
                    'Trades': len(result['trades_df'])
                })
        except Exception as e:
            print(f"\nError running {tf}: {e}")

    if summary:
        print("\n" + "=" * 80)
        print(f" {STRATEGY_NAME} - {instrument} MULTI-TIMEFRAME SUMMARY")
        print("=" * 80)
        for s in summary:
            print(f"{s['TF']:<6} ${s['Return']:>10,.0f} {s['Return %']:>9.2f}% Sharpe: {s['Sharpe (MC)']:>6.2f}")

        instrument_results_dir = RESULTS_BASE / instrument.upper()
        generate_instrument_summary(results_dir=instrument_results_dir,
                                    strategy_name=STRATEGY_NAME, instrument=instrument)
        print(f"Saved: {instrument_results_dir}/SUMMARY.md")

    return results


def main():
    parser = argparse.ArgumentParser(description=f"Run {STRATEGY_NAME} Backtest")
    parser.add_argument('-i', '--instrument', default='AUDUSD', help='Currency pair')
    parser.add_argument('-t', '--timeframe', default='1H', help='Timeframe')
    parser.add_argument('--all', action='store_true', help='Run all timeframes')
    parser.add_argument('--sims', type=int, default=500, help='Monte Carlo simulations')
    parser.add_argument('--cash', type=float, default=1_000_000, help='Initial capital')
    parser.add_argument('--commission', type=float, default=1.0, help='Commission per trade')
    parser.add_argument('--spread', type=float, default=0.0, help='Bid-ask spread')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date')

    args = parser.parse_args()
    kwargs = {'cash': args.cash, 'commission': args.commission, 'spread': args.spread,
              'n_sims': args.sims, 'start_date': args.start}

    if args.all:
        run_multi_timeframe(args.instrument, **kwargs)
    else:
        run_single(args.instrument, args.timeframe, **kwargs)


if __name__ == "__main__":
    main()
