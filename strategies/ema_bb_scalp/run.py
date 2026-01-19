#!/usr/bin/env python3
"""
EMA BB Scalp Strategy Runner

Usage:
    # Single instrument/timeframe
    python run.py -i AUDUSD -t 1H

    # Multi-timeframe for one instrument
    python run.py -i AUDUSD --all

    # Multi-instrument
    python run.py --multi AUDUSD,EURUSD,GBPUSD -t 1H

    # Parameter sweep
    python run.py -i AUDUSD -t 15M --sweep
"""

import sys
import argparse
from pathlib import Path

# Add backtest engine to path
ENGINE_PATH = Path("/Users/williamsmith/Python_local_Mac/04_backtesting/production_backtest_engine/src")
if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))

# Add current strategy directory to path for local imports
STRATEGY_DIR = Path(__file__).parent
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from backtest_engine import (
    Backtester,
    MonteCarloEngine,
    generate_html_report,
    generate_summary_report,
    generate_instrument_summary,
)
from strategy import EMABBScalpADXStrategy

import pandas as pd
import json
import warnings

warnings.filterwarnings("ignore")

# Configuration
STRATEGY_NAME = "EMA BB Scalp ADX"
DATA_DIR = Path("/Users/williamsmith/Python_local_Mac/01_trading_strategies/strategy_research/data")
RESULTS_BASE = STRATEGY_DIR / "results"

# Default pairs available
AVAILABLE_PAIRS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
DEFAULT_TIMEFRAMES = ["15M", "1H", "4H"]


def get_results_dir(instrument: str, timeframe: str) -> Path:
    """Get (and create) results directory for a backtest run."""
    results_dir = RESULTS_BASE / instrument.upper() / timeframe.upper()
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_data(instrument: str, timeframe: str, start_date: str = "2010-01-01") -> pd.DataFrame:
    """Load and resample OHLC data."""
    data_file = DATA_DIR / f"{instrument.upper()}_MASTER.csv"

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print(f"Loading data from {data_file.name}...")
    df = pd.read_csv(data_file)

    # Standardize time column
    time_cols = ['DateTime', 'Date', 'Time', 'time', 'datetime', 'date']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
            df.index.name = 'Time'
            break

    # Filter to specified start date
    df = df.loc[start_date:]

    # Resample to target timeframe
    print(f"Resampling to {timeframe}...")
    tf_map = {
        '15M': '15min', '15m': '15min', '15MIN': '15min',
        '1H': '1h', '1h': '1h', '1HR': '1h',
        '4H': '4h', '4h': '4h', '4HR': '4h',
        '1D': '1d', '1d': '1d', 'D': '1d',
    }
    tf_pandas = tf_map.get(timeframe, timeframe.lower())

    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }
    if 'Volume' in df.columns:
        agg_dict['Volume'] = 'sum'

    df = df.resample(tf_pandas).agg(agg_dict).dropna()
    print(f"Loaded {len(df):,} bars ({df.index[0]} to {df.index[-1]})")

    return df


def run_single(
    instrument: str,
    timeframe: str,
    cash: float = 1_000_000,
    commission: float = 1.0,
    spread: float = 0.0001,
    n_sims: int = 500,
    params: dict = None,
    start_date: str = "2010-01-01"
) -> dict:
    """Run backtest for a single instrument/timeframe combination."""
    print("\n" + "=" * 60)
    print(f" {STRATEGY_NAME} - {instrument} {timeframe}")
    print("=" * 60)

    # Load data
    df = load_data(instrument, timeframe, start_date)

    # Get output directory
    output_dir = get_results_dir(instrument, timeframe)
    print(f"Results directory: {output_dir}")

    # Create and run backtester
    print("Running backtest...")
    bt = Backtester(
        EMABBScalpADXStrategy, df,
        cash=cash,
        commission=commission,
        spread=spread,
        params=params or {}
    )
    stats = bt.run(full_report=True)
    trades_df = stats.get('trades_df', pd.DataFrame())

    if trades_df.empty:
        print("No trades generated!")
        return {'stats': stats, 'mc_result': None}

    print(f"Generated {len(trades_df)} trades")

    # Monte Carlo simulation
    print(f"Running Monte Carlo ({n_sims} simulations)...")
    mc_engine = MonteCarloEngine(trades_df)
    mc_result = mc_engine.run(n_sims=n_sims, label=f"{STRATEGY_NAME}_{timeframe}")

    # Save trades.csv
    trades_df.to_csv(output_dir / "trades.csv", index=False)
    print(f"Saved: trades.csv")

    # Get indicators for plotting
    indicators = bt.strategy.get_indicators()

    # Generate HTML report
    generate_html_report(
        backtest_results=stats,
        trades_df=trades_df,
        price_df=df,
        mc_result=mc_result,
        strategy_name=STRATEGY_NAME,
        instrument=instrument,
        timeframe=timeframe,
        output_path=str(output_dir / "backtest_report.html"),
        dark_theme=True,
        indicators=indicators,
        show_trade_lines=True
    )
    print(f"Saved: backtest_report.html")

    # Generate PNG summary
    generate_summary_report(
        backtest_stats=stats,
        mc_result=mc_result,
        trades_df=trades_df,
        strategy_name=STRATEGY_NAME,
        instrument=instrument,
        timeframe=timeframe,
        output_path=str(output_dir / "backtest_summary.png"),
        dark_theme=True,
        dpi=200
    )
    print(f"Saved: backtest_summary.png")

    # Save stats JSON
    stats_json = {k: v for k, v in stats.items() if k not in ['trades_df', 'equity_curve']}
    for key, value in stats_json.items():
        if hasattr(value, 'item'):
            stats_json[key] = value.item()
    with open(output_dir / "backtest_results.json", "w") as f:
        json.dump(stats_json, f, indent=2)
    print(f"Saved: backtest_results.json")

    # Print summary
    mc_metrics = mc_result.metrics
    print("\n" + "-" * 40)
    print(f"Return:      ${stats.get('total_return', 0):,.2f} ({stats.get('return_pct', 0):.2f}%)")
    print(f"Sharpe (MC): {mc_metrics.sharpe_50:.2f}")
    print(f"Max DD (MC): {mc_metrics.max_dd_50 * 100:.2f}%")
    print(f"Win Rate:    {stats.get('win_rate', 0):.1f}%")
    print(f"Trades:      {len(trades_df)}")
    print("-" * 40)

    return {
        'stats': stats,
        'mc_result': mc_result,
        'output_dir': output_dir,
        'trades_df': trades_df
    }


def run_multi_timeframe(instrument: str, timeframes: list = None, **kwargs) -> dict:
    """Run backtest across multiple timeframes."""
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
                    'TF': tf,
                    'Return': stats.get('total_return', 0),
                    'Return %': stats.get('return_pct', 0),
                    'Sharpe (MC)': mc.sharpe_50,
                    'DD (MC)': mc.max_dd_50 * 100,
                    'Win Rate': stats.get('win_rate', 0),
                    'Trades': len(result['trades_df'])
                })
        except Exception as e:
            print(f"\nError running {tf}: {e}")
            continue

    # Print summary table
    if summary:
        print("\n" + "=" * 80)
        print(f" {STRATEGY_NAME} - {instrument} MULTI-TIMEFRAME SUMMARY")
        print("=" * 80)
        print(f"{'TF':<6} {'Return':>12} {'Return %':>10} {'Sharpe':>8} {'Max DD':>8} {'Win %':>7} {'Trades':>8}")
        print("-" * 80)
        for s in summary:
            print(f"{s['TF']:<6} ${s['Return']:>10,.0f} {s['Return %']:>9.2f}% {s['Sharpe (MC)']:>8.2f} {s['DD (MC)']:>7.2f}% {s['Win Rate']:>6.1f}% {s['Trades']:>8}")
        print("=" * 80)

        # Generate markdown summary
        instrument_results_dir = RESULTS_BASE / instrument.upper()
        print(f"\nGenerating SUMMARY.md...")
        generate_instrument_summary(
            results_dir=instrument_results_dir,
            strategy_name=STRATEGY_NAME,
            instrument=instrument
        )
        print(f"Saved: {instrument_results_dir}/SUMMARY.md")

    return results


def run_sweep(instrument: str, timeframe: str, start_date: str = "2010-01-01", **kwargs):
    """Run parameter sweep optimization."""
    from backtest_engine import ParameterSweep

    print("\n" + "=" * 60)
    print(f" PARAMETER SWEEP: {STRATEGY_NAME} - {instrument} {timeframe}")
    print("=" * 60)

    df = load_data(instrument, timeframe, start_date)

    param_grid = {
        'ema_fast': [20, 30, 40],
        'ema_slow': [50, 60, 80],
        'bb_period': [15, 20, 25],
        'sl_mult': [1.0, 1.1, 1.5],
        'tp_ratio': [1.2, 1.5, 2.0],
    }

    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"Testing {total} parameter combinations...")

    sweep_dir = RESULTS_BASE / instrument.upper() / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    sweep = ParameterSweep(
        strategy_class=EMABBScalpADXStrategy,
        data=df,
        param_grid=param_grid,
        cash=kwargs.get('cash', 1_000_000),
        commission=kwargs.get('commission', 1.0),
        spread=kwargs.get('spread', 0.0001)
    )

    results_df = sweep.run(n_sims=kwargs.get('n_sims', 300))

    print("\nTop 10 Parameter Combinations:")
    print("-" * 80)
    top10 = results_df.head(10)[['params_str', 'sharpe', 'return_pct', 'max_dd_pct', 'win_rate']]
    print(top10.to_string(index=False))

    sweep.generate_report(
        output_path=str(sweep_dir / "parameter_sweep_report.html"),
        strategy_name=STRATEGY_NAME,
        instrument=instrument,
        timeframe=timeframe
    )
    results_df.to_csv(sweep_dir / "sweep_results.csv", index=False)

    print(f"\nSweep results saved to: {sweep_dir}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description=f"Run {STRATEGY_NAME} Backtest")
    parser.add_argument('-i', '--instrument', default='AUDUSD', help='Currency pair')
    parser.add_argument('-t', '--timeframe', default='1H', help='Timeframe (15M, 1H, 4H)')
    parser.add_argument('--all', action='store_true', help='Run all timeframes')
    parser.add_argument('--multi', type=str, help='Comma-separated list of instruments')
    parser.add_argument('--sweep', action='store_true', help='Run parameter sweep')
    parser.add_argument('--sims', type=int, default=500, help='Monte Carlo simulations')
    parser.add_argument('--cash', type=float, default=1_000_000, help='Initial capital')
    parser.add_argument('--commission', type=float, default=1.0, help='Commission per trade')
    parser.add_argument('--spread', type=float, default=0.0001, help='Bid-ask spread')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')

    args = parser.parse_args()

    kwargs = {
        'cash': args.cash,
        'commission': args.commission,
        'spread': args.spread,
        'n_sims': args.sims,
        'start_date': args.start
    }

    if args.sweep:
        run_sweep(args.instrument, args.timeframe, start_date=args.start, **{k: v for k, v in kwargs.items() if k != 'start_date'})
    elif args.multi:
        instruments = [i.strip().upper() for i in args.multi.split(',')]
        for inst in instruments:
            if args.all:
                run_multi_timeframe(inst, **kwargs)
            else:
                run_single(inst, args.timeframe, **kwargs)
    elif args.all:
        run_multi_timeframe(args.instrument, **kwargs)
    else:
        run_single(args.instrument, args.timeframe, **kwargs)


if __name__ == "__main__":
    main()
