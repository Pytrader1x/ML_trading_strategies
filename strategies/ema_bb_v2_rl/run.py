#!/usr/bin/env python3
"""
EMA BB Scalp V2 RL Strategy Runner.

Backtests the strategy with RL exit optimization.
Supports versioned experiments for model and results management.

Usage:
    python run.py -i AUDUSD -t 15M
    python run.py -i AUDUSD -t 15M --no-rl  # Use classical exits
    python run.py -i AUDUSD --all           # All timeframes
    python run.py -i AUDUSD -t 15M --version v1_baseline  # Use specific version
"""

import sys
import argparse
from pathlib import Path

ENGINE_PATH = Path("/Users/williamsmith/Python_local_Mac/04_backtesting/production_backtest_engine/src")
if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))

STRATEGY_DIR = Path(__file__).parent
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from backtest_engine import (
    Backtester, MonteCarloEngine, generate_html_report,
    generate_summary_report, generate_instrument_summary,
)
from strategy import EMABBScalpV2RLStrategy

import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

STRATEGY_NAME = "EMA BB Scalp V2 RL"
DATA_DIR = Path("/Users/williamsmith/Python_local_Mac/01_trading_strategies/ML_trading_strategies/data")
RESULTS_BASE = STRATEGY_DIR / "results"
DEFAULT_TIMEFRAMES = ["15M", "1H", "4H"]

# Global version state (set by main)
_ACTIVE_VERSION = None


def get_version_paths(version: str = None) -> dict:
    """Get paths for a specific experiment version."""
    if version:
        exp_dir = STRATEGY_DIR / "experiments" / version
        return {
            'models': exp_dir / "models",
            'results': exp_dir / "results",
            'model_file': exp_dir / "models" / "exit_policy_final.pt",
        }
    else:
        return {
            'models': STRATEGY_DIR / "models",
            'results': STRATEGY_DIR / "results",
            'model_file': STRATEGY_DIR / "models" / "exit_policy_final.pt",
        }


def get_results_dir(instrument: str, timeframe: str, version: str = None) -> Path:
    version = version or _ACTIVE_VERSION
    if version:
        results_dir = STRATEGY_DIR / "experiments" / version / "results" / instrument.upper() / timeframe.upper()
    else:
        results_dir = RESULTS_BASE / instrument.upper() / timeframe.upper()
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_data(instrument: str, timeframe: str, start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
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
    if end_date:
        df = df.loc[:end_date]
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
               params: dict = None, start_date: str = "2010-01-01", end_date: str = None,
               version: str = None) -> dict:
    version = version or _ACTIVE_VERSION
    version_str = f" [{version}]" if version else ""

    print("\n" + "=" * 60)
    print(f" {STRATEGY_NAME} - {instrument} {timeframe}{version_str}")
    print("=" * 60)

    df = load_data(instrument, timeframe, start_date, end_date)
    output_dir = get_results_dir(instrument, timeframe, version)

    # Set model path based on version
    if version:
        model_path = STRATEGY_DIR / "experiments" / version / "models" / "exit_policy_final.pt"
        if model_path.exists():
            params = params or {}
            params['model_path'] = str(model_path)
            print(f"Using model: {model_path}")

    bt = Backtester(EMABBScalpV2RLStrategy, df, cash=cash, commission=commission,
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


def run_multi_timeframe(instrument: str, timeframes: list = None, version: str = None, **kwargs) -> dict:
    version = version or _ACTIVE_VERSION
    timeframes = timeframes or DEFAULT_TIMEFRAMES
    results = {}
    summary = []

    for tf in timeframes:
        try:
            result = run_single(instrument, tf, version=version, **kwargs)
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


def compare_with_classical(instrument: str, timeframe: str, version: str = None, **kwargs) -> dict:
    """Compare RL exits with classical exits."""
    version = version or _ACTIVE_VERSION
    version_str = f" [{version}]" if version else ""

    print("\n" + "=" * 70)
    print(f" COMPARISON: RL vs Classical - {instrument} {timeframe}{version_str}")
    print("=" * 70)

    # Run with RL exits
    print("\n[RL Exits]")
    params_rl = {'use_rl_exit': True, 'use_ml': True}
    result_rl = run_single(instrument, timeframe, version=version, params={**kwargs.get('params', {}), **params_rl}, **{k: v for k, v in kwargs.items() if k not in ['params', 'version']})

    # Run with classical exits
    print("\n[Classical Exits]")
    params_classical = {'use_rl_exit': False, 'use_ml': True}
    result_classical = run_single(instrument, timeframe, version=version, params={**kwargs.get('params', {}), **params_classical}, **{k: v for k, v in kwargs.items() if k not in ['params', 'version']})

    # Compare
    if result_rl['mc_result'] and result_classical['mc_result']:
        rl_mc = result_rl['mc_result'].metrics
        cl_mc = result_classical['mc_result'].metrics
        rl_stats = result_rl['stats']
        cl_stats = result_classical['stats']

        print("\n" + "=" * 70)
        print(" COMPARISON RESULTS")
        print("=" * 70)
        print(f"{'Metric':<20} {'RL':<15} {'Classical':<15} {'Diff':>10}")
        print("-" * 70)
        print(f"{'Return %':<20} {rl_stats.get('return_pct', 0):>14.2f} {cl_stats.get('return_pct', 0):>14.2f} {rl_stats.get('return_pct', 0) - cl_stats.get('return_pct', 0):>+10.2f}")
        print(f"{'Sharpe (MC)':<20} {rl_mc.sharpe_50:>14.2f} {cl_mc.sharpe_50:>14.2f} {rl_mc.sharpe_50 - cl_mc.sharpe_50:>+10.2f}")
        print(f"{'Max DD (MC)':<20} {rl_mc.max_dd_50 * 100:>13.2f}% {cl_mc.max_dd_50 * 100:>13.2f}% {(rl_mc.max_dd_50 - cl_mc.max_dd_50) * 100:>+10.2f}")
        print(f"{'Win Rate':<20} {rl_stats.get('win_rate', 0):>13.1f}% {cl_stats.get('win_rate', 0):>13.1f}% {rl_stats.get('win_rate', 0) - cl_stats.get('win_rate', 0):>+10.1f}")
        print(f"{'Trades':<20} {len(result_rl['trades_df']):>14} {len(result_classical['trades_df']):>14}")
        print("=" * 70)

    return {'rl': result_rl, 'classical': result_classical}


def main():
    global _ACTIVE_VERSION

    parser = argparse.ArgumentParser(description=f"Run {STRATEGY_NAME} Backtest")
    parser.add_argument('-i', '--instrument', default='AUDUSD', help='Currency pair')
    parser.add_argument('-t', '--timeframe', default='15M', help='Timeframe')
    parser.add_argument('--all', action='store_true', help='Run all timeframes')
    parser.add_argument('--sims', type=int, default=500, help='Monte Carlo simulations')
    parser.add_argument('--cash', type=float, default=1_000_000, help='Initial capital')
    parser.add_argument('--commission', type=float, default=1.0, help='Commission per trade')
    parser.add_argument('--spread', type=float, default=0.0, help='Bid-ask spread')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date')
    parser.add_argument('--end', type=str, default=None, help='End date')
    parser.add_argument('--no-rl', action='store_true', help='Disable RL exits (use classical)')
    parser.add_argument('--no-ml', action='store_true', help='Disable ML entry filtering')
    parser.add_argument('--compare', action='store_true', help='Compare RL vs Classical')
    parser.add_argument('--version', type=str, default=None, help='Experiment version (e.g., v1_baseline)')

    args = parser.parse_args()

    # Set global version
    _ACTIVE_VERSION = args.version

    if args.version:
        version_dir = STRATEGY_DIR / "experiments" / args.version
        if not version_dir.exists():
            print(f"Error: Version '{args.version}' not found at {version_dir}")
            print("Available versions:")
            for d in (STRATEGY_DIR / "experiments").iterdir():
                if d.is_dir() and not d.name.startswith('.'):
                    print(f"  - {d.name}")
            return

    params = {
        'use_rl_exit': not args.no_rl,
        'use_ml': not args.no_ml,
    }

    kwargs = {
        'cash': args.cash,
        'commission': args.commission,
        'spread': args.spread,
        'n_sims': args.sims,
        'start_date': args.start,
        'end_date': args.end,
        'params': params,
        'version': args.version,
    }

    if args.compare:
        compare_with_classical(args.instrument, args.timeframe, **kwargs)
    elif args.all:
        run_multi_timeframe(args.instrument, **kwargs)
    else:
        result = run_single(args.instrument, args.timeframe, **kwargs)

        # Prompt to update results if using a version
        if args.version and result.get('mc_result'):
            print(f"\nTo update registry with these results, run:")
            print(f"  python experiment_manager.py update-results {args.version}")


if __name__ == "__main__":
    main()
