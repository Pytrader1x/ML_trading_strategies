#!/usr/bin/env python3
"""
ML Trading Strategies Analysis

Uses the backtest engine's built-in Portfolio Mode and Monte Carlo to generate
professional portfolio analysis with 12-panel summary PNGs.

Usage:
    # Run portfolio for one strategy
    python analysis/run_analysis.py portfolio -s xgboost_regime -t 15M

    # Run portfolio with specific pairs
    python analysis/run_analysis.py portfolio -s xgboost_regime -p AUD,NZD,CAD -t 1H

    # Run all strategies (15M)
    python analysis/run_analysis.py portfolio --all -t 15M

    # Generate leaderboard from existing results
    python analysis/run_analysis.py leaderboard

    # Quick summary of best performers
    python analysis/run_analysis.py summary

Output Structure:
    analysis/portfolios/{strategy}/{PAIR1_PAIR2_PAIR3}/{timeframe}/
        portfolio_summary.png   # 12-panel visual report
        portfolio_trades.csv    # All trades with instrument column
        portfolio_stats.json    # Stats, correlations, MC metrics
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
STRATEGIES_DIR = PROJECT_DIR / "strategies"
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "portfolios"

ENGINE_PATH = Path("/Users/williamsmith/Python_local_Mac/production_backtest_engine/src")
sys.path.insert(0, str(ENGINE_PATH))

# Available pairs and timeframes
ALL_PAIRS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]
PAIR_ALIASES = {
    "AUD": "AUDUSD", "EUR": "EURUSD", "GBP": "GBPUSD",
    "NZD": "NZDUSD", "CAD": "USDCAD", "CHF": "USDCHF", "JPY": "USDJPY"
}
TIMEFRAMES = ["15M", "1H", "4H"]


def get_strategy_class(strategy_name: str):
    """Dynamically import strategy class."""
    from backtest_engine import Strategy as BaseStrategy

    strategy_dir = STRATEGIES_DIR / strategy_name
    if not strategy_dir.exists():
        raise ValueError(f"Strategy not found: {strategy_name}")

    # Add strategy dir to path
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    # Import strategy module (reload to get fresh import)
    import importlib
    if "strategy" in sys.modules:
        del sys.modules["strategy"]
    strategy_module = importlib.import_module("strategy")

    # Find strategy class (any class that inherits from Strategy)
    for name in dir(strategy_module):
        obj = getattr(strategy_module, name)
        if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
            return obj

    raise ValueError(f"No Strategy class found in {strategy_name}/strategy.py")


def load_data(instrument: str, timeframe: str, start_date: str = "2005-01-01") -> pd.DataFrame:
    """Load and resample OHLC data."""
    data_file = DATA_DIR / f"{instrument.upper()}_MASTER.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)

    # Standardize time column
    for col in ['DateTime', 'Date', 'Time', 'time', 'datetime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
            df.index.name = 'Time'
            break

    df = df.loc[start_date:]

    # Resample
    tf_map = {'15M': '15min', '1H': '1h', '4H': '4h'}
    tf_pandas = tf_map.get(timeframe, timeframe.lower())

    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    if 'Volume' in df.columns:
        agg_dict['Volume'] = 'sum'

    return df.resample(tf_pandas).agg(agg_dict).dropna()


def resolve_pairs(pairs_input: str) -> List[str]:
    """Resolve pair aliases to full names."""
    if pairs_input.upper() == "ALL":
        return ALL_PAIRS

    pairs = []
    for p in pairs_input.split(","):
        p = p.strip().upper()
        if p in PAIR_ALIASES:
            pairs.append(PAIR_ALIASES[p])
        elif p in ALL_PAIRS:
            pairs.append(p)
        else:
            # Try adding USD
            if f"{p}USD" in ALL_PAIRS:
                pairs.append(f"{p}USD")
            elif f"USD{p}" in ALL_PAIRS:
                pairs.append(f"USD{p}")
    return pairs


def get_portfolio_dirname(pairs: List[str]) -> str:
    """Generate directory name from pairs (e.g., AUD_NZD_CAD)."""
    # Extract currency codes
    codes = []
    for pair in sorted(pairs):
        if pair.startswith("USD"):
            codes.append(pair[3:])
        else:
            codes.append(pair[:3])
    return "_".join(codes)


def run_portfolio(
    strategy_name: str,
    pairs: List[str],
    timeframe: str,
    n_sims: int = 500,
    cash: float = 1_000_000,
    start_date: str = "2005-01-01"
) -> Dict:
    """
    Run portfolio backtest using engine's Portfolio Mode.

    Returns portfolio results with 12-panel summary PNG.
    """
    from backtest_engine import Backtester, MonteCarloEngine
    from backtest_engine.portfolio import PortfolioAggregator, PortfolioMonteCarloEngine
    from backtest_engine.portfolio_report import generate_portfolio_summary_report

    print("\n" + "=" * 70)
    print(f"  PORTFOLIO: {strategy_name}")
    print(f"  Pairs: {', '.join(pairs)}")
    print(f"  Timeframe: {timeframe}")
    print("=" * 70)

    # Get strategy class
    strategy_class = get_strategy_class(strategy_name)

    # Run individual backtests
    instrument_results = {}
    instrument_trades = {}

    for i, pair in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] Running {pair}...")

        try:
            df = load_data(pair, timeframe, start_date)
            print(f"  Loaded {len(df):,} bars")

            bt = Backtester(strategy_class, df, cash=cash, commission=1.0, spread=0.0001)
            stats = bt.run(full_report=False)
            trades_df = stats.get('trades_df', pd.DataFrame())

            if trades_df.empty:
                print(f"  No trades for {pair}")
                continue

            print(f"  Generated {len(trades_df)} trades")

            # Run individual MC (faster, 200 sims)
            mc_engine = MonteCarloEngine(trades_df)
            mc_result = mc_engine.run(n_sims=min(n_sims, 200))

            instrument_results[pair] = {
                'stats': stats,
                'trades_df': trades_df,
                'mc_result': mc_result,
                'timeframe': timeframe
            }
            instrument_trades[pair] = trades_df

        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not instrument_results:
        print("\nNo successful backtests!")
        return None

    # Aggregate into portfolio
    print("\n" + "-" * 50)
    print("  Aggregating portfolio...")

    aggregator = PortfolioAggregator(instrument_results, initial_capital=cash)
    portfolio = aggregator.aggregate()

    # Portfolio Monte Carlo
    print(f"  Running Portfolio Monte Carlo ({n_sims} sims)...")
    mc_engine = PortfolioMonteCarloEngine(instrument_trades, capital=cash)
    portfolio.portfolio_mc_result = mc_engine.run(n_sims=n_sims)

    # Create output directory
    pairs_dirname = get_portfolio_dirname(pairs)
    output_dir = OUTPUT_DIR / strategy_name / pairs_dirname / timeframe
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate 12-panel summary PNG
    print("  Generating portfolio_summary.png...")
    summary_path = generate_portfolio_summary_report(
        portfolio_result=portfolio,
        strategy_name=f"{strategy_name} Portfolio",
        output_path=str(output_dir / "portfolio_summary.png"),
        dark_theme=True,
        dpi=200
    )

    # Save portfolio trades
    if not portfolio.portfolio_trades_df.empty:
        portfolio.portfolio_trades_df.to_csv(output_dir / "portfolio_trades.csv", index=False)

    # Save portfolio stats JSON
    stats = portfolio.portfolio_stats
    mc_metrics = portfolio.portfolio_mc_result.metrics if portfolio.portfolio_mc_result else None

    stats_to_save = {
        'strategy': strategy_name,
        'instruments': portfolio.instruments,
        'timeframe': timeframe,
        'generated': datetime.now().isoformat(),
        'portfolio_stats': stats,
        'mc_metrics': {
            'sharpe_mean': mc_metrics.sharpe_mean if mc_metrics else 0,
            'sharpe_50': mc_metrics.sharpe_50 if mc_metrics else 0,
            'sharpe_5': mc_metrics.sharpe_5 if mc_metrics else 0,
            'sharpe_95': mc_metrics.sharpe_95 if mc_metrics else 0,
            'max_dd_50': mc_metrics.max_dd_50 if mc_metrics else 0,
            'max_dd_95': mc_metrics.max_dd_95 if mc_metrics else 0,
            'prob_loss': mc_metrics.prob_loss if mc_metrics else 0,
        } if mc_metrics else None,
        'contribution_pct': portfolio.contribution_pct,
        'diversification_ratio': portfolio.diversification_ratio,
    }

    with open(output_dir / "portfolio_stats.json", 'w') as f:
        json.dump(stats_to_save, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("  PORTFOLIO SUMMARY")
    print("=" * 70)
    print(f"  Pairs:          {len(portfolio.instruments)} ({', '.join(portfolio.instruments)})")
    print(f"  Total Trades:   {stats.get('total_trades', 0)}")
    print(f"  Return:         ${stats.get('total_return', 0):,.2f} ({stats.get('return_pct', 0):.2f}%)")
    print(f"  Sharpe (MC):    {mc_metrics.sharpe_50:.2f}" if mc_metrics else "  Sharpe: N/A")
    print(f"  Max DD (MC):    {mc_metrics.max_dd_50 * 100:.2f}%" if mc_metrics else "  Max DD: N/A")
    print(f"  Win Rate:       {stats.get('win_rate', 0):.1f}%")
    print(f"  Diversification: {portfolio.diversification_ratio:.2f}x")
    print("=" * 70)

    # Contribution breakdown
    print("\n  Contribution by Pair:")
    for pair, pct in sorted(portfolio.contribution_pct.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(abs(pct) / 5) if pct > 0 else "░" * int(abs(pct) / 5)
        sign = "+" if pct > 0 else ""
        print(f"    {pair:<10} {sign}{pct:6.1f}%  {bar}")

    print(f"\n  Output: {output_dir}")

    return {
        'portfolio': portfolio,
        'output_dir': output_dir,
        'stats': stats_to_save
    }


def run_leaderboard():
    """Scan all backtest results and create leaderboard."""
    print("\nScanning backtest results...")

    results = []
    for json_file in STRATEGIES_DIR.rglob("backtest_results.json"):
        try:
            parts = json_file.relative_to(STRATEGIES_DIR).parts
            if len(parts) < 5 or parts[1] != "results":
                continue
            strategy, _, pair, tf = parts[0], parts[1], parts[2], parts[3]

            with open(json_file) as f:
                data = json.load(f)

            results.append({
                'strategy': strategy,
                'pair': pair,
                'timeframe': tf,
                'sharpe': round(data.get('sharpe_ratio', 0), 3),
                'return_pct': round(data.get('return_pct', 0), 2),
                'max_dd_pct': round(data.get('max_drawdown_pct', 0), 2),
                'win_rate': round(data.get('win_rate', 0), 1),
                'profit_factor': round(data.get('profit_factor', 0), 3),
                'total_trades': data.get('total_trades', 0),
            })
        except:
            continue

    if not results:
        print("\nNo backtest results found yet.")
        print("Run some backtests first with: python strategies/<strategy>/run.py -i AUDUSD -t 1H")
        return None

    df = pd.DataFrame(results).sort_values('sharpe', ascending=False)

    # Save outputs
    output_dir = SCRIPT_DIR / "output"
    output_dir.mkdir(exist_ok=True)

    df.to_csv(output_dir / "leaderboard.csv", index=False)

    # Generate markdown
    md = [
        "# ML Strategy Leaderboard",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total:** {len(df)} backtests\n",
        "## Top 20 by Sharpe\n",
        "| # | Strategy | Pair | TF | Sharpe | Return% | MaxDD% | WinRate | Trades |",
        "|:-:|:---------|:----:|:--:|:------:|:-------:|:------:|:-------:|:------:|"
    ]

    for i, row in df.head(20).iterrows():
        wr = "🟢" if row['win_rate'] >= 50 else "🟡" if row['win_rate'] >= 40 else "🔴"
        md.append(f"| {i+1} | {row['strategy']} | {row['pair']} | {row['timeframe']} | "
                  f"**{row['sharpe']:.2f}** | {row['return_pct']:.1f}% | {row['max_dd_pct']:.1f}% | "
                  f"{wr} {row['win_rate']:.0f}% | {row['total_trades']} |")

    with open(output_dir / "LEADERBOARD.md", 'w') as f:
        f.write("\n".join(md))

    print(f"\nFound {len(df)} results")
    print(f"Saved: {output_dir}/leaderboard.csv")
    print(f"Saved: {output_dir}/LEADERBOARD.md")

    print("\n" + "=" * 80)
    print("  TOP 10 BY SHARPE")
    print("=" * 80)
    print(df.head(10).to_string(index=False))

    return df


def get_available_strategies() -> List[str]:
    """Get list of strategies with strategy.py files."""
    strategies = []
    for item in STRATEGIES_DIR.iterdir():
        if item.is_dir() and (item / "strategy.py").exists():
            strategies.append(item.name)
    return sorted(strategies)


def main():
    parser = argparse.ArgumentParser(
        description="ML Trading Strategies Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analysis/run_analysis.py portfolio -s xgboost_regime -t 15M
  python analysis/run_analysis.py portfolio -s lstm_trend -p AUD,NZD,CAD -t 1H
  python analysis/run_analysis.py portfolio --all -t 15M
  python analysis/run_analysis.py leaderboard
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Portfolio command
    port_parser = subparsers.add_parser('portfolio', help='Run portfolio analysis')
    port_parser.add_argument('-s', '--strategy', help='Strategy name')
    port_parser.add_argument('-p', '--pairs', default='ALL', help='Pairs (AUD,NZD,CAD or ALL)')
    port_parser.add_argument('-t', '--timeframe', default='15M', help='Timeframe (15M, 1H, 4H)')
    port_parser.add_argument('--all', action='store_true', help='Run all strategies')
    port_parser.add_argument('--sims', type=int, default=500, help='Monte Carlo sims')

    # Leaderboard command
    subparsers.add_parser('leaderboard', help='Generate leaderboard from results')

    # Summary command
    subparsers.add_parser('summary', help='Quick summary of best performers')

    args = parser.parse_args()

    if args.command == 'portfolio':
        pairs = resolve_pairs(args.pairs)

        if args.all:
            strategies = get_available_strategies()
            if not strategies:
                print("\nNo strategies found in strategies/ directory.")
                print("Create a strategy first with strategy.py file.")
                return
            print(f"\nRunning portfolio for {len(strategies)} strategies...")
            for strat in strategies:
                try:
                    run_portfolio(strat, pairs, args.timeframe, n_sims=args.sims)
                except Exception as e:
                    print(f"Error with {strat}: {e}")
        elif args.strategy:
            run_portfolio(args.strategy, pairs, args.timeframe, n_sims=args.sims)
        else:
            print("Specify -s STRATEGY or --all")

    elif args.command == 'leaderboard':
        run_leaderboard()

    elif args.command == 'summary':
        df = run_leaderboard()
        if df is not None:
            print("\n\nBest by timeframe:")
            for tf in TIMEFRAMES:
                best = df[df['timeframe'] == tf].head(1)
                if not best.empty:
                    r = best.iloc[0]
                    print(f"  {tf}: {r['strategy']}/{r['pair']} Sharpe={r['sharpe']:.2f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
