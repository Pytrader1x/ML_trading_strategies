#!/usr/bin/env python3
"""
Batch runner for all ML strategies on all currency pairs.
Runs 15M, 1H, 4H timeframes for each strategy/pair combination.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import concurrent.futures

# Configuration
STRATEGIES = [
    # Add ML strategies here as they are developed
    # 'xgboost_regime',
    # 'lstm_trend',
    # 'transformer_price',
    # 'dqn_trading',
    # 'stacking_signals',
]

PAIRS = ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY']
START_DATE = '2005-01-01'  # Full data range: 2005-01-03 to 2025-12-05
SIMS = 300  # Reduced for speed
MAX_WORKERS = 3  # Parallel backtests


def run_strategy(strategy: str, pair: str) -> dict:
    """Run a single strategy on a single pair (all timeframes)."""
    strategy_dir = Path(__file__).parent / 'strategies' / strategy
    run_script = strategy_dir / 'run.py'

    if not run_script.exists():
        return {'strategy': strategy, 'pair': pair, 'status': 'SKIP', 'error': 'run.py not found'}

    cmd = [
        sys.executable, str(run_script),
        '-i', pair,
        '--all',
        '--sims', str(SIMS),
        '--start', START_DATE,
        '--spread', '0.0'
    ]

    start_time = datetime.now()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # 15 minute timeout for ML strategies (longer training)
            cwd=Path(__file__).parent
        )
        duration = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            return {
                'strategy': strategy,
                'pair': pair,
                'status': 'OK',
                'duration': f'{duration:.1f}s',
                'output': result.stdout[-500:] if result.stdout else ''
            }
        else:
            return {
                'strategy': strategy,
                'pair': pair,
                'status': 'FAIL',
                'error': result.stderr[-300:] if result.stderr else 'Unknown error',
                'duration': f'{duration:.1f}s'
            }
    except subprocess.TimeoutExpired:
        return {
            'strategy': strategy,
            'pair': pair,
            'status': 'TIMEOUT',
            'error': 'Exceeded 15 minute timeout'
        }
    except Exception as e:
        return {
            'strategy': strategy,
            'pair': pair,
            'status': 'ERROR',
            'error': str(e)
        }


def main():
    if not STRATEGIES:
        print("=" * 80)
        print(" ML TRADING STRATEGIES - BATCH RUNNER")
        print("=" * 80)
        print("\nNo strategies configured yet.")
        print("\nTo add strategies:")
        print("  1. Create a strategy folder in strategies/")
        print("  2. Add strategy.py and run.py")
        print("  3. Add the strategy name to STRATEGIES list in this file")
        print("\nExample structure:")
        print("  strategies/")
        print("    xgboost_regime/")
        print("      strategy.py")
        print("      run.py")
        print("      model.py (optional)")
        print("=" * 80)
        return

    print("=" * 80)
    print(" ML TRADING STRATEGIES - BATCH RUNNER")
    print(" All Strategies x All Pairs x All Timeframes")
    print("=" * 80)
    print(f"Strategies: {len(STRATEGIES)}")
    print(f"Pairs: {len(PAIRS)}")
    print(f"Timeframes: 15M, 1H, 4H")
    print(f"Total runs: {len(STRATEGIES) * len(PAIRS)}")
    print(f"Start date: {START_DATE}")
    print(f"Monte Carlo sims: {SIMS}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print("=" * 80)
    print()

    results = []
    total = len(STRATEGIES) * len(PAIRS)
    completed = 0

    # Create all jobs
    jobs = [(s, p) for p in PAIRS for s in STRATEGIES]

    start_time = datetime.now()

    # Run with parallel workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {executor.submit(run_strategy, s, p): (s, p) for s, p in jobs}

        for future in concurrent.futures.as_completed(future_to_job):
            strategy, pair = future_to_job[future]
            completed += 1

            try:
                result = future.result()
                results.append(result)

                status_icon = '✓' if result['status'] == 'OK' else '✗'
                duration = result.get('duration', '?')
                print(f"[{completed:3d}/{total}] {status_icon} {pair:8s} | {strategy:25s} | {result['status']:7s} | {duration}")

                if result['status'] != 'OK':
                    print(f"           Error: {result.get('error', 'Unknown')[:60]}")

            except Exception as e:
                print(f"[{completed:3d}/{total}] ✗ {pair:8s} | {strategy:25s} | EXCEPTION | {e}")

    total_duration = (datetime.now() - start_time).total_seconds()

    # Summary
    print()
    print("=" * 80)
    print(" SUMMARY")
    print("=" * 80)

    ok_count = sum(1 for r in results if r['status'] == 'OK')
    fail_count = sum(1 for r in results if r['status'] != 'OK')

    print(f"Completed: {ok_count}/{total} successful")
    print(f"Failed: {fail_count}")
    print(f"Total time: {total_duration/60:.1f} minutes")

    if fail_count > 0:
        print("\nFailed runs:")
        for r in results:
            if r['status'] != 'OK':
                print(f"  - {r['pair']}/{r['strategy']}: {r['status']} - {r.get('error', '')[:50]}")

    print("=" * 80)
    print("Results saved to: strategies/<strategy>/results/<PAIR>/SUMMARY.md")
    print("=" * 80)


if __name__ == '__main__':
    main()
