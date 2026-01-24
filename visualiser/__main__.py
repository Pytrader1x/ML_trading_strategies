#!/usr/bin/env python3
"""
Entry point for running the visualizer as a module.

Usage:
    python -m visualiser
    python -m visualiser --strategy ema_bb_v2_rl
    python -m visualiser --port 8080 --speed 20
"""

import argparse
import webbrowser
from pathlib import Path

from . import VisualizerConfig, run_visualizer


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="RL Trade Visualizer - Real-time visualization of RL trading decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m visualiser                        # Run with defaults
  python -m visualiser --strategy my_strat    # Custom strategy
  python -m visualiser --port 8080            # Custom port
  python -m visualiser --no-browser           # Don't auto-open browser
        """
    )

    parser.add_argument(
        '--strategy', '-s',
        type=str,
        default='ema_bb_v2_rl',
        help='Strategy name (directory under strategies/)'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8765,
        help='Server port (default: 8765)'
    )
    parser.add_argument(
        '--speed',
        type=int,
        default=10,
        help='Initial playback speed (default: 10x)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model checkpoint (optional)'
    )
    parser.add_argument(
        '--price-data',
        type=str,
        default=None,
        help='Path to price data CSV (optional)'
    )
    parser.add_argument(
        '--trades-data',
        type=str,
        default=None,
        help='Path to trades data CSV (optional)'
    )
    parser.add_argument(
        '--oos-start',
        type=str,
        default='2022-01-01',
        help='OOS period start date (default: 2022-01-01)'
    )
    parser.add_argument(
        '--oos-end',
        type=str,
        default='2025-12-31',
        help='OOS period end date (default: 2025-12-31)'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not auto-open browser'
    )

    args = parser.parse_args()

    # Build config
    config_kwargs = {
        'strategy_name': args.strategy,
        'port': args.port,
        'initial_speed': args.speed,
        'oos_start': args.oos_start,
        'oos_end': args.oos_end,
        'auto_open_browser': not args.no_browser,
    }

    if args.model:
        config_kwargs['model_path'] = Path(args.model)
    if args.price_data:
        config_kwargs['price_data_path'] = Path(args.price_data)
    if args.trades_data:
        config_kwargs['trades_data_path'] = Path(args.trades_data)

    config = VisualizerConfig(**config_kwargs)

    # Open browser if requested
    if config.auto_open_browser:
        import threading
        import time

        def open_browser():
            time.sleep(1.5)  # Wait for server to start
            webbrowser.open(f'http://localhost:{config.port}')

        threading.Thread(target=open_browser, daemon=True).start()

    # Run visualizer
    run_visualizer(config)


if __name__ == "__main__":
    main()
