#!/usr/bin/env python3
"""
Run Script for EMA BB V2 RL Strategy

Entry point for training and backtesting.
"""

import sys
import argparse
from pathlib import Path

# Add parent dirs to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def main():
    parser = argparse.ArgumentParser(description="EMA BB V2 RL Strategy")
    parser.add_argument("command", choices=["train", "backtest", "eval"],
                        help="Command to run")
    parser.add_argument("-i", "--instrument", default="EURUSD",
                        help="Currency pair")
    parser.add_argument("-t", "--timeframe", default="1H",
                        help="Timeframe")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    parser.add_argument("--model", default="model/best_model.pt",
                        help="Model path")

    args = parser.parse_args()

    if args.command == "train":
        print("Training RL agent...")
        print(f"  Instrument: {args.instrument}")
        print(f"  Timeframe: {args.timeframe}")
        print(f"  Epochs: {args.epochs}")
        print()
        print("TODO: Implement training")

    elif args.command == "backtest":
        print("Running backtest with trained model...")
        print(f"  Instrument: {args.instrument}")
        print(f"  Timeframe: {args.timeframe}")
        print(f"  Model: {args.model}")
        print()
        print("TODO: Implement backtest")

    elif args.command == "eval":
        print("Evaluating trained model...")
        print()
        print("TODO: Implement evaluation")


if __name__ == "__main__":
    main()
