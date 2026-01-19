#!/usr/bin/env python3
"""
Main Entry Point for GPU-Optimized Meta-Labeling + RL Training

This script orchestrates the full training pipeline:
1. Load and preprocess data
2. Train meta-labeler (optional)
3. Train RL agent with PPO

Usage:
    # Local (requires CUDA GPU):
    python run_training.py --instrument AUDUSD --timeframe 1H

    # Train meta-labeler only:
    python run_training.py --instrument AUDUSD --train-meta-labeler

    # Train RL only (with existing meta-labeler):
    python run_training.py --instrument AUDUSD --train-rl

    # Full pipeline:
    python run_training.py --instrument AUDUSD --train-meta-labeler --train-rl

    # On Vast.ai:
    python run_training.py --instrument AUDUSD --train-meta-labeler --train-rl --vastai
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    StrategyConfig, TripleBarrierConfig, RLConfig,
    MetaLabelerConfig, DataConfig, get_default_configs
)
from data_factory import (
    load_and_prepare_data, prepare_gpu_tensors,
    create_train_val_test_split
)
from gpu_env import VectorizedMarket
from meta_labeler import train_meta_labeler
from train_ppo import PPOTrainer, evaluate_policy
from model import create_actor_critic


def check_cuda():
    """Verify CUDA is available and print GPU info."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. This system requires a CUDA-capable GPU.\n"
            "If on Vast.ai, ensure you selected a GPU instance."
        )

    device = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    # Memory info
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"GPU Memory: {total_mem:.1f} GB")

    return device


def main():
    parser = argparse.ArgumentParser(
        description="GPU-Optimized Meta-Labeling + RL Training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        "--instrument", "-i", default="AUDUSD",
        help="Currency pair (default: AUDUSD)"
    )
    parser.add_argument(
        "--timeframe", "-t", default="1H",
        help="Timeframe (15M, 1H, 4H) (default: 1H)"
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Data directory (default: from config)"
    )
    parser.add_argument(
        "--start-date", default="2010-01-01",
        help="Training start date"
    )
    parser.add_argument(
        "--end-date", default=None,
        help="Training end date (default: all available)"
    )

    # Training mode
    parser.add_argument(
        "--train-meta-labeler", action="store_true",
        help="Train the meta-labeler (XGBoost/Transformer)"
    )
    parser.add_argument(
        "--train-rl", action="store_true",
        help="Train the RL agent (PPO)"
    )

    # RL hyperparameters
    parser.add_argument(
        "--num-envs", type=int, default=4096,
        help="Number of parallel environments (default: 4096)"
    )
    parser.add_argument(
        "--total-updates", type=int, default=10000,
        help="Total PPO updates (default: 10000)"
    )
    parser.add_argument(
        "--episode-length", type=int, default=128,
        help="Steps per episode (default: 128)"
    )

    # Meta-labeler type
    parser.add_argument(
        "--meta-labeler-type", choices=["xgboost", "transformer"],
        default="xgboost", help="Meta-labeler model type"
    )

    # Misc
    parser.add_argument(
        "--output-dir", "-o", default="models",
        help="Output directory for models"
    )
    parser.add_argument(
        "--vastai", action="store_true",
        help="Running on Vast.ai (adjusts paths)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Check CUDA
    print("=" * 60)
    print(" GPU-Optimized Meta-Labeling + RL Training")
    print("=" * 60)
    print()

    check_cuda()
    print()

    # Load configs
    configs = get_default_configs()
    strategy_config = configs["strategy"]
    tb_config = configs["triple_barrier"]
    rl_config = configs["rl"]
    ml_config = configs["meta_labeler"]
    data_config = configs["data"]

    # Override from args
    rl_config.num_envs = args.num_envs
    rl_config.total_updates = args.total_updates
    rl_config.episode_length = args.episode_length
    ml_config.model_type = args.meta_labeler_type

    # Determine data path
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.vastai:
        data_dir = Path("/root/data")
    else:
        data_dir = Path(data_config.data_dir)

    data_path = data_dir / f"{args.instrument.upper()}_MASTER.csv"

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Step 1: Load and Prepare Data
    # ==========================================================================
    print("=" * 60)
    print(" Layer 1: Data Factory")
    print("=" * 60)
    print()

    print(f"Loading: {data_path}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Date range: {args.start_date} to {args.end_date or 'end'}")
    print()

    df = load_and_prepare_data(
        data_path,
        args.timeframe,
        strategy_config,
        tb_config,
        start_date=args.start_date,
        end_date=args.end_date
    )

    print()

    # ==========================================================================
    # Step 2: Train Meta-Labeler (Optional)
    # ==========================================================================
    if args.train_meta_labeler:
        print("=" * 60)
        print(" Layer 2: Training Meta-Labeler")
        print("=" * 60)
        print()

        meta_results = train_meta_labeler(
            df, ml_config,
            save_dir=str(output_dir)
        )

        print()

    # ==========================================================================
    # Step 3: Train RL Agent (Optional)
    # ==========================================================================
    if args.train_rl:
        print("=" * 60)
        print(" Layer 3: Training RL Agent (PPO)")
        print("=" * 60)
        print()

        # Prepare GPU tensors
        print("Preparing GPU tensors...")
        gpu_data = prepare_gpu_tensors(df, device="cuda")
        print()

        # Create environment
        print(f"Creating VectorizedMarket with {rl_config.num_envs} environments...")
        env = VectorizedMarket(
            features=gpu_data["features"],
            prices=gpu_data["prices"],
            meta_labels=gpu_data["meta_labels"],
            base_signals=gpu_data["base_signals"],
            atr=gpu_data["atr"],
            config=rl_config
        )
        print()

        # Create trainer
        print("Initializing PPO trainer...")
        trainer = PPOTrainer(env, rl_config)
        print()

        # Train
        print(f"Starting training for {rl_config.total_updates} updates...")
        print(f"Total environment steps: {rl_config.total_updates * rl_config.num_envs * rl_config.episode_length:,}")
        print()

        start_time = time.time()
        history = trainer.train(
            total_updates=rl_config.total_updates,
            save_dir=str(output_dir),
            verbose=args.verbose
        )

        elapsed = time.time() - start_time
        print()

        # Evaluate final policy
        print("Evaluating final policy...")
        eval_results = evaluate_policy(trainer.model, env, n_episodes=100)
        print(f"  Mean Reward: {eval_results['mean_reward']:.4f}")
        print(f"  Std Reward: {eval_results['std_reward']:.4f}")
        print(f"  Mean Trades: {eval_results['mean_trades']:.1f}")
        print()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 60)
    print(" Training Complete")
    print("=" * 60)
    print()
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Files created:")
    for f in output_dir.iterdir():
        print(f"  - {f.name}")
    print()

    if args.train_rl:
        print(f"Training time: {elapsed / 60:.1f} minutes")

    print("To run inference, load the model with:")
    print(f"  model = torch.load('{output_dir}/ppo_final.pt')")
    print()


if __name__ == "__main__":
    main()
