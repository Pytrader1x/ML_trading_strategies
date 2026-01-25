#!/usr/bin/env python3
"""
Out-of-Sample (OOS) Evaluation for v2_exit_guards.

Loads trained model and evaluates on 2022-2025 test data.
"""

import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
import torch

# Import from THIS experiment's frozen config and env
EXPERIMENT_DIR = Path(__file__).parent
STRATEGY_DIR = EXPERIMENT_DIR.parent.parent

sys.path.insert(0, str(EXPERIMENT_DIR))
from config import PPOConfig, Actions
from env import VectorizedExitEnv, EpisodeDataset, TradeEpisode

sys.path.insert(0, str(STRATEGY_DIR))
from model import ActorCritic


def load_episodes(episode_file: Path) -> list:
    """Load pre-computed episodes."""
    with open(episode_file, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        episodes = data['episodes']
        print(f"Loaded {data['n_episodes']} episodes for {data['instrument']} {data['timeframe']}")
    else:
        episodes = data
        print(f"Loaded {len(episodes)} episodes")

    return episodes


def evaluate_oos(
    policy: ActorCritic,
    env: VectorizedExitEnv,
    n_episodes: int = None,
) -> dict:
    """Evaluate policy on OOS data."""
    policy.eval()

    returns = []
    lengths = []
    action_counts = np.zeros(5)

    # Track per-trade details
    trade_details = []

    state = env.reset()
    episode_count = 0
    max_episodes = n_episodes or len(env.dataset)

    print(f"\nEvaluating on {max_episodes} OOS episodes...")

    while episode_count < max_episodes:
        with torch.no_grad():
            action, _, _ = policy.get_action(state, deterministic=True)

        action_np = action.cpu().numpy()
        action_counts[action_np] += 1

        state, reward, done, info = env.step(action)

        if info['episode_returns']:
            returns.extend(info['episode_returns'])
            lengths.extend(info['episode_lengths'])
            episode_count = len(returns)

            if episode_count % 100 == 0:
                print(f"  Evaluated {episode_count}/{max_episodes} episodes...")

    returns = np.array(returns[:max_episodes])
    lengths = np.array(lengths[:max_episodes])

    # Calculate metrics
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe = mean_return / (std_return + 1e-8) * np.sqrt(252)
    win_rate = (returns > 0).mean()
    profit_factor = returns[returns > 0].sum() / (-returns[returns < 0].sum() + 1e-8)
    max_drawdown = (np.maximum.accumulate(np.cumsum(returns)) - np.cumsum(returns)).max()
    total_return = returns.sum()

    metrics = {
        'n_trades': len(returns),
        'mean_return': float(mean_return),
        'std_return': float(std_return),
        'sharpe': float(sharpe),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'max_drawdown': float(max_drawdown),
        'total_return': float(total_return),
        'mean_length': float(lengths.mean()),
        'action_distribution': action_counts / action_counts.sum(),
    }

    return metrics, returns, lengths


def main():
    parser = argparse.ArgumentParser(description="OOS Evaluation for v2_exit_guards")
    parser.add_argument('--checkpoint', type=str, default='models/exit_policy_final_gpu.pt',
                        help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cuda/cpu)')
    parser.add_argument('--n-episodes', type=int, default=None, help='Number of episodes (default: all)')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    # Load OOS episodes
    oos_file = STRATEGY_DIR / "data" / "episodes_test_2022_2025.pkl"
    if not oos_file.exists():
        print(f"OOS file not found: {oos_file}")
        sys.exit(1)

    print("=" * 70)
    print(" OUT-OF-SAMPLE EVALUATION - v2_exit_guards")
    print(" Period: 2022-01-01 to 2025-01-01")
    print("=" * 70)

    episodes = load_episodes(oos_file)

    # Create config and dataset
    config = PPOConfig(n_envs=64, device=device)
    dataset = EpisodeDataset(episodes, device=device)
    env = VectorizedExitEnv(dataset, config)

    # Load trained policy
    checkpoint_path = EXPERIMENT_DIR / args.checkpoint
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"\nLoading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    policy = ActorCritic(config)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.to(device)
    policy.eval()

    print(f"Model trained for {checkpoint.get('total_steps', 'unknown'):,} steps")
    print(f"Best training return: {checkpoint.get('best_return', 'unknown')}")

    # Evaluate
    n_eps = args.n_episodes or len(dataset)
    metrics, returns, lengths = evaluate_oos(policy, env, n_episodes=n_eps)

    # Print results
    print("\n" + "=" * 70)
    print(" OOS RESULTS - v2_exit_guards (2022-2025)")
    print("=" * 70)
    print(f"  Total Trades: {metrics['n_trades']}")
    print(f"  Total Return: {metrics['total_return']:.4f}")
    print(f"  Mean Return:  {metrics['mean_return']:.4f} +/- {metrics['std_return']:.4f}")
    print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"  Win Rate:     {metrics['win_rate'] * 100:.1f}%")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")
    print(f"  Avg Length:   {metrics['mean_length']:.1f} bars")
    print(f"\n  Action Distribution:")
    for i, name in enumerate(Actions.NAMES):
        print(f"    {name}: {metrics['action_distribution'][i] * 100:.1f}%")
    print("=" * 70)

    # Return distribution
    print("\n  Return Distribution:")
    print(f"    Min:    {returns.min():.4f}")
    print(f"    25%:    {np.percentile(returns, 25):.4f}")
    print(f"    Median: {np.median(returns):.4f}")
    print(f"    75%:    {np.percentile(returns, 75):.4f}")
    print(f"    Max:    {returns.max():.4f}")

    # Length distribution
    print("\n  Length Distribution:")
    print(f"    Min:    {lengths.min():.0f} bars")
    print(f"    25%:    {np.percentile(lengths, 25):.0f} bars")
    print(f"    Median: {np.median(lengths):.0f} bars")
    print(f"    75%:    {np.percentile(lengths, 75):.0f} bars")
    print(f"    Max:    {lengths.max():.0f} bars")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
