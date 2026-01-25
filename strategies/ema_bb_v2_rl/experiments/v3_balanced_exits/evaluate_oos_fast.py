#!/usr/bin/env python3
"""
Fast OOS Evaluation for v3_balanced_exits.
No counterfactual computation - just runs model decisions and tracks outcomes.
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).parent
STRATEGY_DIR = EXPERIMENT_DIR.parent.parent

sys.path.insert(0, str(EXPERIMENT_DIR))
from config import PPOConfig, Actions
from env import TradeEpisode

sys.path.insert(0, str(STRATEGY_DIR))
from model import ActorCritic


def load_episodes(episode_file: Path) -> list:
    with open(episode_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        episodes = data['episodes']
        print(f"Loaded {data['n_episodes']} episodes")
    else:
        episodes = data
        print(f"Loaded {len(episodes)} episodes")
    return episodes


def build_state(episode: TradeEpisode, bar: int, sl_atr: float, max_fav: float, max_adv: float, action_hist: list):
    """Build state tensor for single episode at given bar."""
    market = episode.market_tensor[bar]  # (n_features,)

    unrealized_pnl = market[0].item()
    max_fav = max(max_fav, unrealized_pnl)
    max_adv = min(max_adv, unrealized_pnl)

    # Position features
    bars_held_norm = bar / 200.0
    position_features = [bars_held_norm, unrealized_pnl, max_fav, max_adv, sl_atr / 2.0]

    # Market features (first 10)
    market_feats = market[:10].tolist()

    # Entry context
    entry_ctx = [
        episode.entry_atr,
        episode.entry_adx / 100.0,
        episode.entry_rsi / 100.0,
        episode.entry_bb_width,
        episode.entry_ema_diff,
    ]

    # Action history (last 5)
    hist = action_hist[-5:] if len(action_hist) >= 5 else [0.0] * (5 - len(action_hist)) + action_hist

    state = position_features + market_feats + entry_ctx + hist
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0), max_fav, max_adv


def evaluate_episode(policy, episode: TradeEpisode, device: str):
    """Evaluate policy on single episode. Returns (return, length, actions)."""
    max_bars = episode.market_tensor.shape[0]
    sl_atr = 1.1
    max_fav = 0.0
    max_adv = 0.0
    action_hist = []
    actions_taken = []

    for bar in range(max_bars):
        if not episode.valid_mask[bar]:
            # End of episode - return final PnL
            pnl = episode.market_tensor[bar-1, 0].item() if bar > 0 else 0.0
            return pnl, bar, actions_taken

        state, max_fav, max_adv = build_state(episode, bar, sl_atr, max_fav, max_adv, action_hist)
        state = state.to(device)

        with torch.no_grad():
            action, _, _ = policy.get_action(state, deterministic=True)

        action = action.item()
        actions_taken.append(action)
        action_hist.append(action / 4.0)

        unrealized_pnl = episode.market_tensor[bar, 0].item()

        # Process action
        if action == Actions.EXIT:
            return unrealized_pnl, bar + 1, actions_taken

        elif action == Actions.PARTIAL_EXIT:
            return unrealized_pnl, bar + 1, actions_taken

        elif action == Actions.TIGHTEN_SL:
            sl_atr *= 0.75

        elif action == Actions.TRAIL_BREAKEVEN:
            if unrealized_pnl > 0:
                sl_atr = 0.1

        # Check SL hit
        if bar + 1 < max_bars:
            next_pnl = episode.market_tensor[bar + 1, 0].item()
            sl_pnl_threshold = -sl_atr * episode.entry_atr / episode.entry_price
            if next_pnl < sl_pnl_threshold:
                return sl_pnl_threshold, bar + 2, actions_taken

    # Reached end
    return episode.market_tensor[max_bars-1, 0].item(), max_bars, actions_taken


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='models/exit_policy_final.pt')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = args.device

    print("=" * 70)
    print(" FAST OOS EVALUATION - v3_balanced_exits")
    print(" Period: 2022-01-01 to 2025-01-01")
    print("=" * 70)

    # Load OOS data
    oos_file = STRATEGY_DIR / "data" / "episodes_test_2022_2025.pkl"
    episodes = load_episodes(oos_file)

    # Load model
    checkpoint_path = EXPERIMENT_DIR / args.checkpoint
    print(f"Loading model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = PPOConfig(device=device)
    policy = ActorCritic(config)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.to(device)
    policy.eval()

    print(f"Model trained for {checkpoint.get('total_steps', 'N/A'):,} steps")
    print(f"Experiment: {checkpoint.get('experiment', 'unknown')}")

    # Evaluate
    returns = []
    lengths = []
    all_actions = []

    print(f"\nEvaluating {len(episodes)} OOS trades...")

    for i, ep in enumerate(episodes):
        ret, length, actions = evaluate_episode(policy, ep, device)
        returns.append(ret)
        lengths.append(length)
        all_actions.extend(actions)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(episodes)} done...")

    returns = np.array(returns)
    lengths = np.array(lengths)

    # Calculate metrics
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = mean_ret / (std_ret + 1e-8) * np.sqrt(252)
    win_rate = (returns > 0).mean()
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    profit_factor = wins.sum() / (-losses.sum() + 1e-8) if len(losses) > 0 else float('inf')

    cumret = np.cumsum(returns)
    max_dd = (np.maximum.accumulate(cumret) - cumret).max()

    # Action distribution
    action_counts = np.bincount(all_actions, minlength=5)
    action_pct = action_counts / action_counts.sum() * 100

    print("\n" + "=" * 70)
    print(" OOS RESULTS - v3_balanced_exits (2022-2025)")
    print("=" * 70)
    print(f"  Trades:        {len(returns)}")
    print(f"  Total Return:  {returns.sum():.4f}")
    print(f"  Mean Return:   {mean_ret:.4f} +/- {std_ret:.4f}")
    print(f"  Sharpe Ratio:  {sharpe:.2f}")
    print(f"  Win Rate:      {win_rate * 100:.1f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Max Drawdown:  {max_dd:.4f}")
    print(f"  Avg Length:    {lengths.mean():.1f} bars")
    print()
    print("  Action Distribution:")
    for i, name in enumerate(Actions.NAMES):
        print(f"    {name:12s}: {action_pct[i]:5.1f}%")
    print()
    print("  Return Distribution:")
    print(f"    Min:    {returns.min():.4f}")
    print(f"    25%:    {np.percentile(returns, 25):.4f}")
    print(f"    Median: {np.median(returns):.4f}")
    print(f"    75%:    {np.percentile(returns, 75):.4f}")
    print(f"    Max:    {returns.max():.4f}")
    print()
    print("  Length Distribution:")
    print(f"    Min:    {lengths.min():.0f} bars")
    print(f"    Median: {np.median(lengths):.0f} bars")
    print(f"    Max:    {lengths.max():.0f} bars")
    print("=" * 70)

    # Compare to classical strategy
    classical_returns = np.array([ep.classical_pnl for ep in episodes])
    classical_sharpe = classical_returns.mean() / (classical_returns.std() + 1e-8) * np.sqrt(252)
    classical_win = (classical_returns > 0).mean()

    print("\n  vs Classical Strategy:")
    print(f"    Classical Sharpe: {classical_sharpe:.2f} | RL Sharpe: {sharpe:.2f}")
    print(f"    Classical Win:    {classical_win*100:.1f}% | RL Win: {win_rate*100:.1f}%")
    print(f"    Classical Return: {classical_returns.sum():.4f} | RL Return: {returns.sum():.4f}")
    print("=" * 70)

    # v3 specific: Check action diversity
    print("\n  v3 ACTION DIVERSITY CHECK:")
    diverse = all(pct > 1 for pct in action_pct)
    if diverse:
        print("    PASS - All actions used > 1%")
    else:
        print("    FAIL - Policy collapsed to limited actions")
        for i, name in enumerate(Actions.NAMES):
            if action_pct[i] < 1:
                print(f"      {name}: {action_pct[i]:.1f}% (below threshold)")
    print("=" * 70)


if __name__ == "__main__":
    main()
