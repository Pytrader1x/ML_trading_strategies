#!/usr/bin/env python3
"""
Fast OOS Evaluation for v4_no_lookahead.

KEY CHANGES from v3:
1. Action masking: EXIT/PARTIAL blocked until min_exit_bar
2. Proper time-weighted Sharpe calculation
3. Lookahead exploitation detection
4. Warning thresholds for unrealistic metrics
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


def get_action_mask(bar: int, min_exit_bar: int):
    """Get action mask for single episode at given bar."""
    mask = torch.ones(1, 5, dtype=torch.bool)

    if bar < min_exit_bar:
        # Block EXIT and PARTIAL_EXIT
        mask[0, Actions.EXIT] = False
        mask[0, Actions.PARTIAL_EXIT] = False

    return mask


def build_state(episode, bar: int, sl_atr: float, max_fav: float, max_adv: float, action_hist: list):
    """Build state tensor for single episode at given bar."""
    market = episode.market_tensor[bar]

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


def evaluate_episode(policy, episode, device: str, min_exit_bar: int):
    """Evaluate policy on single episode with action masking. Returns (return, length, actions, early_exit_attempts)."""
    max_bars = episode.market_tensor.shape[0]
    sl_atr = 1.1
    max_fav = 0.0
    max_adv = 0.0
    action_hist = []
    actions_taken = []
    early_exit_attempts = 0

    for bar in range(max_bars):
        if not episode.valid_mask[bar]:
            # End of episode - return final PnL
            pnl = episode.market_tensor[bar-1, 0].item() if bar > 0 else 0.0
            return pnl, bar, actions_taken, early_exit_attempts

        state, max_fav, max_adv = build_state(episode, bar, sl_atr, max_fav, max_adv, action_hist)
        state = state.to(device)

        # V4: Apply action mask
        action_mask = get_action_mask(bar, min_exit_bar).to(device)

        with torch.no_grad():
            action, _, _ = policy.get_action(state, deterministic=True, action_mask=action_mask)

        action = action.item()
        actions_taken.append(action)
        action_hist.append(action / 4.0)

        unrealized_pnl = episode.market_tensor[bar, 0].item()

        # V4: Track early exit attempts (should be zero with proper masking)
        if bar < min_exit_bar and action in [Actions.EXIT, Actions.PARTIAL_EXIT]:
            early_exit_attempts += 1
            # Convert to HOLD (shouldn't happen with masking, but safety)
            action = Actions.HOLD

        # Process action
        if action == Actions.EXIT:
            return unrealized_pnl, bar + 1, actions_taken, early_exit_attempts

        elif action == Actions.PARTIAL_EXIT:
            return unrealized_pnl, bar + 1, actions_taken, early_exit_attempts

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
                return sl_pnl_threshold, bar + 2, actions_taken, early_exit_attempts

    # Reached end
    return episode.market_tensor[max_bars-1, 0].item(), max_bars, actions_taken, early_exit_attempts


def calculate_proper_sharpe(returns: np.ndarray, lengths: np.ndarray, timeframe_min: int = 15) -> float:
    """
    Calculate proper time-weighted Sharpe ratio.

    The standard Sharpe calculation assumes uniform time periods. Trading strategies
    have varying trade durations, so we need to account for that.

    Args:
        returns: Array of trade returns (as decimals)
        lengths: Array of trade durations in bars
        timeframe_min: Minutes per bar (15 for 15M timeframe)

    Returns:
        Annualized Sharpe ratio accounting for trade duration
    """
    if len(returns) == 0:
        return 0.0

    # Convert to hourly returns
    hours = lengths * timeframe_min / 60.0
    hours = np.maximum(hours, 1.0)  # Minimum 1 hour to avoid division issues

    hourly_returns = returns / hours

    # Trading hours per year (approximately)
    # 252 trading days * ~10 active hours per day
    trading_hours_per_year = 252 * 10

    mean_hourly = hourly_returns.mean()
    std_hourly = hourly_returns.std()

    if std_hourly < 1e-8:
        return 0.0

    # Annualize
    sharpe = mean_hourly / std_hourly * np.sqrt(trading_hours_per_year)

    return float(sharpe)


def calculate_trade_based_sharpe(returns: np.ndarray, trades_per_year: float = 442) -> float:
    """
    Calculate trade-based Sharpe (simpler, standard approach).

    Args:
        returns: Array of trade returns (as decimals)
        trades_per_year: Expected number of trades per year

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret < 1e-8:
        return 0.0

    # Annualize using trades per year (not sqrt(252))
    sharpe = mean_ret / std_ret * np.sqrt(trades_per_year)

    return float(sharpe)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='models/exit_policy_final.pt')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = args.device

    print("=" * 70)
    print(" FAST OOS EVALUATION - v4_no_lookahead")
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
    min_exit_bar = config.reward.min_exit_bar

    policy = ActorCritic(config)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.to(device)
    policy.eval()

    print(f"Model trained for {checkpoint.get('total_steps', 'N/A'):,} steps")
    print(f"Experiment: {checkpoint.get('experiment', 'unknown')}")
    print(f"min_exit_bar: {min_exit_bar}")

    # Evaluate
    returns = []
    lengths = []
    all_actions = []
    total_early_exit_attempts = 0

    print(f"\nEvaluating {len(episodes)} OOS trades...")

    for i, ep in enumerate(episodes):
        ret, length, actions, early_attempts = evaluate_episode(policy, ep, device, min_exit_bar)
        returns.append(ret)
        lengths.append(length)
        all_actions.extend(actions)
        total_early_exit_attempts += early_attempts

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(episodes)} done...")

    returns = np.array(returns)
    lengths = np.array(lengths)

    # Calculate metrics
    mean_ret = returns.mean()
    std_ret = returns.std()

    # V4: Proper Sharpe calculations
    # Standard (wrong for trades)
    naive_sharpe = mean_ret / (std_ret + 1e-8) * np.sqrt(252)

    # Trade-based (correct)
    # Estimate trades per year from OOS data (3 years, ~1326 trades => ~442/year)
    trades_per_year = len(returns) / 3.0  # OOS is 2022-2025 (3 years)
    trade_sharpe = calculate_trade_based_sharpe(returns, trades_per_year)

    # Time-weighted (most accurate)
    time_weighted_sharpe = calculate_proper_sharpe(returns, lengths, timeframe_min=15)

    win_rate = (returns > 0).mean()
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    profit_factor = wins.sum() / (-losses.sum() + 1e-8) if len(losses) > 0 else float('inf')

    cumret = np.cumsum(returns)
    max_dd = (np.maximum.accumulate(cumret) - cumret).max()

    # Action distribution
    action_counts = np.bincount(all_actions, minlength=5)
    action_pct = action_counts / action_counts.sum() * 100

    # V4: Lookahead metrics
    quick_exits = (lengths <= min_exit_bar).sum()
    quick_exit_rate = quick_exits / len(lengths)
    bar_1_exits = (lengths <= 1).sum()
    bar_2_exits = ((lengths > 1) & (lengths <= 2)).sum()

    print("\n" + "=" * 70)
    print(" OOS RESULTS - v4_no_lookahead (2022-2025)")
    print("=" * 70)
    print(f"  Trades:        {len(returns)}")
    print(f"  Total Return:  {returns.sum():.4f}")
    print(f"  Mean Return:   {mean_ret:.4f} +/- {std_ret:.4f}")

    print()
    print("  Sharpe Ratios:")
    print(f"    Naive (sqrt(252)):     {naive_sharpe:.2f}  (WRONG for trades)")
    print(f"    Trade-based:           {trade_sharpe:.2f}  (sqrt(trades/year))")
    print(f"    Time-weighted:         {time_weighted_sharpe:.2f}  (most accurate)")

    print()
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
    print(f"    Mean:   {lengths.mean():.1f} bars")
    print(f"    Max:    {lengths.max():.0f} bars")
    print("=" * 70)

    # Compare to classical strategy
    classical_returns = np.array([ep.classical_pnl for ep in episodes])
    classical_sharpe = calculate_trade_based_sharpe(classical_returns, trades_per_year)
    classical_win = (classical_returns > 0).mean()

    print("\n  vs Classical Strategy:")
    print(f"    Classical Sharpe: {classical_sharpe:.2f} | RL Sharpe: {trade_sharpe:.2f}")
    print(f"    Classical Win:    {classical_win*100:.1f}% | RL Win: {win_rate*100:.1f}%")
    print(f"    Classical Return: {classical_returns.sum():.4f} | RL Return: {returns.sum():.4f}")
    print("=" * 70)

    # V4: Anti-lookahead checks
    print("\n" + "=" * 70)
    print(" V4 ANTI-LOOKAHEAD VALIDATION")
    print("=" * 70)

    checks_passed = True

    # Check 1: No exits before min_exit_bar
    print(f"\n  1. Early Exit Attempts (bar < {min_exit_bar}):")
    print(f"     Blocked attempts: {total_early_exit_attempts}")
    if total_early_exit_attempts > 0:
        print("     FAIL - Action masking may not be working correctly")
        checks_passed = False
    else:
        print("     PASS - No early exit attempts")

    # Check 2: Quick exits
    print(f"\n  2. Quick Exits (length <= {min_exit_bar} bars):")
    print(f"     Count: {quick_exits} ({quick_exit_rate*100:.1f}%)")
    print(f"     Bar 1: {bar_1_exits}, Bar 2: {bar_2_exits}")
    if quick_exit_rate > 0.1:
        print("     WARNING - High quick exit rate may indicate lookahead")
        checks_passed = False
    else:
        print("     PASS - Quick exit rate acceptable")

    # Check 3: Average length
    print(f"\n  3. Average Episode Length:")
    print(f"     Mean: {lengths.mean():.1f} bars")
    if lengths.mean() < 5:
        print("     FAIL - Avg length < 5 bars indicates possible lookahead")
        checks_passed = False
    elif lengths.mean() < 10:
        print("     WARNING - Avg length somewhat low, monitor carefully")
    else:
        print("     PASS - Average length looks realistic")

    # Check 4: Win rate sanity
    print(f"\n  4. Win Rate Sanity:")
    print(f"     Win Rate: {win_rate*100:.1f}%")
    if win_rate > 0.80:
        print("     FAIL - Win rate > 80% is likely fake (lookahead bias)")
        checks_passed = False
    elif win_rate > 0.65:
        print("     WARNING - Win rate unusually high, verify model")
    else:
        print("     PASS - Win rate looks realistic")

    # Check 5: Sharpe sanity
    print(f"\n  5. Sharpe Ratio Sanity:")
    print(f"     Trade-based Sharpe: {trade_sharpe:.2f}")
    if trade_sharpe > 3.0:
        print("     FAIL - Sharpe > 3.0 is unrealistic")
        checks_passed = False
    elif trade_sharpe > 2.0:
        print("     WARNING - Sharpe unusually high, verify model")
    else:
        print("     PASS - Sharpe ratio looks realistic")

    # Check 6: Action diversity
    print(f"\n  6. Action Diversity:")
    diverse = all(pct > 1 for pct in action_pct)
    if diverse:
        print("     PASS - All actions used > 1%")
    else:
        print("     WARNING - Policy may have collapsed to limited actions")
        for i, name in enumerate(Actions.NAMES):
            if action_pct[i] < 1:
                print(f"       {name}: {action_pct[i]:.1f}% (below threshold)")

    print("\n" + "=" * 70)
    if checks_passed:
        print(" OVERALL: PASS - Model appears to be learning legitimately")
    else:
        print(" OVERALL: FAIL - Possible lookahead bias or other issues detected")
    print("=" * 70)


if __name__ == "__main__":
    main()
