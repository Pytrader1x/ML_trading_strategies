#!/usr/bin/env python3
"""
Fast OOS Evaluation for v6_anti_gaming.

V6 ANTI-GAMING: Fix V5's "breakeven farming" gaming behavior.

KEY CHANGES from v5:
1. ASYMMETRIC BREAKEVEN: Track recovery_exits vs giveback_exits vs profit_exits
2. MFE DECAY: Track MFE at exit to analyze profit giveback
3. TIGHTEN COOLDOWN: Track cooldown blocks
4. NEW METRICS: recovery_rate, giveback_rate, profit_rate

Design Goal: Validate that the model lets winners run instead of farming breakeven.
V5 had 71.6% breakeven exits and only 0.2% profit exits - V6 aims to reverse this.
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
from env import TradeEpisode  # Needed for pickle deserialization

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


def get_action_mask(bar: int, min_exit_bar: int, min_trail_bar: int):
    """Get action mask for single episode at given bar."""
    mask = torch.ones(1, 5, dtype=torch.bool)

    # Block EXIT and PARTIAL_EXIT before min_exit_bar
    if bar < min_exit_bar:
        mask[0, Actions.EXIT] = False
        mask[0, Actions.PARTIAL_EXIT] = False

    # Block TRAIL_BE before min_trail_bar
    if bar < min_trail_bar:
        mask[0, Actions.TRAIL_BREAKEVEN] = False

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


def evaluate_episode(policy, episode, device: str, config):
    """Evaluate policy on single episode with V6 anti-gaming tracking.

    Returns: (return, length, actions, stats_dict)
    """
    min_exit_bar = config.reward.min_exit_bar
    min_trail_bar = config.reward.min_trail_bar
    min_profit_for_trail = config.reward.min_profit_for_trail
    breakeven_buffer = config.reward.breakeven_buffer_pct
    breakeven_band = config.reward.breakeven_band
    tighten_cooldown = config.reward.tighten_cooldown

    max_bars = episode.market_tensor.shape[0]
    sl_atr = 1.1
    max_fav = 0.0
    max_adv = 0.0
    action_hist = []
    actions_taken = []

    # Blocking stats
    early_exit_attempts = 0
    early_trail_attempts = 0
    low_profit_trail_attempts = 0
    breakeven_active = False

    # V6: Tighten cooldown tracking
    last_tighten_bar = -100
    cooldown_blocks = 0

    for bar in range(max_bars):
        if not episode.valid_mask[bar]:
            # End of episode - return final PnL
            pnl = episode.market_tensor[bar-1, 0].item() if bar > 0 else 0.0
            stats = {
                'early_exit_attempts': early_exit_attempts,
                'early_trail_attempts': early_trail_attempts,
                'low_profit_trail_attempts': low_profit_trail_attempts,
                'cooldown_blocks': cooldown_blocks,
                'mfe': max_fav,
                'mae': max_adv,
            }
            return pnl, bar, actions_taken, stats

        state, max_fav, max_adv = build_state(episode, bar, sl_atr, max_fav, max_adv, action_hist)
        state = state.to(device)

        # Apply action mask
        action_mask = get_action_mask(bar, min_exit_bar, min_trail_bar).to(device)

        with torch.no_grad():
            action, _, _ = policy.get_action(state, deterministic=True, action_mask=action_mask)

        action = action.item()
        actions_taken.append(action)
        action_hist.append(action / 4.0)

        unrealized_pnl = episode.market_tensor[bar, 0].item()

        # Track early attempts (should be zero with proper masking)
        if bar < min_exit_bar and action in [Actions.EXIT, Actions.PARTIAL_EXIT]:
            early_exit_attempts += 1
            action = Actions.HOLD  # Safety fallback

        if bar < min_trail_bar and action == Actions.TRAIL_BREAKEVEN:
            early_trail_attempts += 1
            action = Actions.HOLD  # Safety fallback

        # V6: Check TIGHTEN cooldown
        if action == Actions.TIGHTEN_SL:
            if bar - last_tighten_bar < tighten_cooldown:
                cooldown_blocks += 1
                action = Actions.HOLD  # Safety fallback
            else:
                last_tighten_bar = bar

        # Process action
        if action == Actions.EXIT or action == Actions.PARTIAL_EXIT:
            stats = {
                'early_exit_attempts': early_exit_attempts,
                'early_trail_attempts': early_trail_attempts,
                'low_profit_trail_attempts': low_profit_trail_attempts,
                'cooldown_blocks': cooldown_blocks,
                'mfe': max_fav,
                'mae': max_adv,
            }
            return unrealized_pnl, bar + 1, actions_taken, stats

        elif action == Actions.TIGHTEN_SL:
            sl_atr *= 0.75

        elif action == Actions.TRAIL_BREAKEVEN:
            # Check minimum profit requirement
            if unrealized_pnl > min_profit_for_trail:
                breakeven_active = True
            else:
                low_profit_trail_attempts += 1

        # Check SL hit
        if bar + 1 < max_bars:
            next_pnl = episode.market_tensor[bar + 1, 0].item()

            if breakeven_active:
                # True breakeven - exit at buffer (0 in v6)
                if next_pnl < breakeven_buffer:
                    stats = {
                        'early_exit_attempts': early_exit_attempts,
                        'early_trail_attempts': early_trail_attempts,
                        'low_profit_trail_attempts': low_profit_trail_attempts,
                        'cooldown_blocks': cooldown_blocks,
                        'mfe': max_fav,
                        'mae': max_adv,
                    }
                    return breakeven_buffer, bar + 2, actions_taken, stats
            else:
                # Standard ATR-based SL
                sl_pnl_threshold = -sl_atr * episode.entry_atr / episode.entry_price
                if next_pnl < sl_pnl_threshold:
                    stats = {
                        'early_exit_attempts': early_exit_attempts,
                        'early_trail_attempts': early_trail_attempts,
                        'low_profit_trail_attempts': low_profit_trail_attempts,
                        'cooldown_blocks': cooldown_blocks,
                        'mfe': max_fav,
                        'mae': max_adv,
                    }
                    return sl_pnl_threshold, bar + 2, actions_taken, stats

    # Reached end
    stats = {
        'early_exit_attempts': early_exit_attempts,
        'early_trail_attempts': early_trail_attempts,
        'low_profit_trail_attempts': low_profit_trail_attempts,
        'cooldown_blocks': cooldown_blocks,
        'mfe': max_fav,
        'mae': max_adv,
    }
    return episode.market_tensor[max_bars-1, 0].item(), max_bars, actions_taken, stats


def calculate_trade_based_sharpe(returns: np.ndarray, trades_per_year: float = 442) -> float:
    """Calculate trade-based Sharpe (standard approach for trades)."""
    if len(returns) == 0:
        return 0.0

    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret < 1e-8:
        return 0.0

    sharpe = mean_ret / std_ret * np.sqrt(trades_per_year)
    return float(sharpe)


def calculate_proper_sharpe(returns: np.ndarray, lengths: np.ndarray, timeframe_min: int = 15) -> float:
    """Calculate proper time-weighted Sharpe ratio."""
    if len(returns) == 0:
        return 0.0

    hours = lengths * timeframe_min / 60.0
    hours = np.maximum(hours, 1.0)

    hourly_returns = returns / hours
    trading_hours_per_year = 252 * 10

    mean_hourly = hourly_returns.mean()
    std_hourly = hourly_returns.std()

    if std_hourly < 1e-8:
        return 0.0

    sharpe = mean_hourly / std_hourly * np.sqrt(trading_hours_per_year)
    return float(sharpe)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='models/exit_policy_final.pt')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = args.device

    print("=" * 70)
    print(" DEEP OOS EVALUATION - v6_anti_gaming")
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

    print(f"\nModel Info:")
    print(f"  Trained for: {checkpoint.get('total_steps', 'N/A'):,} steps")
    print(f"  Experiment: {checkpoint.get('experiment', 'unknown')}")
    print(f"  min_exit_bar: {config.reward.min_exit_bar}")
    print(f"  min_trail_bar: {config.reward.min_trail_bar}")
    print(f"  min_profit_for_trail: {config.reward.min_profit_for_trail:.4f} (~{config.reward.min_profit_for_trail*10000:.0f} pips)")
    print(f"  breakeven_buffer: {config.reward.breakeven_buffer_pct}")
    print(f"\nV6 Anti-Gaming Parameters:")
    print(f"  recovery_bonus: {config.reward.recovery_bonus}")
    print(f"  giveback_penalty: {config.reward.giveback_penalty}")
    print(f"  breakeven_band: {config.reward.breakeven_band:.4f} (~{config.reward.breakeven_band*10000:.0f} pips)")
    print(f"  mfe_decay_coef: {config.reward.mfe_decay_coef}")
    print(f"  tighten_cooldown: {config.reward.tighten_cooldown} bars")

    # Evaluate
    returns = []
    lengths = []
    all_actions = []
    mfe_list = []
    mae_list = []
    total_early_exit = 0
    total_early_trail = 0
    total_low_profit_trail = 0
    total_cooldown_blocks = 0

    print(f"\nEvaluating {len(episodes)} OOS trades...")

    for i, ep in enumerate(episodes):
        ret, length, actions, stats = evaluate_episode(policy, ep, device, config)
        returns.append(ret)
        lengths.append(length)
        all_actions.extend(actions)
        mfe_list.append(stats['mfe'])
        mae_list.append(stats['mae'])
        total_early_exit += stats['early_exit_attempts']
        total_early_trail += stats['early_trail_attempts']
        total_low_profit_trail += stats['low_profit_trail_attempts']
        total_cooldown_blocks += stats['cooldown_blocks']

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(episodes)} done...")

    returns = np.array(returns)
    lengths = np.array(lengths)
    mfe_arr = np.array(mfe_list)
    mae_arr = np.array(mae_list)

    # Calculate metrics
    mean_ret = returns.mean()
    std_ret = returns.std()

    trades_per_year = len(returns) / 3.0
    trade_sharpe = calculate_trade_based_sharpe(returns, trades_per_year)
    time_weighted_sharpe = calculate_proper_sharpe(returns, lengths, timeframe_min=15)
    naive_sharpe = mean_ret / (std_ret + 1e-8) * np.sqrt(252)

    win_rate = (returns > 0).mean()
    breakeven_band = config.reward.breakeven_band
    breakeven_rate = ((returns >= -breakeven_band) & (returns <= breakeven_band)).mean()
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    profit_factor = wins.sum() / (-losses.sum() + 1e-8) if len(losses) > 0 else float('inf')

    cumret = np.cumsum(returns)
    max_dd = (np.maximum.accumulate(cumret) - cumret).max()

    # Action distribution
    action_counts = np.bincount(all_actions, minlength=5)
    action_pct = action_counts / action_counts.sum() * 100

    # Exit timing analysis
    quick_exits = (lengths <= config.reward.min_exit_bar).sum()
    quick_exit_rate = quick_exits / len(lengths)

    # V6: Exit type analysis (recovery vs giveback vs profit)
    recovery_exits = []   # Breakeven exits after being in loss (and never profit)
    giveback_exits = []   # Breakeven exits after being in profit
    profit_exits = []     # Exits with meaningful profit

    for i in range(len(returns)):
        ret = returns[i]
        mfe = mfe_arr[i]
        mae = mae_arr[i]

        is_breakeven = abs(ret) < breakeven_band
        was_losing = mae < -breakeven_band
        was_winning = mfe > breakeven_band
        is_profit = ret > breakeven_band

        if is_breakeven and was_losing and not was_winning:
            recovery_exits.append(ret)
        elif is_breakeven and was_winning:
            giveback_exits.append(ret)
        elif is_profit:
            profit_exits.append(ret)

    recovery_rate = len(recovery_exits) / len(returns) if len(returns) > 0 else 0
    giveback_rate = len(giveback_exits) / len(returns) if len(returns) > 0 else 0
    profit_exit_rate = len(profit_exits) / len(returns) if len(returns) > 0 else 0

    # Return distribution by exit type
    sl_exits = returns[returns < -0.005]  # Likely SL hits
    breakeven_exits_arr = returns[(returns >= -0.001) & (returns <= 0.001)]
    profit_exits_arr = returns[returns > 0.005]

    print("\n" + "=" * 70)
    print(" OOS RESULTS - v6_anti_gaming (2022-2025)")
    print("=" * 70)
    print(f"  Trades:        {len(returns)}")
    print(f"  Total Return:  {returns.sum():.4f} ({returns.sum()*100:.2f}%)")
    print(f"  Mean Return:   {mean_ret:.4f} +/- {std_ret:.4f}")

    print()
    print("  Sharpe Ratios:")
    print(f"    Naive (sqrt(252)):     {naive_sharpe:.2f}  (WRONG for trades)")
    print(f"    Trade-based:           {trade_sharpe:.2f}  (sqrt(trades/year))")
    print(f"    Time-weighted:         {time_weighted_sharpe:.2f}  (most accurate)")

    print()
    print(f"  Win Rate:      {win_rate * 100:.1f}%")
    print(f"  Breakeven:     {breakeven_rate * 100:.1f}% (within {breakeven_band*10000:.0f} pips)")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Max Drawdown:  {max_dd:.4f}")
    print(f"  Avg Length:    {lengths.mean():.1f} bars")

    print()
    print("  Action Distribution:")
    for i, name in enumerate(Actions.NAMES):
        print(f"    {name:12s}: {action_pct[i]:5.1f}%")

    print()
    print("  V6 Exit Analysis:")
    print(f"    Recovery exits (good): {len(recovery_exits)} ({recovery_rate*100:.1f}%)")
    print(f"    Giveback exits (bad):  {len(giveback_exits)} ({giveback_rate*100:.1f}%)")
    print(f"    Profit exits (good):   {len(profit_exits)} ({profit_exit_rate*100:.1f}%)")

    print()
    print("  Traditional Exit Analysis:")
    print(f"    SL hits (< -0.5%):     {len(sl_exits)} ({len(sl_exits)/len(returns)*100:.1f}%)")
    print(f"    Breakeven-ish:         {len(breakeven_exits_arr)} ({len(breakeven_exits_arr)/len(returns)*100:.1f}%)")
    print(f"    Profit exits (> 0.5%): {len(profit_exits_arr)} ({len(profit_exits_arr)/len(returns)*100:.1f}%)")

    print()
    print("  Return Distribution:")
    print(f"    Min:    {returns.min():.4f}")
    print(f"    25%:    {np.percentile(returns, 25):.4f}")
    print(f"    Median: {np.median(returns):.4f}")
    print(f"    75%:    {np.percentile(returns, 75):.4f}")
    print(f"    Max:    {returns.max():.4f}")

    print()
    print("  MFE/MAE Analysis:")
    print(f"    Avg MFE: {mfe_arr.mean():.4f} ({mfe_arr.mean()*10000:.1f} pips)")
    print(f"    Avg MAE: {mae_arr.mean():.4f} ({mae_arr.mean()*10000:.1f} pips)")
    print(f"    MFE retained: {(returns.mean() / (mfe_arr.mean() + 1e-8))*100:.1f}% (higher is better)")

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
    improvement = ((returns.sum() - classical_returns.sum()) / abs(classical_returns.sum())) * 100 if classical_returns.sum() != 0 else 0
    print(f"    Improvement:      {improvement:+.1f}%")
    print("=" * 70)

    # V6: Enhanced anti-gaming validation
    print("\n" + "=" * 70)
    print(" V6 ANTI-GAMING VALIDATION")
    print("=" * 70)

    checks_passed = True

    # Check 1: Early exit attempts
    print(f"\n  1. Early Exit Attempts (bar < {config.reward.min_exit_bar}):")
    print(f"     Blocked: {total_early_exit}")
    if total_early_exit > 0:
        print("     FAIL - Action masking may not be working")
        checks_passed = False
    else:
        print("     PASS - No early exit attempts")

    # Check 2: Early TRAIL_BE attempts
    print(f"\n  2. Early TRAIL_BE Attempts (bar < {config.reward.min_trail_bar}):")
    print(f"     Blocked: {total_early_trail}")
    if total_early_trail > 0:
        print("     FAIL - TRAIL_BE masking may not be working")
        checks_passed = False
    else:
        print("     PASS - No early TRAIL_BE attempts")

    # Check 3: Low profit TRAIL_BE attempts
    print(f"\n  3. Low Profit TRAIL_BE Attempts (PnL < {config.reward.min_profit_for_trail:.4f}):")
    print(f"     Blocked: {total_low_profit_trail}")
    print(f"     INFO - Model attempted TRAIL_BE {total_low_profit_trail} times without sufficient profit")

    # Check 4: Tighten cooldown blocks
    print(f"\n  4. Tighten Cooldown Blocks (cooldown = {config.reward.tighten_cooldown} bars):")
    print(f"     Blocked: {total_cooldown_blocks}")
    print(f"     INFO - TIGHTEN_SL was rate-limited {total_cooldown_blocks} times")

    # Check 5: Quick exits
    print(f"\n  5. Quick Exits (length <= {config.reward.min_exit_bar} bars):")
    print(f"     Count: {quick_exits} ({quick_exit_rate*100:.1f}%)")
    if quick_exit_rate > 0.2:
        print("     WARNING - High quick exit rate (likely SL hits)")
    else:
        print("     PASS - Quick exit rate acceptable")

    # Check 6: Win rate sanity
    print(f"\n  6. Win Rate Sanity:")
    print(f"     Win Rate: {win_rate*100:.1f}%")
    if win_rate > 0.80:
        print("     FAIL - Win rate > 80% is likely fake")
        checks_passed = False
    elif win_rate > 0.65:
        print("     WARNING - Win rate unusually high")
    else:
        print("     PASS - Win rate looks realistic")

    # Check 7: Sharpe sanity
    print(f"\n  7. Sharpe Ratio Sanity:")
    print(f"     Trade-based Sharpe: {trade_sharpe:.2f}")
    if trade_sharpe > 3.0:
        print("     FAIL - Sharpe > 3.0 is unrealistic")
        checks_passed = False
    elif trade_sharpe > 2.0:
        print("     WARNING - Sharpe unusually high")
    else:
        print("     PASS - Sharpe ratio looks realistic")

    # Check 8: Avg length
    print(f"\n  8. Average Episode Length:")
    print(f"     Mean: {lengths.mean():.1f} bars")
    if lengths.mean() < 3:
        print("     FAIL - Avg length < 3 bars may indicate exploitation")
        checks_passed = False
    elif lengths.mean() < 5:
        print("     WARNING - Avg length somewhat low")
    else:
        print("     PASS - Average length looks realistic")

    # Check 9: Action diversity
    print(f"\n  9. Action Diversity:")
    diverse = all(pct > 1 for pct in action_pct)
    if diverse:
        print("     PASS - All actions used > 1%")
    else:
        print("     WARNING - Policy may have collapsed")
        for i, name in enumerate(Actions.NAMES):
            if action_pct[i] < 1:
                print(f"       {name}: {action_pct[i]:.1f}% (below threshold)")

    # V6: Check 10: Giveback rate (new key metric)
    print(f"\n  10. V6 Giveback Rate (KEY METRIC):")
    print(f"      Giveback exits: {len(giveback_exits)} ({giveback_rate*100:.1f}%)")
    if giveback_rate > 0.4:
        print(f"      FAIL - Giveback rate > 40% indicates model still gaming (V5 had 71.6%)")
        checks_passed = False
    elif giveback_rate > 0.25:
        print("      WARNING - Giveback rate somewhat high")
    else:
        print("      PASS - Giveback rate acceptable")

    # V6: Check 11: Profit exit rate (new key metric)
    print(f"\n  11. V6 Profit Exit Rate (KEY METRIC):")
    print(f"      Profit exits: {len(profit_exits)} ({profit_exit_rate*100:.1f}%)")
    if profit_exit_rate < 0.1:
        print(f"      FAIL - Profit rate < 10% indicates model not letting winners run (V5 had 0.2%)")
        checks_passed = False
    elif profit_exit_rate < 0.2:
        print("      WARNING - Profit rate somewhat low")
    else:
        print("      PASS - Profit exit rate acceptable")

    print("\n" + "=" * 70)
    if checks_passed:
        print(" OVERALL: PASS - V6 anti-gaming measures appear to be working")
    else:
        print(" OVERALL: FAIL - Model may still be gaming breakeven exits")
    print("=" * 70)

    # V6: Summary comparison to V5 targets
    print("\n" + "=" * 70)
    print(" V6 vs V5 COMPARISON")
    print("=" * 70)
    print("                     V5 Actual    V6 Target    V6 Actual")
    print(f"  Giveback rate:     71.6%        <40%         {giveback_rate*100:.1f}%")
    print(f"  Profit rate:       0.2%         >20%         {profit_exit_rate*100:.1f}%")
    print(f"  Avg hold time:     5.2 bars     8-15 bars    {lengths.mean():.1f} bars")
    print(f"  TIGHTEN usage:     47.9%        30-40%       {action_pct[2]:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
