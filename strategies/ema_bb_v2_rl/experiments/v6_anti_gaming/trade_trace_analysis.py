#!/usr/bin/env python3
"""
DEEP TRADE TRACE ANALYSIS - Step by step validation.

Goes through individual trades bar-by-bar to understand exactly
what information the model has and what decisions it makes.

Key Question: Is there ANY way the model could be cheating?
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


ACTION_NAMES = ['HOLD', 'EXIT', 'TIGHTEN_SL', 'TRAIL_BE', 'PARTIAL']


def load_model(checkpoint_path: Path, device: str = "cpu"):
    """Load trained model."""
    config = PPOConfig(device=device)
    config.hidden_dims = [512, 256]  # V5 architecture
    model = ActorCritic(config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def get_action_mask(bar: int, min_exit_bar: int, min_trail_bar: int):
    """Get action mask - what actions are ALLOWED at this bar."""
    mask = torch.ones(1, 5, dtype=torch.bool)

    if bar < min_exit_bar:
        mask[0, Actions.EXIT] = False
        mask[0, Actions.PARTIAL_EXIT] = False

    if bar < min_trail_bar:
        mask[0, Actions.TRAIL_BREAKEVEN] = False

    return mask


def build_state(episode, bar: int, sl_atr: float, max_fav: float, max_adv: float,
                action_hist: list, be_active: bool):
    """Build state tensor - this is EXACTLY what the model sees."""
    market = episode.market_tensor[bar]

    # Feature 0 is unrealized PnL
    unrealized_pnl = market[0].item()
    max_fav = max(max_fav, unrealized_pnl)
    max_adv = min(max_adv, unrealized_pnl)

    # Position features
    bars_held_norm = bar / 200.0
    position_features = [bars_held_norm, unrealized_pnl, max_fav, max_adv, sl_atr / 2.0]

    # Market features (first 10 from tensor)
    market_feats = market[:10].tolist()

    # Entry context (from episode metadata)
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


def trace_single_trade(model, episode, config, trade_idx: int):
    """Trace a single trade step by step."""
    min_exit_bar = config.reward.min_exit_bar
    min_trail_bar = config.reward.min_trail_bar
    min_profit_for_trail = config.reward.min_profit_for_trail

    n_valid = int(episode.valid_mask.sum().item())

    print(f"\n{'='*80}")
    print(f" TRADE #{trade_idx}: {episode.direction} ({'LONG' if episode.direction == 1 else 'SHORT'})")
    print(f"{'='*80}")
    print(f"Entry Price: {episode.entry_price:.5f}")
    print(f"Entry ATR: {episode.entry_atr:.5f}")
    print(f"Entry ADX: {episode.entry_adx:.1f}, RSI: {episode.entry_rsi:.1f}")
    print(f"Max Valid Bars: {n_valid}")
    print(f"Classical Result: {episode.classical_pnl*100:.3f}% ({episode.classical_exit_reason})")
    print()

    # State tracking
    sl_atr = 1.5  # Initial stop loss in ATR
    max_fav = 0.0
    max_adv = 0.0
    action_hist = []
    be_active = False
    be_level = 0.0

    print(f"{'Bar':>4} | {'PnL':>8} | {'MaxFav':>8} | {'Mask':>15} | {'Action':>12} | {'Probs':>35} | Notes")
    print("-" * 120)

    exit_bar = None
    exit_reason = None
    exit_pnl = None

    for bar in range(n_valid):
        # Get current market state
        market = episode.market_tensor[bar]
        current_pnl = market[0].item()

        # Build state (this is what model sees)
        state, max_fav, max_adv = build_state(
            episode, bar, sl_atr, max_fav, max_adv, action_hist, be_active
        )

        # Get action mask
        mask = get_action_mask(bar, min_exit_bar, min_trail_bar)
        mask_str = ""
        if not mask[0, Actions.EXIT]:
            mask_str += "!EXIT "
        if not mask[0, Actions.PARTIAL_EXIT]:
            mask_str += "!PARTIAL "
        if not mask[0, Actions.TRAIL_BREAKEVEN]:
            mask_str += "!TRAIL "
        if not mask_str:
            mask_str = "ALL OK"

        # Get model's decision
        with torch.no_grad():
            logits, value = model(state)

            # Apply mask
            masked_logits = logits.clone()
            masked_logits[~mask] = float('-inf')

            probs = torch.softmax(masked_logits, dim=-1)
            action = probs.argmax(dim=-1).item()

        prob_str = " ".join([f"{ACTION_NAMES[i][:4]}:{probs[0,i].item():.2f}" for i in range(5)])

        # Check for cheating indicators
        notes = []

        # CRITICAL CHECK 1: Can model see future PnL?
        # At bar 0, current_pnl should be ~0 (just entered)
        if bar == 0 and abs(current_pnl) > 0.0001:
            notes.append(f"⚠️ Bar0 PnL={current_pnl:.4f}")

        # CRITICAL CHECK 2: Is model trying to exit when blocked?
        raw_probs = torch.softmax(logits, dim=-1)
        if bar < min_exit_bar and raw_probs[0, Actions.EXIT].item() > 0.3:
            notes.append(f"WANT_EXIT:{raw_probs[0,Actions.EXIT].item():.2f}")

        # Check breakeven activation
        if action == Actions.TRAIL_BREAKEVEN and not be_active:
            if current_pnl >= min_profit_for_trail:
                be_active = True
                be_level = 0.0  # True breakeven
                notes.append(f"BE_ACTIVATED @{current_pnl*100:.3f}%")
            else:
                notes.append(f"BE_BLOCKED (need {min_profit_for_trail*100:.2f}%)")

        # Update stop loss if tightening
        if action == Actions.TIGHTEN_SL:
            sl_atr = max(0.5, sl_atr - 0.25)
            notes.append(f"SL→{sl_atr:.2f}ATR")

        notes_str = " | ".join(notes) if notes else ""

        print(f"{bar:>4} | {current_pnl*100:>7.3f}% | {max_fav*100:>7.3f}% | {mask_str:>15} | {ACTION_NAMES[action]:>12} | {prob_str} | {notes_str}")

        # Record action
        action_hist.append(float(action) / 4.0)

        # Check exit conditions
        if action == Actions.EXIT:
            exit_bar = bar
            exit_reason = "MODEL_EXIT"
            exit_pnl = current_pnl
            break

        if action == Actions.PARTIAL_EXIT:
            exit_bar = bar
            exit_reason = "PARTIAL"
            exit_pnl = current_pnl * 0.5  # Assume 50% partial
            break

        # Check breakeven stop
        if be_active and current_pnl <= be_level:
            exit_bar = bar
            exit_reason = "BREAKEVEN_SL"
            exit_pnl = be_level
            break

        # Check stop loss (simplified)
        sl_pct = sl_atr * episode.entry_atr
        if current_pnl <= -sl_pct:
            exit_bar = bar
            exit_reason = "SL_HIT"
            exit_pnl = -sl_pct
            break

    if exit_bar is None:
        exit_bar = n_valid - 1
        exit_reason = "END"
        exit_pnl = episode.market_tensor[exit_bar, 0].item()

    print("-" * 120)
    print(f"\nTRADE RESULT:")
    print(f"  Exit Bar: {exit_bar}")
    print(f"  Exit Reason: {exit_reason}")
    print(f"  Exit PnL: {exit_pnl*100:.3f}%")
    print(f"  Classical PnL: {episode.classical_pnl*100:.3f}%")
    print(f"  Improvement: {(exit_pnl - episode.classical_pnl)*100:.3f}%")

    return {
        'exit_bar': exit_bar,
        'exit_reason': exit_reason,
        'exit_pnl': exit_pnl,
        'classical_pnl': episode.classical_pnl
    }


def analyze_cheating_potential(results: list):
    """Analyze all traced trades for cheating patterns."""
    print("\n" + "="*80)
    print(" CHEATING ANALYSIS SUMMARY")
    print("="*80)

    bar0_exits = sum(1 for r in results if r['exit_bar'] == 0)
    bar1_exits = sum(1 for r in results if r['exit_bar'] == 1)
    early_exits = sum(1 for r in results if r['exit_bar'] < 3)

    print(f"\nExit Timing:")
    print(f"  Bar 0 exits: {bar0_exits} ({bar0_exits/len(results)*100:.1f}%)")
    print(f"  Bar 1 exits: {bar1_exits} ({bar1_exits/len(results)*100:.1f}%)")
    print(f"  Early exits (<3): {early_exits} ({early_exits/len(results)*100:.1f}%)")

    # Check if early exits correlate with profitable outcomes
    if early_exits > 0:
        early_results = [r for r in results if r['exit_bar'] < 3]
        early_win_rate = np.mean([r['exit_pnl'] > 0 for r in early_results])
        print(f"  Early exit win rate: {early_win_rate*100:.1f}%")

        if early_win_rate > 0.7:
            print("  ⚠️ WARNING: Early exits have suspiciously high win rate!")

    # Compare to classical
    rl_returns = [r['exit_pnl'] for r in results]
    classical_returns = [r['classical_pnl'] for r in results]

    print(f"\nPerformance:")
    print(f"  RL Mean: {np.mean(rl_returns)*100:.3f}%")
    print(f"  Classical Mean: {np.mean(classical_returns)*100:.3f}%")
    print(f"  RL Win Rate: {np.mean(np.array(rl_returns) > 0)*100:.1f}%")
    print(f"  Classical Win Rate: {np.mean(np.array(classical_returns) > 0)*100:.1f}%")

    # Exit reason distribution
    print(f"\nExit Reasons:")
    reasons = {}
    for r in results:
        reasons[r['exit_reason']] = reasons.get(r['exit_reason'], 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count/len(results)*100:.1f}%)")

    print("\n" + "="*80)
    print(" VERDICT")
    print("="*80)

    if bar0_exits == 0 and bar1_exits == 0:
        print("\n✓ NO BAR 0 OR BAR 1 EXITS - Action masking is working")
    else:
        print(f"\n⚠️ CHEATING DETECTED: {bar0_exits} bar 0 exits, {bar1_exits} bar 1 exits")

    avg_exit_bar = np.mean([r['exit_bar'] for r in results])
    if avg_exit_bar > 3:
        print(f"✓ Average exit bar: {avg_exit_bar:.1f} (reasonable)")
    else:
        print(f"⚠️ Average exit bar: {avg_exit_bar:.1f} (suspiciously early)")


def main():
    print("="*80)
    print(" DEEP TRADE TRACE ANALYSIS")
    print(" Step-by-step validation of V5 model decisions")
    print("="*80)

    # Load model
    model_path = EXPERIMENT_DIR / "models" / "exit_policy_final.pt"
    model, config = load_model(model_path, device="cpu")
    print(f"\nLoaded model from {model_path}")
    print(f"min_exit_bar: {config.reward.min_exit_bar}")
    print(f"min_trail_bar: {config.reward.min_trail_bar}")
    print(f"min_profit_for_trail: {config.reward.min_profit_for_trail}")

    # Load episodes
    episode_file = STRATEGY_DIR / "data" / "episodes_test_2022_2025.pkl"
    with open(episode_file, 'rb') as f:
        data = pickle.load(f)
    episodes = data['episodes']
    print(f"Loaded {len(episodes)} OOS episodes")

    # Trace sample trades (first 20 for detailed view)
    print("\n" + "="*80)
    print(" DETAILED TRADE TRACES (First 20)")
    print("="*80)

    detailed_results = []
    for i in range(min(20, len(episodes))):
        result = trace_single_trade(model, episodes[i], config, i+1)
        detailed_results.append(result)

    # Analyze all trades (faster, no printing)
    print("\n" + "="*80)
    print(" ANALYZING ALL TRADES...")
    print("="*80)

    all_results = []
    for i, ep in enumerate(episodes):
        if i % 200 == 0:
            print(f"  {i}/{len(episodes)}")

        # Quick evaluation without detailed printing
        n_valid = int(ep.valid_mask.sum().item())
        sl_atr = 1.5
        max_fav = 0.0
        max_adv = 0.0
        action_hist = []
        be_active = False

        exit_bar = None
        exit_pnl = None
        exit_reason = None

        for bar in range(n_valid):
            current_pnl = ep.market_tensor[bar, 0].item()
            state, max_fav, max_adv = build_state(ep, bar, sl_atr, max_fav, max_adv, action_hist, be_active)
            mask = get_action_mask(bar, config.reward.min_exit_bar, config.reward.min_trail_bar)

            with torch.no_grad():
                logits, _ = model(state)
                masked_logits = logits.clone()
                masked_logits[~mask] = float('-inf')
                action = masked_logits.argmax(dim=-1).item()

            action_hist.append(float(action) / 4.0)

            if action == Actions.TIGHTEN_SL:
                sl_atr = max(0.5, sl_atr - 0.25)

            if action == Actions.TRAIL_BREAKEVEN and not be_active:
                if current_pnl >= config.reward.min_profit_for_trail:
                    be_active = True

            if action == Actions.EXIT:
                exit_bar = bar
                exit_reason = "EXIT"
                exit_pnl = current_pnl
                break

            if action == Actions.PARTIAL_EXIT:
                exit_bar = bar
                exit_reason = "PARTIAL"
                exit_pnl = current_pnl * 0.5
                break

            if be_active and current_pnl <= 0:
                exit_bar = bar
                exit_reason = "BE_SL"
                exit_pnl = 0
                break

            sl_pct = sl_atr * ep.entry_atr
            if current_pnl <= -sl_pct:
                exit_bar = bar
                exit_reason = "SL_HIT"
                exit_pnl = -sl_pct
                break

        if exit_bar is None:
            exit_bar = n_valid - 1
            exit_reason = "END"
            exit_pnl = ep.market_tensor[exit_bar, 0].item()

        all_results.append({
            'exit_bar': exit_bar,
            'exit_reason': exit_reason,
            'exit_pnl': exit_pnl,
            'classical_pnl': ep.classical_pnl
        })

    # Final analysis
    analyze_cheating_potential(all_results)


if __name__ == "__main__":
    main()
