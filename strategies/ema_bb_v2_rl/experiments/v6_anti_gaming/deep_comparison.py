#!/usr/bin/env python3
"""
Deep Comparison: V1 Baseline vs V5 Anti-Cheat

Comprehensive analysis of model evolution showing:
- Exit behavior patterns
- Action distribution changes
- Risk management improvements
- Performance attribution
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Paths
EXPERIMENT_DIR = Path(__file__).parent
STRATEGY_DIR = EXPERIMENT_DIR.parent.parent
V1_DIR = STRATEGY_DIR / "experiments" / "v1_baseline"
V5_DIR = EXPERIMENT_DIR

sys.path.insert(0, str(EXPERIMENT_DIR))
from config import PPOConfig as V5Config, Actions
from env import TradeEpisode

sys.path.insert(0, str(STRATEGY_DIR))
from model import ActorCritic


@dataclass
class TradeResult:
    """Result of a single trade evaluation."""
    episode_idx: int
    final_pnl: float
    exit_bar: int
    exit_reason: str
    actions: List[int]
    action_counts: Dict[str, int]
    max_favorable: float
    max_adverse: float
    classical_pnl: float


def load_episodes(episode_file: Path) -> list:
    """Load trade episodes from pickle file."""
    with open(episode_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data['episodes']
    return data


def load_model(model_path: Path, hidden_dims: List[int], device: str = 'cpu'):
    """Load a model with specified architecture."""
    config = V5Config(device=device)
    config.hidden_dims = hidden_dims

    model = ActorCritic(config)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint.get('total_steps', checkpoint.get('timestep', 0))


def build_state(episode, bar: int, sl_atr: float, max_fav: float, max_adv: float, action_hist: list):
    """Build state tensor for single episode."""
    market = episode.market_tensor[bar]
    unrealized_pnl = market[0].item()
    max_fav = max(max_fav, unrealized_pnl)
    max_adv = min(max_adv, unrealized_pnl)

    bars_held_norm = bar / 200.0
    position_features = [bars_held_norm, unrealized_pnl, max_fav, max_adv, sl_atr / 2.0]
    market_feats = market[:10].tolist()
    entry_ctx = [
        episode.entry_atr,
        episode.entry_adx / 100.0,
        episode.entry_rsi / 100.0,
        episode.entry_bb_width,
        episode.entry_ema_diff,
    ]
    hist = action_hist[-5:] if len(action_hist) >= 5 else [0.0] * (5 - len(action_hist)) + action_hist

    state = position_features + market_feats + entry_ctx + hist
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0), max_fav, max_adv


def evaluate_v1(model, episode, episode_idx: int, device: str) -> TradeResult:
    """Evaluate V1 model - NO action masking, NO guards."""
    max_bars = episode.market_tensor.shape[0]
    sl_atr = 1.1
    max_fav = 0.0
    max_adv = 0.0
    action_hist = []
    actions = []
    action_counts = {name: 0 for name in Actions.NAMES}

    for bar in range(max_bars):
        if not episode.valid_mask[bar]:
            return TradeResult(
                episode_idx=episode_idx,
                final_pnl=episode.market_tensor[bar-1, 0].item() if bar > 0 else 0.0,
                exit_bar=bar,
                exit_reason='END',
                actions=actions,
                action_counts=action_counts,
                max_favorable=max_fav,
                max_adverse=max_adv,
                classical_pnl=episode.classical_pnl
            )

        state, max_fav, max_adv = build_state(episode, bar, sl_atr, max_fav, max_adv, action_hist)
        state = state.to(device)

        with torch.no_grad():
            logits, _ = model(state)
            action = logits.argmax(dim=-1).item()

        actions.append(action)
        action_counts[Actions.NAMES[action]] += 1
        action_hist.append(action / 4.0)

        unrealized_pnl = episode.market_tensor[bar, 0].item()

        # V1: No guards - process all actions immediately
        if action == Actions.EXIT or action == Actions.PARTIAL_EXIT:
            return TradeResult(
                episode_idx=episode_idx,
                final_pnl=unrealized_pnl,
                exit_bar=bar + 1,
                exit_reason='EXIT' if action == Actions.EXIT else 'PARTIAL',
                actions=actions,
                action_counts=action_counts,
                max_favorable=max_fav,
                max_adverse=max_adv,
                classical_pnl=episode.classical_pnl
            )

        elif action == Actions.TIGHTEN_SL:
            sl_atr *= 0.75

        elif action == Actions.TRAIL_BREAKEVEN:
            if unrealized_pnl > 0:
                # V1: Set SL to 0.1 ATR (NOT true breakeven)
                sl_atr = 0.1

        # Check SL
        if bar + 1 < max_bars and episode.valid_mask[bar + 1]:
            next_pnl = episode.market_tensor[bar + 1, 0].item()
            sl_threshold = -sl_atr * episode.entry_atr / episode.entry_price
            if next_pnl < sl_threshold:
                return TradeResult(
                    episode_idx=episode_idx,
                    final_pnl=sl_threshold,
                    exit_bar=bar + 2,
                    exit_reason='SL_HIT',
                    actions=actions,
                    action_counts=action_counts,
                    max_favorable=max_fav,
                    max_adverse=max_adv,
                    classical_pnl=episode.classical_pnl
                )

    return TradeResult(
        episode_idx=episode_idx,
        final_pnl=episode.market_tensor[max_bars-1, 0].item(),
        exit_bar=max_bars,
        exit_reason='END',
        actions=actions,
        action_counts=action_counts,
        max_favorable=max_fav,
        max_adverse=max_adv,
        classical_pnl=episode.classical_pnl
    )


def evaluate_v5(model, episode, episode_idx: int, device: str, config) -> TradeResult:
    """Evaluate V5 model - WITH action masking and anti-cheat guards."""
    min_exit_bar = config.reward.min_exit_bar
    min_trail_bar = config.reward.min_trail_bar
    min_profit_for_trail = config.reward.min_profit_for_trail
    breakeven_buffer = config.reward.breakeven_buffer_pct

    max_bars = episode.market_tensor.shape[0]
    sl_atr = 1.1
    max_fav = 0.0
    max_adv = 0.0
    action_hist = []
    actions = []
    action_counts = {name: 0 for name in Actions.NAMES}
    breakeven_active = False

    for bar in range(max_bars):
        if not episode.valid_mask[bar]:
            return TradeResult(
                episode_idx=episode_idx,
                final_pnl=episode.market_tensor[bar-1, 0].item() if bar > 0 else 0.0,
                exit_bar=bar,
                exit_reason='END',
                actions=actions,
                action_counts=action_counts,
                max_favorable=max_fav,
                max_adverse=max_adv,
                classical_pnl=episode.classical_pnl
            )

        state, max_fav, max_adv = build_state(episode, bar, sl_atr, max_fav, max_adv, action_hist)
        state = state.to(device)

        # V5: Apply action mask
        mask = torch.ones(1, 5, dtype=torch.bool)
        if bar < min_exit_bar:
            mask[0, Actions.EXIT] = False
            mask[0, Actions.PARTIAL_EXIT] = False
        if bar < min_trail_bar:
            mask[0, Actions.TRAIL_BREAKEVEN] = False

        with torch.no_grad():
            logits, _ = model(state)
            masked_logits = logits.clone()
            masked_logits[~mask] = float('-inf')
            action = masked_logits.argmax(dim=-1).item()

        actions.append(action)
        action_counts[Actions.NAMES[action]] += 1
        action_hist.append(action / 4.0)

        unrealized_pnl = episode.market_tensor[bar, 0].item()

        if action == Actions.EXIT or action == Actions.PARTIAL_EXIT:
            return TradeResult(
                episode_idx=episode_idx,
                final_pnl=unrealized_pnl,
                exit_bar=bar + 1,
                exit_reason='EXIT' if action == Actions.EXIT else 'PARTIAL',
                actions=actions,
                action_counts=action_counts,
                max_favorable=max_fav,
                max_adverse=max_adv,
                classical_pnl=episode.classical_pnl
            )

        elif action == Actions.TIGHTEN_SL:
            sl_atr *= 0.75

        elif action == Actions.TRAIL_BREAKEVEN:
            if unrealized_pnl > min_profit_for_trail:
                breakeven_active = True

        # Check SL
        if bar + 1 < max_bars and episode.valid_mask[bar + 1]:
            next_pnl = episode.market_tensor[bar + 1, 0].item()

            if breakeven_active:
                if next_pnl < breakeven_buffer:
                    return TradeResult(
                        episode_idx=episode_idx,
                        final_pnl=breakeven_buffer,
                        exit_bar=bar + 2,
                        exit_reason='BREAKEVEN_SL',
                        actions=actions,
                        action_counts=action_counts,
                        max_favorable=max_fav,
                        max_adverse=max_adv,
                        classical_pnl=episode.classical_pnl
                    )
            else:
                sl_threshold = -sl_atr * episode.entry_atr / episode.entry_price
                if next_pnl < sl_threshold:
                    return TradeResult(
                        episode_idx=episode_idx,
                        final_pnl=sl_threshold,
                        exit_bar=bar + 2,
                        exit_reason='SL_HIT',
                        actions=actions,
                        action_counts=action_counts,
                        max_favorable=max_fav,
                        max_adverse=max_adv,
                        classical_pnl=episode.classical_pnl
                    )

    return TradeResult(
        episode_idx=episode_idx,
        final_pnl=episode.market_tensor[max_bars-1, 0].item(),
        exit_bar=max_bars,
        exit_reason='END',
        actions=actions,
        action_counts=action_counts,
        max_favorable=max_fav,
        max_adverse=max_adv,
        classical_pnl=episode.classical_pnl
    )


def calculate_sharpe(returns: np.ndarray, trades_per_year: float = 442) -> float:
    """Calculate trade-based Sharpe ratio."""
    if len(returns) == 0 or returns.std() < 1e-8:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(trades_per_year)


def analyze_results(results: List[TradeResult], name: str) -> Dict:
    """Compute comprehensive statistics from results."""
    returns = np.array([r.final_pnl for r in results])
    lengths = np.array([r.exit_bar for r in results])
    classical = np.array([r.classical_pnl for r in results])

    # Aggregate action counts
    total_actions = defaultdict(int)
    for r in results:
        for action, count in r.action_counts.items():
            total_actions[action] += count

    total = sum(total_actions.values())
    action_pct = {k: v/total*100 for k, v in total_actions.items()}

    # Exit reason breakdown
    exit_reasons = defaultdict(int)
    for r in results:
        exit_reasons[r.exit_reason] += 1

    # Bar 0 exit analysis
    bar0_exits = sum(1 for r in results if r.exit_bar == 1)
    bar1_exits = sum(1 for r in results if r.exit_bar == 2)

    # Return categories
    large_wins = sum(1 for r in returns if r > 0.005)
    small_wins = sum(1 for r in returns if 0 < r <= 0.005)
    breakeven = sum(1 for r in returns if -0.0005 <= r <= 0.0005)
    small_losses = sum(1 for r in returns if -0.005 <= r < -0.0005)
    large_losses = sum(1 for r in returns if r < -0.005)

    return {
        'name': name,
        'n_trades': len(results),
        'total_return': returns.sum(),
        'mean_return': returns.mean(),
        'std_return': returns.std(),
        'sharpe': calculate_sharpe(returns),
        'win_rate': (returns > 0).mean() * 100,
        'profit_factor': returns[returns > 0].sum() / (-returns[returns < 0].sum() + 1e-8),
        'avg_length': lengths.mean(),
        'median_length': np.median(lengths),
        'max_length': lengths.max(),
        'action_pct': action_pct,
        'exit_reasons': dict(exit_reasons),
        'bar0_exits': bar0_exits,
        'bar1_exits': bar1_exits,
        'large_wins': large_wins,
        'small_wins': small_wins,
        'breakeven': breakeven,
        'small_losses': small_losses,
        'large_losses': large_losses,
        'classical_return': classical.sum(),
        'classical_sharpe': calculate_sharpe(classical),
        'improvement_return': (returns.sum() - classical.sum()) / abs(classical.sum()) * 100 if classical.sum() != 0 else 0,
        'max_drawdown': (np.maximum.accumulate(np.cumsum(returns)) - np.cumsum(returns)).max(),
    }


def create_comparison_plots(v1_stats: Dict, v5_stats: Dict, v1_results: List, v5_results: List, output_dir: Path):
    """Create comprehensive comparison visualizations."""

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('V1 Baseline vs V5 Anti-Cheat: Deep Comparison', fontsize=16, fontweight='bold')

    # 1. Cumulative Returns
    ax1 = fig.add_subplot(3, 3, 1)
    v1_cumret = np.cumsum([r.final_pnl for r in v1_results]) * 100
    v5_cumret = np.cumsum([r.final_pnl for r in v5_results]) * 100
    classical_cumret = np.cumsum([r.classical_pnl for r in v1_results]) * 100

    ax1.plot(v1_cumret, label=f"V1: {v1_cumret[-1]:.1f}%", color='#e74c3c', linewidth=2)
    ax1.plot(v5_cumret, label=f"V5: {v5_cumret[-1]:.1f}%", color='#2ecc71', linewidth=2)
    ax1.plot(classical_cumret, label=f"Classical: {classical_cumret[-1]:.1f}%", color='gray', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.set_title('Equity Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Action Distribution Comparison
    ax2 = fig.add_subplot(3, 3, 2)
    actions = Actions.NAMES
    x = np.arange(len(actions))
    width = 0.35

    v1_pcts = [v1_stats['action_pct'].get(a, 0) for a in actions]
    v5_pcts = [v5_stats['action_pct'].get(a, 0) for a in actions]

    bars1 = ax2.bar(x - width/2, v1_pcts, width, label='V1', color='#e74c3c', alpha=0.8)
    bars2 = ax2.bar(x + width/2, v5_pcts, width, label='V5', color='#2ecc71', alpha=0.8)

    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Action Distribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(actions, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Exit Bar Distribution
    ax3 = fig.add_subplot(3, 3, 3)
    v1_bars = [r.exit_bar for r in v1_results]
    v5_bars = [r.exit_bar for r in v5_results]

    bins = np.arange(0, 31, 1)
    ax3.hist(v1_bars, bins=bins, alpha=0.6, label=f'V1 (avg={np.mean(v1_bars):.1f})', color='#e74c3c')
    ax3.hist(v5_bars, bins=bins, alpha=0.6, label=f'V5 (avg={np.mean(v5_bars):.1f})', color='#2ecc71')
    ax3.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Bar 1 (t+1)')
    ax3.set_xlabel('Exit Bar')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Exit Timing Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Return Distribution
    ax4 = fig.add_subplot(3, 3, 4)
    v1_rets = [r.final_pnl * 100 for r in v1_results]
    v5_rets = [r.final_pnl * 100 for r in v5_results]

    bins = np.linspace(-0.3, 0.5, 50)
    ax4.hist(v1_rets, bins=bins, alpha=0.6, label='V1', color='#e74c3c')
    ax4.hist(v5_rets, bins=bins, alpha=0.6, label='V5', color='#2ecc71')
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Return (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Return Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Exit Reason Breakdown
    ax5 = fig.add_subplot(3, 3, 5)
    all_reasons = set(v1_stats['exit_reasons'].keys()) | set(v5_stats['exit_reasons'].keys())
    reasons = sorted(all_reasons)

    v1_reason_counts = [v1_stats['exit_reasons'].get(r, 0) for r in reasons]
    v5_reason_counts = [v5_stats['exit_reasons'].get(r, 0) for r in reasons]

    x = np.arange(len(reasons))
    bars1 = ax5.bar(x - width/2, v1_reason_counts, width, label='V1', color='#e74c3c', alpha=0.8)
    bars2 = ax5.bar(x + width/2, v5_reason_counts, width, label='V5', color='#2ecc71', alpha=0.8)

    ax5.set_ylabel('Count')
    ax5.set_title('Exit Reasons')
    ax5.set_xticks(x)
    ax5.set_xticklabels(reasons, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Key Metrics Comparison
    ax6 = fig.add_subplot(3, 3, 6)
    metrics = ['Sharpe', 'Win Rate', 'Profit Factor', 'Avg Length']
    v1_vals = [v1_stats['sharpe'], v1_stats['win_rate'], min(v1_stats['profit_factor'], 5), v1_stats['avg_length']]
    v5_vals = [v5_stats['sharpe'], v5_stats['win_rate'], min(v5_stats['profit_factor'], 5), v5_stats['avg_length']]

    x = np.arange(len(metrics))
    bars1 = ax6.bar(x - width/2, v1_vals, width, label='V1', color='#e74c3c', alpha=0.8)
    bars2 = ax6.bar(x + width/2, v5_vals, width, label='V5', color='#2ecc71', alpha=0.8)

    ax6.set_ylabel('Value')
    ax6.set_title('Key Metrics')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Return Categories
    ax7 = fig.add_subplot(3, 3, 7)
    categories = ['Large Win\n(>0.5%)', 'Small Win', 'Breakeven', 'Small Loss', 'Large Loss\n(<-0.5%)']
    v1_cats = [v1_stats['large_wins'], v1_stats['small_wins'], v1_stats['breakeven'], v1_stats['small_losses'], v1_stats['large_losses']]
    v5_cats = [v5_stats['large_wins'], v5_stats['small_wins'], v5_stats['breakeven'], v5_stats['small_losses'], v5_stats['large_losses']]

    x = np.arange(len(categories))
    bars1 = ax7.bar(x - width/2, v1_cats, width, label='V1', color='#e74c3c', alpha=0.8)
    bars2 = ax7.bar(x + width/2, v5_cats, width, label='V5', color='#2ecc71', alpha=0.8)

    ax7.set_ylabel('Count')
    ax7.set_title('Return Categories')
    ax7.set_xticks(x)
    ax7.set_xticklabels(categories, fontsize=8)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Bar 0/1 Exit Analysis
    ax8 = fig.add_subplot(3, 3, 8)
    bar_categories = ['Bar 0\n(Immediate)', 'Bar 1\n(t+1)', 'Bar 2+']
    v1_bar_counts = [v1_stats['bar0_exits'], v1_stats['bar1_exits'], v1_stats['n_trades'] - v1_stats['bar0_exits'] - v1_stats['bar1_exits']]
    v5_bar_counts = [v5_stats['bar0_exits'], v5_stats['bar1_exits'], v5_stats['n_trades'] - v5_stats['bar0_exits'] - v5_stats['bar1_exits']]

    x = np.arange(len(bar_categories))
    bars1 = ax8.bar(x - width/2, v1_bar_counts, width, label='V1', color='#e74c3c', alpha=0.8)
    bars2 = ax8.bar(x + width/2, v5_bar_counts, width, label='V5', color='#2ecc71', alpha=0.8)

    ax8.set_ylabel('Count')
    ax8.set_title('Early Exit Analysis (Anti-Cheat)')
    ax8.set_xticks(x)
    ax8.set_xticklabels(bar_categories)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

    # 9. Summary Table
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')

    summary_text = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║          DEEP COMPARISON SUMMARY                         ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Metric              │    V1 Baseline   │   V5 Anti-Cheat ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Total Return        │  {v1_stats['total_return']*100:>10.2f}%   │  {v5_stats['total_return']*100:>10.2f}%    ║
    ║  Sharpe Ratio        │  {v1_stats['sharpe']:>10.2f}    │  {v5_stats['sharpe']:>10.2f}     ║
    ║  Win Rate            │  {v1_stats['win_rate']:>10.1f}%   │  {v5_stats['win_rate']:>10.1f}%    ║
    ║  Profit Factor       │  {v1_stats['profit_factor']:>10.2f}    │  {v5_stats['profit_factor']:>10.2f}     ║
    ║  Avg Exit Bar        │  {v1_stats['avg_length']:>10.1f}    │  {v5_stats['avg_length']:>10.1f}     ║
    ║  Bar 0 Exits         │  {v1_stats['bar0_exits']:>10d}    │  {v5_stats['bar0_exits']:>10d}     ║
    ║  Large Losses        │  {v1_stats['large_losses']:>10d}    │  {v5_stats['large_losses']:>10d}     ║
    ║  Breakeven Exits     │  {v1_stats['breakeven']:>10d}    │  {v5_stats['breakeven']:>10d}     ║
    ║  Max Drawdown        │  {v1_stats['max_drawdown']*100:>10.2f}%   │  {v5_stats['max_drawdown']*100:>10.2f}%    ║
    ╠══════════════════════════════════════════════════════════╣
    ║  vs Classical        │  {v1_stats['improvement_return']:>+10.1f}%   │  {v5_stats['improvement_return']:>+10.1f}%    ║
    ╚══════════════════════════════════════════════════════════╝
    """
    ax9.text(0.1, 0.5, summary_text, fontfamily='monospace', fontsize=9, va='center', ha='left')

    plt.tight_layout()
    plt.savefig(output_dir / 'v1_vs_v5_deep_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_dir / 'v1_vs_v5_deep_comparison.png'}")
    plt.close()


def main():
    print("=" * 70)
    print(" DEEP COMPARISON: V1 BASELINE vs V5 ANTI-CHEAT")
    print("=" * 70)

    device = 'cpu'

    # Load OOS episodes
    oos_file = STRATEGY_DIR / "data" / "episodes_test_2022_2025.pkl"
    print(f"\nLoading OOS episodes: {oos_file}")
    episodes = load_episodes(oos_file)
    print(f"Loaded {len(episodes)} episodes")

    # Load V1 model
    v1_model_path = V1_DIR / "models" / "exit_policy_final.pt"
    print(f"\nLoading V1 model: {v1_model_path}")
    v1_model, v1_steps = load_model(v1_model_path, hidden_dims=[256, 256], device=device)
    print(f"  Trained for {v1_steps:,} steps")

    # Load V5 model
    v5_model_path = V5_DIR / "models" / "exit_policy_final.pt"
    print(f"\nLoading V5 model: {v5_model_path}")
    v5_model, v5_steps = load_model(v5_model_path, hidden_dims=[512, 256], device=device)
    print(f"  Trained for {v5_steps:,} steps")

    # Load V5 config
    v5_config = V5Config(device=device)

    # Evaluate both models
    print(f"\nEvaluating V1 on {len(episodes)} trades...")
    v1_results = []
    for i, ep in enumerate(episodes):
        result = evaluate_v1(v1_model, ep, i, device)
        v1_results.append(result)
        if (i + 1) % 200 == 0:
            print(f"  V1: {i+1}/{len(episodes)}")

    print(f"\nEvaluating V5 on {len(episodes)} trades...")
    v5_results = []
    for i, ep in enumerate(episodes):
        result = evaluate_v5(v5_model, ep, i, device, v5_config)
        v5_results.append(result)
        if (i + 1) % 200 == 0:
            print(f"  V5: {i+1}/{len(episodes)}")

    # Analyze results
    print("\nAnalyzing results...")
    v1_stats = analyze_results(v1_results, "V1 Baseline")
    v5_stats = analyze_results(v5_results, "V5 Anti-Cheat")

    # Create visualizations
    print("\nCreating comparison plots...")
    output_dir = V5_DIR / "visualizations"
    output_dir.mkdir(exist_ok=True)
    create_comparison_plots(v1_stats, v5_stats, v1_results, v5_results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print(" COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'V1 Baseline':>15} {'V5 Anti-Cheat':>15} {'Change':>12}")
    print("-" * 70)
    print(f"{'Total Return':<25} {v1_stats['total_return']*100:>14.2f}% {v5_stats['total_return']*100:>14.2f}% {(v5_stats['total_return']-v1_stats['total_return'])*100:>+11.2f}%")
    print(f"{'Sharpe Ratio':<25} {v1_stats['sharpe']:>15.2f} {v5_stats['sharpe']:>15.2f} {v5_stats['sharpe']-v1_stats['sharpe']:>+12.2f}")
    print(f"{'Win Rate':<25} {v1_stats['win_rate']:>14.1f}% {v5_stats['win_rate']:>14.1f}% {v5_stats['win_rate']-v1_stats['win_rate']:>+11.1f}%")
    print(f"{'Profit Factor':<25} {v1_stats['profit_factor']:>15.2f} {v5_stats['profit_factor']:>15.2f} {v5_stats['profit_factor']-v1_stats['profit_factor']:>+12.2f}")
    print(f"{'Avg Exit Bar':<25} {v1_stats['avg_length']:>15.1f} {v5_stats['avg_length']:>15.1f} {v5_stats['avg_length']-v1_stats['avg_length']:>+12.1f}")
    print(f"{'Bar 0 Exits':<25} {v1_stats['bar0_exits']:>15d} {v5_stats['bar0_exits']:>15d} {v5_stats['bar0_exits']-v1_stats['bar0_exits']:>+12d}")
    print(f"{'Large Losses (<-0.5%)':<25} {v1_stats['large_losses']:>15d} {v5_stats['large_losses']:>15d} {v5_stats['large_losses']-v1_stats['large_losses']:>+12d}")
    print(f"{'Breakeven Exits':<25} {v1_stats['breakeven']:>15d} {v5_stats['breakeven']:>15d} {v5_stats['breakeven']-v1_stats['breakeven']:>+12d}")
    print(f"{'Max Drawdown':<25} {v1_stats['max_drawdown']*100:>14.2f}% {v5_stats['max_drawdown']*100:>14.2f}% {(v5_stats['max_drawdown']-v1_stats['max_drawdown'])*100:>+11.2f}%")

    print("\n" + "-" * 70)
    print("Action Distribution:")
    print(f"{'Action':<15} {'V1':>10} {'V5':>10} {'Change':>10}")
    for action in Actions.NAMES:
        v1_pct = v1_stats['action_pct'].get(action, 0)
        v5_pct = v5_stats['action_pct'].get(action, 0)
        print(f"{action:<15} {v1_pct:>9.1f}% {v5_pct:>9.1f}% {v5_pct-v1_pct:>+9.1f}%")

    print("\n" + "=" * 70)

    # Return stats for report generation
    return v1_stats, v5_stats, v1_results, v5_results


if __name__ == "__main__":
    v1_stats, v5_stats, v1_results, v5_results = main()
