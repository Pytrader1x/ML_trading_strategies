#!/usr/bin/env python3
"""
Trade Visualization for v5_anti_cheat.

Shows trade decision-making from entry (bar 0) to exit:
- PnL evolution throughout the trade
- Model decisions at each bar with probabilities
- Aggregate statistics on action distribution
- Positive/negative/breakeven trade analysis
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

EXPERIMENT_DIR = Path(__file__).parent
STRATEGY_DIR = EXPERIMENT_DIR.parent.parent

sys.path.insert(0, str(EXPERIMENT_DIR))
from config import PPOConfig, Actions
from env import TradeEpisode

sys.path.insert(0, str(STRATEGY_DIR))
from model import ActorCritic


@dataclass
class TradeHistory:
    """Full history of a single trade."""
    episode_idx: int
    direction: int  # 1=long, -1=short
    entry_price: float
    classical_pnl: float

    # Bar-by-bar history
    bars: List[int]
    pnls: List[float]  # Unrealized PnL at each bar
    actions: List[int]  # Action taken at each bar
    action_probs: List[List[float]]  # Full action probabilities at each bar

    # Outcome
    final_pnl: float
    exit_bar: int
    exit_reason: str  # 'EXIT', 'PARTIAL', 'SL_HIT', 'BREAKEVEN_SL', 'END'


# Colors for actions
ACTION_COLORS = {
    Actions.HOLD: '#4a90d9',         # Blue
    Actions.EXIT: '#e74c3c',          # Red
    Actions.TIGHTEN_SL: '#f39c12',    # Orange
    Actions.TRAIL_BREAKEVEN: '#2ecc71', # Green
    Actions.PARTIAL_EXIT: '#9b59b6',   # Purple
}


def load_episodes(episode_file: Path) -> list:
    """Load trade episodes from pickle file."""
    with open(episode_file, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data['episodes']
    return data


def get_action_mask(bar: int, min_exit_bar: int, min_trail_bar: int) -> torch.Tensor:
    """Get action mask for single episode."""
    mask = torch.ones(1, 5, dtype=torch.bool)
    if bar < min_exit_bar:
        mask[0, Actions.EXIT] = False
        mask[0, Actions.PARTIAL_EXIT] = False
    if bar < min_trail_bar:
        mask[0, Actions.TRAIL_BREAKEVEN] = False
    return mask


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


def evaluate_trade_with_history(
    policy: ActorCritic,
    episode: TradeEpisode,
    episode_idx: int,
    device: str,
    config: PPOConfig
) -> TradeHistory:
    """Evaluate a trade and record full history."""
    min_exit_bar = config.reward.min_exit_bar
    min_trail_bar = config.reward.min_trail_bar
    min_profit_for_trail = config.reward.min_profit_for_trail
    breakeven_buffer = config.reward.breakeven_buffer_pct

    max_bars = episode.market_tensor.shape[0]
    sl_atr = 1.1
    max_fav = 0.0
    max_adv = 0.0
    action_hist = []

    # History tracking
    bars = []
    pnls = []
    actions = []
    action_probs = []

    breakeven_active = False

    for bar in range(max_bars):
        if not episode.valid_mask[bar]:
            # End of episode
            final_pnl = pnls[-1] if pnls else 0.0
            return TradeHistory(
                episode_idx=episode_idx,
                direction=episode.direction,
                entry_price=episode.entry_price,
                classical_pnl=episode.classical_pnl,
                bars=bars,
                pnls=pnls,
                actions=actions,
                action_probs=action_probs,
                final_pnl=final_pnl,
                exit_bar=bar,
                exit_reason='END'
            )

        state, max_fav, max_adv = build_state(episode, bar, sl_atr, max_fav, max_adv, action_hist)
        state = state.to(device)

        unrealized_pnl = episode.market_tensor[bar, 0].item()
        bars.append(bar)
        pnls.append(unrealized_pnl)

        # Get action and probabilities
        action_mask = get_action_mask(bar, min_exit_bar, min_trail_bar).to(device)

        with torch.no_grad():
            logits, _ = policy(state)
            # Apply mask
            masked_logits = logits.clone()
            masked_logits[~action_mask] = float('-inf')
            probs = torch.softmax(masked_logits, dim=-1)
            action = probs.argmax(dim=-1).item()

        actions.append(action)
        action_probs.append(probs[0].cpu().tolist())
        action_hist.append(action / 4.0)

        # Process action
        if action == Actions.EXIT:
            return TradeHistory(
                episode_idx=episode_idx,
                direction=episode.direction,
                entry_price=episode.entry_price,
                classical_pnl=episode.classical_pnl,
                bars=bars,
                pnls=pnls,
                actions=actions,
                action_probs=action_probs,
                final_pnl=unrealized_pnl,
                exit_bar=bar + 1,
                exit_reason='EXIT'
            )

        elif action == Actions.PARTIAL_EXIT:
            return TradeHistory(
                episode_idx=episode_idx,
                direction=episode.direction,
                entry_price=episode.entry_price,
                classical_pnl=episode.classical_pnl,
                bars=bars,
                pnls=pnls,
                actions=actions,
                action_probs=action_probs,
                final_pnl=unrealized_pnl,
                exit_bar=bar + 1,
                exit_reason='PARTIAL'
            )

        elif action == Actions.TIGHTEN_SL:
            sl_atr *= 0.75

        elif action == Actions.TRAIL_BREAKEVEN:
            if unrealized_pnl > min_profit_for_trail:
                breakeven_active = True

        # Check SL hit
        if bar + 1 < max_bars and episode.valid_mask[bar + 1]:
            next_pnl = episode.market_tensor[bar + 1, 0].item()

            if breakeven_active:
                if next_pnl < breakeven_buffer:
                    pnls.append(breakeven_buffer)
                    bars.append(bar + 1)
                    actions.append(-1)  # Special marker for SL
                    action_probs.append([0, 0, 0, 0, 0])
                    return TradeHistory(
                        episode_idx=episode_idx,
                        direction=episode.direction,
                        entry_price=episode.entry_price,
                        classical_pnl=episode.classical_pnl,
                        bars=bars,
                        pnls=pnls,
                        actions=actions,
                        action_probs=action_probs,
                        final_pnl=breakeven_buffer,
                        exit_bar=bar + 2,
                        exit_reason='BREAKEVEN_SL'
                    )
            else:
                sl_threshold = -sl_atr * episode.entry_atr / episode.entry_price
                if next_pnl < sl_threshold:
                    pnls.append(sl_threshold)
                    bars.append(bar + 1)
                    actions.append(-1)  # Special marker for SL
                    action_probs.append([0, 0, 0, 0, 0])
                    return TradeHistory(
                        episode_idx=episode_idx,
                        direction=episode.direction,
                        entry_price=episode.entry_price,
                        classical_pnl=episode.classical_pnl,
                        bars=bars,
                        pnls=pnls,
                        actions=actions,
                        action_probs=action_probs,
                        final_pnl=sl_threshold,
                        exit_bar=bar + 2,
                        exit_reason='SL_HIT'
                    )

    # Reached max bars
    return TradeHistory(
        episode_idx=episode_idx,
        direction=episode.direction,
        entry_price=episode.entry_price,
        classical_pnl=episode.classical_pnl,
        bars=bars,
        pnls=pnls,
        actions=actions,
        action_probs=action_probs,
        final_pnl=pnls[-1] if pnls else 0.0,
        exit_bar=max_bars,
        exit_reason='END'
    )


def plot_single_trade(trade: TradeHistory, save_path: Optional[Path] = None):
    """
    Plot a single trade showing:
    - PnL evolution with action markers
    - Action probability heatmap over time
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    fig.suptitle(f'Trade #{trade.episode_idx} | Final PnL: {trade.final_pnl*100:.2f}% | Exit: {trade.exit_reason}',
                 fontsize=14, fontweight='bold')

    # Top: PnL curve with action markers
    ax1 = axes[0]

    # PnL line
    pnls_pct = [p * 100 for p in trade.pnls]
    ax1.plot(trade.bars, pnls_pct, 'k-', linewidth=2, label='PnL', zorder=1)

    # Fill profit/loss regions
    ax1.fill_between(trade.bars, 0, pnls_pct, where=[p > 0 for p in pnls_pct],
                     color='#2ecc71', alpha=0.2, label='Profit')
    ax1.fill_between(trade.bars, 0, pnls_pct, where=[p < 0 for p in pnls_pct],
                     color='#e74c3c', alpha=0.2, label='Loss')

    # Action markers
    for i, (bar, pnl, action) in enumerate(zip(trade.bars, pnls_pct, trade.actions)):
        if action < 0:  # SL hit marker
            ax1.scatter(bar, pnl, marker='X', s=200, c='red', zorder=3, edgecolors='black', linewidths=1)
        elif action != Actions.HOLD:
            color = ACTION_COLORS.get(action, 'gray')
            ax1.scatter(bar, pnl, marker='o', s=100, c=color, zorder=3, edgecolors='black', linewidths=1)

    # Zero line
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Entry and exit markers
    ax1.axvline(x=0, color='blue', linestyle='-', alpha=0.7, linewidth=2, label='Entry')
    ax1.axvline(x=trade.exit_bar - 1, color='red', linestyle='-', alpha=0.7, linewidth=2, label='Exit')

    ax1.set_xlabel('Bar (15-min)', fontsize=11)
    ax1.set_ylabel('PnL (%)', fontsize=11)
    ax1.set_title('PnL Evolution with Model Decisions', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Legend with action colors
    legend_patches = [
        mpatches.Patch(color=ACTION_COLORS[Actions.HOLD], label='HOLD'),
        mpatches.Patch(color=ACTION_COLORS[Actions.EXIT], label='EXIT'),
        mpatches.Patch(color=ACTION_COLORS[Actions.TIGHTEN_SL], label='TIGHTEN_SL'),
        mpatches.Patch(color=ACTION_COLORS[Actions.TRAIL_BREAKEVEN], label='TRAIL_BE'),
        mpatches.Patch(color=ACTION_COLORS[Actions.PARTIAL_EXIT], label='PARTIAL'),
    ]
    ax1.legend(handles=legend_patches, loc='upper right', fontsize=9)

    # Bottom: Action probability heatmap
    ax2 = axes[1]

    # Filter out SL hit bars (action=-1)
    valid_bars = [b for b, a in zip(trade.bars, trade.actions) if a >= 0]
    valid_probs = [p for a, p in zip(trade.actions, trade.action_probs) if a >= 0]

    if valid_probs:
        probs_array = np.array(valid_probs).T  # (5, n_bars)
        im = ax2.imshow(probs_array, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

        ax2.set_yticks(range(5))
        ax2.set_yticklabels(Actions.NAMES)
        ax2.set_xlabel('Bar (15-min)', fontsize=11)
        ax2.set_ylabel('Action', fontsize=11)
        ax2.set_title('Action Probabilities (Darker = Higher)', fontsize=12)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Probability', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    plt.close()


def plot_trade_summary(trades: List[TradeHistory], save_path: Optional[Path] = None):
    """
    Plot aggregate trade summary:
    - Return distribution (histogram)
    - Action distribution by bar position
    - Win/Loss/Breakeven breakdown
    - PnL contribution by exit type
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'V5 Anti-Cheat OOS Trade Analysis (n={len(trades)})', fontsize=16, fontweight='bold')

    returns = [t.final_pnl * 100 for t in trades]

    # 1. Return Distribution
    ax1 = axes[0, 0]

    bins = np.linspace(min(returns), max(returns), 50)
    ax1.hist(returns, bins=bins, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
    ax1.axvline(x=np.mean(returns), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
    ax1.axvline(x=np.median(returns), color='orange', linestyle='-', linewidth=2, label=f'Median: {np.median(returns):.2f}%')

    ax1.set_xlabel('Return (%)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Return Distribution', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Action Distribution by Bar
    ax2 = axes[0, 1]

    # Aggregate action counts by bar position
    max_bar_to_plot = 30
    action_by_bar = {i: [0] * 5 for i in range(max_bar_to_plot)}

    for trade in trades:
        for bar, action in zip(trade.bars, trade.actions):
            if bar < max_bar_to_plot and action >= 0:
                action_by_bar[bar][action] += 1

    bars_x = list(range(max_bar_to_plot))
    bottom = np.zeros(max_bar_to_plot)

    for action_idx in range(5):
        counts = [action_by_bar[b][action_idx] for b in bars_x]
        ax2.bar(bars_x, counts, bottom=bottom, color=ACTION_COLORS[action_idx],
                label=Actions.NAMES[action_idx], alpha=0.8)
        bottom += counts

    ax2.set_xlabel('Bar Position', fontsize=11)
    ax2.set_ylabel('Action Count', fontsize=11)
    ax2.set_title('Action Distribution by Bar Position', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Win/Loss/Breakeven Breakdown
    ax3 = axes[1, 0]

    wins = sum(1 for r in returns if r > 0.05)
    losses = sum(1 for r in returns if r < -0.05)
    breakeven = sum(1 for r in returns if -0.05 <= r <= 0.05)

    categories = ['Wins (>0.05%)', 'Losses (<-0.05%)', 'Breakeven']
    counts = [wins, losses, breakeven]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']

    bars = ax3.bar(categories, counts, color=colors, edgecolor='black')

    # Add percentage labels
    total = len(returns)
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold')

    ax3.set_ylabel('Number of Trades', fontsize=11)
    ax3.set_title('Trade Outcome Distribution', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. PnL Contribution by Exit Type
    ax4 = axes[1, 1]

    exit_pnls = {}
    for trade in trades:
        reason = trade.exit_reason
        if reason not in exit_pnls:
            exit_pnls[reason] = []
        exit_pnls[reason].append(trade.final_pnl * 100)

    exit_types = list(exit_pnls.keys())
    total_pnls = [sum(exit_pnls[e]) for e in exit_types]
    avg_pnls = [np.mean(exit_pnls[e]) for e in exit_types]
    counts_by_exit = [len(exit_pnls[e]) for e in exit_types]

    x = np.arange(len(exit_types))
    width = 0.35

    bars1 = ax4.bar(x - width/2, total_pnls, width, label='Total PnL (%)', color='#3498db', alpha=0.8)
    ax4.set_ylabel('Total PnL (%)', fontsize=11)

    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, counts_by_exit, width, label='Count', color='#f39c12', alpha=0.8)
    ax4_twin.set_ylabel('Trade Count', fontsize=11)

    ax4.set_xticks(x)
    ax4.set_xticklabels(exit_types, rotation=15)
    ax4.set_title('PnL and Count by Exit Type', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    plt.close()


def plot_decision_waterfall(trades: List[TradeHistory], save_path: Optional[Path] = None):
    """
    Plot decision waterfall showing cumulative PnL by action type.
    Shows how much each action type contributes to overall performance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('V5 Decision Impact Analysis', fontsize=14, fontweight='bold')

    # 1. PnL by Action at Exit
    ax1 = axes[0]

    pnl_by_action = {i: [] for i in range(5)}
    for trade in trades:
        # Get the final action (the one that closed the trade)
        if trade.exit_reason in ['EXIT', 'PARTIAL']:
            for i in range(len(trade.actions) - 1, -1, -1):
                if trade.actions[i] in [Actions.EXIT, Actions.PARTIAL_EXIT]:
                    pnl_by_action[trade.actions[i]].append(trade.final_pnl * 100)
                    break
        elif trade.exit_reason == 'BREAKEVEN_SL':
            pnl_by_action[Actions.TRAIL_BREAKEVEN].append(trade.final_pnl * 100)
        elif trade.exit_reason == 'SL_HIT':
            # Attribute to TIGHTEN_SL if it was used, else HOLD
            tighten_used = any(a == Actions.TIGHTEN_SL for a in trade.actions)
            if tighten_used:
                pnl_by_action[Actions.TIGHTEN_SL].append(trade.final_pnl * 100)
            else:
                pnl_by_action[Actions.HOLD].append(trade.final_pnl * 100)

    action_names = Actions.NAMES
    total_pnls = [sum(pnl_by_action[i]) for i in range(5)]
    avg_pnls = [np.mean(pnl_by_action[i]) if pnl_by_action[i] else 0 for i in range(5)]
    counts = [len(pnl_by_action[i]) for i in range(5)]

    colors = [ACTION_COLORS[i] for i in range(5)]
    bars = ax1.bar(action_names, total_pnls, color=colors, edgecolor='black', alpha=0.8)

    # Add labels
    for bar, count, avg in zip(bars, counts, avg_pnls):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'n={count}\navg={avg:.2f}%', ha='center', fontsize=9)

    ax1.set_ylabel('Total PnL (%)', fontsize=11)
    ax1.set_title('Total PnL Contribution by Exit Trigger', fontsize=12)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Cumulative PnL by Trade (sorted)
    ax2 = axes[1]

    sorted_returns = sorted([t.final_pnl * 100 for t in trades])
    cumulative = np.cumsum(sorted_returns)

    # Color by positive/negative contribution
    colors_sorted = ['#2ecc71' if r > 0 else '#e74c3c' for r in sorted_returns]

    ax2.fill_between(range(len(cumulative)), 0, cumulative,
                     where=[c >= 0 for c in cumulative], color='#2ecc71', alpha=0.3)
    ax2.fill_between(range(len(cumulative)), 0, cumulative,
                     where=[c < 0 for c in cumulative], color='#e74c3c', alpha=0.3)
    ax2.plot(cumulative, 'k-', linewidth=2)

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Trade # (sorted by return)', fontsize=11)
    ax2.set_ylabel('Cumulative PnL (%)', fontsize=11)
    ax2.set_title('Cumulative PnL Waterfall (Sorted)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add annotations
    final_pnl = cumulative[-1]
    ax2.annotate(f'Final: {final_pnl:.2f}%', xy=(len(cumulative)-1, final_pnl),
                xytext=(len(cumulative)*0.7, final_pnl*1.1),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    plt.close()


def plot_example_trades(trades: List[TradeHistory], output_dir: Path, n_examples: int = 6):
    """
    Plot example trades: 2 wins, 2 losses, 2 breakeven.
    """
    wins = [t for t in trades if t.final_pnl > 0.001]
    losses = [t for t in trades if t.final_pnl < -0.001]
    breakevens = [t for t in trades if -0.001 <= t.final_pnl <= 0.001]

    examples = []

    # Best and worst win
    if wins:
        wins_sorted = sorted(wins, key=lambda t: t.final_pnl, reverse=True)
        examples.append(('best_win', wins_sorted[0]))
        if len(wins_sorted) > 1:
            examples.append(('typical_win', wins_sorted[len(wins_sorted)//2]))

    # Best and worst loss
    if losses:
        losses_sorted = sorted(losses, key=lambda t: t.final_pnl)
        examples.append(('worst_loss', losses_sorted[0]))
        if len(losses_sorted) > 1:
            examples.append(('typical_loss', losses_sorted[len(losses_sorted)//2]))

    # Breakeven examples
    if breakevens:
        examples.append(('breakeven_1', breakevens[0]))
        if len(breakevens) > 1:
            examples.append(('breakeven_2', breakevens[len(breakevens)//2]))

    for name, trade in examples[:n_examples]:
        save_path = output_dir / f'trade_example_{name}.png'
        plot_single_trade(trade, save_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='models/exit_policy_final.pt')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n-trades', type=int, default=500, help='Number of trades to analyze')
    parser.add_argument('--output-dir', type=str, default='visualizations')
    args = parser.parse_args()

    output_dir = EXPERIMENT_DIR / args.output_dir
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(" V5 ANTI-CHEAT TRADE VISUALIZATION")
    print("=" * 70)

    # Load OOS data
    oos_file = STRATEGY_DIR / "data" / "episodes_test_2022_2025.pkl"
    print(f"Loading: {oos_file}")
    episodes = load_episodes(oos_file)
    print(f"Loaded {len(episodes)} episodes")

    # Load model
    checkpoint_path = EXPERIMENT_DIR / args.checkpoint
    print(f"Loading model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)

    config = PPOConfig(device=args.device)
    policy = ActorCritic(config)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.to(args.device)
    policy.eval()

    # Evaluate trades with full history
    n_trades = min(args.n_trades, len(episodes))
    print(f"\nEvaluating {n_trades} trades with full history...")

    trades = []
    for i in range(n_trades):
        trade = evaluate_trade_with_history(policy, episodes[i], i, args.device, config)
        trades.append(trade)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_trades} done...")

    print(f"\nGenerating visualizations...")

    # 1. Summary plot
    plot_trade_summary(trades, output_dir / 'v5_trade_summary.png')

    # 2. Decision waterfall
    plot_decision_waterfall(trades, output_dir / 'v5_decision_waterfall.png')

    # 3. Example trades
    plot_example_trades(trades, output_dir)

    # 4. Print statistics
    returns = [t.final_pnl * 100 for t in trades]
    win_rate = sum(1 for r in returns if r > 0) / len(returns)

    print("\n" + "=" * 70)
    print(" VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"  Trades analyzed:   {len(trades)}")
    print(f"  Total Return:      {sum(returns):.2f}%")
    print(f"  Win Rate:          {win_rate*100:.1f}%")
    print(f"  Output directory:  {output_dir}")
    print()
    print("  Generated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"    - {f.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
