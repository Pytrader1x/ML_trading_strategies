"""
Metrics calculator for trading performance analysis.
"""

from typing import Dict, List
import numpy as np


class MetricsCalculator:
    """
    Calculates comprehensive trading metrics from trade history.

    Includes Sharpe ratio, win rate, profit factor, drawdown, and more.
    """

    def __init__(self, position_size: float = 2_000_000):
        """
        Initialize the metrics calculator.

        Args:
            position_size: Notional position size for return calculations
        """
        self.position_size = position_size
        self.reset()

    def reset(self):
        """Reset all tracked statistics."""
        self.stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pips': 0.0,
            'total_pnl': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'long_trades': 0,
            'short_trades': 0,
            'long_wins': 0,
            'short_wins': 0,
            'long_pnl': 0.0,
            'short_pnl': 0.0,
            'total_bars_held': 0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'pnl_history': [],
            'partial_count': 0,
            'sl_count': 0,
            'trail_be_count': 0,
            'tighten_sl_count': 0,
            'rl_exit_count': 0
        }

    def update_from_trade(
        self,
        pnl_dollars: float,
        pnl_pips: float,
        direction: int,
        bars_held: int,
        exit_reason: str,
        remaining_position_size: float,
        partial_count: int = 0
    ):
        """
        Update statistics from a completed trade.

        Args:
            pnl_dollars: Total P&L in dollars
            pnl_pips: P&L for this exit portion in pips
            direction: Trade direction (1=long, -1=short)
            bars_held: Number of bars the trade was held
            exit_reason: Reason for exit (RL_EXIT, SL_HIT, etc.)
            remaining_position_size: Position size at exit (fraction of original)
            partial_count: Number of partial exits in this trade
        """
        s = self.stats

        s['trades'] += 1
        s['total_pips'] += pnl_pips * remaining_position_size
        s['total_pnl'] += pnl_dollars
        s['pnl_history'].append(pnl_dollars)
        s['total_bars_held'] += bars_held

        # Win/Loss tracking
        if pnl_dollars > 0:
            s['wins'] += 1
            s['gross_profit'] += pnl_dollars
            s['max_win'] = max(s['max_win'], pnl_dollars)
            s['consecutive_wins'] += 1
            s['consecutive_losses'] = 0
            s['max_consecutive_wins'] = max(s['max_consecutive_wins'], s['consecutive_wins'])
        else:
            s['losses'] += 1
            s['gross_loss'] += abs(pnl_dollars)
            s['max_loss'] = min(s['max_loss'], pnl_dollars)
            s['consecutive_losses'] += 1
            s['consecutive_wins'] = 0
            s['max_consecutive_losses'] = max(s['max_consecutive_losses'], s['consecutive_losses'])

        # Long/Short tracking
        if direction == 1:
            s['long_trades'] += 1
            s['long_pnl'] += pnl_dollars
            if pnl_dollars > 0:
                s['long_wins'] += 1
        else:
            s['short_trades'] += 1
            s['short_pnl'] += pnl_dollars
            if pnl_dollars > 0:
                s['short_wins'] += 1

        # Exit reason tracking
        if 'SL_HIT' in exit_reason:
            s['sl_count'] += 1
        elif 'TRAIL_BE' in exit_reason:
            s['trail_be_count'] += 1
        elif 'TIGHTEN' in exit_reason:
            s['tighten_sl_count'] += 1
        elif 'RL_EXIT' in exit_reason or 'RL_PARTIAL' in exit_reason:
            s['rl_exit_count'] += 1

        # Partial count
        s['partial_count'] += partial_count

    def add_partial_pnl(self, pnl_pips: float, pnl_dollars: float, position_size: float):
        """Track P&L from a partial exit."""
        s = self.stats
        s['total_pips'] += pnl_pips * position_size
        s['total_pnl'] += pnl_dollars

    def compute_metrics(self) -> Dict:
        """
        Compute all derived trading metrics.

        Returns:
            Dictionary of computed metrics
        """
        s = self.stats
        n = s['trades']

        if n == 0:
            return self._empty_metrics()

        # Basic rates
        win_rate = (s['wins'] / n * 100) if n > 0 else 0
        long_pct = (s['long_trades'] / n * 100) if n > 0 else 0
        short_pct = (s['short_trades'] / n * 100) if n > 0 else 0
        long_win_rate = (s['long_wins'] / s['long_trades'] * 100) if s['long_trades'] > 0 else 0
        short_win_rate = (s['short_wins'] / s['short_trades'] * 100) if s['short_trades'] > 0 else 0

        # Averages
        avg_win = s['gross_profit'] / s['wins'] if s['wins'] > 0 else 0
        avg_loss = s['gross_loss'] / s['losses'] if s['losses'] > 0 else 0
        avg_trade = s['total_pnl'] / n
        avg_duration = s['total_bars_held'] / n

        # Profit Factor
        profit_factor = s['gross_profit'] / s['gross_loss'] if s['gross_loss'] > 0 else (999.9 if s['gross_profit'] > 0 else 0)

        # Sharpe Ratio (annualized, assuming 15M bars)
        sharpe_ratio = self._compute_sharpe_ratio()

        # Return % (on notional)
        return_pct = (s['total_pnl'] / self.position_size * 100)

        # Max Drawdown
        max_drawdown = self._compute_max_drawdown()

        # Exit type percentages
        rl_exit_pct = (s['rl_exit_count'] / n * 100) if n > 0 else 0
        sl_hit_pct = (s['sl_count'] / n * 100) if n > 0 else 0
        partial_per_trade = s['partial_count'] / n if n > 0 else 0

        return {
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'avg_win': round(avg_win, 0),
            'avg_loss': round(avg_loss, 0),
            'avg_trade': round(avg_trade, 0),
            'avg_duration': round(avg_duration, 1),
            'long_pct': round(long_pct, 1),
            'short_pct': round(short_pct, 1),
            'long_win_rate': round(long_win_rate, 1),
            'short_win_rate': round(short_win_rate, 1),
            'expectancy': round(avg_trade, 0),
            'return_pct': round(return_pct, 3),
            'max_drawdown': round(max_drawdown, 0),
            'rl_exit_pct': round(rl_exit_pct, 1),
            'sl_hit_pct': round(sl_hit_pct, 1),
            'partial_per_trade': round(partial_per_trade, 2),
            'max_consec_wins': s['max_consecutive_wins'],
            'max_consec_losses': s['max_consecutive_losses'],
            'max_win': round(s['max_win'], 0),
            'max_loss': round(s['max_loss'], 0),
            'long_pnl': round(s['long_pnl'], 0),
            'short_pnl': round(s['short_pnl'], 0)
        }

    def _compute_sharpe_ratio(self) -> float:
        """Compute annualized Sharpe ratio."""
        s = self.stats
        n = s['trades']

        if len(s['pnl_history']) < 2:
            return 0.0

        bars_per_year = 35040  # 365.25 * 24 * 4

        returns = np.array(s['pnl_history'])
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return > 0:
            trades_per_year = bars_per_year / max(1, s['total_bars_held'] / n)
            return (mean_return / std_return) * np.sqrt(trades_per_year)

        return 0.0

    def _compute_max_drawdown(self) -> float:
        """Compute maximum drawdown from P&L history."""
        s = self.stats

        if not s['pnl_history']:
            return 0.0

        max_drawdown = 0.0
        running_pnl = 0
        peak = 0

        for pnl in s['pnl_history']:
            running_pnl += pnl
            peak = max(peak, running_pnl)
            drawdown = peak - running_pnl
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _empty_metrics(self) -> Dict:
        """Return empty metrics when no trades."""
        return {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_trade': 0.0,
            'avg_duration': 0.0,
            'long_pct': 0.0,
            'short_pct': 0.0,
            'long_win_rate': 0.0,
            'short_win_rate': 0.0,
            'expectancy': 0.0,
            'return_pct': 0.0,
            'max_drawdown': 0.0,
            'rl_exit_pct': 0.0,
            'sl_hit_pct': 0.0,
            'partial_per_trade': 0.0,
            'max_consec_wins': 0,
            'max_consec_losses': 0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'long_pnl': 0.0,
            'short_pnl': 0.0
        }

    def get_stats(self) -> Dict:
        """Get raw statistics dictionary."""
        return self.stats.copy()
