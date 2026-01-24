"""
Trade dataclass and related types for the RL Trade Visualizer.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class Trade:
    """
    Represents an active or completed trade.

    Tracks entry/exit details, P&L, position sizing for partial exits,
    and complete action history from the RL agent.
    """

    # Entry details
    entry_time: str
    entry_price: float
    entry_idx: int
    direction: int  # 1 = long, -1 = short
    sl: float  # Original stop loss
    tp: float  # Take profit
    entry_atr: float

    # Exit details (None until trade closes)
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_idx: Optional[int] = None
    pnl_pips: Optional[float] = None
    pnl_dollars: Optional[float] = None

    # Maximum excursions (tracking best/worst during trade)
    max_favorable_pips: float = 0.0
    max_adverse_pips: float = 0.0

    # Position sizing (1.0 = full position, decreases with partial exits)
    position_size: float = 1.0
    realized_pnl_pips: float = 0.0
    realized_pnl_dollars: float = 0.0
    partial_exits: List = field(default_factory=list)

    # Action tracking
    action_counts: Dict = field(default_factory=lambda: {
        'HOLD': 0, 'EXIT': 0, 'TIGHTEN_SL': 0, 'TRAIL_BE': 0, 'PARTIAL': 0
    })

    # Complete action history with full details
    action_history: List = field(default_factory=list)

    # Exit metadata
    exit_reason: str = ""  # RL_EXIT, SL_HIT, TRAIL_BE_HIT, TIGHTEN_SL_HIT, TP_HIT
    sl_tightened: bool = False
    trailed_to_be: bool = False

    def is_active(self) -> bool:
        """Check if trade is still active."""
        return self.exit_time is None

    def total_pnl_dollars(self) -> float:
        """Get total P&L including realized and unrealized."""
        if self.pnl_dollars is not None:
            return self.pnl_dollars
        return self.realized_pnl_dollars

    def add_action(
        self,
        action: str,
        bar: int,
        time: str,
        time_pretty: str,
        price: float,
        pnl_pips: float,
        pnl_dollars: float,
        position_size_pct: float,
        position_m: float,
        cumulative_pnl: float,
        sl: float,
        note: str = "",
        **extra_fields
    ):
        """Add an action to the history."""
        action_record = {
            'action': action,
            'bar': bar,
            'time': time,
            'time_pretty': time_pretty,
            'price': price,
            'pnl_pips': pnl_pips,
            'pnl_dollars': pnl_dollars,
            'position_size_pct': position_size_pct,
            'position_m': position_m,
            'cumulative_pnl': cumulative_pnl,
            'sl': sl,
            'note': note,
            **extra_fields
        }
        self.action_history.append(action_record)

    def to_dict(self, current_price: float = None, current_pnl_pips: float = None, config=None) -> Dict:
        """
        Convert trade to dictionary for JSON serialization.

        Args:
            current_price: Current market price (for active trades)
            current_pnl_pips: Current unrealized P&L in pips (for active trades)
            config: VisualizerConfig for position/pip values
        """
        # Import here to avoid circular import
        from .config import POSITION_SIZE, PIP_VALUE

        is_active = self.is_active()

        # Calculate P&L for active trades
        if is_active and current_price is not None:
            unrealized_pnl = (current_pnl_pips or 0) * PIP_VALUE / 10000 * self.position_size
            total_pnl = self.realized_pnl_dollars + unrealized_pnl
            pnl_pips = current_pnl_pips or 0
            pnl_dollars = total_pnl
        else:
            pnl_pips = self.pnl_pips or 0
            pnl_dollars = self.pnl_dollars or 0
            unrealized_pnl = 0

        # Build partial exit details
        partial_details = []
        for i, p in enumerate(self.partial_exits):
            partial_details.append({
                'num': i + 1,
                'pips': float(p['pips']),
                'dollars': float(p['dollars']),
                'size_pct': float(p['size'] * 100),
                'time': str(p.get('time', '')),
                'time_pretty': p.get('time_pretty', '')
            })

        return {
            'is_active': is_active,
            'direction': int(self.direction),
            'entry_price': float(self.entry_price),
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'current_price': float(current_price) if current_price else None,
            'pnl_pips': float(pnl_pips),
            'pnl_dollars': float(pnl_dollars),
            'unrealized_pnl': float(unrealized_pnl) if is_active else 0,
            'exit_reason': str(self.exit_reason) if self.exit_reason else '',
            'action_counts': {k: int(v) for k, v in self.action_counts.items()},
            'sl_tightened': bool(self.sl_tightened),
            'trailed_to_be': bool(self.trailed_to_be),
            'num_partials': int(len(self.partial_exits)),
            'partial_details': partial_details,
            'partial_pnl': float(self.realized_pnl_dollars),
            'position_size_pct': float(self.position_size * 100),
            'sl_price': float(self.sl),
            'tp_price': float(self.tp),
            'mfe': float(self.max_favorable_pips),
            'mae': float(self.max_adverse_pips),
            'entry_time': str(self.entry_time),
            'exit_time': str(self.exit_time) if self.exit_time else None,
            'action_history': list(self.action_history)
        }
