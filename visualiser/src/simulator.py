"""
Trade Simulator for RL Exit Optimization visualization.

Simulates trading on historical data with RL exit decisions.
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from .config import POSITION_SIZE, PIP_VALUE, WINDOW_BARS, VisualizerConfig
from .trade import Trade
from .metrics import MetricsCalculator
from .model_wrapper import ActorCriticWithActivations
from .utils import format_datetime_pretty


class TradeSimulator:
    """
    Simulates trading on historical data with RL exit decisions.

    Manages trade entries, RL-based exit logic, and produces
    visualization data for the web interface.
    """

    def __init__(
        self,
        price_data: pd.DataFrame,
        trades_data: pd.DataFrame,
        model: ActorCriticWithActivations,
        config: VisualizerConfig = None,
        device: str = "cpu"
    ):
        """
        Initialize the trade simulator.

        Args:
            price_data: OHLC price data with datetime index
            trades_data: Historical trade entries with entry_time, direction, sl, tp
            model: RL model for exit decisions
            config: Visualizer configuration
            device: PyTorch device string
        """
        self.price_data = price_data.copy()
        self.trades_data = trades_data
        self.model = model
        self.config = config or VisualizerConfig()
        self.device = torch.device(device)

        # Import Actions enum from strategy
        strategy_dir = self.config.model_path.parent.parent
        import sys
        if str(strategy_dir) not in sys.path:
            sys.path.insert(0, str(strategy_dir))
        from config import Actions
        self.Actions = Actions

        # Create local index mapping
        self.price_data['local_idx'] = range(len(price_data))

        # Create timestamp to local index mapping
        self.ts_to_idx = {str(ts): idx for idx, ts in enumerate(price_data.index)}

        # Pre-compute trade entries by LOCAL index
        self.trade_entries = {}
        for _, trade in trades_data.iterrows():
            entry_time = str(trade['entry_time'])
            if entry_time in self.ts_to_idx:
                local_idx = self.ts_to_idx[entry_time]
                self.trade_entries[local_idx] = trade

        # Current state
        self.current_idx = 0
        self.active_trade: Optional[Trade] = None

        # Track completed trades
        self.completed_trades: List[Trade] = []
        self.entries: List[Dict] = []
        self.exits: List[Dict] = []
        self.action_markers: List[Dict] = []  # Track all RL actions for chart display

        # State tracking for RL
        self.bars_held = 0
        self.max_favorable = 0.0
        self.max_adverse = 0.0
        self.action_history = [0] * 5
        self.current_sl_price = 0.0

        # Activation history for timeline visualization
        self.activation_history: List[Dict] = []
        # Neuron-action correlation tracking (across all trades)
        self.action_neuron_stats = {i: [] for i in range(5)}  # action_id -> list of activation vectors

        # Metrics calculator
        self.metrics = MetricsCalculator(position_size=self.config.position_size)

        # Add technical indicators
        self._add_indicators()

    def _add_indicators(self):
        """Add technical indicators for state computation."""
        df = self.price_data

        # ATR
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # ADX (simplified)
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-10))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()

        # BB position
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['bb_upper'] = sma + 2 * std
        df['bb_lower'] = sma - 2 * std
        df['bb_pos'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # EMA diff
        df['ema_fast'] = df['Close'].ewm(span=30).mean()
        df['ema_slow'] = df['Close'].ewm(span=50).mean()
        df['ema_diff'] = (df['ema_fast'] - df['ema_slow']) / df['Close']

        # Fill NaN
        df.bfill(inplace=True)
        df.fillna(0, inplace=True)

    def reset(self):
        """Reset simulator to beginning."""
        self.current_idx = 0
        self.active_trade = None
        self.completed_trades = []
        self.entries = []
        self.exits = []
        self.action_markers = []
        self.bars_held = 0
        self.max_favorable = 0.0
        self.max_adverse = 0.0
        self.action_history = [0] * 5
        self.current_sl_price = 0.0
        self.activation_history = []
        # Note: action_neuron_stats NOT reset - accumulates across session
        self.metrics.reset()

    def build_state(self) -> torch.Tensor:
        """Build state tensor for RL model (25 dimensions)."""
        row = self.price_data.iloc[self.current_idx]

        if self.active_trade is None:
            return torch.zeros(1, 25, device=self.device)

        # Compute unrealized PnL
        current_price = row['Close']
        entry_price = self.active_trade.entry_price
        direction = self.active_trade.direction
        pnl = (current_price - entry_price) / entry_price * direction

        # Update MFE/MAE
        self.max_favorable = max(self.max_favorable, pnl)
        self.max_adverse = min(self.max_adverse, pnl)

        # Position features (5)
        bars_held_norm = self.bars_held / 200.0
        sl_distance = abs(entry_price - self.current_sl_price) / entry_price if entry_price > 0 else 0

        # Helper for NaN handling
        def safe(val, default=0.0):
            return default if (np.isnan(val) or np.isinf(val)) else val

        # Market features (10)
        market_features = [
            pnl,
            self.max_favorable,
            self.max_adverse,
            safe(row['atr'] / current_price),
            safe(row['adx'] / 100.0, 0.25),
            safe(row['rsi'] / 100.0, 0.5),
            safe(row['bb_pos'], 0.5),
            safe(row['ema_diff']),
            safe((row['Close'] - row['Open']) / row['Open']),
            0.0  # volume placeholder
        ]

        # Entry context (5)
        entry_context = [
            safe(self.active_trade.entry_atr / entry_price),
            safe(row['adx'] / 100.0, 0.25),
            safe(row['rsi'] / 100.0, 0.5),
            safe((row['bb_upper'] - row['bb_lower']) / current_price),
            safe(row['ema_diff'])
        ]

        # Action history (5)
        action_hist = [a / 4.0 for a in self.action_history]

        # Combine all features
        state = [bars_held_norm, pnl, self.max_favorable, self.max_adverse, sl_distance]
        state.extend(market_features)
        state.extend(entry_context)
        state.extend(action_hist)

        # Clean any remaining NaN/Inf
        state = [0.0 if (np.isnan(x) or np.isinf(x)) else x for x in state]

        return torch.tensor([state], dtype=torch.float32, device=self.device)

    def step_to_next_trade(self) -> Optional[Dict]:
        """Jump to next significant event."""
        result = None
        max_steps = 5000

        if self.active_trade is not None:
            while self.active_trade is not None and max_steps > 0:
                result = self.step()
                if result is None:
                    break
                max_steps -= 1
        else:
            while self.active_trade is None and max_steps > 0:
                result = self.step()
                if result is None:
                    break
                max_steps -= 1
                if self.active_trade is not None:
                    break

        return result

    def step(self) -> Optional[Dict]:
        """Advance one bar and return visualization data."""
        if self.current_idx >= len(self.price_data) - 1:
            return None

        row = self.price_data.iloc[self.current_idx]
        timestamp = str(row.name)

        action = 0
        probs = [1.0, 0, 0, 0, 0]
        model_output = {'value': 0, 'entropy': 0, 'confidence': 1.0}
        activations = None

        # Check for new trade entry
        if self.active_trade is None and self.current_idx in self.trade_entries:
            self._enter_trade(row, timestamp)

        # If in trade, get RL decision
        current_trade_info = None
        if self.active_trade is not None:
            action, probs, model_output, activations, current_trade_info = \
                self._process_active_trade(row, timestamp)

        # Build output
        candles = self._get_candle_window(timestamp)
        trade_zone = self._build_trade_zone(timestamp) if self.active_trade else None
        trade_history = self._build_trade_history()

        self.current_idx += 1

        return {
            'timestamp': timestamp,
            'candles': candles,
            'entries': self._filter_to_window(self.entries),
            'exits': self._filter_to_window(self.exits),
            'action_markers': self._filter_to_window(self.action_markers),
            'action': action,
            'probs': probs,
            'model_output': model_output,
            'activations': activations,
            'trade_zone': trade_zone,
            'current_trade': current_trade_info,
            'stats': self.metrics.get_stats(),
            'metrics': self.metrics.compute_metrics(),
            'trade_history': trade_history,
            # Enhanced visualization data
            'activation_timeline': self.activation_history[-100:],  # Last 100 bars
            'action_correlations': self._compute_action_correlations()
        }

    def _enter_trade(self, row: pd.Series, timestamp: str):
        """Enter a new trade."""
        trade_row = self.trade_entries[self.current_idx]
        entry_atr = row['atr'] if not np.isnan(row['atr']) else 0.001

        self.active_trade = Trade(
            entry_time=timestamp,
            entry_price=trade_row['entry_price'],
            entry_idx=self.current_idx,
            direction=int(trade_row['direction']),
            sl=trade_row['sl'],
            tp=trade_row['tp'],
            entry_atr=entry_atr
        )

        self.entries.append({
            'time': timestamp,
            'price': float(trade_row['entry_price']),
            'direction': int(trade_row['direction'])
        })

        self.bars_held = 0
        self.max_favorable = 0.0
        self.max_adverse = 0.0
        self.current_sl_price = trade_row['sl']

    def _process_active_trade(self, row, timestamp):
        """Process RL decision for active trade."""
        state_tensor = self.build_state()
        result = self.model.get_action_with_details(state_tensor, deterministic=True)

        action = result['action']
        probs = result['probs']
        model_output = {
            'value': result['value'],
            'entropy': result['entropy'],
            'confidence': result['confidence']
        }
        activations = result['activations']
        activations_full = result.get('activations_full')

        # Track activation history for timeline visualization
        if activations_full:
            self.activation_history.append({
                'bar': self.bars_held,
                'activations': activations_full,
                'action': action
            })
            # Track neuron-action correlations (limit to 1000 samples per action)
            if len(self.action_neuron_stats[action]) < 1000:
                self.action_neuron_stats[action].append(activations_full)

        self.action_history = self.action_history[1:] + [action]

        action_names = ['HOLD', 'EXIT', 'TIGHTEN_SL', 'TRAIL_BE', 'PARTIAL']
        self.active_trade.action_counts[action_names[action]] += 1

        current_price = row['Close']
        pnl_pips = (current_price - self.active_trade.entry_price) * 10000 * self.active_trade.direction
        pnl_dollars = pnl_pips * self.config.position_size / 10000

        self.active_trade.max_favorable_pips = max(self.active_trade.max_favorable_pips, pnl_pips)
        self.active_trade.max_adverse_pips = min(self.active_trade.max_adverse_pips, pnl_pips)

        unrealized_pnl_dollars = pnl_dollars * self.active_trade.position_size
        total_pnl_dollars = self.active_trade.realized_pnl_dollars + unrealized_pnl_dollars

        current_trade_info = self._build_current_trade_info(
            current_price, pnl_pips, total_pnl_dollars, unrealized_pnl_dollars
        )

        exit_trade, exit_price, exit_reason = self._process_action(
            action, row, timestamp, current_price, pnl_pips, pnl_dollars,
            unrealized_pnl_dollars, total_pnl_dollars
        )

        if self.bars_held > 0 and not exit_trade:
            sl_hit, sl_price = self._check_sl_hit(row)
            if sl_hit:
                exit_trade = True
                exit_price = sl_price
                exit_reason = self._get_sl_exit_reason()

        if exit_trade:
            self._process_exit(timestamp, exit_price, exit_reason)
            current_trade_info = None
        else:
            self.bars_held += 1

        return action, probs, model_output, activations, current_trade_info

    def _process_action(self, action, row, timestamp, current_price, pnl_pips, pnl_dollars,
                        unrealized_pnl_dollars, total_pnl_dollars):
        """Process RL action and return (exit_trade, exit_price, exit_reason)."""
        Actions = self.Actions
        exit_trade = False
        exit_price = current_price
        exit_reason = ""
        old_sl = self.current_sl_price

        if action == Actions.EXIT:
            exit_trade = True
            exit_reason = "RL_EXIT"
            self._record_action(
                'EXIT', timestamp, current_price, pnl_pips,
                pnl_dollars * self.active_trade.position_size,
                total_pnl_dollars, f"Full exit @ {current_price:.5f}",
                closed_size=self.active_trade.position_size  # Close ALL remaining
            )
        elif action == Actions.PARTIAL_EXIT:
            if self.active_trade.position_size > 0.25:
                self._process_partial_exit(timestamp, current_price, pnl_pips, pnl_dollars)
            else:
                exit_trade = True
                exit_reason = "RL_PARTIAL_FINAL"
                self._record_action(
                    'EXIT', timestamp, current_price, pnl_pips,
                    pnl_dollars * self.active_trade.position_size,
                    total_pnl_dollars, f"Final exit @ {current_price:.5f}",
                    closed_size=self.active_trade.position_size  # Close ALL remaining
                )
        elif action == Actions.TIGHTEN_SL:
            self._process_tighten_sl(row, timestamp, current_price, pnl_pips,
                                     unrealized_pnl_dollars, total_pnl_dollars, old_sl)
        elif action == Actions.TRAIL_BREAKEVEN:
            if pnl_pips > 0:
                self._process_trail_be(timestamp, current_price, pnl_pips,
                                       unrealized_pnl_dollars, total_pnl_dollars, old_sl)
        else:
            if self.bars_held == 0 or self.bars_held % 5 == 0:
                self._record_action(
                    'HOLD', timestamp, current_price, pnl_pips,
                    unrealized_pnl_dollars, total_pnl_dollars, f"Hold @ bar {self.bars_held}"
                )

        return exit_trade, exit_price, exit_reason

    def _process_partial_exit(self, timestamp, current_price, pnl_pips, pnl_dollars):
        """Process a partial exit."""
        # Close 50% of CURRENT position
        partial_size = self.active_trade.position_size * 0.5
        partial_pnl_dollars = pnl_pips * self.config.position_size * partial_size / 10000

        self._record_action(
            'PARTIAL', timestamp, current_price, pnl_pips, partial_pnl_dollars,
            self.active_trade.realized_pnl_dollars + partial_pnl_dollars,
            f"Close {partial_size*100:.0f}% @ {current_price:.5f}",
            closed_size=partial_size,  # Pass the SIZE BEING CLOSED
            remaining_pct=(self.active_trade.position_size - partial_size) * 100
        )

        self.active_trade.partial_exits.append({
            'price': current_price, 'pips': pnl_pips, 'dollars': partial_pnl_dollars,
            'size': partial_size, 'time': timestamp,
            'time_pretty': format_datetime_pretty(timestamp)
        })

        self.active_trade.realized_pnl_pips += pnl_pips * partial_size
        self.active_trade.realized_pnl_dollars += partial_pnl_dollars
        self.active_trade.position_size -= partial_size
        self.metrics.add_partial_pnl(pnl_pips, partial_pnl_dollars, partial_size)

    def _process_tighten_sl(self, row, timestamp, current_price, pnl_pips,
                            unrealized_pnl_dollars, total_pnl_dollars, old_sl):
        """Process tighten SL action."""
        if self.active_trade.direction == 1:
            new_sl = current_price - (current_price - self.current_sl_price) * 0.5
            if new_sl > self.current_sl_price:
                self.current_sl_price = new_sl
                self.active_trade.sl_tightened = True
        else:
            new_sl = current_price + (self.current_sl_price - current_price) * 0.5
            if new_sl < self.current_sl_price:
                self.current_sl_price = new_sl
                self.active_trade.sl_tightened = True

        self._record_action(
            'TIGHTEN_SL', timestamp, current_price, pnl_pips,
            unrealized_pnl_dollars, total_pnl_dollars,
            f"SL {old_sl:.5f} → {self.current_sl_price:.5f}",
            old_sl=old_sl, new_sl=self.current_sl_price
        )

    def _process_trail_be(self, timestamp, current_price, pnl_pips,
                          unrealized_pnl_dollars, total_pnl_dollars, old_sl):
        """Process trail to breakeven action."""
        self.current_sl_price = self.active_trade.entry_price
        self.active_trade.trailed_to_be = True
        self._record_action(
            'TRAIL_BE', timestamp, current_price, pnl_pips,
            unrealized_pnl_dollars, total_pnl_dollars,
            f"Trail to BE @ {self.active_trade.entry_price:.5f}",
            old_sl=old_sl, new_sl=self.active_trade.entry_price
        )

    def _record_action(self, action, timestamp, price, pnl_pips, pnl_dollars,
                       cumulative_pnl, note, closed_size=None, **extra_fields):
        """
        Record an action to the trade's action history.

        Args:
            closed_size: For PARTIAL/EXIT, the fraction being closed (e.g., 0.5 for 50%)
                        If None, defaults to current position_size (for non-exit actions)
        """
        # For PARTIAL/EXIT actions, closed_size should be passed explicitly
        # For other actions (HOLD, TIGHTEN, TRAIL), show remaining position
        if closed_size is not None:
            display_size = closed_size
        else:
            display_size = self.active_trade.position_size

        self.active_trade.add_action(
            action=action, bar=self.bars_held, time=timestamp,
            time_pretty=format_datetime_pretty(timestamp),
            price=price, pnl_pips=pnl_pips, pnl_dollars=pnl_dollars,
            position_size_pct=display_size * 100,
            position_m=self.config.position_size * display_size / 1_000_000,
            cumulative_pnl=cumulative_pnl, sl=self.current_sl_price, note=note,
            **extra_fields
        )

        # Add to action_markers for chart visualization (skip HOLD actions)
        if action != 'HOLD':
            self.action_markers.append({
                'time': str(timestamp),
                'price': float(price),
                'action': action,
                'pnl_pips': float(pnl_pips),
                'pnl_dollars': float(pnl_dollars),
                'direction': int(self.active_trade.direction),
                'note': note
            })

    def _check_sl_hit(self, row) -> Tuple[bool, float]:
        """Check if stop loss was hit."""
        if self.active_trade.direction == 1:
            if row['Low'] <= self.current_sl_price:
                return True, self.current_sl_price
        else:
            if row['High'] >= self.current_sl_price:
                return True, self.current_sl_price
        return False, 0.0

    def _get_sl_exit_reason(self) -> str:
        """Get exit reason based on SL type."""
        if self.active_trade.trailed_to_be:
            return "TRAIL_BE_HIT"
        elif self.active_trade.sl_tightened:
            return "TIGHTEN_SL_HIT"
        return "SL_HIT"

    def _process_exit(self, timestamp, exit_price, exit_reason):
        """Process trade exit."""
        exit_pnl_pips = (exit_price - self.active_trade.entry_price) * 10000 * self.active_trade.direction
        remaining_size = self.active_trade.position_size
        exit_pnl_dollars = exit_pnl_pips * self.config.position_size * remaining_size / 10000

        total_pnl_pips = self.active_trade.realized_pnl_pips + (exit_pnl_pips * remaining_size)
        total_pnl_dollars = self.active_trade.realized_pnl_dollars + exit_pnl_dollars

        self.active_trade.exit_time = timestamp
        self.active_trade.exit_price = exit_price
        self.active_trade.exit_idx = self.current_idx
        self.active_trade.pnl_pips = total_pnl_pips
        self.active_trade.pnl_dollars = total_pnl_dollars
        self.active_trade.exit_reason = exit_reason

        self.exits.append({
            'time': str(timestamp), 'price': float(exit_price),
            'pnl': float(total_pnl_pips), 'pnl_dollars': float(total_pnl_dollars),
            'exit_reason': str(exit_reason), 'direction': int(self.active_trade.direction),
            'had_partials': bool(len(self.active_trade.partial_exits) > 0)
        })

        self.metrics.update_from_trade(
            pnl_dollars=total_pnl_dollars, pnl_pips=exit_pnl_pips,
            direction=self.active_trade.direction, bars_held=self.bars_held,
            exit_reason=exit_reason, remaining_position_size=remaining_size,
            partial_count=len(self.active_trade.partial_exits)
        )

        self.completed_trades.append(self.active_trade)
        self.active_trade = None

    def _build_current_trade_info(self, current_price, pnl_pips, total_pnl_dollars, unrealized_pnl_dollars):
        """Build current trade info dictionary."""
        t = self.active_trade
        return {
            'direction': int(t.direction), 'entry_price': float(t.entry_price),
            'current_price': float(current_price), 'pnl_pips': float(pnl_pips),
            'pnl_dollars': float(total_pnl_dollars), 'unrealized_pnl': float(unrealized_pnl_dollars),
            'realized_pnl': float(t.realized_pnl_dollars), 'bars_held': int(self.bars_held),
            'mfe': float(t.max_favorable_pips), 'mae': float(t.max_adverse_pips),
            'action_counts': {k: int(v) for k, v in t.action_counts.items()},
            'sl_tightened': bool(t.sl_tightened), 'trailed_to_be': bool(t.trailed_to_be),
            'current_sl': float(self.current_sl_price), 'original_sl': float(t.sl),
            'position_size': float(t.position_size),
            'position_value_m': float(self.config.position_size * t.position_size / 1_000_000),
            'partial_exits': int(len(t.partial_exits))
        }

    def _get_candle_window(self, timestamp):
        """Get candle data for the visible window."""
        window_bars = self.config.window_bars
        start_idx = max(0, self.current_idx - window_bars)
        window = self.price_data.iloc[start_idx:self.current_idx + 1]
        return {
            'timestamps': [str(t) for t in window.index],
            'open': window['Open'].tolist(), 'high': window['High'].tolist(),
            'low': window['Low'].tolist(), 'close': window['Close'].tolist()
        }

    def _filter_to_window(self, items):
        """Filter entries/exits to visible window."""
        window_bars = self.config.window_bars
        start_idx = max(0, self.current_idx - window_bars)
        window_start = str(self.price_data.index[start_idx])
        return [e for e in items if e['time'] >= window_start]

    def _build_trade_zone(self, timestamp):
        """Build trade zone data for chart."""
        t = self.active_trade
        return {
            'x0': str(t.entry_time), 'x1': str(timestamp),
            'entry_price': float(t.entry_price), 'sl': float(self.current_sl_price),
            'original_sl': float(t.sl), 'tp': float(t.tp),
            'sl_changed': bool(abs(self.current_sl_price - t.sl) > 0.00001),
            'direction': int(t.direction)
        }

    def _build_trade_history(self):
        """Build trade history for display."""
        trade_history = []

        if self.active_trade is not None:
            current_price = self.price_data.iloc[self.current_idx]['Close']
            current_pnl_pips = (current_price - self.active_trade.entry_price) * 10000 * self.active_trade.direction

            trade_dict = self.active_trade.to_dict(current_price, current_pnl_pips)
            trade_dict['entry_time_pretty'] = format_datetime_pretty(self.active_trade.entry_time)
            trade_dict['bars_held'] = self.bars_held
            trade_history.append(trade_dict)

        max_completed = 14 if self.active_trade else 15
        for t in reversed(self.completed_trades[-max_completed:]):
            trade_dict = t.to_dict()
            trade_dict['entry_time_pretty'] = format_datetime_pretty(t.entry_time)
            trade_dict['exit_time_pretty'] = format_datetime_pretty(t.exit_time) if t.exit_time else ''
            trade_history.append(trade_dict)

        return trade_history

    def _compute_action_correlations(self) -> Dict:
        """
        Compute neuron-action correlations for visualization.

        Returns which neurons are most active when each action is chosen.
        """
        correlations = {}
        action_names = ['HOLD', 'EXIT', 'TIGHTEN_SL', 'TRAIL_BE', 'PARTIAL']

        for action_id, name in enumerate(action_names):
            samples = self.action_neuron_stats[action_id]
            if len(samples) < 5:
                # Not enough samples yet
                correlations[name] = {'top_neurons': [], 'mean_activation': 0.0, 'sample_count': len(samples)}
                continue

            # Compute mean activation per neuron for this action
            samples_array = np.array(samples)  # shape: (n_samples, n_neurons)
            mean_activations = np.mean(samples_array, axis=0)

            # Find top 10 most active neurons for this action
            top_indices = np.argsort(mean_activations)[-10:][::-1]
            top_neurons = [
                {'neuron': int(idx), 'activation': float(mean_activations[idx])}
                for idx in top_indices
            ]

            correlations[name] = {
                'top_neurons': top_neurons,
                'mean_activation': float(np.mean(mean_activations)),
                'sample_count': len(samples)
            }

        return correlations
