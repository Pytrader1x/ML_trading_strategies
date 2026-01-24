"""
Configuration constants for the RL Trade Visualizer.

This module can be customized for different strategies.
"""

from pathlib import Path

# ============================================================================
# Path Configuration
# ============================================================================

# Base paths
REPO_ROOT = Path(__file__).parent.parent.parent  # ML_trading_strategies/
VISUALISER_DIR = Path(__file__).parent.parent  # visualiser/
DATA_DIR = REPO_ROOT / "data"

# ============================================================================
# Visualization Settings
# ============================================================================

WINDOW_BARS = 96  # 96 bars = 24 hours of 15M data
DEFAULT_PORT = 8765
DEFAULT_SPEED = 10

# ============================================================================
# FX Trading Constants (can be overridden per strategy)
# ============================================================================

POSITION_SIZE = 2_000_000  # $2M notional position
PIP_SIZE = 0.0001  # Standard pip for AUDUSD
PIP_VALUE = POSITION_SIZE * PIP_SIZE  # $200 per pip for 2M position

# ============================================================================
# Out-of-Sample Period (default)
# ============================================================================

OOS_START = '2022-01-01'
OOS_END = '2025-12-31'

# ============================================================================
# Action Names and Colors (PPO-style discrete actions)
# ============================================================================

ACTION_NAMES = ['HOLD', 'EXIT', 'TIGHTEN_SL', 'TRAIL_BREAKEVEN', 'PARTIAL_EXIT']
ACTION_COLORS = ['#8b949e', '#f85149', '#d29922', '#58a6ff', '#a371f7']

# Mapping for trade action display
ACTION_DISPLAY = {
    0: {'name': 'HOLD', 'color': '#8b949e'},
    1: {'name': 'EXIT', 'color': '#f85149'},
    2: {'name': 'TIGHTEN_SL', 'color': '#d29922'},
    3: {'name': 'TRAIL_BE', 'color': '#58a6ff'},
    4: {'name': 'PARTIAL', 'color': '#a371f7'},
}


class VisualizerConfig:
    """
    Configuration for a specific visualization session.

    Allows customization of paths, trading parameters, and display settings.
    """

    def __init__(
        self,
        strategy_name: str = "ema_bb_v2_rl",
        model_path: Path = None,
        price_data_path: Path = None,
        trades_data_path: Path = None,
        position_size: float = 2_000_000,
        pip_size: float = 0.0001,
        oos_start: str = '2022-01-01',
        oos_end: str = '2025-12-31',
        port: int = 8765,
        window_bars: int = 96,
        initial_speed: int = 10,
        auto_open_browser: bool = True
    ):
        """
        Initialize visualizer configuration.

        Args:
            strategy_name: Name of the strategy being visualized
            model_path: Path to trained model checkpoint
            price_data_path: Path to OHLC price data CSV
            trades_data_path: Path to trade signals CSV
            position_size: Notional position size in base currency
            pip_size: Pip size for the instrument
            oos_start: Start date for out-of-sample period
            oos_end: End date for out-of-sample period
            port: WebSocket server port
            window_bars: Number of bars to display in chart
        """
        self.strategy_name = strategy_name

        # Default paths based on strategy
        strategy_dir = REPO_ROOT / "strategies" / strategy_name

        self.model_path = model_path or strategy_dir / "models" / "exit_policy_insample_2005_2021.pt"
        self.price_data_path = price_data_path or DATA_DIR / "AUDUSD_15M.csv"
        self.trades_data_path = trades_data_path or strategy_dir / "data" / "trades_test_2022_2025.csv"

        self.position_size = position_size
        self.pip_size = pip_size
        self.pip_value = position_size * pip_size

        self.oos_start = oos_start
        self.oos_end = oos_end

        self.port = port
        self.window_bars = window_bars
        self.initial_speed = initial_speed
        self.auto_open_browser = auto_open_browser

    def __repr__(self):
        return f"VisualizerConfig(strategy='{self.strategy_name}', port={self.port})"
