"""
RL Trade Visualizer

A modular, reusable visualization system for RL-based trading strategies.

Usage:
    # From command line (from repo root):
    python -m visualiser --strategy ema_bb_v2_rl

    # Programmatically:
    from visualiser import VisualizerConfig, run_visualizer

    config = VisualizerConfig.from_strategy("ema_bb_v2_rl")
    run_visualizer(config)

Structure:
    visualiser/
    ├── __init__.py          # This file - package exports
    ├── __main__.py          # Entry point for `python -m visualiser`
    ├── VISUALIZER.md        # Documentation
    ├── templates/
    │   └── index.html       # HTML template
    ├── static/
    │   ├── css/styles.css   # CSS styles
    │   └── js/app.js        # JavaScript frontend
    └── src/
        ├── config.py        # Configuration classes
        ├── utils.py         # Utility functions
        ├── trade.py         # Trade dataclass
        ├── metrics.py       # Performance metrics
        ├── model_wrapper.py # Model wrapper with activation hooks
        ├── simulator.py     # Trade simulation engine
        └── server.py        # WebSocket server
"""

from .src.config import (
    VisualizerConfig,
    POSITION_SIZE,
    PIP_VALUE,
    WINDOW_BARS,
    ACTION_NAMES,
    ACTION_COLORS,
)
from .src.trade import Trade
from .src.metrics import MetricsCalculator
from .src.model_wrapper import ActorCriticWithActivations, load_model, get_device
from .src.simulator import TradeSimulator
from .src.server import VisualizationServer, run_server
from .src.utils import format_datetime_pretty, format_pnl, format_pips, kill_existing_server

__version__ = "2.0.0"
__all__ = [
    # Config
    'VisualizerConfig',
    'POSITION_SIZE',
    'PIP_VALUE',
    'WINDOW_BARS',
    'ACTION_NAMES',
    'ACTION_COLORS',
    # Data structures
    'Trade',
    # Metrics
    'MetricsCalculator',
    # Model
    'ActorCriticWithActivations',
    'load_model',
    'get_device',
    # Simulation
    'TradeSimulator',
    # Server
    'VisualizationServer',
    'run_server',
    # Utilities
    'format_datetime_pretty',
    'format_pnl',
    'format_pips',
    'kill_existing_server',
]


def run_visualizer(config: VisualizerConfig = None, **kwargs):
    """
    Run the visualizer with the given configuration.

    Args:
        config: VisualizerConfig instance, or None to use defaults
        **kwargs: Additional arguments passed to VisualizerConfig if config is None

    Example:
        run_visualizer()  # Use default config
        run_visualizer(config=my_config)
        run_visualizer(strategy_name="ema_bb_v2_rl", port=8080)
    """
    import asyncio
    from pathlib import Path

    if config is None:
        config = VisualizerConfig(**kwargs)

    # Kill any existing server on the port
    kill_existing_server(config.port)

    # Load HTML template
    template_path = Path(__file__).parent / 'templates' / 'index.html'
    html_template = template_path.read_text()

    # Run the server
    try:
        asyncio.run(run_server(config, html_template))
    except KeyboardInterrupt:
        print("\nShutting down...")
