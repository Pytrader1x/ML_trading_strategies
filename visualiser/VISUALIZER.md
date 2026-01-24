# RL Trade Visualizer

A modular, real-time visualization system for RL-based trading strategies. Watch your trained RL agent make exit decisions on out-of-sample (OOS) data with full transparency into model internals.

## Features

- **Real-time Candlestick Charts** - Interactive Plotly charts with entry/exit markers, SL/TP zones
- **RL Agent Decisions** - See the agent's action (HOLD, EXIT, TIGHTEN_SL, TRAIL_BE, PARTIAL) in real-time
- **Probability Distribution** - View the probability distribution over all actions
- **Neural Activations** - Heatmap of encoder layer activations
- **Model Outputs** - Value estimate, entropy, and confidence metrics
- **Performance Metrics** - 20+ metrics including Sharpe ratio, win rate, profit factor
- **Detailed Trade Log** - Card-based log with full action timeline per trade

## Quick Start

```bash
# From the repo root
python -m visualiser

# With custom options
python -m visualiser --strategy ema_bb_v2_rl --port 8080
```

Then open http://localhost:8765 in your browser.

## Architecture

```
visualiser/
├── __init__.py          # Package exports and run_visualizer()
├── __main__.py          # CLI entry point
├── VISUALIZER.md        # This documentation
├── templates/
│   └── index.html       # HTML template
├── static/
│   ├── css/styles.css   # Styling (dark theme)
│   └── js/app.js        # Frontend JavaScript
└── src/
    ├── config.py        # Configuration classes
    ├── utils.py         # Utility functions
    ├── trade.py         # Trade dataclass
    ├── metrics.py       # Performance calculations
    ├── model_wrapper.py # Model with activation hooks
    ├── simulator.py     # Trade simulation engine
    └── server.py        # WebSocket server
```

## Components

### Configuration (`src/config.py`)

```python
from visualiser import VisualizerConfig

# Default config for ema_bb_v2_rl
config = VisualizerConfig()

# Custom strategy
config = VisualizerConfig.from_strategy("my_strategy")

# Full customization
config = VisualizerConfig(
    strategy_name="my_strategy",
    model_path=Path("models/my_model.pt"),
    price_data_path=Path("data/AUDUSD_15M.csv"),
    trades_data_path=Path("data/trades.csv"),
    position_size=2_000_000,
    port=8765,
    oos_start="2022-01-01",
    oos_end="2025-12-31",
)
```

### Trade Simulator (`src/simulator.py`)

The `TradeSimulator` class handles:

1. **State Building** - Constructs the 25-dimensional state vector for the RL model
2. **Action Processing** - Executes HOLD, EXIT, TIGHTEN_SL, TRAIL_BE, PARTIAL actions
3. **SL Management** - Tracks original and modified stop-loss levels
4. **Partial Exit Handling** - Manages position sizing after partial exits
5. **Statistics Tracking** - Maintains comprehensive session statistics

### Model Wrapper (`src/model_wrapper.py`)

The `ActorCriticWithActivations` class wraps your trained model with:

- **Activation Hooks** - Captures intermediate activations for visualization
- **Detailed Output** - Returns action, probabilities, value, entropy, confidence

### Metrics Calculator (`src/metrics.py`)

Computes 20+ trading metrics:

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Annualized risk-adjusted return |
| Win Rate | Percentage of winning trades |
| Profit Factor | Gross profit / Gross loss |
| Max Drawdown | Largest peak-to-trough decline |
| Avg Win/Loss | Average winning/losing trade |
| Long/Short WR | Win rate by direction |
| RL Exit % | Percentage of RL-initiated exits |
| SL Hit % | Percentage of stop-loss exits |

### Server (`src/server.py`)

WebSocket server with:

- **Real-time Streaming** - Push updates to connected clients
- **Playback Control** - Play, pause, step, reset, speed adjustment
- **Static File Serving** - CSS and JavaScript assets
- **Next Trade Jump** - Skip to next significant event

## Action Space

The RL agent has 5 discrete actions:

| Action | Description |
|--------|-------------|
| **HOLD** | Maintain current position |
| **EXIT** | Close entire position immediately |
| **TIGHTEN_SL** | Move stop-loss 50% closer to current price |
| **TRAIL_BE** | Move stop-loss to break-even (entry price) |
| **PARTIAL** | Close 50% of remaining position |

## State Space (25 dimensions)

| Category | Features |
|----------|----------|
| Position | bars_held_norm, pnl, max_favorable, max_adverse, sl_distance |
| Market | pnl, mfe, mae, atr, adx, rsi, bb_pos, ema_diff, return, vol |
| Entry Context | entry_atr, entry_adx, entry_rsi, entry_bb_width, entry_ema_diff |
| Action History | last 5 actions (normalized) |

## Reward Function

The model was trained with a reward function that considers:

```
R = R_pnl + R_sharpe + R_regret + R_time

R_pnl    = Realized PnL on exit
R_sharpe = Differential Sharpe ratio contribution
R_regret = Penalty for suboptimal exits (vs MFE)
R_time   = Small time decay per bar held
```

## Customization

### Using with a New Strategy

1. Create your strategy in `strategies/my_strategy/`
2. Ensure it has `model.py` with `ActorCritic` class
3. Ensure it has `config.py` with `PPOConfig` class
4. Train and save your model to `models/exit_policy.pt`
5. Generate trades data to `data/trades.csv`

```python
from visualiser import VisualizerConfig, run_visualizer

config = VisualizerConfig(
    strategy_name="my_strategy",
    model_path=Path("strategies/my_strategy/models/my_model.pt"),
    trades_data_path=Path("strategies/my_strategy/data/trades.csv"),
)
run_visualizer(config)
```

### Modifying the Frontend

- **Styling**: Edit `static/css/styles.css`
- **JavaScript**: Edit `static/js/app.js`
- **HTML Layout**: Edit `templates/index.html`

### Adding New Metrics

Edit `src/metrics.py`:

```python
class MetricsCalculator:
    @staticmethod
    def compute(stats: Dict) -> Dict:
        # Add your metric here
        my_metric = compute_my_metric(stats)

        return {
            ...
            'my_metric': my_metric,
        }
```

Then add display in `static/js/app.js` and `templates/index.html`.

## Keyboard Shortcuts (Future)

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| → | Step forward |
| N | Next trade |
| R | Reset |

## Troubleshooting

### "Connecting..." but no data

1. Ensure the model file exists at the expected path
2. Check that trades data has entries within the OOS date range
3. Verify the WebSocket connection in browser DevTools

### Model not loading

1. Check that `model.py` and `config.py` exist in the strategy directory
2. Ensure `PPOConfig` and `ActorCritic` classes are importable
3. Verify model checkpoint format matches what `load_model` expects

### Static files not loading

1. Ensure CSS/JS files exist in `visualiser/static/`
2. Check browser DevTools Network tab for 404 errors

## Dependencies

- `torch` - Model inference
- `pandas` - Data handling
- `numpy` - Numerical operations
- `aiohttp` - WebSocket server

## License

Part of the ML Trading Strategies framework.
