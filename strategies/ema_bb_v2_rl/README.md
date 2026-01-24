# EMA BB V2 + Reinforcement Learning Strategy

RL-enhanced version of EMA BB Scalp V2 strategy. Uses classical strategy signals as base, with RL for trade filtering (meta-labeling), position sizing, and entry/exit timing optimization.

## Directory Structure

```
ema_bb_v2_rl/
├── __init__.py     # Package exports
├── config.py       # All hyperparameters
├── env.py          # Gymnasium trading environment
├── model.py        # Neural network architectures
├── strategy.py     # Backtest integration
├── train.py        # Training entry point
├── run.py          # CLI runner
├── README.md       # This file
└── model/          # Trained model checkpoints (gitignored)
```

## Organization Guidelines

### File Purposes

| File | Purpose |
|------|---------|
| `config.py` | Centralized configuration - all hyperparameters live here |
| `env.py` | Gymnasium environment definition |
| `model.py` | Network architectures (MLP, LSTM, etc.) |
| `strategy.py` | Wrapper for backtest engine integration |
| `train.py` | Training loop and logic |
| `run.py` | CLI interface for training/backtesting |

### Adding New Code

- **New indicators/features**: Add to `env.py` in `_precompute_features()`
- **New network architectures**: Add to `model.py`
- **New hyperparameters**: Add to appropriate dataclass in `config.py`
- **New CLI commands**: Add to `run.py`

### Model Artifacts

- Store trained models in `model/` subdirectory
- Use descriptive names: `{algorithm}_{pair}_{date}.pt`
- Keep only production-ready checkpoints in version control

### Data Files

- Do not store data files in this directory
- Reference data from the shared project data directory

## Usage

```bash
# Train
python run.py train -i EURUSD -t 1H --epochs 100

# Backtest
python run.py backtest -i EURUSD -t 1H --model model/best_model.pt

# Evaluate
python run.py eval --model model/best_model.pt
```

## Status

Work in progress. Core structure is in place; training implementation pending.
