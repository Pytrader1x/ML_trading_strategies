# RL Exit Optimizer - Development Progress

## 1. Session - 2026-01-24
Completed visualizer enhancements (trade cards, TOTAL row, metrics panel).

## 2. Session - 2026-01-25: v2 Experiment Structure

### 2.1 Restructured Experiment Directory
Created isolated experiment structure where each version has its own:
- `config.py` - FROZEN configuration snapshot
- `env.py` - Environment with version-specific guards
- `train.py` - Training script using local config
- `experiment.yaml` - Metadata and parameters
- `models/` - Trained models
- `results/` - Backtest results

### 2.2 Created v2_exit_guards Experiment

**Problem Identified**: Model exits immediately at Bar 0 with ~0 pips profit.

**Root Cause Analysis**:
1. EXIT action had NO guards (min_profit=0, min_bars=0)
2. Time penalty (time_coef=0.005) grows with bars held
3. Model learned: immediate EXIT = 0 reward but avoids all penalties
4. This is a degenerate local optimum

**Solution - v2 Changes**:
| Parameter | v1 | v2 | Rationale |
|-----------|-----|-----|-----------|
| min_profit_for_exit | 0.0 | 0.001 | Block exits unless ~10 pips profit |
| min_bars_for_exit | 0 | 2 | Must hold at least 2 bars |
| min_profit_for_partial | 0.001 | 0.002 | ~20 pips for partial |
| time_coef | 0.005 | 0.002 | Less incentive for early exits |
| regret_coef | 0.5 | 0.8 | Penalize missing optimal more |
| entropy_coef_end | 0.001 | 0.005 | More exploration at end |
| entropy_anneal_steps | 5M | 7M | Slower decay |

### 2.3 Files Created

```
experiments/v2_exit_guards/
├── config.py          # Frozen config with EXIT guards enabled
├── env.py             # Environment with guard statistics tracking
├── train.py           # Training script (uses local config)
├── experiment.yaml    # Experiment metadata
├── README.md          # Documentation
├── models/            # (empty, populated after training)
└── results/           # (empty, populated after training)
```

### 2.4 GPU Training Strategy Confirmed

**Is it PPO?** YES - Confirmed in train.py:
- Clipped surrogate objective (PPO-Clip)
- GAE for advantage estimation
- Entropy annealing (0.05 → 0.005)
- KL divergence early stopping
- Value function clipping

**GPU Saturation**:
- 64 parallel environments (vectorized)
- 2048 steps × 64 envs = 131,072 samples per update
- All tensors on GPU memory
- Pre-computed episode tensors eliminate Python bottlenecks

### 2.5 Registry Updated
Added v2_exit_guards to experiments/registry.json with status "pending".

## Next Steps

1. Deploy v2_exit_guards to Vast.ai RTX 4090
2. Train for 10M timesteps with W&B monitoring
3. Sync results and update registry
4. Compare v1 vs v2 performance

## Training Command
```bash
# From v2_exit_guards directory:
python train.py --wandb --device cuda

# Or via Vast.ai MCP:
# Ask Claude to deploy training
```
