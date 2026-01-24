# EMA BB V2 RL Strategy

PPO-based reinforcement learning model for optimizing trade exits. Combines classical EMA/Bollinger Band entry signals with learned exit timing.

---

## CRITICAL: Action Guards for Training

### The Problem: Immediate 0-Profit Exits

The model learns to **immediately PARTIAL/EXIT at Bar 0 with 0 pips** because:

1. **Time penalty avoidance**: `time_coef` penalty grows over time. Exiting at Bar 0 avoids all future time penalties.
2. **Zero-cost exits**: With 0 pips, reward ≈ 0 (not negative), so it's "safe"
3. **No action cost**: PARTIAL had no cost, unlike TIGHTEN_SL or TRAIL_BE

This is a **local optimum** the model finds, but it's terrible for real trading:
- Closes 50% position at 0.0 pips (no benefit)
- Pays spread/commission for nothing
- Reduces position before trend develops

### The Solution: Action Guards

Guards are now implemented in **both** training (`env.py`) and visualization (`simulator.py`).

#### Training Environment Guards (`config.py` → `RewardConfig`)

```python
# Minimum normalized PnL required (0.001 ≈ 10 pips for typical FX)
min_profit_for_partial: float = 0.001
min_profit_for_exit: float = 0.0       # EXIT always allowed (can set higher)

# Minimum bars held before action allowed
min_bars_for_partial: int = 1          # Must hold at least 1 bar
min_bars_for_exit: int = 0             # EXIT allowed anytime

# Penalty for attempting blocked actions
invalid_action_penalty: float = 0.01   # Negative reward for invalid attempts
```

#### How It Works in Training

```python
# In VectorizedExitEnv.step():

# PARTIAL only allowed if: profit >= threshold AND bars >= minimum
partial_allowed = (unrealized_pnl >= min_profit_for_partial) & \
                  (current_bar >= min_bars_for_partial)

if partial_allowed:
    # Execute the partial exit
    rewards = unrealized_pnl * reward_scale
    dones = True
else:
    # Block! Convert to HOLD + apply penalty
    rewards -= invalid_action_penalty
    # Position stays open, model learns this action was bad
```

#### Visualizer Guards (`visualiser/src/config.py`)

```python
# VisualizerConfig
min_profit_for_partial = 1.0   # Minimum pips (not normalized)
min_bars_before_partial = 1    # Minimum bars
```

### Retraining Required

After changing these parameters, you **must retrain** for the model to learn:
1. Don't attempt PARTIAL at Bar 0 (will be blocked + penalized)
2. Wait for profit before taking partial profits
3. HOLD is the correct action when not in sufficient profit

```bash
# Retrain with guards enabled
python train.py --version v2_with_guards --episodes data/episodes_train_2005_2021.pkl
```

### Tuning the Guards

| Parameter | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| `min_profit_for_partial` | 0.002 (20 pips) | 0.001 (10 pips) | 0.0005 (5 pips) |
| `min_bars_for_partial` | 3 | 1 | 0 |
| `invalid_action_penalty` | 0.02 | 0.01 | 0.005 |

---

## Quick Start

```bash
# Run backtest with latest baseline
python run.py -i AUDUSD -t 15M --version v1_baseline

# Compare RL vs classical exits
python run.py -i AUDUSD -t 15M --compare --version v1_baseline

# Train a new model
python train.py --version v2_experiment --episodes data/episodes_train_2005_2021.pkl
```

---

## Experiment Versioning

All experiments are versioned for reproducibility and comparison. Each version contains its own models, results, and configuration snapshot.

### Directory Structure

```
ema_bb_v2_rl/
├── config.py, model.py, env.py      # Shared code
├── train.py, run.py, strategy.py    # Shared code (version-aware)
├── experiment_manager.py            # Version management CLI
│
├── data/                            # Shared training data
│   ├── episodes_train_2005_2021.pkl
│   └── episodes_test_2022_2025.pkl
│
└── experiments/                     # Versioned experiments
    ├── registry.json               # Central experiment registry
    ├── COMPARISON.md               # Auto-generated comparison
    │
    ├── v1_baseline/                # Version 1
    │   ├── experiment.yaml         # Config snapshot
    │   ├── models/                 # Trained weights
    │   │   └── exit_policy_final.pt
    │   ├── results/                # Backtest outputs
    │   │   └── AUDUSD/15M/
    │   └── README.md               # Version notes
    │
    └── v2_*/                       # Future versions...
```

### Creating a New Version

```bash
# Create from parent (recommended)
python experiment_manager.py create v2_entropy --parent v1_baseline -d "Lower entropy decay"

# Create from scratch
python experiment_manager.py create v2_new -d "Fresh experiment"
```

This creates:
- Directory structure with `models/`, `results/`
- `experiment.yaml` config snapshot (copied from parent if specified)
- `README.md` template

### Training a Version

```bash
# Train with version-specific output paths
python train.py --version v2_entropy --episodes data/episodes_train_2005_2021.pkl

# The model saves to: experiments/v2_entropy/models/exit_policy_final.pt
```

### Running Backtests

```bash
# Backtest specific version
python run.py -i AUDUSD -t 15M --version v2_entropy

# Results save to: experiments/v2_entropy/results/AUDUSD/15M/
```

### Comparing Versions

```bash
# List all versions
python experiment_manager.py list

# Compare two versions
python experiment_manager.py compare v1_baseline v2_entropy

# Generate comparison dashboard
python experiment_manager.py dashboard
```

### Updating Results

After training and backtesting, update the registry:

```bash
python experiment_manager.py update-results v2_entropy
```

---

## Version Naming Convention

Format: `v{N}_{short_description}`

Examples:
- `v1_baseline` - Initial baseline
- `v2_entropy` - Modified entropy schedule
- `v3_larger_net` - Larger network architecture
- `v4_reward_shape` - Different reward shaping

---

## What to Track Per Version

### Required
- `experiment.yaml` - Full config snapshot (PPO params, reward params)
- `models/exit_policy_final.pt` - Trained policy weights
- `results/*/backtest_results.json` - Backtest metrics

### Recommended
- `README.md` - Hypothesis, findings, observations
- `results/training/training_metrics.json` - Training curves

### Optional
- W&B run link for live monitoring
- Intermediate checkpoints

---

## Configuration

Edit `experiments/{version}/experiment.yaml` before training:

```yaml
training:
  ppo:
    n_envs: 64
    learning_rate: 0.0003
    entropy_coef_start: 0.05    # Modify for exploration
    entropy_coef_end: 0.001
    hidden_dims: [256, 256]     # Network size

  reward:
    w_realized: 1.0
    w_mtm: 0.1
    risk_coef: 0.3              # Drawdown penalty
    regret_coef: 0.5            # Missing optimal exit penalty
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `config.py` | PPOConfig, RewardConfig, Actions |
| `model.py` | ActorCritic network |
| `env.py` | VectorizedExitEnv |
| `train.py` | PPO training loop |
| `strategy.py` | Backtest interface |
| `run.py` | Backtest runner |
| `extract_episodes.py` | Generate training episodes |
| `experiment_manager.py` | Version management CLI |

---

## Experiment Manager Commands

```bash
# Create new version
python experiment_manager.py create NAME [--parent PARENT] [-d "description"]

# List all versions
python experiment_manager.py list

# Compare two versions
python experiment_manager.py compare V1 V2

# Set active version
python experiment_manager.py set-active NAME

# Update results after backtest
python experiment_manager.py update-results NAME

# Generate comparison dashboard
python experiment_manager.py dashboard
```
