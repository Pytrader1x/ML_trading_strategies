# v2_exit_guards Experiment

**Status**: Pending
**Parent**: v1_baseline
**Created**: 2026-01-25

---

## Problem Statement

The v1_baseline model learned to **EXIT immediately at Bar 0 with ~0 pips profit**. This is a degenerate local optimum where the model avoids all time penalties by exiting before they accumulate.

**Root Cause Analysis:**
1. EXIT action had NO guards (min_profit=0, min_bars=0)
2. Time penalty (`time_coef=0.005`) grows with bars held
3. At Bar 0 with ~0 pips: reward ≈ 0, no penalties
4. Model learned: immediate EXIT = predictable 0 reward, avoids penalties

---

## Hypothesis

By enabling EXIT guards and reducing time penalty, the model will:
1. Be **forced to HOLD** when attempting early exits (guards block them)
2. Learn that HOLD is correct when not yet profitable
3. Eventually discover profitable exit patterns
4. Hold longer and capture more of the optimal PnL

---

## Changes from v1_baseline

| Parameter | v1_baseline | v2_exit_guards | Rationale |
|-----------|-------------|----------------|-----------|
| `min_profit_for_exit` | 0.0 | **0.001** | Block exits unless ~10 pips profit |
| `min_bars_for_exit` | 0 | **2** | Must hold at least 2 bars |
| `min_profit_for_partial` | 0.001 | **0.002** | ~20 pips for partial |
| `min_bars_for_partial` | 1 | **2** | 2 bars minimum |
| `invalid_action_penalty` | 0.01 | **0.02** | Stronger penalty for invalid attempts |
| `time_coef` | 0.005 | **0.002** | Less incentive for early exits |
| `regret_coef` | 0.5 | **0.8** | Penalize missing optimal more |
| `entropy_coef_end` | 0.001 | **0.005** | More exploration at end |
| `entropy_anneal_steps` | 5M | **7M** | Slower entropy decay |

---

## Expected Behavior Changes

1. **Average episode length should INCREASE** (model holds longer)
2. **EXIT action % should DECREASE initially** (blocked by guards)
3. **HOLD action % should INCREASE**
4. **Training curves may look worse initially** (exploring more)
5. **Final performance should be better** (actual profitable exits)

---

## Training

### Local (for testing)
```bash
cd experiments/v2_exit_guards
python train.py --timesteps 1000000 --device cpu
```

### GPU (Vast.ai)
```bash
# From strategy root
python deploy_vastai.py --experiment v2_exit_guards
```

### Full Training
```bash
cd experiments/v2_exit_guards
python train.py --wandb --device cuda
```

---

## Directory Structure

```
v2_exit_guards/
├── config.py          # FROZEN config snapshot
├── env.py             # Environment with EXIT guards
├── train.py           # Training script
├── experiment.yaml    # Experiment metadata
├── README.md          # This file
├── models/            # Trained models (populated after training)
│   └── exit_policy_final.pt
└── results/           # Backtest results (populated after evaluation)
    └── training/
        └── training_metrics.json
```

---

## Monitoring During Training

Watch for these indicators that the fix is working:

1. **Episode length > 5 bars** (v1 was ~1.1 bars)
2. **Block rate initially high** then decreasing as model learns
3. **Entropy stays higher longer** (exploring more actions)
4. **Returns may dip initially** then improve as model finds good exits

---

## After Training

1. Update results in experiment.yaml
2. Run backtest: `python run.py -i AUDUSD -t 15M --version v2_exit_guards`
3. Compare with v1: `python experiment_manager.py compare v1_baseline v2_exit_guards`
4. Update registry: `python experiment_manager.py update-results v2_exit_guards`

---

## Risk

If guards are too strict, model may:
- Never find valid exits
- Rely entirely on SL hits
- Have worse Sharpe than v1

**Mitigation**: Guard parameters are conservative (10 pips, 2 bars). Can be relaxed if needed.
