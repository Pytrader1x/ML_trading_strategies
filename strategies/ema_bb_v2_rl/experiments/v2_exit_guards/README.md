# v2_exit_guards Experiment

**Status**: Pending
**Parent**: v1_baseline
**Created**: 2026-01-25

---

## Key Innovation: Counterfactual Rewards

Instead of hard guards that block exits, we use **counterfactual rewards** to teach the model when exits are good vs bad.

### How It Works

When the model exits at bar T with `pnl_exit`:

1. **Look at what happens AFTER** (next 20 bars)
2. **Defensive bonus**: If price drops below `pnl_exit` → reward (good exit, avoided loss)
3. **Opportunity cost**: If price rises above `pnl_exit` → penalty (bad exit, missed gain)
4. **Tiny exit cost**: Small transaction cost discourages truly wasteful exits

### The Math

```
future_worst = min(next 20 bars)
future_best = max(next 20 bars)

defensive_bonus = max(0, pnl_exit - future_worst) × 0.5
opportunity_cost = max(0, future_best - pnl_exit) × 0.5
exit_cost = 0.0001

total_adjustment = defensive_bonus - opportunity_cost - exit_cost
```

---

## Why Not Hard Guards?

We initially implemented hard guards (min_profit, min_bars requirements), but:

**Problem with guards**: They block ALL early exits, including legitimate defensive exits.

**Example**: Trade at +5 pips starting to reverse
- Hard guard blocks PARTIAL (waiting for +10 pips)
- Price drops to -20 pips
- We lose 25 pips instead of saving 5 pips

**Counterfactual rewards solve this**:
- Early defensive exit → rewarded (avoided loss)
- Wasteful 0-pip exit → penalized (missed opportunity)
- Model learns the DIFFERENCE

---

## Expected Behavior

| Scenario | Counterfactual Reward | Model Learns |
|----------|----------------------|--------------|
| Exit at 0 pips, price goes +30 | -15 (missed opportunity) | BAD - don't exit early |
| Exit at +10 pips, price drops to -20 | +15 (avoided 30 pip loss) | GOOD - defensive exit |
| Exit at +10 pips, price goes +15 | -2.5 (missed 5 pips) | OK - slight regret |
| Exit at optimal point | ~0 (balanced) | PERFECT |

---

## Training

```bash
# Local test (1M steps)
cd experiments/v2_exit_guards
python train.py --timesteps 1000000 --device cpu

# Full training on Vast.ai
python train.py --wandb --device cuda
```

---

## Monitoring

During training, watch these metrics:

1. **Episode length**: Should be > 3 bars (not immediate exits)
2. **defensive_rate**: % of exits that avoided future loss
3. **premature_rate**: % of exits that missed future gain
4. **avg_defensive_bonus**: Average reward for defensive exits
5. **avg_opportunity_cost**: Average penalty for premature exits

The environment tracks these via `env.get_counterfactual_stats()`.

---

## Files

```
v2_exit_guards/
├── config.py          # Counterfactual reward params (no hard guards)
├── env.py             # _compute_counterfactual_reward() method
├── train.py           # Training script
├── experiment.yaml    # Experiment metadata
├── README.md          # This file
├── models/            # Trained models (after training)
└── results/           # Training metrics (after training)
```

---

## Config Parameters

```python
# Counterfactual rewards
defensive_coef: 0.5              # Reward for avoiding loss
regret_coef: 0.5                 # Penalty for missing gain
counterfactual_lookforward: 20   # Bars to look ahead
exit_cost: 0.0001                # Tiny transaction cost

# Equal weights (defensive = regret) → Sharpe-optimized
```

---

## Comparison to v1

| Aspect | v1_baseline | v2_exit_guards |
|--------|-------------|----------------|
| EXIT guards | None | None (but counterfactual rewards) |
| PARTIAL guards | min_profit=0.001, min_bars=1 | None (but counterfactual rewards) |
| Defensive exits | Blocked if early | **Rewarded if avoided loss** |
| Wasteful exits | Allowed (0 cost) | **Penalized (opportunity cost)** |
| Time penalty | 0.005 | 0.002 (reduced) |
| Transaction cost | None | 0.0001 per exit |
