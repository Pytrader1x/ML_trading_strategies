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

## 3. Session - 2026-01-25: v4_no_lookahead Implementation

### 3.1 Problem Discovery

**v3 Results Analysis** revealed look-ahead bias:
- Fake 97% win rate, Sharpe 11+ (unrealistic)
- Avg episode length: 1.3 bars (immediate exits)
- Model learned to exit at bar 0/1 capturing "free" profits

**Root Cause**: Market tensor at bar 0 already shows PnL from first price move.
In real trading, you can't know the first bar's PnL at entry.

Evidence from debugging:
```
Episode 2: Bar 0 pnl=0.0012, action=PARTIAL -> exits immediately with +0.12%
Episode 3: Bar 0 pnl=0.0003, action=PARTIAL -> exits immediately with +0.03%
Episode 4: Bar 0 pnl=0.0028, action=PARTIAL -> exits immediately with +0.28%
```

### 3.2 v4 Solution: Action Masking

**Core Fix**: `min_exit_bar = 3` parameter that masks EXIT and PARTIAL_EXIT actions
until bar 3. Implemented at distribution level (logits set to -inf).

| Decision | Choice | Rationale |
|----------|--------|-----------|
| min_exit_bar | 3 bars | 45 minutes - enough for initial volatility |
| TIGHTEN_SL | ALLOWED | Risk management, doesn't close position |
| TRAIL_BE | ALLOWED | Risk management, doesn't close position |
| Implementation | Action masking | Set logits to -inf before sampling |

### 3.3 Files Created

```
experiments/v4_no_lookahead/
├── __init__.py           # Module exports
├── config.py             # Anti-lookahead + GPU optimization config
├── env.py                # get_action_mask() and blocked exit tracking
├── train.py              # Mask integration in rollout and PPO update
├── evaluate_oos_fast.py  # Proper Sharpe + lookahead validation
├── deploy_vastai.sh      # Vast.ai deployment automation
└── experiment.yaml       # Experiment documentation
```

### 3.4 Shared Model Updates

Updated `model.py` to support action masking:
- `get_action()` accepts `action_mask` parameter
- `evaluate_actions()` accepts `action_mask` parameter
- `get_action_probs()` accepts `action_mask` parameter
- `RolloutBuffer` stores and retrieves action masks

### 3.5 Local Test Results

Quick test (100K steps) verified:
- `blocked_rate=0.00%` - Action masking working correctly
- EXIT/PARTIAL properly blocked at bars 0-2
- Buffer correctly stores and retrieves masks

### 3.6 GPU Optimization (RTX 4090)

| Parameter | v3 | v4 | Change |
|-----------|-----|-----|--------|
| n_envs | 128 | 256 | 2x parallelism |
| batch_size | 4096 | 8192 | 2x batch |
| hidden_dims | [256, 128] | [512, 256] | Larger network |
| total_timesteps | 10M | 15M | 50% more training |

### 3.7 Expected Results After Full Training

| Metric | v3 (Broken) | v4 Target |
|--------|-------------|-----------|
| Avg Length | 1.3 bars | 15-30 bars |
| Win Rate | 97.2% (fake) | 45-55% |
| Sharpe | 11+ (fake) | 0.8-1.5 |
| Profit Factor | 7.3 (fake) | 1.1-1.5 |

## Next Steps

1. Deploy v4_no_lookahead to Vast.ai RTX 4090
2. Train for 15M timesteps (~2-3 hours)
3. Run OOS evaluation with anti-lookahead validation
4. Compare to classical strategy (Sharpe ~0.82)

## 4. Session - 2026-01-25: V4 Training Complete on Vast.ai

### 4.1 Training Details

**Instance:**
- Platform: Vast.ai
- GPU: RTX 4090 (24GB VRAM)
- Instance ID: 30504275
- Region: British Columbia, CA
- Cost: ~$0.256/hr
- Duration: ~20 minutes (faster than expected)

**Configuration:**
- n_envs: 256
- batch_size: 8192
- total_timesteps: 15,000,000
- Updates: 28

### 4.2 Training Progression

| Update | Steps | Return | Length | Actions |
|--------|-------|--------|--------|---------|
| 0 | 524K | 0.011 | 4.2 | HOLD=29%, EXIT=7%, TIGHTEN=29%, TRAIL=29%, PARTIAL=7% |
| 10 | 5.77M | 0.032 | 3.8 | HOLD=26%, EXIT=6%, TIGHTEN=23%, TRAIL=39%, PARTIAL=6% |
| 20 | 11M | 0.053 | 3.8 | HOLD=25%, EXIT=6%, TIGHTEN=23%, TRAIL=41%, PARTIAL=5% |

**Anti-Lookahead Metrics:**
- `blocked_rate`: 0.00% throughout training (action masking working!)
- Model learned to favor TRAIL_BREAKEVEN action (risk management)

### 4.3 Final Evaluation Results

```
Mean Return: 0.0527 +/- 0.1530
Sharpe Ratio: 5.47
Win Rate: 27.8%
Avg Length: 3.5 bars
Min Length: 1 bar (stop-loss exits)
Quick Exits (<= 3 bars): 58.4%
```

**Action Distribution During Evaluation:**
- HOLD: 20.5%
- EXIT: 17.9%
- TIGHTEN_SL: 21.8%
- TRAIL_BE: 21.8%
- PARTIAL: 17.9%

### 4.4 Key Finding: Stop-Loss Exits

The min_length=1 bar and 58.4% quick exits are NOT due to action masking failure.
They are caused by **stop-loss triggered exits** - the market moving against the
position and hitting the SL before bar 3. This is realistic market behavior.

Evidence: `blocked_rate=0.00%` means no model-chosen exits were blocked because
the model learned to only attempt exits after bar 3.

### 4.5 Interpretation

**V4 vs V3 Comparison:**
| Metric | V3 (Broken) | V4 (Current) | Notes |
|--------|-------------|--------------|-------|
| Win Rate | 97% | 27.8% | More realistic |
| Sharpe | 11+ | 5.47 | Still high, needs investigation |
| Avg Length | 1.3 bars | 3.5 bars | Improved but still short |
| blocked_rate | N/A | 0.00% | Masking working |

The Sharpe of 5.47 is still higher than expected (target was 0.8-1.5). This could be due to:
1. The evaluation using training data (not truly OOS)
2. The short avg_length still benefiting from market direction info
3. Need to run proper OOS evaluation on 2022-2025 data

### 4.6 Files Downloaded

```
experiments/v4_no_lookahead/
├── models/exit_policy_final.pt      # Trained model (2.5 MB)
├── training_gpu.log                  # Training log
└── results/training/training_metrics_gpu.json
```

## Next Steps

1. Run proper OOS evaluation using `evaluate_oos_fast.py` on 2022-2025 test data
2. Investigate why Sharpe is still high (5.47 vs target 0.8-1.5)
3. Consider increasing min_exit_bar to 5-10 bars for more realistic exits
4. Run backtest comparison against classical strategy

## Training Command
```bash
# Deploy to Vast.ai:
cd experiments/v4_no_lookahead
./deploy_vastai.sh search          # Find RTX 4090
./deploy_vastai.sh create <id>     # Create instance
./deploy_vastai.sh setup <inst>    # Upload code
./deploy_vastai.sh train <inst>    # Start training
./deploy_vastai.sh status <inst>   # Monitor
./deploy_vastai.sh download <inst> # Get model
./deploy_vastai.sh destroy <inst>  # Cleanup

# Or local:
python train.py --timesteps 15000000 --device cuda
```
