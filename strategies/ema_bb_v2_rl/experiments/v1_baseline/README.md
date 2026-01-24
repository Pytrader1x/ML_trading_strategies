# v1_baseline - Initial PPO Exit Optimizer

## Description

Baseline PPO model for optimizing trade exits. Uses default hyperparameters from the literature with adjustments for trading domain.

## Key Features

- **5-action discrete space**: HOLD, EXIT, TIGHTEN_SL, TRAIL_BE, PARTIAL
- **Dense rewards**: Mark-to-market + realized P&L
- **Entropy annealing**: 0.05 → 0.001 over 5M steps
- **Network**: 2×256 MLP with layer normalization

## Results (AUDUSD 15M)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 0.69 |
| Return % | 9.46% |
| Max Drawdown | 2.93% |
| Win Rate | 42.7% |
| Trades | 1,325 |
| Profit Factor | 1.09 |

## Training Details

- **Episodes**: 48MB in-sample (2005-2021), 11MB out-of-sample (2022-2025)
- **Total timesteps**: 10M
- **Training time**: ~4 hours on RTX 4090
- **Checkpoint**: `models/exit_policy_final.pt` (1.7MB)

## Notes

Initial version focused on validating the RL exit approach. Future versions should explore:
- Larger entropy bonus for more exploration
- Different network architectures
- Action-specific reward shaping
