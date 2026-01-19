# ML Trading Strategies

A systematic backtesting framework for evaluating **machine learning-based trading strategies** across multiple currency pairs and timeframes.

## Overview

This repository contains ML-driven trading strategies backtested on **7 currency pairs** across **3 timeframes** (15M, 1H, 4H) using a 20-year historical dataset (2005-2025).

**Goal:** Develop and validate ML models for trading with robust out-of-sample performance (Sharpe > 0.5).

---

## Quick Start

```bash
# Run a single ML strategy on one pair
python strategies/lstm_trend/run.py -i AUDUSD -t 1H

# Run all timeframes for a pair
python strategies/lstm_trend/run.py -i AUDUSD --all

# Run ALL strategies on ALL pairs (full batch)
python run_all_backtests.py
```

---

## Directory Structure

```
ML_trading_strategies/
├── data/                          # Historical price data (CSV)
│   └── {PAIR}_MASTER.csv          # 1-minute OHLC data
│
├── strategies/                    # ML Strategy implementations
│   └── {strategy_name}/
│       ├── strategy.py            # Strategy class with ML model
│       ├── model.py               # ML model definition (optional)
│       ├── train.py               # Training script (optional)
│       ├── run.py                 # Entry point
│       └── results/               # Backtest outputs
│           └── {PAIR}/
│               ├── SUMMARY.md     # Performance comparison
│               ├── 15M/           # 15-minute results
│               ├── 1H/            # 1-hour results
│               └── 4H/            # 4-hour results
│
├── analysis/                      # Portfolio analysis tools
│   ├── run_analysis.py            # Portfolio aggregation & leaderboard
│   ├── output/                    # Leaderboard CSVs
│   └── portfolios/                # Portfolio backtest results
│
├── docs/                          # Documentation and examples
├── run_all_backtests.py           # Batch runner
└── README.md
```

---

## Output Files

Each backtest generates:

| File | Description |
|------|-------------|
| `backtest_report.html` | Interactive chart with trades, equity curve |
| `backtest_summary.png` | Visual summary (equity, drawdown, stats) |
| `backtest_results.json` | Raw metrics (Sharpe, win rate, P&L, etc.) |
| `trades.csv` | Trade-by-trade log |
| `SUMMARY.md` | Multi-timeframe comparison table |

---

## ML Strategy Categories

### Supervised Learning
| Strategy | Description | Status |
|----------|-------------|--------|
| `xgboost_regime` | XGBoost regime classification | Planned |
| `lstm_trend` | LSTM trend prediction | Planned |
| `transformer_price` | Transformer price forecasting | Planned |

### Reinforcement Learning
| Strategy | Description | Status |
|----------|-------------|--------|
| `dqn_trading` | Deep Q-Network for trade decisions | Planned |
| `ppo_portfolio` | PPO for portfolio allocation | Planned |

### Ensemble Methods
| Strategy | Description | Status |
|----------|-------------|--------|
| `stacking_signals` | Stacked ensemble of indicators | Planned |
| `voting_classifier` | Multiple model voting | Planned |

---

## Key Metrics

We focus on these metrics for strategy evaluation:

| Metric | Target | Description |
|--------|--------|-------------|
| **Sharpe Ratio** | > 0.5 | Risk-adjusted return (higher = better) |
| **Max Drawdown** | < 25% | Largest peak-to-trough decline |
| **Win Rate** | > 40% | Percentage of profitable trades |
| **Profit Factor** | > 1.2 | Gross profit / gross loss |
| **Trade Count** | > 100 | Statistical significance |
| **OOS Performance** | ~IIS | Out-of-sample vs in-sample consistency |

---

## Currency Pairs

| Pair | Data Range | Notes |
|------|------------|-------|
| AUDUSD | 2005-2025 | Australian Dollar |
| EURUSD | 2005-2025 | Euro (most liquid) |
| GBPUSD | 2005-2025 | British Pound |
| NZDUSD | 2005-2025 | New Zealand Dollar |
| USDCAD | 2005-2025 | Canadian Dollar |
| USDCHF | 2005-2025 | Swiss Franc |
| USDJPY | 2005-2025 | Japanese Yen (pip=0.01) |

**Note:** USDJPY uses pip size 0.01 (vs 0.0001 for others). Some strategies need adjustment.

---

## Creating a New ML Strategy

```python
# strategies/my_ml_strategy/strategy.py
from backtest_engine import Strategy
import joblib

class MyMLStrategy(Strategy):
    def init(self):
        # Load pre-trained model
        self.model = joblib.load('model.pkl')

        # Calculate features on self.data (full DataFrame)
        self.data['ema_20'] = self.data['Close'].ewm(span=20).mean()
        self.data['rsi'] = self._calc_rsi(14)
        self.features = self.data[['ema_20', 'rsi']].values

    def next(self, i: int, record):
        if self.broker.active_trade:
            return

        # Get prediction
        X = self.features[i:i+1]
        pred = self.model.predict(X)[0]

        close = record['Close']
        if pred == 1:  # Buy signal
            sl = close - 0.0050
            tp = close + 0.0100
            self.broker.buy(i, close, size=100000, sl=sl, tp=tp)
        elif pred == -1:  # Sell signal
            sl = close + 0.0050
            tp = close - 0.0100
            self.broker.sell(i, close, size=100000, sl=sl, tp=tp)
```

---

## ML Best Practices

### Walk-Forward Validation
```python
# Use expanding or rolling window for training
train_end = int(len(data) * 0.7)
train_data = data[:train_end]
test_data = data[train_end:]

# Retrain periodically (e.g., every 6 months)
```

### Feature Engineering
- Use technical indicators as features
- Include lagged returns, volatility measures
- Normalize features (z-score or min-max)
- Avoid lookahead bias

### Model Selection
- Start simple (Logistic Regression, XGBoost)
- Graduate to complex models (LSTM, Transformer) if needed
- Ensemble multiple models for robustness

---

## Backtest Engine

Requires the [Production Backtest Engine](https://github.com/Pytrader1x/production-backtest-engine).

```bash
# Install the engine (update path in run.py files as needed)
git clone https://github.com/Pytrader1x/production-backtest-engine.git
```

Key features:
- Vectorized indicator calculation
- Single position at a time
- SL/TP as absolute prices
- Monte Carlo simulation (500 sims)
- HTML report generation

---

## Rules

1. **No overfitting**: Use proper train/test splits and cross-validation
2. **Out-of-sample testing**: Mandatory walk-forward validation for all ML strategies
3. **Realistic costs**: Include spread and commission
4. **Statistical significance**: Minimum 100+ trades
5. **Robust metrics**: Prefer Sharpe > 0.5, max DD < 25%
6. **Feature hygiene**: No lookahead bias, proper lagging

---

*Generated by Production Backtest Engine*
