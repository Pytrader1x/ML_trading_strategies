# ML Trading Strategies

Train machine learning models to discover alpha and generate profitable trading signals. Validated using the [Production Backtest Engine](https://github.com/Pytrader1x/production-backtest-engine).

## Objective

Build ML/RL models that learn to trade FX markets profitably:
- Train models on 20 years of historical data (2005-2025)
- Generate trading signals with edge (alpha)
- Validate with rigorous backtesting + Monte Carlo simulation
- Target: Sharpe > 1.0, Max DD < 20%, Win Rate > 50%

---

## Directory Structure

```
ML_trading_strategies/
├── data/                           # Historical price data
│   └── {PAIR}_MASTER.csv           # 1-minute OHLC (2005-2025)
│
├── strategies/                     # ML strategy implementations
│   └── {strategy_name}/
│       ├── strategy.py             # Strategy class (backtest interface)
│       ├── model.py                # ML model definition
│       ├── train.py                # Training script
│       ├── run.py                  # Entry point for backtesting
│       ├── experiment_manager.py   # Version management CLI
│       │
│       ├── experiments/            # Versioned experiments
│       │   ├── registry.json       # Central tracking
│       │   └── v1_baseline/
│       │       ├── experiment.yaml # Config snapshot
│       │       ├── models/         # Trained weights
│       │       └── results/        # Backtest outputs
│       │
│       └── data/                   # Strategy-specific training data
│
└── README.md
```

---

## Available Data

| Pair | Records | Date Range | File Size |
|------|---------|------------|-----------|
| AUDUSD | ~10M | 2005-2025 | 369 MB |
| EURUSD | ~10M | 2005-2025 | 376 MB |
| GBPUSD | ~10M | 2005-2025 | 297 MB |
| NZDUSD | ~10M | 2005-2025 | 293 MB |
| USDCAD | ~10M | 2005-2025 | 292 MB |
| USDCHF | ~10M | 2005-2025 | 295 MB |
| USDJPY | ~10M | 2005-2025 | 36 MB |

**Note:** USDJPY uses pip size 0.01 (vs 0.0001 for others).

---

## Creating a Strategy

### 1. Strategy Class (strategy.py)

Inherit from `Strategy` and implement `init` + `next`:

```python
from backtest_engine import Strategy
import joblib

class MyMLStrategy(Strategy):
    def init(self):
        # Load trained model
        self.model = joblib.load('model/best_model.pkl')

        # Calculate features
        self.data['sma_20'] = self.data['Close'].rolling(20).mean()
        self.data['rsi'] = self._calc_rsi(14)
        self.features = self.data[['sma_20', 'rsi']].values

    def next(self, i, record):
        if i < 50 or self.broker.active_trade:
            return

        # Get prediction
        X = self.features[i:i+1]
        pred = self.model.predict(X)[0]

        price = record['Close']
        if pred == 1:  # Buy signal
            sl = price - 0.0050
            tp = price + 0.0100
            self.broker.buy(i, price, size=100000, sl=sl, tp=tp)
        elif pred == -1:  # Sell signal
            sl = price + 0.0050
            tp = price - 0.0100
            self.broker.sell(i, price, size=100000, sl=sl, tp=tp)
```

### 2. Training Script (train.py)

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv('../../data/EURUSD_MASTER.csv', parse_dates=['Time'], index_col='Time')
df = df.resample('1h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()

# Create features and labels
df['returns'] = df['Close'].pct_change()
df['sma_20'] = df['Close'].rolling(20).mean()
df['label'] = (df['returns'].shift(-1) > 0).astype(int)

# Train/test split (walk-forward)
train = df['2005':'2020']
test = df['2021':'2025']

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(train[['sma_20']].dropna(), train['label'].dropna())

# Save
joblib.dump(model, 'model/best_model.pkl')
```

### 3. Run Script (run.py)

```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
from backtest_engine import Backtester
from strategy import MyMLStrategy

# Load and resample data
df = pd.read_csv('../../data/EURUSD_MASTER.csv', parse_dates=['Time'], index_col='Time')
df = df.resample('1h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()

# Run backtest
bt = Backtester(MyMLStrategy, df, cash=1_000_000, commission=0.50, spread=0.0001)
bt.run_monte_carlo(output_dir='results/EURUSD/1H', n_sims=500, open_browser=True)
```

---

## ML Strategy Ideas

### Supervised Learning
| Strategy | Approach |
|----------|----------|
| `xgboost_regime` | XGBoost to classify market regimes |
| `lstm_trend` | LSTM for trend direction prediction |
| `transformer_price` | Transformer for price movement |

### Reinforcement Learning
| Strategy | Approach |
|----------|----------|
| `ppo_trader` | PPO agent learns entry/exit timing |
| `dqn_sizing` | DQN for dynamic position sizing |

### Meta-Learning
| Strategy | Approach |
|----------|----------|
| `meta_labeler` | Train model to filter base signals |
| `ensemble_vote` | Combine multiple model predictions |

---

## Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Sharpe Ratio** | > 1.0 | Risk-adjusted return |
| **Max Drawdown** | < 20% | Worst peak-to-trough |
| **Win Rate** | > 50% | % profitable trades |
| **Profit Factor** | > 1.5 | Gross profit / gross loss |
| **Trades** | > 100 | Statistical significance |

---

## Backtest Engine

Requires the [Production Backtest Engine](https://github.com/Pytrader1x/production-backtest-engine).

```bash
git clone https://github.com/Pytrader1x/production-backtest-engine.git
pip install -e production-backtest-engine/
```

**Key Features:**
- Event-driven backtesting (no look-ahead bias)
- Monte Carlo simulation (500+ sims)
- Interactive HTML reports
- PNG summary charts
- Parameter optimization

### Quick Reference

```python
from backtest_engine import Backtester, Strategy, MonteCarloEngine

# Run backtest
bt = Backtester(MyStrategy, df, cash=1_000_000)
stats = bt.run()

# Monte Carlo validation
bt.run_monte_carlo(output_dir='results', n_sims=500)

# Access metrics
sharpe = stats.get('sharpe_ratio', 0)
max_dd = stats.get('max_drawdown_pct', 0)
win_rate = stats.get('win_rate', 0)
```

### Output Files

```
results/{PAIR}/{TIMEFRAME}/
├── backtest_summary.png    # Visual summary
├── backtest_report.html    # Interactive dashboard
├── trades.csv              # Trade log
└── backtest_results.json   # Raw metrics
```

---

## Rules

1. **No overfitting** - Use proper train/test splits
2. **Walk-forward validation** - Train on past, test on future
3. **Out-of-sample testing** - Reserve 2021-2025 for final validation
4. **Feature hygiene** - No lookahead bias, proper lagging
5. **Realistic costs** - Include spread (0.0001) and commission ($0.50)
6. **Statistical significance** - Minimum 100+ trades
7. **Version experiments** - Use versioned directories for model iteration (see below)

---

## Experiment Versioning

For strategies with iterative model development (especially RL), use versioned experiments for reproducibility and comparison.

### Structure

```
strategies/{strategy_name}/
├── experiments/
│   ├── registry.json          # Central tracking
│   ├── COMPARISON.md          # Auto-generated comparison
│   │
│   ├── v1_baseline/           # Version 1
│   │   ├── experiment.yaml    # Config snapshot
│   │   ├── models/            # Trained weights
│   │   └── results/           # Backtest outputs
│   │
│   └── v2_*/                  # Future versions...
│
├── data/                      # Shared training data
└── config.py, train.py, etc.  # Shared code
```

### Usage

```bash
cd strategies/ema_bb_v2_rl

# List versions
python experiment_manager.py list

# Create new version from parent
python experiment_manager.py create v2_experiment --parent v1_baseline

# Train specific version
python train.py --version v2_experiment --episodes data/episodes_train.pkl

# Backtest specific version
python run.py -i AUDUSD -t 15M --version v2_experiment

# Compare versions
python experiment_manager.py compare v1_baseline v2_experiment
```

### Version Naming

Format: `v{N}_{short_description}`

Examples: `v1_baseline`, `v2_entropy`, `v3_larger_net`

---

## Quick Start

```bash
# 1. Create new strategy
mkdir -p strategies/my_strategy/model
cd strategies/my_strategy

# 2. Train model
python train.py

# 3. Run backtest
python run.py -i EURUSD -t 1H

# 4. Run all pairs/timeframes
python run.py --all
```
