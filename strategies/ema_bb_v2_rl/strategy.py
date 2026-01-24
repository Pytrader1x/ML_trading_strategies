"""
Backtest Strategy Wrapper for EMA BB V2 RL

Wraps trained RL model for use with the backtest engine.
"""

import torch
import numpy as np
from pathlib import Path

# TODO: Import from backtest_engine when ready
# from backtest_engine import Strategy


class EmaBBV2RLStrategy:
    """
    RL-enhanced EMA BB Scalp V2 Strategy.

    Uses trained RL model to:
    - Filter base strategy signals
    - Optimize entry timing
    - Manage position sizing
    """

    def __init__(self, model_path: str = "model/best_model.pt"):
        self.model_path = Path(model_path)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init(self):
        """Initialize strategy - called once before backtest."""
        # Load trained model
        if self.model_path.exists():
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
        else:
            print(f"Warning: Model not found at {self.model_path}")
            print("Using random actions (for testing)")

        # Calculate features
        self._precompute_features()

    def _precompute_features(self):
        """Calculate features for RL model input."""
        df = self.data.copy()

        # EMA BB V2 indicators
        df['ema_8'] = df['Close'].ewm(span=8).mean()
        df['ema_21'] = df['Close'].ewm(span=21).mean()

        # Bollinger Bands
        df['bb_mid'] = df['Close'].rolling(20).mean()
        df['bb_std'] = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

        # Price features
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()

        # Normalize
        feature_cols = ['ema_8', 'ema_21', 'bb_upper', 'bb_lower', 'returns', 'volatility']
        self.features = df[feature_cols].values

    def next(self, i: int, record):
        """Called for each bar - trading logic."""
        if i < 50:  # Warmup period
            return

        # Get observation
        obs = self._get_observation(i)

        # Get action from model
        if self.model is not None:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action_probs, _ = self.model(obs_tensor)
                action = torch.argmax(action_probs, dim=-1).item()
        else:
            action = np.random.randint(0, 4)  # Random for testing

        # Execute action
        price = record['Close']

        if action == 1 and not self.broker.active_trade:  # Buy
            sl = price - 0.0050
            tp = price + 0.0100
            self.broker.buy(i, price, size=100000, sl=sl, tp=tp)

        elif action == 2 and not self.broker.active_trade:  # Sell
            sl = price + 0.0050
            tp = price - 0.0100
            self.broker.sell(i, price, size=100000, sl=sl, tp=tp)

        elif action == 3 and self.broker.active_trade:  # Close
            self.broker.close(i, price, reason="RL Signal")

    def _get_observation(self, i: int) -> np.ndarray:
        """Get observation for RL model."""
        features = self.features[i]

        # Add position state
        position = 0
        if self.broker.active_trade:
            position = self.broker.active_trade.get('direction', 0)

        balance_ratio = self.broker.cash / self.broker.initial_cash

        return np.concatenate([
            features,
            [position, balance_ratio, 0.0]  # position, balance, unrealized
        ]).astype(np.float32)
