"""
RL Trading Environment for EMA BB V2

Gymnasium-compatible environment for training RL agents.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class TradingEnv(gym.Env):
    """
    Trading environment that wraps EMA BB V2 signals.

    Observation space:
        - Price features (OHLC, returns, volatility)
        - Technical indicators (EMA, BB, etc.)
        - Position state (flat, long, short)
        - Account state (balance, unrealized PnL)

    Action space:
        - 0: Hold / Do nothing
        - 1: Buy / Go long
        - 2: Sell / Go short
        - 3: Close position

    Or continuous for position sizing:
        - [-1, 1] where sign = direction, magnitude = size
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        initial_balance: float = 100_000,
        max_position_size: int = 100_000,
        spread: float = 0.0001,
        commission: float = 0.50,
        lookback_window: int = 50,
        reward_scaling: float = 1.0,
    ):
        super().__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.spread = spread
        self.commission = commission
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling

        # Calculate features
        self._precompute_features()

        # Action space: discrete (hold, buy, sell, close)
        self.action_space = spaces.Discrete(4)

        # Observation space
        n_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features + 3,),  # features + position + balance + unrealized
            dtype=np.float32
        )

        self.reset()

    def _precompute_features(self):
        """Calculate all features upfront for speed."""
        # TODO: Add EMA BB V2 indicators
        df = self.df.copy()

        # Basic features
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()

        # Normalize
        df = df.dropna()
        self.features = df[['returns', 'volatility', 'sma_20', 'sma_50']].values
        self.prices = df['Close'].values
        self.n_steps = len(self.prices)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0  # -1 short, 0 flat, 1 long
        self.position_price = 0.0
        self.position_size = 0
        self.total_reward = 0.0
        self.trades = []

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        features = self.features[self.current_step]
        position_state = np.array([
            self.position,
            self.balance / self.initial_balance,
            self._unrealized_pnl() / self.initial_balance
        ])
        return np.concatenate([features, position_state]).astype(np.float32)

    def _unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.position == 0:
            return 0.0
        current_price = self.prices[self.current_step]
        if self.position == 1:  # Long
            return (current_price - self.position_price) * self.position_size
        else:  # Short
            return (self.position_price - current_price) * self.position_size

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        current_price = self.prices[self.current_step]
        reward = 0.0

        # Execute action
        if action == 1 and self.position <= 0:  # Buy
            if self.position == -1:  # Close short first
                reward += self._close_position(current_price)
            self._open_position(1, current_price)

        elif action == 2 and self.position >= 0:  # Sell
            if self.position == 1:  # Close long first
                reward += self._close_position(current_price)
            self._open_position(-1, current_price)

        elif action == 3 and self.position != 0:  # Close
            reward += self._close_position(current_price)

        # Move to next step
        self.current_step += 1

        # Check if done
        terminated = self.current_step >= self.n_steps - 1
        truncated = self.balance <= 0

        # Add step reward (unrealized PnL change)
        if self.position != 0:
            reward += self._unrealized_pnl() * 0.001  # Small reward for holding

        reward *= self.reward_scaling
        self.total_reward += reward

        obs = self._get_observation()
        info = {
            "balance": self.balance,
            "position": self.position,
            "total_reward": self.total_reward
        }

        return obs, reward, terminated, truncated, info

    def _open_position(self, direction: int, price: float):
        """Open a new position."""
        self.position = direction
        self.position_price = price + (self.spread if direction == 1 else -self.spread)
        self.position_size = self.max_position_size
        self.balance -= self.commission

    def _close_position(self, price: float) -> float:
        """Close current position and return realized PnL."""
        if self.position == 0:
            return 0.0

        exit_price = price - (self.spread if self.position == 1 else -self.spread)

        if self.position == 1:
            pnl = (exit_price - self.position_price) * self.position_size
        else:
            pnl = (self.position_price - exit_price) * self.position_size

        self.balance += pnl - self.commission
        self.trades.append({
            "entry": self.position_price,
            "exit": exit_price,
            "pnl": pnl,
            "direction": self.position
        })

        self.position = 0
        self.position_price = 0.0
        self.position_size = 0

        return pnl

    def render(self):
        """Render current state."""
        print(f"Step: {self.current_step}, Balance: ${self.balance:,.2f}, "
              f"Position: {self.position}, Unrealized: ${self._unrealized_pnl():,.2f}")
