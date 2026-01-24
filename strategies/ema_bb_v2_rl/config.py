"""
Configuration for EMA BB V2 RL Strategy

All hyperparameters in one place.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Data loading and preprocessing."""
    pairs: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"
    ])
    timeframe: str = "1H"
    train_start: str = "2005-01-01"
    train_end: str = "2020-12-31"
    test_start: str = "2021-01-01"
    test_end: str = "2025-01-01"


@dataclass
class EnvConfig:
    """RL Environment settings."""
    initial_balance: float = 100_000
    max_position_size: int = 100_000
    spread: float = 0.0001
    commission: float = 0.50
    lookback_window: int = 50
    reward_scaling: float = 1.0


@dataclass
class ModelConfig:
    """Neural network architecture."""
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    use_lstm: bool = False
    lstm_hidden: int = 128


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    algorithm: str = "PPO"  # PPO, A2C, DQN
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_envs: int = 8  # Parallel environments
    eval_freq: int = 10_000
    save_freq: int = 50_000
    seed: int = 42


@dataclass
class Config:
    """Master config combining all settings."""
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Default config instance
config = Config()
