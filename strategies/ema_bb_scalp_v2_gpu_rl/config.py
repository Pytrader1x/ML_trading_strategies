"""
Configuration classes for the GPU-optimized Meta-Labeling + RL system.

All hyperparameters are centralized here for easy tuning.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class StrategyConfig:
    """
    Original EMA BB Scalp V2 strategy parameters.
    These define the base signal generation logic.
    """
    # EMA parameters
    ema_fast: int = 30
    ema_slow: int = 50

    # Bollinger Bands
    bb_period: int = 20
    bb_dev: float = 2.0

    # Trend confirmation
    trend_conf_bars: int = 3

    # ADX filter
    adx_period: int = 14
    adx_min: float = 20.0
    adx_max: float = 50.0
    use_adx_filter: bool = True

    # ATR for volatility
    atr_period: int = 14

    # Risk management (original)
    sl_coef: float = 1.1
    tp_ratio: float = 1.5

    # Trading limits
    max_trades_per_day: int = 10
    cooldown_bars: int = 1

    @property
    def warmup_bars(self) -> int:
        """Minimum bars needed before generating signals."""
        return max(self.ema_slow, self.bb_period, self.adx_period, self.trend_conf_bars) + 10


@dataclass
class TripleBarrierConfig:
    """
    Triple Barrier Method configuration for meta-labeling.

    The Triple Barrier defines three exit conditions:
    1. Upper barrier: Take profit (TP)
    2. Lower barrier: Stop loss (SL)
    3. Vertical barrier: Maximum holding time

    A trade is labeled as "Win" (1) if it exits via TP first,
    "Loss" (0) if it exits via SL or times out with negative return.
    """
    # Profit take distance as multiple of ATR
    profit_take_atr_mult: float = 2.0

    # Stop loss distance as multiple of ATR
    stop_loss_atr_mult: float = 1.5

    # Maximum holding period in bars
    max_holding_bars: int = 48

    # Minimum return to consider a "win" when time expires
    min_return_threshold: float = 0.0005  # 5 pips for FX

    # Whether to use high/low for barrier touches (more realistic)
    # If False, uses close prices only
    use_high_low: bool = True


@dataclass
class RLConfig:
    """
    Reinforcement Learning hyperparameters for PPO training.
    Optimized for GPU training on RTX 4090/5090.
    """
    # Environment settings
    num_envs: int = 4096  # Parallel environments (must be power of 2 for efficiency)
    episode_length: int = 128  # Steps per rollout before reset

    # Observation space
    obs_dim: int = 10  # 7 market features + meta_label + position + unrealized_pnl

    # Action space
    action_dim: int = 4  # 0=Hold, 1=Long, 2=Short, 3=Close

    # Network architecture
    hidden_size: int = 256
    num_layers: int = 2
    use_layer_norm: bool = True

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    clip_epsilon: float = 0.2  # PPO clipping parameter
    entropy_coef: float = 0.01  # Entropy bonus for exploration
    value_coef: float = 0.5  # Value loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    # Training schedule
    n_epochs: int = 10  # PPO epochs per update
    batch_size: int = 256  # Mini-batch size for SGD
    total_updates: int = 10000  # Total training updates

    # Device settings
    device: str = "cuda"
    use_mixed_precision: bool = True  # FP16 for faster training
    use_torch_compile: bool = True  # torch.compile optimization

    # Reward shaping
    meta_label_reward_scale: float = 0.1  # Bonus for meta-aligned entries
    bad_entry_penalty: float = 0.05  # Penalty for meta-bad entries
    holding_cost: float = 0.001  # Small cost per step in position

    # Checkpointing
    save_interval: int = 1000  # Save every N updates
    log_interval: int = 100  # Log every N updates

    @property
    def batch_count(self) -> int:
        """Number of mini-batches per epoch."""
        total_samples = self.num_envs * self.episode_length
        return total_samples // self.batch_size


@dataclass
class MetaLabelerConfig:
    """
    Configuration for the Meta-Labeler (XGBoost GPU or Transformer).
    The meta-labeler predicts P(success) for each candidate entry signal.
    """
    # Model type: "xgboost" or "transformer"
    model_type: str = "xgboost"

    # XGBoost GPU parameters
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 500
    xgb_tree_method: str = "hist"  # GPU-accelerated histogram method
    xgb_device: str = "cuda"
    xgb_early_stopping_rounds: int = 50
    xgb_eval_metric: str = "auc"

    # Transformer parameters (alternative to XGBoost)
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_seq_len: int = 20  # Lookback window
    transformer_dropout: float = 0.1

    # Training settings
    train_val_split: float = 0.2
    random_seed: int = 42

    # Feature columns (from data_factory)
    feature_columns: List[str] = field(default_factory=lambda: [
        'adx', 'rsi', 'atr_pct', 'bb_width', 'ema_diff', 'dist_fast', 'dist_slow'
    ])

    # Model paths
    model_save_path: str = "models/meta_labeler.json"
    transformer_save_path: str = "models/meta_labeler_transformer.pt"


@dataclass
class DataConfig:
    """
    Data loading and preprocessing configuration.
    """
    # Data paths
    data_dir: str = "/Users/williamsmith/Python_local_Mac/01_trading_strategies/strategy_research/data"

    # Default instruments and timeframes
    default_instrument: str = "AUDUSD"
    default_timeframe: str = "1H"
    available_timeframes: List[str] = field(default_factory=lambda: ["15M", "1H", "4H"])

    # Date ranges
    train_start: str = "2010-01-01"
    train_end: str = "2022-01-01"
    val_start: str = "2022-01-01"
    val_end: str = "2023-01-01"
    test_start: str = "2023-01-01"
    test_end: str = "2025-01-01"

    # Normalization
    normalize_features: bool = True
    clip_outliers: bool = True
    outlier_std: float = 5.0


def get_default_configs():
    """Return all default configurations."""
    return {
        "strategy": StrategyConfig(),
        "triple_barrier": TripleBarrierConfig(),
        "rl": RLConfig(),
        "meta_labeler": MetaLabelerConfig(),
        "data": DataConfig(),
    }
