"""
EMA BB Scalp V2 GPU RL Strategy

GPU-optimized Meta-Labeling + Reinforcement Learning system for the EMA BB Scalp strategy.
Designed for training on 4096+ parallel environments using CUDA.

Components:
- data_factory: Vectorized feature generation + Triple Barrier labeling
- gpu_env: VectorizedMarket CUDA environment
- model: ActorCritic neural network
- meta_labeler: XGBoost GPU for entry quality prediction
- train_ppo: PPO training loop with GAE
"""

from .config import StrategyConfig, TripleBarrierConfig, RLConfig, MetaLabelerConfig

__version__ = "0.1.0"
__all__ = [
    "StrategyConfig",
    "TripleBarrierConfig",
    "RLConfig",
    "MetaLabelerConfig",
]
