"""
EMA BB V2 RL Strategy - PPO-based Exit Optimization

This package implements a reinforcement learning approach to optimal trade exits.
The classical EMA + BB + ADX entry logic is preserved, while exits are learned
by a PPO agent trained to maximize risk-adjusted returns.

Key Components:
- VectorizedExitEnv: GPU-optimized parallel environment
- ActorCritic: Policy network for exit decisions
- PPOTrainer: Training loop with W&B logging
- EMABBScalpV2RLStrategy: Strategy with RL exit integration
"""

from .strategy import EMABBScalpV2RLStrategy
from .config import PPOConfig, RewardConfig
from .model import ActorCritic
from .env import VectorizedExitEnv

__all__ = [
    'EMABBScalpV2RLStrategy',
    'PPOConfig',
    'RewardConfig',
    'ActorCritic',
    'VectorizedExitEnv',
]
