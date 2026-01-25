"""
v4_no_lookahead - Anti-Lookahead Bias Experiment

Fixes the look-ahead bias discovered in v3 where the model learned to exit
at bar 0/1 to capture "free" profits visible in the market_tensor.

Key Changes from v3:
1. Action masking: EXIT and PARTIAL blocked until min_exit_bar (default 3)
2. Proper Sharpe calculation accounting for trade duration
3. GPU optimization for RTX 4090 (256 envs, 8192 batch)
4. Monitoring for blocked exit attempts
"""

from .config import PPOConfig, RewardConfig, Actions
from .env import VectorizedExitEnv, EpisodeDataset, TradeEpisode

__all__ = [
    'PPOConfig',
    'RewardConfig',
    'Actions',
    'VectorizedExitEnv',
    'EpisodeDataset',
    'TradeEpisode',
]
