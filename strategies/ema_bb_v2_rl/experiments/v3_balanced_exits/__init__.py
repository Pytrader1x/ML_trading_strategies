"""
v3_balanced_exits experiment.

Balanced counterfactual rewards to prevent policy collapse.

Key Changes from v2:
- REDUCED regret_coef (0.3 vs 2.0)
- ASYMMETRIC coefficients (defensive > regret)
- BOUNDED counterfactual window (8 bars vs 20)
- RESTORED time penalty (0.005 vs 0.002)
- NO hold bonus
- EXIT bonus for decisive action
- Diversity loss to force minimum action probability
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
