"""
V6 Anti-Gaming Experiment

FROZEN SNAPSHOT - Do not modify this experiment after training begins.

V6 ANTI-GAMING: Fix V5's "breakeven farming" gaming behavior.

V5 Problem: Agent exits 71.6% of trades at breakeven, never lets winners run (0.2% profit exits)

V6 Solution:
1. ASYMMETRIC BREAKEVEN REWARDS: Reward recovering losers to scratch, penalize giving back winners
   - recovery_bonus: 0.2 (reward scratch after loss)
   - giveback_penalty: 0.3 (penalize scratch after profit)
   - breakeven_band: 0.001 (10 pips)

2. MFE DECAY PENALTY: Penalize giving back too much of maximum favorable excursion
   - mfe_decay_coef: 0.4
   - min_mfe_for_decay: 0.001 (10 pips)

3. TIGHTEN COOLDOWN: Prevent TIGHTEN_SL spam
   - tighten_cooldown: 2 bars

4. MODERATE INCREASES from V5:
   - regret_coef: 0.5 (was 0.4)
   - min_exit_bar: 2 (was 1)
   - min_trail_bar: 2 (was 1)
   - min_profit_for_trail: 0.0005 (was 0.0002, ~5 pips vs 2 pips)

Design Goal: Force the model to let winners run instead of farming breakeven exits.

Target improvements over V5:
- Giveback exits: 71.6% -> <40%
- Profit exits: 0.2% -> >20%
- Avg hold time: 5.2 bars -> 8-15 bars
- TIGHTEN usage: 47.9% -> 30-40%

Data Split:
- Train: 2005-2021 (5,638 trades)
- Test OOS: 2022-2025 (1,326 trades)

Training: 15M timesteps on RTX 4090 with 256 parallel environments.
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

__version__ = "6.0.0"
__experiment__ = "v6_anti_gaming"
