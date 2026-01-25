#!/usr/bin/env python3
"""
Training script for v4_no_lookahead experiment.

KEY CHANGES from v3:
1. ACTION MASKING: Block EXIT/PARTIAL until min_exit_bar (default 3)
2. Track blocked exit attempts (lookahead exploitation)
3. GPU optimization for RTX 4090 (256 envs, 8192 batch)
4. Larger network [512, 256]
5. More training: 15M timesteps

Usage:
    cd experiments/v4_no_lookahead
    python train.py --wandb

    # Or from strategy root:
    python experiments/v4_no_lookahead/train.py --wandb
"""

import sys
import argparse
import pickle
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Import from THIS experiment's frozen config and env
EXPERIMENT_DIR = Path(__file__).parent
STRATEGY_DIR = EXPERIMENT_DIR.parent.parent

# Add experiment dir FIRST - for local config and env
sys.path.insert(0, str(EXPERIMENT_DIR))
from config import PPOConfig, Actions, RewardConfig
from env import VectorizedExitEnv, EpisodeDataset, TradeEpisode

# Import shared modules from strategy root (model only)
sys.path.insert(0, str(STRATEGY_DIR))
from model import ActorCritic, RolloutBuffer


class PPOTrainer:
    """PPO Trainer with v4 action masking (anti-lookahead) configuration.

    KEY v4 ADDITIONS:
    - Action masking: EXIT/PARTIAL blocked until min_exit_bar
    - Track blocked exit attempts
    - GPU optimization for RTX 4090
    """

    def __init__(
        self,
        env: VectorizedExitEnv,
        policy: ActorCritic,
        config: PPOConfig,
        use_wandb: bool = False,
        run_name: Optional[str] = None,
    ):
        self.env = env
        self.policy = policy.to(config.device)
        self.config = config
        self.device = torch.device(config.device)
        self.use_wandb = use_wandb and HAS_WANDB

        # Optimizer
        self.optimizer = optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )

        # Rollout buffer with action masking support
        self.buffer = RolloutBuffer(
            buffer_size=config.n_steps,
            n_envs=config.n_envs,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            device=config.device,
        )

        # Training state
        self.total_steps = 0
        self.update_count = 0
        self.best_return = float('-inf')

        # Metrics
        self.episode_returns = []
        self.episode_lengths = []

        # Models directory (this experiment's models/)
        self._models_dir = EXPERIMENT_DIR / "models"

        # Initialize W&B
        if self.use_wandb:
            run_name = run_name or f"v4_no_lookahead_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=run_name,
                config={
                    'experiment': 'v4_no_lookahead',
                    'n_envs': config.n_envs,
                    'n_steps': config.n_steps,
                    'batch_size': config.batch_size,
                    'learning_rate': config.learning_rate,
                    'gamma': config.gamma,
                    'gae_lambda': config.gae_lambda,
                    'clip_epsilon': config.clip_epsilon,
                    'entropy_coef_start': config.entropy_coef_start,
                    'entropy_coef_end': config.entropy_coef_end,
                    'total_timesteps': config.total_timesteps,
                    'hidden_dims': config.hidden_dims,
                    # V4 anti-lookahead
                    'min_exit_bar': config.reward.min_exit_bar,
                    'use_action_masking': config.reward.use_action_masking,
                    # Counterfactual rewards
                    'time_coef': config.reward.time_coef,
                    'regret_coef': config.reward.regret_coef,
                    'defensive_coef': config.reward.defensive_coef,
                    'counterfactual_window': config.reward.counterfactual_window,
                    # Diversity loss
                    'min_action_prob': config.reward.min_action_prob,
                    'diversity_coef': config.reward.diversity_coef,
                },
            )
            wandb.watch(policy, log='gradients', log_freq=100)

    def train(self) -> Dict[str, list]:
        """Main training loop with action masking."""
        config = self.config
        num_updates = config.total_timesteps // (config.n_steps * config.n_envs)

        print(f"\n{'=' * 70}")
        print(f" V4_NO_LOOKAHEAD Training - {config.total_timesteps:,} timesteps")
        print(f"{'=' * 70}")
        print(f"  Device: {config.device}")
        print(f"  Envs: {config.n_envs}")
        print(f"  Steps per rollout: {config.n_steps}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Updates: {num_updates}")
        print(f"  W&B: {'Enabled' if self.use_wandb else 'Disabled'}")
        print(f"\n  --- V4 Anti-Lookahead Protection ---")
        print(f"  min_exit_bar: {config.reward.min_exit_bar} (EXIT/PARTIAL blocked until bar {config.reward.min_exit_bar})")
        print(f"  use_action_masking: {config.reward.use_action_masking}")
        print(f"\n  --- V3 Counterfactual Rewards (kept) ---")
        print(f"  defensive_coef: {config.reward.defensive_coef}")
        print(f"  regret_coef: {config.reward.regret_coef}")
        print(f"  counterfactual_window: {config.reward.counterfactual_window} bars")
        print(f"\n  --- GPU Optimization (RTX 4090) ---")
        print(f"  n_envs: {config.n_envs} (up from 128)")
        print(f"  batch_size: {config.batch_size} (up from 4096)")
        print(f"  hidden_dims: {config.hidden_dims}")
        print(f"{'=' * 70}\n")

        state = self.env.reset()
        metrics_history = defaultdict(list)

        for update in range(num_updates):
            # Learning Rate Schedule
            frac = 1.0 - update / num_updates
            lr = config.learning_rate * frac if config.lr_schedule == "linear" else config.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Entropy Annealing
            entropy_coef = self._get_entropy_coef()

            # Collect Rollout with ACTION MASKING (v4)
            self.policy.eval()
            with torch.no_grad():
                for step in range(config.n_steps):
                    # V4: Get action mask from environment
                    action_mask = self.env.get_action_mask()

                    # V4: Pass mask to policy for proper sampling
                    action, log_prob, value = self.policy.get_action(
                        state,
                        action_mask=action_mask,
                    )
                    next_state, reward, done, info = self.env.step(action)

                    # V4: Store action mask in buffer for PPO update
                    self.buffer.add(
                        state, action, reward, done, log_prob, value,
                        action_mask=action_mask,
                    )

                    state = next_state
                    self.total_steps += config.n_envs

                    if info['episode_returns']:
                        self.episode_returns.extend(info['episode_returns'])
                        self.episode_lengths.extend(info['episode_lengths'])

                # Bootstrap value
                _, _, last_value = self.policy.get_action(state)

            # Compute returns and advantages
            self.buffer.compute_returns_and_advantages(
                last_value, config.gamma, config.gae_lambda
            )

            # PPO Update with DIVERSITY LOSS and ACTION MASKING
            self.policy.train()
            update_metrics = self._ppo_update(entropy_coef)

            # Store metrics
            for k, v in update_metrics.items():
                metrics_history[k].append(v)

            self.update_count += 1
            self.buffer.reset()

            # Logging
            if update % config.log_interval == 0:
                self._log_progress(update, num_updates, update_metrics, lr, entropy_coef)

            # Save Checkpoint
            if update % config.save_interval == 0 and update > 0:
                self._save_checkpoint(update)

        # Final save
        self._save_checkpoint(num_updates, final=True)

        if self.use_wandb:
            wandb.finish()

        return dict(metrics_history)

    def _ppo_update(self, entropy_coef: float) -> Dict[str, float]:
        """Perform PPO update with action masking (v4).

        v4 ADDITION: Use stored action masks during evaluation.
        """
        config = self.config

        policy_losses = []
        value_losses = []
        entropy_values = []
        clip_fractions = []
        approx_kls = []
        diversity_losses = []

        for epoch in range(config.n_epochs):
            for batch in self.buffer.get_batches(config.batch_size):
                states = batch['states']
                actions = batch['actions']
                old_log_probs = batch['log_probs']
                returns = batch['returns']
                advantages = batch['advantages']
                old_values = batch['values']
                action_masks = batch['action_masks']  # V4: Retrieved from buffer

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Forward pass with action mask (v4)
                log_probs, entropy, values = self.policy.evaluate_actions(
                    states, actions,
                    action_mask=action_masks,  # V4: Pass mask for consistent log_prob
                )

                # Policy loss (PPO-Clip)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                values_clipped = old_values + torch.clamp(
                    values - old_values, -config.clip_epsilon, config.clip_epsilon
                )
                value_loss1 = (values - returns) ** 2
                value_loss2 = (values_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Diversity loss (from v3) - apply with mask
                action_probs = self.policy.get_action_probs(states, action_mask=action_masks)
                min_prob = config.reward.min_action_prob

                # Only penalize valid actions below minimum
                # For masked actions, their prob is 0 which is fine
                below_min = (action_probs < min_prob).float() * action_masks.float()
                diversity_loss = (below_min * (min_prob - action_probs)).sum(dim=1).mean()

                # Total loss
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    + entropy_coef * entropy_loss
                    + config.reward.diversity_coef * diversity_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), config.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_values.append(-entropy_loss.item())
                diversity_losses.append(diversity_loss.item())

                clip_frac = ((ratio - 1).abs() > config.clip_epsilon).float().mean().item()
                clip_fractions.append(clip_frac)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                approx_kls.append(approx_kl)

                # Early stopping on KL
                if approx_kl > config.target_kl:
                    break

            if approx_kl > config.target_kl:
                break

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_values),
            'clip_fraction': np.mean(clip_fractions),
            'approx_kl': np.mean(approx_kls),
            'diversity_loss': np.mean(diversity_losses),
        }

    def _get_entropy_coef(self) -> float:
        """Get annealed entropy coefficient."""
        config = self.config
        progress = min(1.0, self.total_steps / config.entropy_anneal_steps)
        return config.entropy_coef_start + progress * (config.entropy_coef_end - config.entropy_coef_start)

    def _log_progress(
        self,
        update: int,
        total_updates: int,
        metrics: Dict[str, float],
        lr: float,
        entropy_coef: float,
    ):
        """Log training progress with lookahead stats."""
        recent_returns = self.episode_returns[-100:] if self.episode_returns else [0]
        recent_lengths = self.episode_lengths[-100:] if self.episode_lengths else [0]

        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        mean_length = np.mean(recent_lengths)

        # Get counterfactual stats from env
        cf_stats = self.env.get_counterfactual_stats()
        action_dist = self.env.get_action_distribution()

        # V4: Get lookahead stats
        lookahead_stats = self.env.get_lookahead_stats()

        print(f"Update {update:5d}/{total_updates} | Steps: {self.total_steps:,}")
        print(f"  Return: {mean_return:8.3f} +/- {std_return:6.3f}")
        print(f"  Length: {mean_length:6.1f} | LR: {lr:.2e} | Ent: {entropy_coef:.4f}")
        print(f"  Loss: {metrics['policy_loss']:.4f} | KL: {metrics['approx_kl']:.4f} | Clip: {metrics['clip_fraction']:.3f}")
        print(f"  Diversity Loss: {metrics['diversity_loss']:.4f}")
        print(f"  Counterfactual: def_rate={cf_stats['defensive_rate']:.2%} prem_rate={cf_stats['premature_rate']:.2%}")
        print(f"  Lookahead: blocked_rate={lookahead_stats['blocked_rate']:.2%} "
              f"blocked={lookahead_stats['blocked_exits']:.0f} total_exits={lookahead_stats['total_exit_attempts']:.0f}")
        print(f"  Actions: HOLD={action_dist['HOLD']:.1%} EXIT={action_dist['EXIT']:.1%} "
              f"TIGHTEN={action_dist['TIGHTEN_SL']:.1%} TRAIL={action_dist['TRAIL_BE']:.1%} "
              f"PARTIAL={action_dist['PARTIAL']:.1%}")
        print()

        if self.use_wandb:
            wandb.log({
                'train/episode_return': mean_return,
                'train/episode_return_std': std_return,
                'train/episode_length': mean_length,
                'train/policy_loss': metrics['policy_loss'],
                'train/value_loss': metrics['value_loss'],
                'train/entropy': metrics['entropy'],
                'train/clip_fraction': metrics['clip_fraction'],
                'train/approx_kl': metrics['approx_kl'],
                'train/diversity_loss': metrics['diversity_loss'],
                'train/learning_rate': lr,
                'train/entropy_coef': entropy_coef,
                'train/total_steps': self.total_steps,
                # V4 lookahead stats
                'lookahead/blocked_rate': lookahead_stats['blocked_rate'],
                'lookahead/blocked_exits': lookahead_stats['blocked_exits'],
                'lookahead/total_exit_attempts': lookahead_stats['total_exit_attempts'],
                # Counterfactual stats
                'counterfactual/defensive_rate': cf_stats['defensive_rate'],
                'counterfactual/premature_rate': cf_stats['premature_rate'],
                'counterfactual/avg_defensive_bonus': cf_stats['avg_defensive_bonus'],
                'counterfactual/avg_opportunity_cost': cf_stats['avg_opportunity_cost'],
                'counterfactual/total_exits': cf_stats['total_exits'],
                # Action distribution
                'actions/HOLD': action_dist['HOLD'],
                'actions/EXIT': action_dist['EXIT'],
                'actions/TIGHTEN_SL': action_dist['TIGHTEN_SL'],
                'actions/TRAIL_BE': action_dist['TRAIL_BE'],
                'actions/PARTIAL': action_dist['PARTIAL'],
            }, step=self.total_steps)

        # V4: Check for lookahead exploitation warning
        if mean_length < 5:
            print(f"  WARNING: Avg length {mean_length:.1f} < 5 bars - possible lookahead exploitation!")

        if mean_return > self.best_return and len(self.episode_returns) > 50:
            self.best_return = mean_return

    def _save_checkpoint(self, update: int, final: bool = False):
        """Save model checkpoint."""
        self._models_dir.mkdir(parents=True, exist_ok=True)

        if final:
            path = self._models_dir / "exit_policy_final.pt"
        else:
            path = self._models_dir / f"exit_policy_{update:06d}.pt"

        torch.save({
            'update': update,
            'total_steps': self.total_steps,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_return': self.best_return,
            'experiment': 'v4_no_lookahead',
            'min_exit_bar': self.config.reward.min_exit_bar,
        }, path)

        print(f"  Saved checkpoint: {path}")

        if self.use_wandb:
            wandb.save(str(path))


def load_episodes(episode_file: Path) -> list:
    """Load pre-computed episodes."""
    with open(episode_file, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        episodes = data['episodes']
        print(f"Loaded {data['n_episodes']} episodes for {data['instrument']} {data['timeframe']}")
    else:
        episodes = data
        print(f"Loaded {len(episodes)} episodes")

    return episodes


def evaluate_policy(
    policy: ActorCritic,
    env: VectorizedExitEnv,
    n_episodes: int = 500,
) -> Dict[str, float]:
    """Evaluate trained policy with anti-lookahead metrics."""
    policy.eval()

    returns = []
    lengths = []
    action_counts = np.zeros(5)
    early_exit_attempts = 0
    min_exit_bar = env.min_exit_bar

    state = env.reset()
    episode_count = 0

    while episode_count < n_episodes:
        with torch.no_grad():
            # V4: Use action mask during evaluation too
            action_mask = env.get_action_mask()
            action, _, _ = policy.get_action(state, deterministic=True, action_mask=action_mask)

        action_counts[action.cpu().numpy()] += 1
        state, _, _, info = env.step(action)

        if info['episode_returns']:
            returns.extend(info['episode_returns'])
            lengths.extend(info['episode_lengths'])
            episode_count = len(returns)

    returns = np.array(returns[:n_episodes])
    lengths = np.array(lengths[:n_episodes])

    # V4: Check for quick exits that might indicate lookahead
    quick_exits = (lengths <= min_exit_bar).sum()

    metrics = {
        'mean_return': float(returns.mean()),
        'std_return': float(returns.std()),
        'sharpe': float(returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)),
        'win_rate': float((returns > 0).mean()),
        'mean_length': float(lengths.mean()),
        'min_length': float(lengths.min()),
        'quick_exits': int(quick_exits),
        'quick_exit_rate': float(quick_exits / len(lengths)),
        'action_distribution': action_counts / action_counts.sum(),
    }

    print("\n" + "=" * 70)
    print(" EVALUATION RESULTS - v4_no_lookahead")
    print("=" * 70)
    print(f"  Mean Return: {metrics['mean_return']:.4f} +/- {metrics['std_return']:.4f}")
    print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"  Win Rate: {metrics['win_rate'] * 100:.1f}%")
    print(f"  Avg Length: {metrics['mean_length']:.1f} bars")
    print(f"  Min Length: {metrics['min_length']:.0f} bars (should be >= {min_exit_bar})")
    print(f"  Quick Exits (<= {min_exit_bar} bars): {metrics['quick_exits']} ({metrics['quick_exit_rate']*100:.1f}%)")
    print(f"  Action Distribution:")
    for i, name in enumerate(Actions.NAMES):
        print(f"    {name}: {metrics['action_distribution'][i] * 100:.1f}%")
    print("=" * 70)

    # V4: Warning if still exploiting
    if metrics['mean_length'] < 5:
        print("\n  WARNING: Avg length < 5 bars - model may still be exploiting lookahead!")
    if metrics['quick_exit_rate'] > 0.1:
        print(f"\n  WARNING: {metrics['quick_exit_rate']*100:.1f}% quick exits - check for lookahead bias!")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train v4_no_lookahead RL Exit Policy")
    parser.add_argument('--timesteps', type=int, default=15_000_000, help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=256, help='Parallel environments')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--run-name', type=str, default=None, help='W&B run name')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()

    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Load episodes from strategy data directory
    episode_file = STRATEGY_DIR / "data" / "episodes_train_2005_2021.pkl"
    if not episode_file.exists():
        print(f"Episode file not found: {episode_file}")
        print("Run extract_episodes.py first")
        sys.exit(1)

    episodes = load_episodes(episode_file)

    # Create config from THIS experiment's frozen config
    config = PPOConfig(
        n_envs=args.n_envs,
        total_timesteps=args.timesteps,
        device=device,
    )

    # Create dataset and environment
    print("Creating dataset...")
    dataset = EpisodeDataset(episodes, device=device)
    print(f"Dataset: {len(dataset)} episodes on {device}")

    env = VectorizedExitEnv(dataset, config)

    # Create policy (using shared model.py)
    policy = ActorCritic(config)

    # Train
    trainer = PPOTrainer(
        env, policy, config,
        use_wandb=args.wandb,
        run_name=args.run_name,
    )

    metrics = trainer.train()

    # Final evaluation
    evaluate_policy(policy, env, n_episodes=1000)

    # Save metrics
    results_dir = EXPERIMENT_DIR / "results" / "training"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = results_dir / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({k: [float(v) for v in vs] for k, vs in metrics.items()}, f)
    print(f"Saved training metrics to {metrics_file}")


if __name__ == "__main__":
    main()
