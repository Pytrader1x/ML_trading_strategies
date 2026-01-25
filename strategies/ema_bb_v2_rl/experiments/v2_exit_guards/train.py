#!/usr/bin/env python3
"""
Training script for v2_exit_guards experiment.

This script uses the FROZEN config from this experiment directory.
Run from this directory or specify full paths.

Usage:
    cd experiments/v2_exit_guards
    python train.py --wandb

    # Or from strategy root:
    python experiments/v2_exit_guards/train.py --wandb
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

# Import from THIS experiment's frozen config
EXPERIMENT_DIR = Path(__file__).parent
STRATEGY_DIR = EXPERIMENT_DIR.parent.parent

# Add experiment dir to path for local imports
sys.path.insert(0, str(EXPERIMENT_DIR))
from config import PPOConfig, Actions, RewardConfig

# Import shared modules from strategy root
sys.path.insert(0, str(STRATEGY_DIR))
from model import ActorCritic, RolloutBuffer
from env import VectorizedExitEnv, EpisodeDataset, TradeEpisode


class PPOTrainer:
    """PPO Trainer with v2 exit guards configuration."""

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

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=config.n_steps,
            n_envs=config.n_envs,
            state_dim=config.state_dim,
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
            run_name = run_name or f"v2_exit_guards_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=run_name,
                config={
                    'experiment': 'v2_exit_guards',
                    'n_envs': config.n_envs,
                    'n_steps': config.n_steps,
                    'learning_rate': config.learning_rate,
                    'gamma': config.gamma,
                    'gae_lambda': config.gae_lambda,
                    'clip_epsilon': config.clip_epsilon,
                    'entropy_coef_start': config.entropy_coef_start,
                    'entropy_coef_end': config.entropy_coef_end,
                    'total_timesteps': config.total_timesteps,
                    'hidden_dims': config.hidden_dims,
                    # V2 specific
                    'time_coef': config.reward.time_coef,
                    'regret_coef': config.reward.regret_coef,
                    'min_profit_for_exit': config.reward.min_profit_for_exit,
                    'min_bars_for_exit': config.reward.min_bars_for_exit,
                    'min_profit_for_partial': config.reward.min_profit_for_partial,
                    'min_bars_for_partial': config.reward.min_bars_for_partial,
                    'invalid_action_penalty': config.reward.invalid_action_penalty,
                },
            )
            wandb.watch(policy, log='gradients', log_freq=100)

    def train(self) -> Dict[str, list]:
        """Main training loop."""
        config = self.config
        num_updates = config.total_timesteps // (config.n_steps * config.n_envs)

        print(f"\n{'=' * 60}")
        print(f" V2_EXIT_GUARDS Training - {config.total_timesteps:,} timesteps")
        print(f"{'=' * 60}")
        print(f"  Device: {config.device}")
        print(f"  Envs: {config.n_envs}")
        print(f"  Steps per rollout: {config.n_steps}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Updates: {num_updates}")
        print(f"  W&B: {'Enabled' if self.use_wandb else 'Disabled'}")
        print(f"\n  --- V2 Key Changes ---")
        print(f"  EXIT guards: min_profit={config.reward.min_profit_for_exit}, min_bars={config.reward.min_bars_for_exit}")
        print(f"  PARTIAL guards: min_profit={config.reward.min_profit_for_partial}, min_bars={config.reward.min_bars_for_partial}")
        print(f"  time_coef: {config.reward.time_coef} (v1 was 0.005)")
        print(f"  regret_coef: {config.reward.regret_coef} (v1 was 0.5)")
        print(f"  entropy decay: {config.entropy_coef_start} -> {config.entropy_coef_end} over {config.entropy_anneal_steps:,} steps")
        print(f"{'=' * 60}\n")

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

            # Collect Rollout
            self.policy.eval()
            with torch.no_grad():
                for step in range(config.n_steps):
                    action, log_prob, value = self.policy.get_action(state)
                    next_state, reward, done, info = self.env.step(action)

                    self.buffer.add(state, action, reward, done, log_prob, value)

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

            # PPO Update
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
        """Perform PPO update with multiple epochs."""
        config = self.config

        policy_losses = []
        value_losses = []
        entropy_values = []
        clip_fractions = []
        approx_kls = []

        for epoch in range(config.n_epochs):
            for batch in self.buffer.get_batches(config.batch_size):
                states = batch['states']
                actions = batch['actions']
                old_log_probs = batch['log_probs']
                returns = batch['returns']
                advantages = batch['advantages']
                old_values = batch['values']

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Forward pass
                log_probs, entropy, values = self.policy.evaluate_actions(states, actions)

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

                # Total loss
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    + entropy_coef * entropy_loss
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
        """Log training progress."""
        recent_returns = self.episode_returns[-100:] if self.episode_returns else [0]
        recent_lengths = self.episode_lengths[-100:] if self.episode_lengths else [0]

        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        mean_length = np.mean(recent_lengths)

        print(f"Update {update:5d}/{total_updates} | Steps: {self.total_steps:,}")
        print(f"  Return: {mean_return:8.3f} +/- {std_return:6.3f}")
        print(f"  Length: {mean_length:6.1f} | LR: {lr:.2e} | Ent: {entropy_coef:.4f}")
        print(f"  Loss: {metrics['policy_loss']:.4f} | KL: {metrics['approx_kl']:.4f} | Clip: {metrics['clip_fraction']:.3f}")
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
                'train/learning_rate': lr,
                'train/entropy_coef': entropy_coef,
                'train/total_steps': self.total_steps,
            }, step=self.total_steps)

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
            'experiment': 'v2_exit_guards',
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
    """Evaluate trained policy."""
    policy.eval()

    returns = []
    lengths = []
    action_counts = np.zeros(5)

    state = env.reset()
    episode_count = 0

    while episode_count < n_episodes:
        with torch.no_grad():
            action, _, _ = policy.get_action(state, deterministic=True)

        action_counts[action.cpu().numpy()] += 1
        state, _, _, info = env.step(action)

        if info['episode_returns']:
            returns.extend(info['episode_returns'])
            lengths.extend(info['episode_lengths'])
            episode_count = len(returns)

    returns = np.array(returns[:n_episodes])
    lengths = np.array(lengths[:n_episodes])

    metrics = {
        'mean_return': float(returns.mean()),
        'std_return': float(returns.std()),
        'sharpe': float(returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)),
        'win_rate': float((returns > 0).mean()),
        'mean_length': float(lengths.mean()),
        'action_distribution': action_counts / action_counts.sum(),
    }

    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS - v2_exit_guards")
    print("=" * 60)
    print(f"  Mean Return: {metrics['mean_return']:.4f} +/- {metrics['std_return']:.4f}")
    print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"  Win Rate: {metrics['win_rate'] * 100:.1f}%")
    print(f"  Avg Length: {metrics['mean_length']:.1f} bars")
    print(f"  Action Distribution:")
    for i, name in enumerate(Actions.NAMES):
        print(f"    {name}: {metrics['action_distribution'][i] * 100:.1f}%")
    print("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train v2_exit_guards RL Exit Policy")
    parser.add_argument('--timesteps', type=int, default=10_000_000, help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=64, help='Parallel environments')
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
