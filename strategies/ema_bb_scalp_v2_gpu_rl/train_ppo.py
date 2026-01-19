"""
Layer 3b: PPO Training Loop with GAE

This module implements the PPO (Proximal Policy Optimization) training loop
with Generalized Advantage Estimation (GAE).

Key optimizations for GPU:
1. Pre-allocated rollout buffers (no Python lists)
2. GAE computation on GPU
3. Mixed precision training (FP16)
4. torch.compile for model optimization

The trainer operates on the VectorizedMarket environment, processing
thousands of environments in parallel.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import time

from .config import RLConfig
from .gpu_env import VectorizedMarket
from .model import ActorCritic, create_actor_critic


class PPOTrainer:
    """
    PPO training loop optimized for GPU.

    Handles:
    - Rollout collection with pre-allocated buffers
    - GAE advantage estimation
    - PPO clipped surrogate loss
    - Mixed precision training
    - Checkpointing and logging
    """

    def __init__(
        self,
        env: VectorizedMarket,
        config: RLConfig,
        model: Optional[ActorCritic] = None
    ):
        """
        Initialize PPO trainer.

        Args:
            env: VectorizedMarket environment
            config: RLConfig with hyperparameters
            model: Optional pre-trained model
        """
        self.env = env
        self.config = config
        self.device = torch.device(config.device)

        # Initialize or use provided model
        if model is None:
            self.model = create_actor_critic(config, compile_model=config.use_torch_compile)
        else:
            self.model = model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )

        # Learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.total_updates,
            eta_min=config.learning_rate * 0.1
        )

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_mixed_precision)

        # Pre-allocate rollout buffers
        self._allocate_buffers()

        # Tracking
        self.global_step = 0
        self.update_count = 0

        print(f"PPOTrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Num envs: {config.num_envs}")
        print(f"  Episode length: {config.episode_length}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Mixed precision: {config.use_mixed_precision}")

    def _allocate_buffers(self):
        """Pre-allocate all rollout storage on GPU for zero-copy operations."""
        N = self.config.num_envs
        T = self.config.episode_length
        obs_dim = self.config.obs_dim

        # Observation buffer: (T, N, obs_dim)
        self.obs_buffer = torch.zeros(T, N, obs_dim, device=self.device)

        # Action buffer: (T, N)
        self.action_buffer = torch.zeros(T, N, dtype=torch.long, device=self.device)

        # Reward buffer: (T, N)
        self.reward_buffer = torch.zeros(T, N, device=self.device)

        # Done buffer: (T, N)
        self.done_buffer = torch.zeros(T, N, dtype=torch.bool, device=self.device)

        # Value buffer: (T, N)
        self.value_buffer = torch.zeros(T, N, device=self.device)

        # Log probability buffer: (T, N)
        self.logprob_buffer = torch.zeros(T, N, device=self.device)

        # GAE buffers
        self.advantage_buffer = torch.zeros(T, N, device=self.device)
        self.return_buffer = torch.zeros(T, N, device=self.device)

        print(f"Allocated {T * N * (obs_dim + 6) * 4 / 1e6:.1f} MB for rollout buffers")

    def collect_rollout(self, next_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Collect one rollout of experience from the environment.

        Args:
            next_obs: Initial observations (num_envs, obs_dim)

        Returns:
            dict with final_value and final_obs for GAE computation
        """
        obs = next_obs

        for t in range(self.config.episode_length):
            # Get action and value from policy (no gradient computation)
            with torch.no_grad():
                action, logprob, _, value = self.model.get_action_and_value(obs)

            # Step environment
            next_obs, reward, done, info = self.env.step(action)

            # Store in pre-allocated buffers (zero-copy)
            self.obs_buffer[t] = obs
            self.action_buffer[t] = action
            self.reward_buffer[t] = reward
            self.done_buffer[t] = done
            self.value_buffer[t] = value
            self.logprob_buffer[t] = logprob

            # Auto-reset done environments
            if done.any():
                next_obs = self.env.reset(env_mask=done)

            obs = next_obs
            self.global_step += self.config.num_envs

        # Compute final value for GAE bootstrap
        with torch.no_grad():
            _, _, _, final_value = self.model.get_action_and_value(obs)

        return {
            "final_value": final_value,
            "final_obs": obs
        }

    def compute_gae(self, final_value: torch.Tensor):
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE provides a balance between bias and variance in advantage estimation.
        Lambda=1 gives high variance (Monte Carlo), lambda=0 gives high bias (TD).

        Updates self.advantage_buffer and self.return_buffer in-place.
        """
        T = self.config.episode_length
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda

        # Initialize GAE to zero
        gae = torch.zeros(self.config.num_envs, device=self.device)

        # Backward iteration through rollout
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = final_value
                next_non_terminal = 1.0 - self.done_buffer[t].float()
            else:
                next_value = self.value_buffer[t + 1]
                next_non_terminal = 1.0 - self.done_buffer[t].float()

            # TD error: delta = r + gamma * V(s') * (1-done) - V(s)
            delta = (
                self.reward_buffer[t] +
                gamma * next_value * next_non_terminal -
                self.value_buffer[t]
            )

            # GAE: A = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
            gae = delta + gamma * gae_lambda * next_non_terminal * gae

            # Store advantage and return
            self.advantage_buffer[t] = gae
            self.return_buffer[t] = gae + self.value_buffer[t]

    def update(self) -> Dict[str, float]:
        """
        Perform PPO update on collected rollout.

        Uses mini-batch SGD with clipped surrogate loss.

        Returns:
            dict with loss metrics
        """
        T = self.config.episode_length
        N = self.config.num_envs
        batch_size = self.config.batch_size
        n_epochs = self.config.n_epochs
        total_samples = T * N

        # Flatten batches for SGD
        b_obs = self.obs_buffer.reshape(total_samples, -1)
        b_actions = self.action_buffer.reshape(total_samples)
        b_logprobs = self.logprob_buffer.reshape(total_samples)
        b_advantages = self.advantage_buffer.reshape(total_samples)
        b_returns = self.return_buffer.reshape(total_samples)
        b_values = self.value_buffer.reshape(total_samples)

        # Normalize advantages (improves training stability)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Tracking metrics
        total_pg_loss = 0
        total_v_loss = 0
        total_entropy = 0
        total_clipfrac = 0
        n_updates = 0

        for epoch in range(n_epochs):
            # Shuffle indices for mini-batches
            indices = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    _, newlogprob, entropy, newvalue = self.model.get_action_and_value(
                        b_obs[mb_idx],
                        b_actions[mb_idx]
                    )

                    # Compute policy ratio
                    logratio = newlogprob - b_logprobs[mb_idx]
                    ratio = logratio.exp()

                    # Clipped surrogate loss
                    mb_advantages = b_advantages[mb_idx]
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss (clipped)
                    v_loss_unclipped = (newvalue - b_returns[mb_idx]) ** 2
                    v_clipped = b_values[mb_idx] + torch.clamp(
                        newvalue - b_values[mb_idx],
                        -self.config.clip_epsilon,
                        self.config.clip_epsilon
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_idx]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    # Entropy loss (encourages exploration)
                    entropy_loss = entropy.mean()

                    # Total loss
                    loss = (
                        pg_loss +
                        self.config.value_coef * v_loss -
                        self.config.entropy_coef * entropy_loss
                    )

                # Backward pass with mixed precision
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Track metrics
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy_loss.item()
                total_clipfrac += clipfrac.item()
                n_updates += 1

        # Learning rate scheduling
        self.scheduler.step()

        self.update_count += 1

        return {
            "pg_loss": total_pg_loss / n_updates,
            "v_loss": total_v_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "clipfrac": total_clipfrac / n_updates,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train(
        self,
        total_updates: Optional[int] = None,
        save_dir: str = "models",
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Main training loop.

        Args:
            total_updates: Number of PPO updates (default from config)
            save_dir: Directory for checkpoints
            verbose: Whether to print progress

        Returns:
            dict with training history
        """
        total_updates = total_updates or self.config.total_updates
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        history = {
            "pg_loss": [],
            "v_loss": [],
            "entropy": [],
            "avg_reward": [],
            "avg_episode_pnl": [],
        }

        # Initial reset
        next_obs = self.env.reset()

        start_time = time.time()
        pbar = tqdm(range(total_updates), desc="Training PPO")

        for update in pbar:
            # Collect rollout
            rollout_info = self.collect_rollout(next_obs)
            next_obs = rollout_info["final_obs"]

            # Compute GAE
            self.compute_gae(rollout_info["final_value"])

            # PPO update
            metrics = self.update()

            # Compute episode statistics
            avg_reward = self.reward_buffer.sum(dim=0).mean().item()
            avg_episode_pnl = self.reward_buffer.sum(dim=0).mean().item()

            # Store history
            history["pg_loss"].append(metrics["pg_loss"])
            history["v_loss"].append(metrics["v_loss"])
            history["entropy"].append(metrics["entropy"])
            history["avg_reward"].append(avg_reward)
            history["avg_episode_pnl"].append(avg_episode_pnl)

            # Update progress bar
            pbar.set_postfix({
                "pg": f"{metrics['pg_loss']:.3f}",
                "v": f"{metrics['v_loss']:.3f}",
                "ent": f"{metrics['entropy']:.3f}",
                "rew": f"{avg_reward:.2f}",
            })

            # Logging
            if verbose and (update + 1) % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                sps = self.global_step / elapsed
                print(f"\nUpdate {update + 1}/{total_updates}")
                print(f"  Steps: {self.global_step:,} ({sps:.0f} SPS)")
                print(f"  PG Loss: {metrics['pg_loss']:.4f}")
                print(f"  V Loss: {metrics['v_loss']:.4f}")
                print(f"  Entropy: {metrics['entropy']:.4f}")
                print(f"  Clip Frac: {metrics['clipfrac']:.3f}")
                print(f"  Avg Reward: {avg_reward:.4f}")
                print(f"  LR: {metrics['lr']:.2e}")

            # Checkpointing
            if (update + 1) % self.config.save_interval == 0:
                self.save(save_dir / f"ppo_checkpoint_{update + 1}.pt")

        # Final save
        self.save(save_dir / "ppo_final.pt")

        elapsed = time.time() - start_time
        print(f"\nTraining complete!")
        print(f"  Total time: {elapsed / 60:.1f} minutes")
        print(f"  Total steps: {self.global_step:,}")
        print(f"  Steps/second: {self.global_step / elapsed:.0f}")

        return history

    def save(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "update_count": self.update_count,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.update_count = checkpoint["update_count"]

        print(f"Checkpoint loaded: {path}")
        print(f"  Resumed from step {self.global_step}")


def evaluate_policy(
    model: ActorCritic,
    env: VectorizedMarket,
    n_episodes: int = 100,
    deterministic: bool = True
) -> Dict[str, float]:
    """
    Evaluate trained policy.

    Args:
        model: Trained ActorCritic model
        env: VectorizedMarket environment
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions

    Returns:
        dict with evaluation metrics
    """
    model.eval()

    total_rewards = []
    total_trades = []
    total_pnl = []

    episodes_done = 0
    obs = env.reset()

    while episodes_done < n_episodes:
        with torch.no_grad():
            action, _, _, _ = model.get_action_and_value(obs, deterministic=deterministic)

        obs, reward, done, info = env.step(action)

        if done.any():
            # Record metrics for completed episodes
            done_idx = done.nonzero(as_tuple=True)[0]
            for idx in done_idx:
                total_rewards.append(info["episode_pnl"][idx].item())
                total_trades.append(info["total_trades"][idx].item())

            episodes_done += done.sum().item()
            obs = env.reset(env_mask=done)

    model.train()

    return {
        "mean_reward": sum(total_rewards) / len(total_rewards),
        "std_reward": torch.tensor(total_rewards).std().item(),
        "mean_trades": sum(total_trades) / len(total_trades),
        "n_episodes": len(total_rewards),
    }
