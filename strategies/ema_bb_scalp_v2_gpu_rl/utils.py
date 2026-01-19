"""
Utility Functions for GPU-Optimized Meta-Labeling + RL

This module provides helper functions for:
- GPU memory management
- Logging and visualization
- Model serialization
- Performance benchmarking
"""

import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable
from contextlib import contextmanager


# =============================================================================
# GPU Memory Management
# =============================================================================

def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    total = props.total_memory / 1e9
    allocated = torch.cuda.memory_allocated(device) / 1e9
    cached = torch.cuda.memory_reserved(device) / 1e9
    free = total - cached

    return {
        "total_gb": total,
        "allocated_gb": allocated,
        "cached_gb": cached,
        "free_gb": free,
        "utilization_pct": (allocated / total) * 100
    }


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@contextmanager
def gpu_memory_tracker(label: str = ""):
    """Context manager to track GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated() / 1e6

    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        end_mem = torch.cuda.memory_allocated() / 1e6
        delta = end_mem - start_mem
        print(f"{label}: {delta:+.1f} MB (total: {end_mem:.1f} MB)")


# =============================================================================
# Timing Utilities
# =============================================================================

class Timer:
    """Simple timer for benchmarking."""

    def __init__(self):
        self.times = {}
        self.start_times = {}

    def start(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_times[name]

        if name not in self.times:
            self.times[name] = []
        self.times[name].append(elapsed)

        return elapsed

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary."""
        summary = {}
        for name, times in self.times.items():
            summary[name] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "total": np.sum(times),
                "count": len(times)
            }
        return summary


@contextmanager
def timed(name: str = ""):
    """Context manager for timing code blocks."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed*1000:.2f} ms")


# =============================================================================
# Performance Benchmarking
# =============================================================================

def benchmark_env_throughput(
    env,
    n_steps: int = 1000,
    warmup_steps: int = 100
) -> Dict[str, float]:
    """
    Benchmark environment step throughput.

    Returns steps per second and other metrics.
    """
    # Warmup
    obs = env.reset()
    for _ in range(warmup_steps):
        actions = torch.randint(0, 4, (env.num_envs,), device=env.device)
        obs, _, done, _ = env.step(actions)
        if done.any():
            obs = env.reset(env_mask=done)

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(n_steps):
        actions = torch.randint(0, 4, (env.num_envs,), device=env.device)
        obs, _, done, _ = env.step(actions)
        if done.any():
            obs = env.reset(env_mask=done)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_env_steps = n_steps * env.num_envs
    steps_per_second = total_env_steps / elapsed

    return {
        "total_env_steps": total_env_steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps_per_second,
        "steps_per_second_per_env": steps_per_second / env.num_envs
    }


def benchmark_model_inference(
    model,
    obs_dim: int,
    batch_sizes: List[int] = [256, 1024, 4096, 16384],
    n_iters: int = 100,
    device: str = "cuda"
) -> Dict[int, Dict[str, float]]:
    """
    Benchmark model inference speed at different batch sizes.
    """
    model.eval()
    results = {}

    for batch_size in batch_sizes:
        obs = torch.randn(batch_size, obs_dim, device=device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(obs)

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(n_iters):
            with torch.no_grad():
                _ = model(obs)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        samples_per_second = (batch_size * n_iters) / elapsed
        latency_ms = (elapsed / n_iters) * 1000

        results[batch_size] = {
            "samples_per_second": samples_per_second,
            "latency_ms": latency_ms,
            "elapsed_seconds": elapsed
        }

    model.train()
    return results


# =============================================================================
# Logging and Visualization
# =============================================================================

def print_training_summary(history: Dict[str, List[float]], last_n: int = 100):
    """Print summary of training history."""
    print("\nTraining Summary (last {} updates):".format(last_n))
    print("-" * 50)

    for key, values in history.items():
        if len(values) > 0:
            recent = values[-last_n:] if len(values) >= last_n else values
            mean = np.mean(recent)
            std = np.std(recent)
            min_val = np.min(recent)
            max_val = np.max(recent)
            print(f"{key:20s}: {mean:10.4f} +/- {std:.4f} (min: {min_val:.4f}, max: {max_val:.4f})")


def save_training_history(history: Dict[str, List[float]], path: str):
    """Save training history to numpy file."""
    np.savez(path, **{k: np.array(v) for k, v in history.items()})
    print(f"Training history saved to: {path}")


def load_training_history(path: str) -> Dict[str, np.ndarray]:
    """Load training history from numpy file."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


# =============================================================================
# Model Utilities
# =============================================================================

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable
    }


def model_summary(model: torch.nn.Module, input_shape: tuple = None):
    """Print model architecture summary."""
    print(f"\nModel: {model.__class__.__name__}")
    print("-" * 60)

    for name, param in model.named_parameters():
        print(f"{name:40s} {str(list(param.shape)):20s} {param.numel():>10,d}")

    params = count_parameters(model)
    print("-" * 60)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")


# =============================================================================
# Data Utilities
# =============================================================================

def compute_running_stats(
    values: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    count: int,
    momentum: float = 0.99
) -> tuple:
    """
    Update running mean and variance (for normalization).

    Uses Welford's online algorithm for numerical stability.
    """
    batch_mean = values.mean(dim=0)
    batch_var = values.var(dim=0)
    batch_count = values.shape[0]

    # Update running stats
    total_count = count + batch_count

    delta = batch_mean - running_mean
    new_mean = running_mean + delta * batch_count / total_count

    m_a = running_var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta ** 2 * count * batch_count / total_count
    new_var = M2 / total_count

    return new_mean, new_var, total_count


# =============================================================================
# Environment Utilities
# =============================================================================

def compute_returns(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """Compute discounted returns from rewards."""
    returns = torch.zeros_like(rewards)
    running_return = 0

    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


def normalize_rewards(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize rewards to zero mean and unit variance."""
    return (rewards - rewards.mean()) / (rewards.std() + eps)
