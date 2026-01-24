#!/usr/bin/env python3
"""
Experiment Version Manager for RL Exit Optimizer.

Manages versioned experiments for comparing different hyperparameter configurations.

Usage:
    python experiment_manager.py create v2_entropy --parent v1_baseline
    python experiment_manager.py list
    python experiment_manager.py compare v1_baseline v2_entropy
    python experiment_manager.py set-active v2_entropy
    python experiment_manager.py dashboard
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
import subprocess

import yaml

STRATEGY_DIR = Path(__file__).parent
EXPERIMENTS_DIR = STRATEGY_DIR / "experiments"
REGISTRY_FILE = EXPERIMENTS_DIR / "registry.json"


def load_registry() -> dict:
    """Load experiment registry."""
    if not REGISTRY_FILE.exists():
        return {
            "schema_version": "1.0",
            "active_version": None,
            "versions": {}
        }
    with open(REGISTRY_FILE) as f:
        return json.load(f)


def save_registry(registry: dict):
    """Save experiment registry."""
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=STRATEGY_DIR
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def create_version(name: str, parent: Optional[str] = None, description: str = "") -> Path:
    """
    Create a new experiment version directory.

    Args:
        name: Version name (e.g., 'v2_entropy')
        parent: Parent version to copy config from
        description: Brief description of the experiment

    Returns:
        Path to the new version directory
    """
    registry = load_registry()

    if name in registry['versions']:
        raise ValueError(f"Version '{name}' already exists")

    version_dir = EXPERIMENTS_DIR / name
    version_dir.mkdir(parents=True, exist_ok=False)

    # Create subdirectories
    (version_dir / "models").mkdir()
    (version_dir / "results" / "training").mkdir(parents=True)

    # Copy parent config or create from defaults
    if parent and parent in registry['versions']:
        parent_config = EXPERIMENTS_DIR / parent / "experiment.yaml"
        if parent_config.exists():
            with open(parent_config) as f:
                config = yaml.safe_load(f)
            config['meta']['version'] = name.split('_')[0] if '_' in name else name
            config['meta']['name'] = '_'.join(name.split('_')[1:]) if '_' in name else name
            config['meta']['description'] = description or f"Derived from {parent}"
            config['meta']['created'] = datetime.now().isoformat()
            config['meta']['git_commit'] = get_git_commit()
            config['meta']['status'] = "pending"
            config['changes_from_previous'] = [f"Forked from {parent}"]
        else:
            config = _create_default_config(name, description)
    else:
        config = _create_default_config(name, description)

    # Save experiment.yaml
    with open(version_dir / "experiment.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Create README
    readme = f"""# {name}

## Description

{description or 'New experiment version. Update this description.'}

## Changes from Previous Version

- [ ] Document your changes here

## Results

_Results will be populated after training and evaluation._

## Notes

_Add any observations or learnings here._
"""
    with open(version_dir / "README.md", 'w') as f:
        f.write(readme)

    # Update registry
    registry['versions'][name] = {
        "path": f"experiments/{name}",
        "status": "pending",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "description": description or "New experiment",
        "git_commit": get_git_commit(),
        "parent": parent
    }
    save_registry(registry)

    print(f"Created new experiment version: {name}")
    print(f"  Directory: {version_dir}")
    print(f"  Config: {version_dir / 'experiment.yaml'}")
    if parent:
        print(f"  Parent: {parent}")
    print("\nNext steps:")
    print(f"  1. Edit experiments/{name}/experiment.yaml with your changes")
    print(f"  2. Run: python train.py --version {name} --episodes data/episodes_train_2005_2021.pkl")
    print(f"  3. Run: python run.py -i AUDUSD -t 15M --version {name}")

    return version_dir


def _create_default_config(name: str, description: str) -> dict:
    """Create default experiment config."""
    return {
        'meta': {
            'version': name.split('_')[0] if '_' in name else name,
            'name': '_'.join(name.split('_')[1:]) if '_' in name else name,
            'description': description or 'New experiment',
            'created': datetime.now().isoformat(),
            'author': 'william',
            'git_commit': get_git_commit(),
            'status': 'pending'
        },
        'training': {
            'train_episodes': 'data/episodes_train_2005_2021.pkl',
            'test_episodes': 'data/episodes_test_2022_2025.pkl',
            'ppo': {
                'n_envs': 64,
                'n_steps': 2048,
                'learning_rate': 0.0003,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'target_kl': 0.015,
                'n_epochs': 10,
                'batch_size': 2048,
                'total_timesteps': 10000000,
                'hidden_dims': [256, 256],
                'entropy_coef_start': 0.05,
                'entropy_coef_end': 0.001,
            },
            'reward': {
                'w_realized': 1.0,
                'w_mtm': 0.1,
                'risk_coef': 0.3,
                'regret_coef': 0.5,
                'reward_scale': 100.0,
            },
            'device': 'cuda'
        },
        'evaluation': {
            'instruments': ['AUDUSD'],
            'timeframes': ['15M'],
        },
        'results_summary': {},
        'changes_from_previous': ['Initial version']
    }


def list_versions():
    """List all experiment versions."""
    registry = load_registry()

    print("\n" + "=" * 80)
    print(" EXPERIMENT VERSIONS")
    print("=" * 80)
    print(f"{'Name':<25} {'Status':<12} {'Created':<12} {'Sharpe':>8} {'Return':>10}")
    print("-" * 80)

    for name, info in registry['versions'].items():
        status = info.get('status', 'unknown')
        created = info.get('created', 'unknown')
        sharpe = info.get('sharpe', '-')
        ret = info.get('return_pct', '-')

        # Mark active version
        marker = "*" if name == registry.get('active_version') else " "

        sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else str(sharpe)
        ret_str = f"{ret:.2f}%" if isinstance(ret, (int, float)) else str(ret)

        print(f"{marker}{name:<24} {status:<12} {created:<12} {sharpe_str:>8} {ret_str:>10}")

    print("=" * 80)
    print(f"Active version: {registry.get('active_version', 'None')}")
    print(f"Total versions: {len(registry['versions'])}")


def compare_versions(v1: str, v2: str):
    """Compare two experiment versions."""
    registry = load_registry()

    if v1 not in registry['versions'] or v2 not in registry['versions']:
        missing = v1 if v1 not in registry['versions'] else v2
        raise ValueError(f"Version '{missing}' not found")

    info1 = registry['versions'][v1]
    info2 = registry['versions'][v2]

    # Load configs
    config1_path = EXPERIMENTS_DIR / v1 / "experiment.yaml"
    config2_path = EXPERIMENTS_DIR / v2 / "experiment.yaml"

    config1 = yaml.safe_load(open(config1_path)) if config1_path.exists() else {}
    config2 = yaml.safe_load(open(config2_path)) if config2_path.exists() else {}

    print("\n" + "=" * 80)
    print(f" COMPARISON: {v1} vs {v2}")
    print("=" * 80)

    # Results comparison
    print("\n[RESULTS]")
    print(f"{'Metric':<20} {v1:<15} {v2:<15} {'Diff':>12}")
    print("-" * 60)

    metrics = ['sharpe', 'return_pct', 'max_dd_pct', 'win_rate', 'total_trades']
    for metric in metrics:
        val1 = info1.get(metric, '-')
        val2 = info2.get(metric, '-')

        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = val2 - val1
            diff_str = f"{diff:+.2f}" if abs(diff) < 100 else f"{diff:+.0f}"
        else:
            diff_str = "-"

        val1_str = f"{val1:.2f}" if isinstance(val1, (int, float)) else str(val1)
        val2_str = f"{val2:.2f}" if isinstance(val2, (int, float)) else str(val2)

        print(f"{metric:<20} {val1_str:<15} {val2_str:<15} {diff_str:>12}")

    # Config differences
    print("\n[CONFIG CHANGES]")
    if config1 and config2:
        ppo1 = config1.get('training', {}).get('ppo', {})
        ppo2 = config2.get('training', {}).get('ppo', {})

        for key in set(ppo1.keys()) | set(ppo2.keys()):
            if ppo1.get(key) != ppo2.get(key):
                print(f"  {key}: {ppo1.get(key)} -> {ppo2.get(key)}")

        reward1 = config1.get('training', {}).get('reward', {})
        reward2 = config2.get('training', {}).get('reward', {})

        for key in set(reward1.keys()) | set(reward2.keys()):
            if reward1.get(key) != reward2.get(key):
                print(f"  reward.{key}: {reward1.get(key)} -> {reward2.get(key)}")

    print("=" * 80)


def set_active_version(name: str):
    """Set the active experiment version."""
    registry = load_registry()

    if name not in registry['versions']:
        raise ValueError(f"Version '{name}' not found")

    registry['active_version'] = name
    save_registry(registry)

    print(f"Active version set to: {name}")


def update_results(name: str):
    """Update registry with results from completed experiment."""
    registry = load_registry()

    if name not in registry['versions']:
        raise ValueError(f"Version '{name}' not found")

    version_dir = EXPERIMENTS_DIR / name

    # Look for backtest results
    results_pattern = version_dir / "results" / "*" / "*" / "backtest_results.json"
    import glob
    result_files = list(version_dir.glob("results/*/*/backtest_results.json"))

    if result_files:
        with open(result_files[0]) as f:
            results = json.load(f)

        registry['versions'][name].update({
            'sharpe': results.get('sharpe_ratio'),
            'return_pct': results.get('return_pct'),
            'max_dd_pct': abs(results.get('max_drawdown_pct', 0)),
            'win_rate': results.get('win_rate'),
            'total_trades': results.get('total_trades'),
            'status': 'completed'
        })

        # Also update experiment.yaml
        config_file = version_dir / "experiment.yaml"
        if config_file.exists():
            with open(config_file) as f:
                config = yaml.safe_load(f)
            config['meta']['status'] = 'completed'
            config['results_summary'] = {
                'sharpe_ratio': results.get('sharpe_ratio'),
                'return_pct': results.get('return_pct'),
                'max_drawdown_pct': abs(results.get('max_drawdown_pct', 0)),
                'win_rate': results.get('win_rate'),
                'total_trades': results.get('total_trades'),
            }
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        save_registry(registry)
        print(f"Updated results for {name}")
        print(f"  Sharpe: {results.get('sharpe_ratio'):.2f}")
        print(f"  Return: {results.get('return_pct'):.2f}%")
    else:
        print(f"No backtest results found for {name}")


def generate_dashboard():
    """Generate comparison dashboard markdown."""
    registry = load_registry()

    dashboard = f"""# Experiment Comparison Dashboard

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary

| Version | Status | Sharpe | Return % | Max DD % | Win Rate | Trades |
|---------|--------|--------|----------|----------|----------|--------|
"""

    completed = []
    for name, info in registry['versions'].items():
        status = info.get('status', 'unknown')
        sharpe = info.get('sharpe', '-')
        ret = info.get('return_pct', '-')
        dd = info.get('max_dd_pct', '-')
        wr = info.get('win_rate', '-')
        trades = info.get('total_trades', '-')

        def fmt(val, decimals=2):
            if isinstance(val, (int, float)):
                return f"{val:.{decimals}f}" if decimals else str(int(val))
            return str(val)

        dashboard += f"| {name} | {status} | {fmt(sharpe)} | {fmt(ret)}% | {fmt(dd)}% | {fmt(wr, 1)}% | {fmt(trades, 0)} |\n"

        if status == 'completed' and isinstance(sharpe, (int, float)):
            completed.append((name, sharpe))

    # Best performer
    if completed:
        best = max(completed, key=lambda x: x[1])
        dashboard += f"\n## Best Performing: {best[0]} (Sharpe: {best[1]:.2f})\n"

    # Change log
    dashboard += "\n## Change Log\n\n"
    for name, info in registry['versions'].items():
        desc = info.get('description', 'No description')
        dashboard += f"- **{name}**: {desc}\n"

    # Save
    output_path = EXPERIMENTS_DIR / "COMPARISON.md"
    with open(output_path, 'w') as f:
        f.write(dashboard)

    print(f"Generated dashboard: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Experiment Version Manager")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Create command
    create_parser = subparsers.add_parser('create', help='Create new version')
    create_parser.add_argument('name', help='Version name (e.g., v2_entropy)')
    create_parser.add_argument('--parent', '-p', help='Parent version to fork from')
    create_parser.add_argument('--description', '-d', default='', help='Description')

    # List command
    subparsers.add_parser('list', help='List all versions')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two versions')
    compare_parser.add_argument('v1', help='First version')
    compare_parser.add_argument('v2', help='Second version')

    # Set active command
    active_parser = subparsers.add_parser('set-active', help='Set active version')
    active_parser.add_argument('name', help='Version to activate')

    # Update results command
    update_parser = subparsers.add_parser('update-results', help='Update version results')
    update_parser.add_argument('name', help='Version name')

    # Dashboard command
    subparsers.add_parser('dashboard', help='Generate comparison dashboard')

    args = parser.parse_args()

    if args.command == 'create':
        create_version(args.name, args.parent, args.description)
    elif args.command == 'list':
        list_versions()
    elif args.command == 'compare':
        compare_versions(args.v1, args.v2)
    elif args.command == 'set-active':
        set_active_version(args.name)
    elif args.command == 'update-results':
        update_results(args.name)
    elif args.command == 'dashboard':
        generate_dashboard()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
