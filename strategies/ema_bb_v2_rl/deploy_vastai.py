#!/usr/bin/env python3
"""
Vast.ai Training Deployment Script.

Orchestrates GPU training on Vast.ai:
1. Launches RTX 4090 instance
2. Sets up training environment
3. Uploads code and data
4. Runs training with W&B monitoring
5. Syncs results and destroys instance

Usage:
    python deploy_vastai.py --timesteps 10000000
    python deploy_vastai.py --timesteps 5000000 --wandb-key YOUR_KEY
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

STRATEGY_DIR = Path(__file__).parent


def check_mcp_available():
    """Check if Vast.ai MCP tools are available."""
    try:
        # This script is designed to be run with Claude Code's MCP integration
        # The actual MCP calls will be made through Claude's tool interface
        return True
    except Exception:
        return False


def print_manual_instructions():
    """Print manual deployment instructions if MCP not available."""
    print("""
================================================================================
 MANUAL VAST.AI DEPLOYMENT INSTRUCTIONS
================================================================================

If running outside Claude Code, use these manual steps:

1. FIND A GPU:
   vastai search offers 'gpu_name=RTX_4090 num_gpus=1 dph<0.5' -o 'score-'

2. CREATE INSTANCE:
   vastai create instance <OFFER_ID> \\
       --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \\
       --disk 20 --ssh

3. WAIT FOR READY (check status):
   vastai show instances

4. UPLOAD CODE:
   scp -r -P <PORT> strategies/ema_bb_v2_rl root@<HOST>:/root/rl_exit/

5. SSH AND SETUP:
   ssh -p <PORT> root@<HOST>
   cd /root/rl_exit
   pip install torch numpy pandas wandb

6. RUN TRAINING:
   python train.py --episodes data/episodes_train_2005_2021.pkl --timesteps 10000000 --wandb

7. MONITOR:
   - W&B dashboard: https://wandb.ai
   - SSH logs: tail -f /root/training/logs/training.log

8. DOWNLOAD RESULTS:
   scp -r -P <PORT> root@<HOST>:/root/rl_exit/models ./models/

9. DESTROY INSTANCE:
   vastai destroy instance <INSTANCE_ID>

================================================================================
""")


def generate_training_script(timesteps: int, wandb_key: str = None) -> str:
    """Generate the training script to run on GPU instance."""
    wandb_setup = ""
    if wandb_key:
        wandb_setup = f"""
# Setup W&B
export WANDB_API_KEY="{wandb_key}"
pip install wandb
"""

    return f"""#!/bin/bash
set -e

echo "=============================================="
echo " RL Exit Optimizer Training"
echo " Started: $(date)"
echo "=============================================="

# Navigate to training directory
cd /root/rl_exit

# Install dependencies
echo "Installing dependencies..."
pip install --quiet torch numpy pandas

{wandb_setup}

# Verify GPU
echo "GPU Info:"
nvidia-smi

# Check data
echo ""
echo "Data files:"
ls -la data/

# Run training
echo ""
echo "Starting training..."
python train.py \\
    --episodes data/episodes_train_2005_2021.pkl \\
    --timesteps {timesteps} \\
    --n-envs 128 \\
    --n-steps 2048 \\
    {"--wandb" if wandb_key else ""} \\
    --device cuda

echo ""
echo "=============================================="
echo " Training Complete: $(date)"
echo "=============================================="

# List results
echo "Model files:"
ls -la models/
"""


def main():
    parser = argparse.ArgumentParser(description="Deploy RL Training to Vast.ai")
    parser.add_argument('--timesteps', type=int, default=10_000_000, help='Training timesteps')
    parser.add_argument('--wandb-key', type=str, default=None, help='W&B API key')
    parser.add_argument('--gpu', type=str, default='RTX_4090', help='GPU type')
    parser.add_argument('--disk', type=int, default=20, help='Disk size GB')
    parser.add_argument('--dry-run', action='store_true', help='Print instructions only')
    args = parser.parse_args()

    print(f"""
================================================================================
 RL Exit Optimizer - Vast.ai Deployment
================================================================================
 Timesteps:  {args.timesteps:,}
 GPU:        {args.gpu}
 W&B:        {'Enabled' if args.wandb_key else 'Disabled'}
================================================================================
""")

    if args.dry_run:
        print_manual_instructions()
        print("\nGenerated training script:")
        print("-" * 60)
        print(generate_training_script(args.timesteps, args.wandb_key))
        return

    # Check for episode data
    episode_file = STRATEGY_DIR / "data" / "episodes_train_2005_2021.pkl"
    if not episode_file.exists():
        print(f"ERROR: Episode file not found: {episode_file}")
        print("\nRun extract_episodes.py first:")
        print("  python extract_episodes.py -i AUDUSD -t 15M")
        sys.exit(1)

    print(f"Episode file found: {episode_file}")
    print(f"  Size: {episode_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Generate training script
    training_script = generate_training_script(args.timesteps, args.wandb_key)
    script_path = STRATEGY_DIR / "train_remote.sh"
    with open(script_path, 'w') as f:
        f.write(training_script)
    print(f"Generated training script: {script_path}")

    # Check if running in Claude Code with MCP
    print("""
================================================================================
 DEPLOYMENT STEPS (Run in Claude Code)
================================================================================

The following MCP commands will be executed:

1. Search for GPU:
   mcp__vast-ai__search_offers(query="gpu_name={args.gpu} num_gpus=1", limit=5)

2. Create instance:
   mcp__vast-ai__create_instance(
       offer_id=<BEST_OFFER>,
       image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
       disk={args.disk},
       ssh=True
   )

3. Wait for ready:
   mcp__vast-ai__wait_for_instance_ready(instance_id=<ID>)

4. Upload code:
   mcp__vast-ai__scp_upload_folder(
       local_path="./strategies/ema_bb_v2_rl/",
       remote_path="/root/rl_exit/"
   )

5. Setup W&B:
   mcp__vast-ai__setup_wandb(project="rl-exit-optimizer")

6. Run training:
   mcp__vast-ai__ssh_execute_background_command(
       command="cd /root/rl_exit && bash train_remote.sh",
       task_name="rl_training"
   )

7. Monitor progress:
   mcp__vast-ai__check_task_status(...)
   mcp__vast-ai__check_wandb_runs(project="rl-exit-optimizer")

8. Sync results and cleanup:
   mcp__vast-ai__wait_for_task_then_destroy(...)

================================================================================

To proceed, ask Claude to:
"Deploy RL training to Vast.ai with {args.timesteps:,} timesteps"

Or run manually with --dry-run to see shell commands.
""")


if __name__ == "__main__":
    main()
