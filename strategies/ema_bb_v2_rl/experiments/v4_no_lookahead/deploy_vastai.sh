#!/bin/bash
# Deploy v4_no_lookahead to Vast.ai RTX 4090 for full training
#
# Prerequisites:
# 1. Vast.ai CLI installed: pip install vastai
# 2. API key configured: vastai set api-key YOUR_API_KEY
#
# Usage:
#   ./deploy_vastai.sh search       # Find RTX 4090 offers
#   ./deploy_vastai.sh create ID    # Create instance from offer ID
#   ./deploy_vastai.sh setup ID     # Upload code and install deps
#   ./deploy_vastai.sh train ID     # Start training
#   ./deploy_vastai.sh status ID    # Check training status
#   ./deploy_vastai.sh download ID  # Download trained model
#   ./deploy_vastai.sh destroy ID   # Terminate instance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATEGY_DIR="$(dirname $(dirname $SCRIPT_DIR))"
PROJECT_NAME="v4_no_lookahead"

case "$1" in
    search)
        echo "Searching for RTX 4090 offers..."
        vastai search offers "gpu_name=RTX_4090 num_gpus=1 reliability>0.95 cuda_vers>=12.0" --order 'dph_base' | head -20
        echo ""
        echo "To create an instance, run: $0 create <offer_id>"
        ;;

    create)
        if [ -z "$2" ]; then
            echo "Usage: $0 create <offer_id>"
            exit 1
        fi
        echo "Creating instance from offer $2..."
        vastai create instance $2 \
            --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
            --disk 30 \
            --ssh \
            --env '-e WANDB_MODE=offline'
        echo ""
        echo "Instance created. Wait for it to boot, then run: $0 setup <instance_id>"
        echo "Check status with: vastai show instances"
        ;;

    setup)
        if [ -z "$2" ]; then
            echo "Usage: $0 setup <instance_id>"
            exit 1
        fi
        echo "Getting instance $2 SSH info..."
        SSH_INFO=$(vastai ssh-url $2)
        echo "SSH: $SSH_INFO"

        # Parse SSH info - format is ssh://root@host:port
        HOST=$(echo $SSH_INFO | sed 's|ssh://root@||' | cut -d: -f1)
        PORT=$(echo $SSH_INFO | sed 's|ssh://root@||' | cut -d: -f2)

        echo "Creating directories on remote..."
        ssh -p $PORT root@$HOST "mkdir -p /root/rl_exit"

        echo "Uploading training code..."
        # Upload v4 experiment
        scp -P $PORT -r $SCRIPT_DIR root@$HOST:/root/rl_exit/v4_no_lookahead/

        # Upload shared model.py
        scp -P $PORT $STRATEGY_DIR/model.py root@$HOST:/root/rl_exit/

        # Upload training data
        echo "Uploading training data (this may take a while)..."
        scp -P $PORT $STRATEGY_DIR/data/episodes_train_2005_2021.pkl root@$HOST:/root/rl_exit/data/
        scp -P $PORT $STRATEGY_DIR/data/episodes_test_2022_2025.pkl root@$HOST:/root/rl_exit/data/

        echo "Installing dependencies..."
        ssh -p $PORT root@$HOST "pip install wandb numpy"

        echo ""
        echo "Setup complete! Start training with: $0 train $2"
        ;;

    train)
        if [ -z "$2" ]; then
            echo "Usage: $0 train <instance_id>"
            exit 1
        fi
        echo "Getting instance $2 SSH info..."
        SSH_INFO=$(vastai ssh-url $2)
        HOST=$(echo $SSH_INFO | sed 's|ssh://root@||' | cut -d: -f1)
        PORT=$(echo $SSH_INFO | sed 's|ssh://root@||' | cut -d: -f2)

        echo "Starting training on remote GPU..."
        ssh -p $PORT root@$HOST "cd /root/rl_exit && nohup python v4_no_lookahead/train.py \
            --timesteps 15000000 \
            --n-envs 256 \
            --device cuda \
            > training.log 2>&1 &"

        echo ""
        echo "Training started! Monitor with: $0 status $2"
        echo "Expected time: 2-3 hours on RTX 4090"
        ;;

    status)
        if [ -z "$2" ]; then
            echo "Usage: $0 status <instance_id>"
            exit 1
        fi
        echo "Getting instance $2 SSH info..."
        SSH_INFO=$(vastai ssh-url $2)
        HOST=$(echo $SSH_INFO | sed 's|ssh://root@||' | cut -d: -f1)
        PORT=$(echo $SSH_INFO | sed 's|ssh://root@||' | cut -d: -f2)

        echo "Checking training status..."
        ssh -p $PORT root@$HOST "tail -50 /root/rl_exit/training.log"
        ;;

    download)
        if [ -z "$2" ]; then
            echo "Usage: $0 download <instance_id>"
            exit 1
        fi
        echo "Getting instance $2 SSH info..."
        SSH_INFO=$(vastai ssh-url $2)
        HOST=$(echo $SSH_INFO | sed 's|ssh://root@||' | cut -d: -f1)
        PORT=$(echo $SSH_INFO | sed 's|ssh://root@||' | cut -d: -f2)

        mkdir -p $SCRIPT_DIR/models

        echo "Downloading trained model..."
        scp -P $PORT root@$HOST:/root/rl_exit/v4_no_lookahead/models/exit_policy_final.pt $SCRIPT_DIR/models/

        echo "Downloading training log..."
        scp -P $PORT root@$HOST:/root/rl_exit/training.log $SCRIPT_DIR/training_gpu.log

        echo ""
        echo "Download complete! Model saved to: $SCRIPT_DIR/models/exit_policy_final.pt"
        echo "Now run OOS evaluation: python evaluate_oos_fast.py"
        ;;

    destroy)
        if [ -z "$2" ]; then
            echo "Usage: $0 destroy <instance_id>"
            exit 1
        fi
        echo "Destroying instance $2..."
        vastai destroy instance $2
        echo "Instance destroyed."
        ;;

    *)
        echo "V4 No-Lookahead Vast.ai Deployment Script"
        echo ""
        echo "Usage: $0 <command> [instance_id]"
        echo ""
        echo "Commands:"
        echo "  search          Search for RTX 4090 GPU offers"
        echo "  create <id>     Create instance from offer ID"
        echo "  setup <id>      Upload code and data to instance"
        echo "  train <id>      Start training on remote GPU"
        echo "  status <id>     Check training progress"
        echo "  download <id>   Download trained model"
        echo "  destroy <id>    Terminate instance"
        echo ""
        echo "Typical workflow:"
        echo "  1. $0 search           # Find cheapest RTX 4090"
        echo "  2. $0 create 12345     # Create from offer ID"
        echo "  3. Wait 2-3 minutes for boot"
        echo "  4. $0 setup <inst_id>  # Upload code"
        echo "  5. $0 train <inst_id>  # Start training"
        echo "  6. $0 status <inst_id> # Check progress (every ~30 min)"
        echo "  7. $0 download <inst_id> # Get trained model"
        echo "  8. $0 destroy <inst_id>  # Clean up"
        ;;
esac
