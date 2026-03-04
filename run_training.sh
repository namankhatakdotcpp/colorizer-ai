#!/usr/bin/env bash

# PyTorch Distributed Data Parallel Launcher
# Optimally configured to spawn NCCL-linked training engines across 8 GPUs natively via torchrun.

# Strict bash constraints
set -e

# Default to the primary config if none provided mathematically via terminal argument
CONFIG_PATH=${1:-"configs/default.yaml"}

echo "====================================================================="
echo "Initializing PyTorch Distributed Training Sequence (DDP)"
echo "Targeting 8 GPU Node Topology..."
echo "Applying Configuration Profile: $CONFIG_PATH"
echo "====================================================================="

# Torchrun is the official PyTorch elastic launcher replacing torch.distributed.launch
# It automatically provisions LOCAL_RANK, RANK, and WORLD_SIZE environment variables internally to train.py

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    training/train.py --config "$CONFIG_PATH"
