#!/usr/bin/env bash

# Top-level execution wrapper for the Modular Multi-Stage Pipeline

# Strict bash constraints
set -e

STAGE=${1:-"colorizer"}

echo "====================================================================="
echo "Initializing PyTorch Distributed Training Sequence (DDP)"
echo "Targeting 8 GPU Node Topology..."
echo "Executing Stage: $STAGE"
echo "====================================================================="

if [ "$STAGE" == "colorizer" ]; then
    H_SCRIPT="training/train_colorizer.py"
elif [ "$STAGE" == "sr" ]; then
    H_SCRIPT="training/train_sr.py"
elif [ "$STAGE" == "depth" ]; then
    H_SCRIPT="training/train_depth.py"
elif [ "$STAGE" == "contrast" ]; then
    H_SCRIPT="training/train_micro_contrast.py"
else
    echo "Invalid Stage specified. Choose: colorizer, sr, depth, contrast"
    exit 1
fi

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    $H_SCRIPT
