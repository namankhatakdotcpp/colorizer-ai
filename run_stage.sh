#!/usr/bin/env bash

# Top-level execution wrapper for the Modular Multi-Stage Pipeline

# Strict bash constraints
set -euo pipefail

STAGE=${1:-"colorizer"}

echo "====================================================================="
echo "Initializing PyTorch Distributed Training Sequence (DDP)"
echo "Executing Stage: $STAGE"
echo "====================================================================="

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
    NPROC_PER_NODE=${NPROC_PER_NODE:-4}
fi

export PYTHONPATH="${PYTHONPATH:-.}"

echo "Targeting ${NPROC_PER_NODE} GPU process(es)..."

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
    --nproc_per_node="$NPROC_PER_NODE" \
    $H_SCRIPT
