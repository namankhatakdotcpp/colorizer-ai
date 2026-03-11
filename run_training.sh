#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
COLORIZER_DATA_DIR="${COLORIZER_DATA_DIR:-datasets/flickr2k}"
RGB_INPUT_DIR="${RGB_INPUT_DIR:-datasets/flickr2k/rgb}"
SR_DATA_DIR="${SR_DATA_DIR:-datasets/div2k}"
DEPTH_DATA_DIR="${DEPTH_DATA_DIR:-datasets/coco}"
CONTRAST_DATA_DIR="${CONTRAST_DATA_DIR:-datasets/flickr2k}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-100}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-20}"
STAGE4_EPOCHS="${STAGE4_EPOCHS:-20}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
fi

export PYTHONPATH="${PYTHONPATH:-.}"

echo "[1/6] Preprocessing LAB dataset"
if [[ ! -d "${COLORIZER_DATA_DIR}/L" || ! -d "${COLORIZER_DATA_DIR}/AB" ]]; then
  if [[ ! -d "$RGB_INPUT_DIR" ]]; then
    echo "ERROR: RGB_INPUT_DIR not found: $RGB_INPUT_DIR"
    echo "Place RGB training images there, or set RGB_INPUT_DIR to your dataset path."
    exit 1
  fi
  python datasets/preprocess_lab.py --input-dir "$RGB_INPUT_DIR" --output-dir "$COLORIZER_DATA_DIR" --img-size 256
else
  echo "Found preprocessed dataset at ${COLORIZER_DATA_DIR}/L and ${COLORIZER_DATA_DIR}/AB"
fi

echo "[2/6] Stage 1 - Colorizer training"
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  training/train_colorizer.py \
  --epochs "$STAGE1_EPOCHS" \
  --data-root "$COLORIZER_DATA_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR"

echo "[3/6] Verify stage1 checkpoint size"
CHECKPOINT_DIR_ENV="$CHECKPOINT_DIR" python - <<'PY'
import os
from pathlib import Path
p = Path(os.environ["CHECKPOINT_DIR_ENV"]) / "stage1_colorizer_latest.pth"
if not p.exists():
    raise SystemExit("ERROR: missing stage1 checkpoint")
size_mb = p.stat().st_size / (1024 * 1024)
print(f"stage1 checkpoint size: {size_mb:.2f} MB")
if size_mb < 20:
    raise SystemExit("ERROR: stage1 checkpoint too small (<20MB)")
PY

echo "[4/6] Stage 2 - Super Resolution training"
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  training/train_sr.py \
  --epochs "$STAGE2_EPOCHS" \
  --data-root "$SR_DATA_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR"

echo "[5/6] Stage 3 - Depth training"
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  training/train_depth.py \
  --epochs "$STAGE3_EPOCHS" \
  --data-root "$DEPTH_DATA_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR"

echo "[6/6] Stage 4 - Contrast training"
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  training/train_micro_contrast.py \
  --epochs "$STAGE4_EPOCHS" \
  --data-root "$CONTRAST_DATA_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR"

echo "Training pipeline completed."
