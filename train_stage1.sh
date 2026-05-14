#!/usr/bin/env bash
# ============================================================
#  train_stage1.sh  — Auto-detecting GPU launch script for
#  Stage 1 colorizer training.
#
#  Usage:
#    chmod +x train_stage1.sh
#    ./train_stage1.sh                       # auto-detect GPUs
#    EPOCHS=150 ./train_stage1.sh            # override epochs
#    DATA_ROOTS="datasets/flickr2k datasets/coco" ./train_stage1.sh
#
#  For multi-GPU servers (original fault-tolerant behavior):
#    GPUS=0,1,2,3 NUM_GPUS=4 ./train_stage1.sh
# ============================================================
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR}"

# ─── Configurable defaults ─────────────────────────────────
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
LR="${LR:-1e-4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
LOG_FILE="${LOG_FILE:-train_stage1.log}"
STOP_FILE="STOP_TRAINING"

# ✅ FIX: All three dataset roots → uses full 39K images
DATA_ROOTS="${DATA_ROOTS:-datasets/flickr2k datasets/coco datasets/div2k}"

GAN_START_EPOCH="${GAN_START_EPOCH:-30}"
GAN_WEIGHT="${GAN_WEIGHT:-0.1}"

# Multi-GPU server overrides (only used when NUM_GPUS > 1)
GPUS="${GPUS:-}"          # e.g. "0,1,2,3" — leave empty for auto-detect
NUM_GPUS="${NUM_GPUS:-}"  # leave empty for auto-detect

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*" | tee -a "$LOG_FILE"
}

# ─── Detect GPU count ──────────────────────────────────────
detect_gpus() {
  python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0"
}

if [[ -z "$NUM_GPUS" ]]; then
  NUM_GPUS=$(detect_gpus)
fi

# Build common training args
# shellcheck disable=SC2206
TRAIN_ARGS=(
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --image-size "$IMAGE_SIZE"
  --lr "$LR"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --data-roots $DATA_ROOTS
  --gan-start-epoch "$GAN_START_EPOCH"
  --gan-weight "$GAN_WEIGHT"
  --optimizer adamw
  --gradient-clip 1.0
  --perceptual-weight 0.6
  --colorfulness-weight 1.5
  --focal-chroma-weight 0.8
  --histogram-weight 0.4
  --resume auto
)

log "=========================================================="
log "  colorizer-ai  Stage 1 Training Launcher"
log "=========================================================="
log "  GPUs detected : $NUM_GPUS"
log "  Epochs        : $EPOCHS"
log "  Batch size    : $BATCH_SIZE"
log "  Dataset roots : $DATA_ROOTS"
log "  Checkpoint dir: $CHECKPOINT_DIR"
log "=========================================================="

mkdir -p "$CHECKPOINT_DIR"

# ─── Launch based on GPU count ─────────────────────────────

if [[ "$NUM_GPUS" -ge 2 ]]; then
  # ── Multi-GPU: torchrun with fault-tolerant restart loop ─
  log "Multi-GPU mode: $NUM_GPUS GPUs"
  [[ -n "$GPUS" ]] && export CUDA_VISIBLE_DEVICES="$GPUS"

  while true; do
    [[ -f "$STOP_FILE" ]] && { log "STOP_TRAINING detected. Exiting."; break; }

    log "Launching with torchrun (nproc=$NUM_GPUS)..."
    torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
      training/train_colorizer.py "${TRAIN_ARGS[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}

    [[ $exit_code -eq 0 ]] && { log "Training complete (exit 0)."; break; }
    [[ -f "$STOP_FILE" ]] && { log "STOP_TRAINING detected after crash. Exiting."; break; }

    log "Crashed (exit $exit_code). Restarting in 15s..."
    sleep 15
  done

elif [[ "$NUM_GPUS" -eq 1 ]]; then
  # ── Single GPU: no torchrun, no DDP ──────────────────────
  log "Single-GPU mode — using --debug-single (no DDP required)."
  python training/train_colorizer.py \
    "${TRAIN_ARGS[@]}" \
    --debug-single \
    "$@" 2>&1 | tee -a "$LOG_FILE"

else
  # ── No GPU ───────────────────────────────────────────────
  log "ERROR: No CUDA GPU detected!"
  log ""
  log "  The training script requires a GPU. Your options:"
  log "  1. Fix CUDA: run setup_env.sh to rebuild the conda environment"
  log "  2. Use a cloud GPU: Vast.ai / RunPod / Lambda / Google Colab"
  log ""
  log "  To check if torch can see your GPU:"
  log "    python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.device_count())\""
  log ""
  exit 1
fi

log ""
log "=========================================================="
log "  Stage 1 complete. Checkpoints in: $CHECKPOINT_DIR/"
log ""
log "  DEPLOY with EMA checkpoint (better quality):"
log "    $CHECKPOINT_DIR/stage1_colorizer_ema_best.pth"
log ""
log "  Update configs/pipeline.yaml:"
log "    checkpoints:"
log "      colorizer: stage1_colorizer_ema_best.pth"
log "=========================================================="
