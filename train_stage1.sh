#!/usr/bin/env bash

set -uo pipefail

# Configurable GPU assignment for torchrun workers.
GPUS="2,3,4,5"
NUM_GPUS=4
LOG_FILE="train_stage1.log"
STOP_FILE="STOP_TRAINING"

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*" | tee -a "$LOG_FILE"
}

# Returns 0 when any target GPU looks busy, else 1.
are_target_gpus_busy() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi

  local report
  report="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "$report" ]]; then
    return 1
  fi

  local gpu_id line util mem
  IFS=',' read -r -a gpu_ids <<< "$GPUS"
  for gpu_id in "${gpu_ids[@]}"; do
    gpu_id="${gpu_id// /}"
    line="$(awk -F',' -v idx="$gpu_id" '$1+0==idx {print $0}' <<< "$report" | head -n1)"
    if [[ -z "$line" ]]; then
      continue
    fi

    util="$(awk -F',' '{gsub(/ /,"",$2); print $2+0}' <<< "$line")"
    mem="$(awk -F',' '{gsub(/ /,"",$3); print $3+0}' <<< "$line")"

    # Consider GPU busy if utilization is significant or memory is already occupied.
    if (( util > 20 || mem > 1000 )); then
      return 0
    fi
  done

  return 1
}

log "Starting Stage1 fault-tolerant launcher (GPUS=$GPUS, NUM_GPUS=$NUM_GPUS)."

while true; do
  if [[ -f "$STOP_FILE" ]]; then
    log "Detected $STOP_FILE. Exiting restart loop."
    break
  fi

  log "Launching Stage1 training..."
  CUDA_VISIBLE_DEVICES="$GPUS" \
    torchrun --nproc_per_node="$NUM_GPUS" training/train_colorizer.py "$@" 2>&1 | tee -a "$LOG_FILE"
  exit_code=${PIPESTATUS[0]}

  if [[ $exit_code -eq 0 ]]; then
    log "Training finished successfully (exit code 0). Exiting launcher."
    break
  fi

  log "Training crashed (exit code $exit_code). Preparing restart."

  if [[ -f "$STOP_FILE" ]]; then
    log "Detected $STOP_FILE after crash. Exiting restart loop."
    break
  fi

  if are_target_gpus_busy; then
    log "Target GPUs appear busy. Sleeping 60s before restart."
    sleep 60
  else
    log "Target GPUs appear available. Sleeping 15s before restart."
    sleep 15
  fi

done
