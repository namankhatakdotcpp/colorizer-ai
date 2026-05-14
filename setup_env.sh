#!/usr/bin/env bash
# ============================================================
#  setup_env.sh  — Clean conda environment setup for colorizer-ai
#
#  Fixes:
#    ✅ ClobberError (jpeg vs libjpeg-turbo conflict)
#    ✅ Mixed channels (defaults + pytorch conflict)
#    ✅ torch missing / inconsistent
#    ✅ OpenCV / OpenMP binary conflicts
#
#  Usage:
#    chmod +x setup_env.sh
#    ./setup_env.sh
#
#  After setup:
#    conda activate colorizer
#    ./train_stage1.sh
# ============================================================
set -euo pipefail

ENV_NAME="colorizer"
PYTHON_VERSION="3.10"

# Detect CUDA version for PyTorch wheel selection
detect_cuda_version() {
  if command -v nvcc &>/dev/null; then
    nvcc --version | grep "release" | awk '{print $6}' | cut -c2- | cut -d'.' -f1,2
  elif command -v nvidia-smi &>/dev/null; then
    nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2
  else
    echo ""
  fi
}

CUDA_VER=$(detect_cuda_version)
echo "============================================================"
echo "  colorizer-ai Environment Setup"
echo "============================================================"
echo "  Environment name : $ENV_NAME"
echo "  Python version   : $PYTHON_VERSION"
echo "  CUDA detected    : ${CUDA_VER:-'none (CPU only)'}"
echo "============================================================"
echo ""

# Select the correct PyTorch channel based on CUDA version
if [[ -z "$CUDA_VER" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  TORCH_VERSION="torch==2.3.1"
  TORCHVISION_VERSION="torchvision==0.18.1"
  echo "⚠️  No GPU detected. Installing CPU-only PyTorch."
  echo "   Training will NOT work. Fix your CUDA driver first!"
  echo ""
elif [[ "$CUDA_VER" == "11"* ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu118"
  TORCH_VERSION="torch==2.3.1+cu118"
  TORCHVISION_VERSION="torchvision==0.18.1+cu118"
elif [[ "$CUDA_VER" == "12"* ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu121"
  TORCH_VERSION="torch==2.3.1+cu121"
  TORCHVISION_VERSION="torchvision==0.18.1+cu121"
else
  TORCH_INDEX="https://download.pytorch.org/whl/cu121"
  TORCH_VERSION="torch==2.3.1+cu121"
  TORCHVISION_VERSION="torchvision==0.18.1+cu121"
  echo "⚠️  Unknown CUDA version '$CUDA_VER'. Defaulting to CUDA 12.1 wheels."
fi

echo "  PyTorch build    : $TORCH_VERSION"
echo ""

# ─── Step 1: Remove broken environment if it exists ──────
if conda env list | grep -q "^${ENV_NAME}\b"; then
  echo "[1/5] Removing existing '$ENV_NAME' environment to clear conflicts..."
  conda env remove -n "$ENV_NAME" -y
  echo "      Done."
else
  echo "[1/5] No existing '$ENV_NAME' environment found — fresh install."
fi

# ─── Step 2: Create clean base environment ────────────────
echo ""
echo "[2/5] Creating fresh conda environment '$ENV_NAME' (Python $PYTHON_VERSION)..."
# ✅ KEY FIX: Use ONLY the conda-forge channel. This prevents the
# jpeg vs libjpeg-turbo ClobberError caused by mixing defaults+pytorch channels.
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" \
  -c conda-forge \
  --override-channels \
  -y \
  libjpeg-turbo \
  libpng \
  libwebp \
  libgl \
  libglib
echo "      Done."

# ─── Step 3: Install PyTorch via pip (NOT conda) ─────────
# ✅ KEY FIX: Always use pip for PyTorch to avoid conda channel conflicts.
# Never do `conda install pytorch` — it pulls in its own libjpeg which
# conflicts with conda-forge's libjpeg-turbo.
echo ""
echo "[3/5] Installing PyTorch via pip (avoids conda channel conflicts)..."
conda run -n "$ENV_NAME" pip install \
  "$TORCH_VERSION" \
  "$TORCHVISION_VERSION" \
  --index-url "$TORCH_INDEX" \
  --no-deps
echo "      Done."

# ─── Step 4: Install all project dependencies via pip ────
echo ""
echo "[4/5] Installing project dependencies..."
conda run -n "$ENV_NAME" pip install \
  numpy==1.26.4 \
  pillow==10.3.0 \
  scikit-image==0.22.0 \
  opencv-python-headless==4.10.0.84 \
  fastapi==0.111.0 \
  uvicorn[standard]==0.30.1 \
  python-multipart==0.0.9 \
  slowapi==0.1.9 \
  prometheus-client==0.20.0 \
  pydantic-settings==2.3.4 \
  python-dotenv==1.0.1 \
  PyYAML==6.0.2 \
  torchmetrics==1.4.0 \
  pytorch-fid==0.3.0 \
  tqdm==4.66.4 \
  scipy==1.13.1 \
  matplotlib==3.9.1 \
  lpips==0.1.4
echo "      Done."

# ─── Step 5: Validate GPU visibility ─────────────────────
echo ""
echo "[5/5] Validating environment..."
conda run -n "$ENV_NAME" python - <<'PYCHECK'
import sys
print(f"  Python : {sys.version.split()[0]}")

import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU count      : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}          : {props.name} ({props.total_memory // 1024**2} MB VRAM)")
else:
    print("  ⚠️  CUDA NOT available. Check your driver with: nvidia-smi")

import cv2
print(f"  OpenCV : {cv2.__version__}")

import numpy as np
print(f"  NumPy  : {np.__version__}")

import PIL
print(f"  Pillow : {PIL.__version__}")

print("\n  ✅ Environment validation passed!")
PYCHECK

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Activate with:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  Then start training:"
echo "    ./train_stage1.sh"
echo ""
echo "  If CUDA is still not visible after activation, run:"
echo "    nvidia-smi                      # check driver"
echo "    nvcc --version                  # check toolkit"
echo "    python -c 'import torch; print(torch.cuda.is_available())'"
echo "============================================================"
