# PyTorch Multi-GPU Distributed Training Setup Guide

This guide outlines the complete environment initialization sequence required to deploy this Unified Config-Driven PyTorch architecture onto a fresh remote Linux server (e.g., AWS EC2, GCP Compute Engine, Lambda Labs, or an on-premise cluster) equipped with multiple NVIDIA GPUs.

---

### Step 1: Clone the Repository
SSH into your remote instance and pull down the project source code.

```bash
git clone https://github.com/your-username/colorizer-ai.git
cd colorizer-ai
```

### Step 2: Create a Dedicated Python Virtual Environment
To prevent dependency conflicts with system-level packages safely isolate your PyTorch environment:

```bash
# Ensure venv is installed (Ubuntu/Debian example)
sudo apt update && sudo apt install -y python3-venv

# Create and activate environment
python3 -m venv venv
source venv/bin/activate

# Upgrade essential build tools
pip install --upgrade pip setuptools wheel
```

### Step 3: Install Core Dependencies
Install PyTorch strictly configured for CUDA acceleration. Check the official [PyTorch matrix](https://pytorch.org/get-started/locally/) to match your server's exact CUDA version if `cu118` is outdated.

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required numerical, vision, and configuration utilities
pip install pyyaml pillow numpy opencv-python scikit-image tqdm tensorboard lpips
```

### Step 4: Verify Hardware and Software Alignment
Confirm that the operating system recognizes all NVIDIA GPUs and that PyTorch has successfully mapped them identically. 

```bash
# Check raw hardware visibility (should list all 8 GPUs)
nvidia-smi

# Check PyTorch CUDA mapping (should output "True" and "8")
python -c "import torch; print('CUDA built:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

### Step 5: Dataset Hydration and Preprocessing
Before executing training, ensure your raw datasets are staged and fast-loading tensors mathematically computed if applicable (e.g., Colorization LAB split caching):

```bash
# Move your raw datasets into the appropriate directories
mkdir -p data/colorization/rgb_images
# [SCP/Wget your dataset into data/colorization/rgb_images/]

# Run modular caching script (Uses tqdm to display progress)
python datasets/preprocess_lab.py
```

### Step 6: Launch Distributed Data Parallel (DDP) Training
Leverage the standardized deployment orchestrator to elasticize your workload over the PCI-e switch natively:

```bash
# Ensure execution privileges
chmod +x run_training.sh

# Target the desired YAML configuration blueprint
./run_training.sh configs/colorizer.yaml
```

*Note: Use `tmux` or `screen` to run Step 6 if you expect the SSH connection to drop during the multi-day run duration.*
