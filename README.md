# 📸 DSLR AI Pipeline  
### End-to-End Computational Photography Engine (Pure CNN)

---

## 🚀 Overview

This project implements a **multi-stage CNN-based DSLR simulation pipeline** that transforms grayscale images into high-resolution, depth-aware, HDR-enhanced outputs.

The system is designed using **pure convolutional architectures (no GAN dependency)** and follows a structured curriculum training strategy across multiple independent modules.

The goal is to approximate DSLR-style image characteristics using deep learning and computational photography techniques.

---

## 🏗 Architecture Pipeline
L Channel (256x256)
    ↓
UNet Colorizer
    ↓
RRDB Super Resolution (4x)
    ↓
Micro-Contrast Enhancement
    ↓
Zero-DCE HDR Enhancement
    ↓
Depth Estimation (MiDaS-style)
    ↓
Dynamic Filter Network (Depth-Aware Bokeh)
    ↓
Final DSLR-like Output (1024x1024)

---

## 🧠 Modules

### 1️⃣ UNet Colorizer
- LAB color space prediction
- Composite loss:
  - L1 Loss
  - Perceptual Loss (VGG16)
  - SSIM Loss
- Mixed precision training
- Best-checkpoint saving with validation loop

---

### 2️⃣ Super Resolution (RRDBNet)
- 4x upscaling (256 → 1024)
- Residual-in-Residual Dense Blocks
- PixelShuffle upscaling
- Patch-based training
- L1 + Perceptual Loss
- PSNR-based checkpointing

---

### 3️⃣ Micro Contrast Enhancer
- Residual CNN architecture
- High-frequency enhancement
- Laplacian & Sobel-based losses
- Detail refinement without hallucination

---

### 4️⃣ Zero-DCE HDR Enhancement
- 7-layer curve estimation network
- Zero-reference training
- Losses:
  - Spatial Consistency
  - Exposure Control
  - Color Constancy
  - Illumination Smoothness

---

### 5️⃣ Depth Estimation (MiDaS-style)
- ResNet50 encoder
- Multi-scale decoder
- Scale-and-Shift Invariant Loss (SSIL)
- Pretrained backbone fine-tuning

---

### 6️⃣ Dynamic Filter Network (Bokeh Rendering)
- Depth-aware separable kernel prediction
- Edge-preserving blur
- Sobel-based focus constraint
- Patch-based optimization for memory efficiency

---

## 🏋️ Training Strategy

Each module is trained **independently** before sequential fine-tuning.

Curriculum training avoids gradient collapse across the multi-stage pipeline.

Training supports:
- CUDA (NVIDIA GPUs)
- Apple MPS
- Mixed precision (autocast)

---

## 🖥 Tech Stack

- PyTorch
- FastAPI
- Docker
- Async Worker Architecture (Planned)
- Redis (Planned)
- Mixed Precision Training

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/dslr-ai-pipeline.git
cd dslr-ai-pipeline
```
## Install backend dependencies:

pip install -r backend/requirements.txt

## Run with Docker:

docker-compose up --build

---

## 📊 Current Status

- ✅ **Baseline ResNet18 Colorizer trained**
- 🔄 **UNet Colorizer upgrade in progress**
- 🔜 **Super Resolution module implementation**
- 🔜 **Multi-stage inference integration**

---

## ⚠️ Notes

- Model weights and datasets are not included in this repository.
- Training the full 1024×1024 pipeline requires dedicated GPU hardware.
- Designed for research and educational purposes.

---

## 🎯 Project Goals

- Simulate DSLR-style depth separation
- Enhance micro-contrast and dynamic range
- Maintain physically plausible image transformations
- Build a modular, production-ready AI imaging pipeline

---

## 👨‍💻 Author

Developed as an advanced deep learning research project focused on computational photography and multi-stage CNN systems.

---

## 📜 License

MIT License

---

## 🖼 Sample Results

| Input (Grayscale) | Output (Enhanced DSLR-style) |
|-------------------|------------------------------|
| ![input](assets/input.jpg) | ![output](assets/output.jpg) |
