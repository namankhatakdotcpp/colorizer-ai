📸 DSLR AI Pipeline – End-to-End Computational Photography Engine
🚀 Overview

This project implements a multi-stage CNN-based DSLR simulation pipeline that transforms grayscale images into high-resolution, depth-aware, HDR-enhanced images.

The system is built using pure CNN architectures (no GAN dependency) and follows a curriculum training strategy across multiple modules.

🏗 Architecture Pipeline
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
🧠 Modules
1️⃣ UNet Colorizer

LAB color space prediction

L1 + Perceptual + SSIM loss

Mixed precision training

2️⃣ Super Resolution (RRDBNet)

4x upscaling

PixelShuffle architecture

Patch-based training

3️⃣ Micro Contrast Enhancer

Residual CNN

Laplacian & Sobel-based loss

4️⃣ Zero-DCE HDR

Zero-reference curve-based enhancement

Spatial, exposure & color constancy losses

5️⃣ Depth Estimation

ResNet50 encoder

Scale-and-shift invariant loss

6️⃣ Dynamic Filter Network

Depth-aware separable kernel prediction

Edge-preserving bokeh rendering

🖥 Tech Stack

PyTorch

FastAPI

Docker

MPS / CUDA

Redis / Async Worker (Planned)

🏋️ Training Strategy

Each module is trained independently before sequential fine-tuning.

Curriculum training prevents gradient explosion across the 6-stage pipeline.

⚡ Deployment

FastAPI acts as a lightweight API layer.
Heavy inference runs in asynchronous GPU workers.

📦 Installation
git clone https://github.com/YOUR_USERNAME/dslr-ai-pipeline.git
cd dslr-ai-pipeline
pip install -r backend/requirements.txt

Run:

docker-compose up --build
🎯 Current Status

✅ ResNet18 baseline colorizer trained

🔄 Upgrading to UNet

🔜 Super Resolution module in progress

📈 Future Work

Patch-based DFN optimization

Distributed training support

Model quantization for production

🧑‍💻 Author

Built as a deep learning research project focused on computational photography and DSLR simulation.
