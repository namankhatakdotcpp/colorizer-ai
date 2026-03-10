# Colorizer-AI 🎨📸

**A Computational Photography Pipeline for Image Colorization and Enhancement**

Colorizer-AI is a deep learning pipeline that converts black-and-white images into high-quality, high-resolution, DSLR-style images using a sequence of specialized neural networks.

The system performs:

* 🎨 Image Colorization
* 🔍 Super Resolution (4× Upscaling)
* 🧠 Depth Estimation
* ✨ Micro-Contrast Enhancement

The pipeline is designed to mimic a **modern AI Image Signal Processor (ISP)** similar to those used in smartphone computational photography systems.

---

# 🚀 Pipeline Overview

Input grayscale image passes through four models sequentially:

```
Grayscale Image
        │
        ▼
┌────────────────────┐
│  UNet Colorizer    │
│  (L → AB channels) │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ RRDB SuperRes      │
│ (4× Upscaling)     │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ Depth Estimation   │
│ (Dynamic Filter)   │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ Micro Contrast Net │
│ Edge Enhancement   │
└─────────┬──────────┘
          ▼
     Final Image
```

---

# 🧠 Models Used

## 1️⃣ Colorizer

* Architecture: **UNet**
* Input: L channel (grayscale)
* Output: AB color channels
* Loss:

  * L1
  * VGG Perceptual
  * SSIM

Dataset:

* COCO
* Places365

---

## 2️⃣ Super Resolution

* Architecture: **RRDB (ESRGAN style)**
* Upscaling: **4×**

Training Strategy:

* HR images → RandomCrop(256)
* Bicubic downscale → LR images

Dataset:

* DIV2K
* Flickr2K

Loss:

* L1
* Perceptual Loss
* GAN (optional)

---

## 3️⃣ Depth Estimation

* Architecture: **Dynamic Filter Network**
* Predicts depth maps from RGB images

Used for:

* Scene understanding
* Spatial feature enhancement

---

## 4️⃣ Micro-Contrast Enhancement

Enhances:

* local gradients
* edges
* textures

Creates DSLR-style sharpness.

---

# 📂 Repository Structure

```
colorizer-ai/

configs/
    colorizer.yaml
    sr.yaml
    depth.yaml
    micro_contrast.yaml

datasets/
    dataset_colorizer.py
    dataset_sr.py
    preprocess_lab.py
    scripts/download_datasets.sh

models/
    unet_colorizer.py
    rrdb_sr.py
    depth_model.py
    micro_contrast_model.py

training/
    train.py
    train_colorizer.py
    train_sr.py
    train_depth.py
    train_micro_contrast.py

utils/
    config_loader.py
    losses.py
    tracker.py

run_stage.sh
run_training.sh
inference_pipeline.py
ARCHITECTURE.md
SETUP.md
```

---

# 📦 Dataset Setup

Download datasets automatically:

```
./datasets/scripts/download_datasets.sh
```

Datasets used:

| Dataset   | Purpose          |
| --------- | ---------------- |
| COCO      | colorization     |
| Places365 | colorization     |
| DIV2K     | super resolution |
| Flickr2K  | super resolution |

Total size: **~50GB**

---

# ⚡ Preprocess LAB Dataset

Before training the colorizer:

```
python datasets/preprocess_lab.py
```

This converts RGB images to:

```
L channel
AB channels
```

This speeds up training **2-3×**.

---

# 🏋️ Training Models

Train models **in this order**:

### 1️⃣ Colorizer

```
./run_stage.sh colorizer
```

### 2️⃣ Super Resolution

```
./run_stage.sh sr
```

### 3️⃣ Depth Estimation

```
./run_stage.sh depth
```

### 4️⃣ Micro Contrast

```
./run_stage.sh contrast
```

---

# 🖥 Multi-GPU Training

The project supports **Distributed Data Parallel (DDP)**.

Example:

```
torchrun --nproc_per_node=8 train.py
```

Optimizations included:

* Mixed Precision (AMP)
* Gradient Accumulation
* Distributed Samplers
* Optimized DataLoaders

---

# 🖼 Inference

Run the complete pipeline on an image:

```
python inference_pipeline.py path/to/image.jpg
```

Output will be saved to:

```
outputs/
```

---

# 🧪 Hardware

The pipeline is optimized for:

```
8 × NVIDIA RTX A6000
48GB VRAM each
```

Training speed estimate:

| Model            | Time    |
| ---------------- | ------- |
| Colorizer        | 3-4 hrs |
| Super Resolution | 6-8 hrs |
| Depth            | ~2 hrs  |
| Micro Contrast   | ~2 hrs  |

---

# 📊 Metrics

Models are evaluated using:

* PSNR
* SSIM
* LPIPS

Training progress is tracked with **TensorBoard**.

---

# 📜 Documentation

Detailed architecture explanation:

```
ARCHITECTURE.md
```

Environment setup guide:

```
SETUP.md
```

---

# 🎯 Future Improvements

Planned upgrades:

* SwinIR Super Resolution
* Diffusion Colorization
* Real-time inference
* GAN texture synthesis

---

# 👨‍💻 Author

**Naman**

Built as a research project exploring computational photography pipelines and deep learning based image enhancement.

---

## 🧠 System Architecture

The pipeline processes an image through four neural networks sequentially.

![Pipeline](assets/pipeline.png)

## Training, Resume, and Inference (DDP)

### Train full pipeline with one command

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_training.sh
```

This runs:
1. LAB preprocessing
2. Stage 1 colorizer training
3. Stage 1 checkpoint size verification
4. Stage 2 SR training
5. Stage 3 depth training
6. Stage 4 contrast training

### Resume training

All stage trainers auto-resume when `*_latest.pth` exists in `checkpoints/`.

Examples:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 training/train_colorizer.py
```

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 training/train_sr.py
```

### Dataset format for colorizer

Expected preprocessed structure:

```text
datasets/flickr2k/
  L/
  AB/
```

To generate it from RGB images:

```bash
python datasets/preprocess_lab.py --input-dir datasets/flickr2k/rgb --output-dir datasets/flickr2k --img-size 256
```

### Inference

Single-stage colorization:

```bash
python inference_pipeline.py test.jpg
```

Output:

```text
outputs/colorized_test.jpg
```

Optional full pipeline (if stage2/3/4 checkpoints exist):

```bash
python inference_pipeline.py test.jpg --full-pipeline
```

### Pipeline smoke test

```bash
python scripts/test_pipeline.py --image test.jpg
```
