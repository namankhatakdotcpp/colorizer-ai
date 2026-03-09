#!/usr/bin/env bash
# Dataset Downloader for Colorizer-AI
# Ensure you run this inside the 'datasets/scripts' directory or project root!

set -e

echo "====================================================================="
echo "Initializing Dataset Acquisition Pipeline"
echo "Targeting DIV2K, Flickr2K, and COCO (Colorization) repositories..."
echo "====================================================================="

# Base directories mapping exactly to the Architecture
mkdir -p data/colorization/rgb_images
mkdir -p data/sr/train
mkdir -p data/depth/train
mkdir -p data/enhance/train

# -------------------------------------------------------------------------
# 1. Super Resolution Datasets (DIV2K & Flickr2K)
# -------------------------------------------------------------------------
echo ">> Downloading DIV2K (High Resolution)..."
# Using wget to gracefully handle massive multi-GB zip files
wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -O data/sr/train/DIV2K_train_HR.zip
echo ">> Unzipping DIV2K..."
unzip -q data/sr/train/DIV2K_train_HR.zip -d data/sr/train/
rm data/sr/train/DIV2K_train_HR.zip

echo ">> Downloading Flickr2K (High Resolution)..."
# Assuming standard open-source mirroring
wget -c https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar -O data/sr/train/Flickr2K.tar
echo ">> Extracting Flickr2K..."
tar -xf data/sr/train/Flickr2K.tar -C data/sr/train/
rm data/sr/train/Flickr2K.tar

# -------------------------------------------------------------------------
# 2. Colorization Dataset (COCO 2017 Unlabeled or Val for quick start)
# -------------------------------------------------------------------------
echo ">> Downloading COCO 2017 Validation Set (For Colorization Base)..."
wget -c http://images.cocodataset.org/zips/val2017.zip -O data/colorization/val2017.zip
echo ">> Unzipping COCO..."
unzip -q data/colorization/val2017.zip -d data/colorization/rgb_images/
rm data/colorization/val2017.zip

# -------------------------------------------------------------------------
# 3. Depth & Micro-Contrast (Placeholder/Dummy links for architecture)
# -------------------------------------------------------------------------
# In a real academic environment, you would download NYU-Depth V2 or MIT5K here.
# For now, we will symlink the COCO images so the pipeline can instantly test successfully.
echo ">> Symlinking COCO to Depth and Contrast stages for verified testing..."
ln -sfn $(pwd)/data/colorization/rgb_images/val2017 $(pwd)/data/depth/train/images
ln -sfn $(pwd)/data/colorization/rgb_images/val2017 $(pwd)/data/enhance/train/images

echo "====================================================================="
echo "Dataset Acquisition Complete!"
echo "NEXT STEP: Run 'python datasets/preprocess_lab.py' to generate LAB caches!"
echo "====================================================================="
