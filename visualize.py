"""
Before/After Visualization Grid Generator

Creates comparison grids showing:
[Input] [Stage4 Output] [GAN Refined] [Ground Truth]

Usage:
    # Single image
    python visualize.py --image 000123.png
    
    # All images in ground_truth
    python visualize.py --all
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse


def load_image(path: str, size: tuple = (256, 256)) -> np.ndarray:
    """Load and resize image."""
    if not os.path.exists(path):
        print(f"[Visualize] Warning: {path} not found")
        return None
    
    try:
        img = cv2.imread(path)
        if img is None:
            print(f"[Visualize] Error: Could not read {path}")
            return None
        img = cv2.resize(img, size)
        return img
    except Exception as e:
        print(f"[Visualize] Error loading {path}: {e}")
        return None


def add_label(img: np.ndarray, label: str, position: tuple = (10, 25), 
              font_size: float = 0.7, color: tuple = (255, 255, 255)) -> np.ndarray:
    """Add text label to image."""
    img_copy = img.copy()
    cv2.putText(
        img_copy, 
        label, 
        position,
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_size, 
        color, 
        2
    )
    return img_copy


def create_grid(img_list: list, labels: list, grid_width: int = 4) -> np.ndarray:
    """
    Create horizontal comparison grid.
    
    Args:
        img_list: List of images
        labels: List of labels for each image
        grid_width: Number of images per row
        
    Returns:
        Stacked image grid
    """
    if not img_list or not labels:
        print("[Visualize] Empty image list")
        return None
    
    # Add labels to images
    labeled_imgs = []
    for img, label in zip(img_list, labels):
        if img is None:
            continue
        img_copy = img.copy()
        cv2.putText(
            img_copy, 
            label, 
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        labeled_imgs.append(img_copy)
    
    if not labeled_imgs:
        print("[Visualize] No valid images to grid")
        return None
    
    # Pad with empty images if needed
    while len(labeled_imgs) < grid_width:
        h, w, c = labeled_imgs[0].shape
        labeled_imgs.append(np.zeros((h, w, c), dtype=np.uint8))
    
    # Stack horizontally
    grid = np.hstack(labeled_imgs[:grid_width])
    return grid


def visualize_single(sample_name: str, size: tuple = (256, 256)) -> bool:
    """
    Create comparison grid for single image.
    
    Simplified to load only:
        - outputs/generated/{sample_name} (GAN refined output)
        - data/ground_truth/{sample_name} (ground truth reference)
    
    Returns:
        True if successful
    """
    print(f"\n[Visualize] Processing: {sample_name}")
    
    # Load images - simplified to only generated and ground truth
    generated = load_image(f"outputs/generated/{sample_name}", size)
    gt = load_image(f"data/ground_truth/{sample_name}", size)
    
    # Need at least generated and GT
    if generated is None or gt is None:
        print(f"[Visualize] Error: Missing required images for {sample_name}")
        if generated is None:
            print(f"           - {sample_name} not in outputs/generated/")
        if gt is None:
            print(f"           - {sample_name} not in data/ground_truth/")
        return False
    
    # Create grid with only generated and ground truth
    img_list = [generated, gt]
    labels = ["Generated (GAN)", "Ground Truth"]
    
    grid = create_grid(img_list, labels, grid_width=2)
    
    if grid is None:
        return False
    
    # Save output
    os.makedirs("outputs/visuals", exist_ok=True)
    output_path = f"outputs/visuals/comparison_{sample_name}"
    
    try:
        cv2.imwrite(output_path, grid)
        print(f"✓ Saved: {output_path}")
        print(f"  Grid size: {grid.shape}")
        return True
    except Exception as e:
        print(f"[Visualize] Error saving {output_path}: {e}")
        return False


def visualize_all(size: tuple = (256, 256), limit: int = None) -> int:
    """
    Create comparison grids for all images in ground_truth.
    
    Args:
        size: Image size for grid
        limit: Max number of images to process (None for all)
        
    Returns:
        Number of successfully processed images
    """
    gt_dir = Path("data/ground_truth")
    
    if not gt_dir.exists():
        print(f"[Visualize] Error: {gt_dir} does not exist")
        return 0
    
    images = sorted(gt_dir.glob("*"))
    if limit:
        images = images[:limit]
    
    print(f"\n[Visualize] Found {len(images)} images in {gt_dir}")
    
    success_count = 0
    for img_path in images:
        if visualize_single(img_path.name, size):
            success_count += 1
    
    print(f"\n[Visualize] Successfully processed {success_count}/{len(images)} images")
    return success_count


# ────────────────── MAIN ──────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate before/after comparison grids"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        default=None,
        help="Single image filename (e.g., 000123.png)"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Process all images in data/ground_truth"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of images when using --all"
    )
    parser.add_argument(
        "--size", 
        type=int, 
        default=256,
        help="Image size for grid (default: 256)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  Before/After Comparison Grid Generator")
    print("="*60)
    
    if args.all:
        # Process all images
        visualize_all(size=(args.size, args.size), limit=args.limit)
    elif args.image:
        # Process single image
        visualize_single(args.image, size=(args.size, args.size))
    else:
        # Default: process first image in ground_truth
        gt_dir = Path("data/ground_truth")
        if gt_dir.exists():
            images = sorted(gt_dir.glob("*"))
            if images:
                visualize_single(images[0].name, size=(args.size, args.size))
            else:
                print("[Visualize] No images found in data/ground_truth")
        else:
            print(f"[Visualize] Error: {gt_dir} does not exist")
    
    print("\n✨ Grid images saved to: outputs/visuals/")
    print("="*60 + "\n")
