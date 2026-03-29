"""
LPIPS + FID Metrics for GAN Refinement Evaluation

Metrics:
- LPIPS: Perceptual similarity (lower = better)
- FID: Fréchet Inception Distance (lower = better)

Usage:
    python metrics.py
    
Expected outputs improve with GAN training:
    Without GAN: higher LPIPS, higher FID
    With GAN:    lower LPIPS, lower FID
"""

import os
import torch
import lpips
from PIL import Image
from pathlib import Path
from torchvision import transforms
from pytorch_fid import fid_score

# ────────────────── CONFIG ──────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Metrics] Device: {device}")

# LPIPS model (uses AlexNet features, good for perceptual similarity)
lpips_model = lpips.LPIPS(net='alex').to(device)

# Image preprocessing: resize + normalize
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


# ────────────────── LPIPS ──────────────────
def compute_lpips(fake_dir: str, real_dir: str) -> float:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).
    
    Args:
        fake_dir: Directory with generated/refined images
        real_dir: Directory with ground truth images
        
    Returns:
        Mean LPIPS score (lower is better, range ~0.0-1.0)
    """
    total = 0
    count = 0
    
    fake_dir = Path(fake_dir)
    real_dir = Path(real_dir)
    
    if not fake_dir.exists():
        print(f"[LPIPS] Error: {fake_dir} does not exist")
        return None
    if not real_dir.exists():
        print(f"[LPIPS] Error: {real_dir} does not exist")
        return None

    for fname in os.listdir(fake_dir):
        fake_path = fake_dir / fname
        real_path = real_dir / fname

        if not real_path.exists():
            continue
        
        try:
            img_fake = transform(Image.open(fake_path).convert("RGB")).unsqueeze(0).to(device)
            img_real = transform(Image.open(real_path).convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                dist = lpips_model(img_fake, img_real)

            total += dist.item()
            count += 1
            
        except Exception as e:
            print(f"[LPIPS] Skipped {fname}: {e}")
            continue

    if count == 0:
        print("[LPIPS] No matching image pairs found")
        return None
        
    lpips_score = total / count
    print(f"[LPIPS] Processed {count} image pairs")
    return lpips_score


# ────────────────── FID ──────────────────
def compute_fid(fake_dir: str, real_dir: str) -> float:
    """
    Compute FID (Fréchet Inception Distance).
    
    Args:
        fake_dir: Directory with generated/refined images
        real_dir: Directory with ground truth images
        
    Returns:
        FID score (lower is better, typical range 0-200)
    """
    fake_dir = Path(fake_dir)
    real_dir = Path(real_dir)
    
    if not fake_dir.exists():
        print(f"[FID] Error: {fake_dir} does not exist")
        return None
    if not real_dir.exists():
        print(f"[FID] Error: {real_dir} does not exist")
        return None

    try:
        fid_score_value = fid_score.calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=16,
            device=device,
            dims=2048
        )
        return fid_score_value
    except Exception as e:
        print(f"[FID] Error computing FID: {e}")
        return None


# ────────────────── MAIN ──────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  GAN Refinement Metrics Evaluation")
    print("="*50 + "\n")
    
    # Paths
    fake_dir = "outputs/generated"
    real_dir = "data/ground_truth"
    
    # Ensure output directory exists
    os.makedirs(fake_dir, exist_ok=True)
    
    print(f"Generated images dir: {fake_dir}")
    print(f"Ground truth dir: {real_dir}\n")
    
    # Compute metrics
    print("[Metrics] Computing LPIPS...")
    lpips_score = compute_lpips(fake_dir, real_dir)
    
    print("[Metrics] Computing FID...")
    fid = compute_fid(fake_dir, real_dir)
    
    # Display results
    print("\n" + "="*50)
    print("            RESULTS")
    print("="*50)
    
    if lpips_score is not None:
        print(f"LPIPS: {lpips_score:.4f} (lower is better)")
    else:
        print("LPIPS: N/A")
    
    if fid is not None:
        print(f"FID:   {fid:.2f} (lower is better)")
    else:
        print("FID: N/A")
    
    print("\n📊 Expected with GAN training:")
    print("   Without GAN → high LPIPS, high FID")
    print("   With GAN   → ↓ lower LPIPS, ↓ lower FID")
    print("="*50 + "\n")
