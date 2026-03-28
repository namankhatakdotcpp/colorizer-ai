#!/usr/bin/env python3
"""Quick diagnostic to verify ColorizerStage is working correctly."""

import cv2
import numpy as np
import torch
from pathlib import Path
from models.unet_colorizer import UNetColorizer


def diagnose_colorizer():
    """Verify the colorizer produces non-black output and correct AB scaling."""
    
    print("=" * 70)
    print("COLORIZER DIAGNOSTIC")
    print("=" * 70)
    
    # Paths
    root = Path(__file__).parent
    checkpoint_path = root / "checkpoints" / "stage1_colorizer_latest.pth"
    test_img_path = root / "assets" / "sample.jpg"
    
    # Check checkpoint exists
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"✓ Checkpoint: {checkpoint_path.name}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetColorizer(in_channels=1, out_channels=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        # Try directly as state dict
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError:
            # If that fails and it's a dict with model weights, extract them
            if isinstance(checkpoint, dict):
                # Find the key containing model weights
                for key in checkpoint:
                    if key not in ["epoch", "optimizer_state_dict", "best_metric", "loss"]:
                        model.load_state_dict(checkpoint[key])
                        break
            else:
                raise
    
    model.eval()
    print(f"✓ Model loaded on {device}")
    
    # Load test image
    if not test_img_path.exists():
        print(f"⚠ Test image not found, creating synthetic test...")
        l_test = np.random.rand(256, 256).astype(np.float32) * 0.7 + 0.2
    else:
        img = cv2.imread(str(test_img_path), cv2.IMREAD_GRAYSCALE)
        l_test = img.astype(np.float32) / 255.0
    
    print(f"✓ Test image shape: {l_test.shape}")
    
    # Create input tensor (1, 1, H, W)
    l_tensor = torch.from_numpy(l_test[np.newaxis, np.newaxis, :, :]).float().to(device)
    
    # Run inference
    with torch.inference_mode():
        ab_pred = model(l_tensor)  # [1, 2, H, W], range [-1, 1]
    
    print(f"✓ Model output shape: {ab_pred.shape}")
    
    ab_np = ab_pred.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    ab_raw_min = ab_np.min()
    ab_raw_max = ab_np.max()
    print(f"  AB raw [-1, 1] range: [{ab_raw_min:.4f}, {ab_raw_max:.4f}]")
    
    # CRITICAL: Scale to [-128, 128]
    ab_128 = np.clip(ab_np, -1.0, 1.0) * 128.0
    print(f"  AB after *128:      [{ab_128.min():.1f}, {ab_128.max():.1f}]")
    
    # Apply color boost
    color_boost = 1.4
    ab_boosted = ab_128 * color_boost
    print(f"  AB after boost x{color_boost}: [{ab_boosted.min():.1f}, {ab_boosted.max():.1f}]")
    
    # Clip to valid range
    ab_final = np.clip(ab_boosted, -110.0, 110.0)
    print(f"  AB final clipped:    [{ab_final.min():.1f}, {ab_final.max():.1f}]")
    
    # Reconstruct LAB
    H, W = l_test.shape
    lab = np.zeros((H, W, 3), dtype=np.float32)
    lab[:, :, 0] = l_test * 100.0
    lab[:, :, 1] = ab_final[:, :, 0]
    lab[:, :, 2] = ab_final[:, :, 1]
    
    # Convert to RGB
    from skimage.color import lab2rgb
    rgb = np.clip(lab2rgb(lab), 0.0, 1.0)
    
    rgb_min = rgb.min()
    rgb_max = rgb.max()
    rgb_mean = rgb.mean()
    
    print(f"\n✓ RGB output range:  [{rgb_min:.4f}, {rgb_max:.4f}], mean={rgb_mean:.4f}")
    
    # Check for black output
    if rgb_max < 0.05:
        print("  ❌ PROBLEM: Output is mostly black!")
        return False
    
    if rgb_max < 0.3:
        print(f"  ⚠ WARNING: Output is quite dark (max={rgb_max:.4f})")
    else:
        print(f"  ✓ Output has good brightness")
    
    # Compute chroma (saturation)
    chroma = np.sqrt(ab_final[:, :, 0]**2 + ab_final[:, :, 1]**2)
    mean_chroma = chroma.mean()
    vivid_pixels = (chroma > 20).sum() / chroma.size * 100
    
    print(f"\n✓ Color metrics:")
    print(f"  Mean Chroma: {mean_chroma:.2f} (goal: >15)")
    print(f"  Vivid pixels (chroma>20): {vivid_pixels:.1f}% (goal: >60%)")
    
    if mean_chroma < 5:
        print("  ❌ PROBLEM: Colors are too weak!")
        return False
    
    # Success criteria
    print("\n" + "=" * 70)
    if rgb_max > 0.3 and mean_chroma > 10:
        print("✅ DIAGNOSIS: COLORIZER IS WORKING CORRECTLY")
        print("   - Output has proper brightness")
        print("   - Colors are adequately saturated")
        return True
    else:
        print("❌ DIAGNOSIS: COLORIZER HAS ISSUES")
        return False


if __name__ == "__main__":
    success = diagnose_colorizer()
    exit(0 if success else 1)
