#!/usr/bin/env python3
"""
CHECKPOINT & AB VALUE DIAGNOSTIC
Compare both checkpoints and verify AB scaling is correct

Usage: python diagnose.py
"""
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from models.unet_colorizer import UNetColorizer

print("=" * 80)
print("CHECKPOINT & AB VALUE DIAGNOSTIC")
print("=" * 80)

# Test BOTH checkpoints and compare
for ckpt_name in [
    "checkpoints/stage1_colorizer_latest.pth",
    "checkpoints/stage1_colorizer_finetuned_latest.pth",
]:
    if not Path(ckpt_name).exists():
        print(f"\n❌ NOT FOUND: {ckpt_name}")
        continue

    print(f"\n📊 Testing: {ckpt_name}")
    print("-" * 80)
    
    model = UNetColorizer(in_channels=1, out_channels=2)
    ckpt = torch.load(ckpt_name, map_location="cpu")
    
    # Extract state dict
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    
    # Strip "module." prefix if present
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Run with random grayscale input
    dummy = torch.rand(1, 1, 256, 256)
    with torch.no_grad():
        out = model(dummy)

    ab_raw = out.squeeze().permute(1, 2, 0).numpy()
    ab_125 = ab_raw * 128.0  # Scale to [-128, 128]
    
    chroma_raw = np.sqrt(np.mean(ab_raw[:, :, 0]**2 + ab_raw[:, :, 1]**2))
    chroma_scaled = np.sqrt(np.mean(ab_125[:, :, 0]**2 + ab_125[:, :, 1]**2))

    print(f"  Model output range:          [{ab_raw.min():.4f}, {ab_raw.max():.4f}]  (tanh [-1,1])")
    print(f"  After *128 scaling:          [{ab_125.min():.1f}, {ab_125.max():.1f}]  (LAB [-128,128])")
    print(f"  Mean chroma (raw):           {chroma_raw:.4f}")
    print(f"  Mean chroma (scaled *128):   {chroma_scaled:.2f}")
    
    # Determine grade
    if chroma_scaled > 30.0:
        grade = "✅ EXCELLENT"
    elif chroma_scaled > 20.0:
        grade = "✅ GOOD"
    elif chroma_scaled > 12.0:
        grade = "⚠️  FAIR"
    else:
        grade = "❌ POOR"
    
    print(f"  Grade:                       {grade}")
    print(f"  Checkpoint epoch:            {ckpt.get('epoch', 'unknown')}")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print("""
Use the checkpoint with the HIGHEST mean chroma when scaled by 128.

If stage1_colorizer_latest.pth has chroma > 20 when scaled:
  ✅ Use: stage1_colorizer_latest.pth

If stage1_colorizer_finetuned_latest.pth has chroma > stage1_colorizer_latest.pth:
  ⚠️  Check why finetuned has higher chroma (it may be better)

The correct checkpoint should produce AB values in range [-60, 80] or similar
AFTER applying color_boost (typically 1.4-1.6).

If AB values are still small ([-12, 29]), the checkpoint may be using
different training data or has different output characteristics.
""")
print("=" * 80)
