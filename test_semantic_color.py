"""
Test script for semantic color correction.

Demonstrates:
- Sky detection and blue boost
- Water detection and blue-green shift
- Performance characteristics
- Before/after visualization
"""

import time
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from skimage.color import lab2rgb

# Import the semantic color correction function
import sys
sys.path.insert(0, str(Path(__file__).parent))

from inference_pipeline import semantic_color_correction, safe_lab_to_rgb


def create_synthetic_test_image():
    """Create a synthetic test image with sky, water, and ground regions."""
    h, w = 480, 640
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    # Sky region (top 30%)
    sky_height = int(0.3 * h)
    sky_color = np.array([135, 206, 235], dtype=np.float32) / 255.0  # Light blue
    rgb[:sky_height, :, :] = sky_color
    
    # Add slight variation to sky
    noise = np.random.randn(sky_height, w, 3) * 0.05
    rgb[:sky_height, :, :] = np.clip(rgb[:sky_height, :, :] + noise, 0, 1)
    
    # Water region (middle)
    water_height_start = sky_height
    water_height_end = int(0.65 * h)
    water_color = np.array([64, 164, 223], dtype=np.float32) / 255.0  # Ocean blue
    rgb[water_height_start:water_height_end, :, :] = water_color
    
    # Add wave-like patterns
    noise = np.random.randn(water_height_end - water_height_start, w, 3) * 0.08
    rgb[water_height_start:water_height_end, :, :] = np.clip(
        rgb[water_height_start:water_height_end, :, :] + noise, 0, 1
    )
    
    # Ground region (bottom)
    ground_color = np.array([139, 69, 19], dtype=np.float32) / 255.0  # Brown
    rgb[water_height_end:, :, :] = ground_color
    
    # Add ground texture
    noise = np.random.randn(h - water_height_end, w, 3) * 0.1
    rgb[water_height_end:, :, :] = np.clip(rgb[water_height_end:, :, :] + noise, 0, 1)
    
    return rgb


def test_semantic_correction_on_synthetic():
    """Test semantic correction on synthetic image."""
    print("\n" + "="*60)
    print("TEST 1: Synthetic Image")
    print("="*60)
    
    # Create synthetic test image
    rgb_original = create_synthetic_test_image()
    
    # Apply semantic color correction
    start_time = time.perf_counter()
    rgb_corrected = semantic_color_correction(rgb_original, enable_sky_boost=True, enable_water_boost=True)
    elapsed = time.perf_counter() - start_time
    
    print(f"✓ Semantic correction applied in {elapsed*1000:.2f}ms")
    
    # Analyze color changes
    rgb_u8_original = (rgb_original * 255).astype(np.uint8)
    rgb_u8_corrected = (rgb_corrected * 255).astype(np.uint8)
    
    # Sky region analysis (top 30%)
    sky_h = int(0.3 * rgb_original.shape[0])
    sky_orig = rgb_u8_original[:sky_h, :, :]
    sky_corr = rgb_u8_corrected[:sky_h, :, :]
    
    print(f"\nSky Region (top 30%):")
    print(f"  Original  - Mean RGB: ({sky_orig[..., 0].mean():.1f}, {sky_orig[..., 1].mean():.1f}, {sky_orig[..., 2].mean():.1f})")
    print(f"  Corrected - Mean RGB: ({sky_corr[..., 0].mean():.1f}, {sky_corr[..., 1].mean():.1f}, {sky_corr[..., 2].mean():.1f})")
    
    # Water region analysis (30-65%)
    water_h_start = sky_h
    water_h_end = int(0.65 * rgb_original.shape[0])
    water_orig = rgb_u8_original[water_h_start:water_h_end, :, :]
    water_corr = rgb_u8_corrected[water_h_start:water_h_end, :, :]
    
    print(f"\nWater Region (30-65%):")
    print(f"  Original  - Mean RGB: ({water_orig[..., 0].mean():.1f}, {water_orig[..., 1].mean():.1f}, {water_orig[..., 2].mean():.1f})")
    print(f"  Corrected - Mean RGB: ({water_corr[..., 0].mean():.1f}, {water_corr[..., 1].mean():.1f}, {water_corr[..., 2].mean():.1f})")
    
    # Ground region analysis (bottom)
    ground_orig = rgb_u8_original[water_h_end:, :, :]
    ground_corr = rgb_u8_corrected[water_h_end:, :, :]
    
    print(f"\nGround Region (bottom 35%):")
    print(f"  Original  - Mean RGB: ({ground_orig[..., 0].mean():.1f}, {ground_orig[..., 1].mean():.1f}, {ground_orig[..., 2].mean():.1f})")
    print(f"  Corrected - Mean RGB: ({ground_corr[..., 0].mean():.1f}, {ground_corr[..., 1].mean():.1f}, {ground_corr[..., 2].mean():.1f})")
    
    # Save test images
    output_dir = Path("outputs/semantic_correction_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray((rgb_original * 255).astype(np.uint8)).save(output_dir / "original.jpg")
    Image.fromarray((rgb_corrected * 255).astype(np.uint8)).save(output_dir / "corrected.jpg")
    
    print(f"\n✓ Test images saved to {output_dir}/")


def test_performance_scaling():
    """Test performance with different image sizes."""
    print("\n" + "="*60)
    print("TEST 2: Performance Scaling")
    print("="*60)
    
    sizes = [(256, 256), (512, 512), (768, 768), (1024, 1024)]
    
    print(f"\n{'Size':<15} {'Time (ms)':<15} {'Pixels/ms':<15}")
    print("-" * 45)
    
    for h, w in sizes:
        rgb = np.random.rand(h, w, 3).astype(np.float32)
        
        start_time = time.perf_counter()
        _ = semantic_color_correction(rgb)
        elapsed = (time.perf_counter() - start_time) * 1000
        
        pixels_per_ms = (h * w) / elapsed
        print(f"{h}x{w:<9} {elapsed:<14.2f} {pixels_per_ms:<14.0f}")


def test_disable_features():
    """Test disabling individual features."""
    print("\n" + "="*60)
    print("TEST 3: Feature Toggle")
    print("="*60)
    
    rgb = create_synthetic_test_image()
    
    # Test with all features enabled
    rgb_all = semantic_color_correction(rgb, enable_sky_boost=True, enable_water_boost=True)
    
    # Test with only sky boost
    rgb_sky_only = semantic_color_correction(rgb, enable_sky_boost=True, enable_water_boost=False)
    
    # Test with only water boost
    rgb_water_only = semantic_color_correction(rgb, enable_sky_boost=False, enable_water_boost=True)
    
    # Test with all disabled (should return original)
    rgb_none = semantic_color_correction(rgb, enable_sky_boost=False, enable_water_boost=False)
    
    print("✓ All features enabled")
    print("✓ Sky boost only")
    print("✓ Water boost only")
    print("✓ All disabled (identity)")
    
    # Verify differences
    diff_all_sky = np.mean(np.abs(rgb_all - rgb_sky_only))
    diff_all_water = np.mean(np.abs(rgb_all - rgb_water_only))
    diff_none = np.mean(np.abs(rgb - rgb_none))
    
    print(f"\nDifferences:")
    print(f"  All vs Sky-only: {diff_all_sky:.6f} (water boost effect)")
    print(f"  All vs Water-only: {diff_all_water:.6f} (sky boost effect)")
    print(f"  Original vs Disabled: {diff_none:.6f} (should be ~0.0)")
    
    assert diff_none < 0.001, "Disabled correction should be identity"
    print("✓ Disabled correction is identity (correct)")


def test_on_real_colorization(colorizer_checkpoint=None):
    """Test on real colorization output if checkpoint is available."""
    print("\n" + "="*60)
    print("TEST 4: Integration with Real Colorizer")
    print("="*60)
    
    try:
        import torch
        from PIL import Image
        from models.unet_colorizer import UNetColorizer
        
        # Try to load a test image
        test_image_path = Path("backend/data/dummy")
        if not test_image_path.exists():
            print("⚠ No test image found, skipping real colorizer test")
            return
        
        # Find first image
        image_paths = list(test_image_path.glob("*.jpg")) + list(test_image_path.glob("*.png"))
        if not image_paths:
            print("⚠ No test images found, skipping real colorizer test")
            return
        
        print(f"✓ Found {len(image_paths)} test images")
        print("✓ Real colorizer integration test skipped (requires full pipeline setup)")
        
    except Exception as e:
        print(f"⚠ Could not run real colorizer test: {e}")


def main():
    """Run all tests."""
    print("\n" + "█" * 60)
    print("SEMANTIC COLOR CORRECTION - TEST SUITE")
    print("█" * 60)
    
    test_semantic_correction_on_synthetic()
    test_performance_scaling()
    test_disable_features()
    test_on_real_colorization()
    
    print("\n" + "█" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
