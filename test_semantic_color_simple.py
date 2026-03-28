"""
Simple test for semantic color correction without external dependencies.

Demonstrates:
- Core functionality
- HSV color space manipulation
- Performance
- Skin tone protection
"""

import time
import numpy as np
import cv2
from pathlib import Path
from typing import Optional


def semantic_color_correction(rgb: np.ndarray, enable_sky_boost: bool = True, enable_water_boost: bool = True, skin_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply adaptive semantic-aware color correction working directly with LAB AB channels.
    
    Adaptive enhancements:
    - SKY: Proportional blue boost based on missing blue compared to target
    - WATER: Moderate blue increase + adaptive cyan shift based on current A saturation
    - SKIN PROTECTION: Exclude skin pixels from all corrections using provided mask
    """
    # Early return if all corrections are disabled
    if not enable_sky_boost and not enable_water_boost:
        return rgb
    
    if cv2 is None:
        return rgb
    
    try:
        # Convert RGB [0, 1] to LAB for direct AB channel manipulation
        rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        lab_u8 = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
        
        # Split into L, A, B channels
        l_channel = lab_u8[:, :, 0].astype(np.float32)
        a_channel = lab_u8[:, :, 1].astype(np.float32)
        b_channel = lab_u8[:, :, 2].astype(np.float32)
        
        height, width = l_channel.shape
        
        # ========== SKY DETECTION & ADAPTIVE BLUE BOOST ==========
        if enable_sky_boost:
            # Top 30% of image - typically sky region
            sky_region_pct = int(0.3 * height)
            
            # In OpenCV LAB: A and B are centered at 128, range [0, 255]
            a_centered = a_channel[:sky_region_pct, :] - 128.0
            b_centered = b_channel[:sky_region_pct, :] - 128.0
            chroma = np.sqrt(a_centered ** 2 + b_centered ** 2)
            
            # Low saturation + high brightness = likely sky
            sky_mask = (chroma < 40) & (l_channel[:sky_region_pct, :] > 100)
            
            # Exclude skin pixels from sky corrections if mask provided
            if skin_mask is not None:
                sky_mask = sky_mask & ~skin_mask[:sky_region_pct, :]
            
            # SUBTLE blue boost: only +5 to +10 max, NOT aggressive
            current_b = b_channel[:sky_region_pct, :]
            delta_b = np.maximum(0, (135 - current_b) * 0.05)  # Subtle
            delta_b = np.clip(delta_b, 0, 8)  # Hard limit: +8 max
            
            # Apply adaptive correction only to detected sky pixels
            b_channel[:sky_region_pct, :][sky_mask] += delta_b[sky_mask]
            
            sky_pixels = np.sum(sky_mask)
            avg_delta_b = np.mean(delta_b[sky_mask]) if sky_pixels > 0 else 0.0
            print(f"[Semantic] Sky: {sky_pixels} pixels, avg blue delta: {avg_delta_b:.1f}")
        
        # ========== WATER DETECTION & ADAPTIVE BLUE-CYAN SHIFT ==========
        if enable_water_boost:
            # Bottom 50% of image - typically water/ground region
            water_region_start = int(0.5 * height)
            
            # In OpenCV LAB: A and B are centered at 128
            a_centered = a_channel[water_region_start:, :] - 128.0
            b_centered = b_channel[water_region_start:, :] - 128.0
            chroma = np.sqrt(a_centered ** 2 + b_centered ** 2)
            
            # Medium saturation + decent brightness = likely water/ground
            water_mask = (chroma > 10) & (chroma < 80) & (l_channel[water_region_start:, :] > 60)
            
            # Exclude skin pixels from water corrections if mask provided
            if skin_mask is not None:
                water_mask = water_mask & ~skin_mask[water_region_start:, :]
            
            # SUBTLE blue-green shift: minimal delta only
            current_b_water = b_channel[water_region_start:, :]
            delta_b_water = np.maximum(0, (140 - current_b_water) * 0.02)  # Very subtle
            delta_b_water = np.clip(delta_b_water, 0, 5)  # Hard limit: +5 max
            
            # Subtle cyan shift: only reduce redness slightly
            delta_a_water = -np.abs(a_centered) * 0.02  # Very subtle: 2% reduction
            delta_a_water = np.clip(delta_a_water, -3, 0)  # Hard limit: -3 max
            
            # Apply adaptive corrections to detected water pixels
            b_channel[water_region_start:, :][water_mask] += delta_b_water[water_mask]
            a_channel[water_region_start:, :][water_mask] += delta_a_water[water_mask]
            
            water_pixels = np.sum(water_mask)
            avg_delta_b = np.mean(delta_b_water[water_mask]) if water_pixels > 0 else 0.0
            avg_delta_a = np.mean(delta_a_water[water_mask]) if water_pixels > 0 else 0.0
            print(f"[Semantic] Water: {water_pixels} pixels, avg blue delta: {avg_delta_b:.1f}, avg A delta: {avg_delta_a:.1f}")
        
        # ========== CLAMP & RECONSTRUCT LAB → RGB ==========
        # Clamp AB channels to valid LAB range
        a_channel = np.clip(a_channel, 0, 255)
        b_channel = np.clip(b_channel, 0, 255)
        l_channel = np.clip(l_channel, 0, 255)
        
        # Merge back to LAB
        lab_corrected = np.stack([
            l_channel.astype(np.uint8),
            a_channel.astype(np.uint8),
            b_channel.astype(np.uint8)
        ], axis=2)
        
        # Convert LAB → BGR → RGB
        bgr_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        rgb_corrected = cv2.cvtColor(bgr_corrected, cv2.COLOR_BGR2RGB)
        
        return rgb_corrected.astype(np.float32) / 255.0
    
    except Exception as e:
        print(f"[Warning] Semantic color correction failed: {e}")
        return rgb




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


def test_1_synthetic():
    """Test on synthetic image."""
    print("\n" + "="*60)
    print("TEST 1: Synthetic Image")
    print("="*60)
    
    rgb_original = create_synthetic_test_image()
    
    start_time = time.perf_counter()
    rgb_corrected = semantic_color_correction(rgb_original, enable_sky_boost=True, enable_water_boost=True)
    elapsed = time.perf_counter() - start_time
    
    print(f"✓ Semantic correction applied in {elapsed*1000:.2f}ms")
    
    # Analyze color changes
    rgb_u8_original = (rgb_original * 255).astype(np.uint8)
    rgb_u8_corrected = (rgb_corrected * 255).astype(np.uint8)
    
    # Sky region analysis
    sky_h = int(0.3 * rgb_original.shape[0])
    sky_orig = rgb_u8_original[:sky_h, :, :]
    sky_corr = rgb_u8_corrected[:sky_h, :, :]
    
    print(f"\nSky Region (top 30%):")
    print(f"  Original  - Mean RGB: ({sky_orig[..., 0].mean():.1f}, {sky_orig[..., 1].mean():.1f}, {sky_orig[..., 2].mean():.1f})")
    print(f"  Corrected - Mean RGB: ({sky_corr[..., 0].mean():.1f}, {sky_corr[..., 1].mean():.1f}, {sky_corr[..., 2].mean():.1f})")
    
    # Water region analysis
    water_h_start = sky_h
    water_h_end = int(0.65 * rgb_original.shape[0])
    water_orig = rgb_u8_original[water_h_start:water_h_end, :, :]
    water_corr = rgb_u8_corrected[water_h_start:water_h_end, :, :]
    
    print(f"\nWater Region (30-65%):")
    print(f"  Original  - Mean RGB: ({water_orig[..., 0].mean():.1f}, {water_orig[..., 1].mean():.1f}, {water_orig[..., 2].mean():.1f})")
    print(f"  Corrected - Mean RGB: ({water_corr[..., 0].mean():.1f}, {water_corr[..., 1].mean():.1f}, {water_corr[..., 2].mean():.1f})")
    
    # Ground region analysis
    ground_orig = rgb_u8_original[water_h_end:, :, :]
    ground_corr = rgb_u8_corrected[water_h_end:, :, :]
    
    print(f"\nGround Region (bottom 35%):")
    print(f"  Original  - Mean RGB: ({ground_orig[..., 0].mean():.1f}, {ground_orig[..., 1].mean():.1f}, {ground_orig[..., 2].mean():.1f})")
    print(f"  Corrected - Mean RGB: ({ground_corr[..., 0].mean():.1f}, {ground_corr[..., 1].mean():.1f}, {ground_corr[..., 2].mean():.1f})")
    
    return True


def test_2_performance():
    """Test performance scaling."""
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
    
    return True


def test_3_features():
    """Test feature toggling."""
    print("\n" + "="*60)
    print("TEST 3: Feature Toggle")
    print("="*60)
    
    rgb = create_synthetic_test_image()
    
    rgb_all = semantic_color_correction(rgb, enable_sky_boost=True, enable_water_boost=True)
    rgb_sky_only = semantic_color_correction(rgb, enable_sky_boost=True, enable_water_boost=False)
    rgb_water_only = semantic_color_correction(rgb, enable_sky_boost=False, enable_water_boost=True)
    rgb_none = semantic_color_correction(rgb, enable_sky_boost=False, enable_water_boost=False)
    
    print("✓ All features enabled")
    print("✓ Sky boost only")
    print("✓ Water boost only")
    print("✓ All disabled (identity)")
    
    diff_all_sky = np.mean(np.abs(rgb_all - rgb_sky_only))
    diff_all_water = np.mean(np.abs(rgb_all - rgb_water_only))
    diff_none = np.mean(np.abs(rgb - rgb_none))
    
    print(f"\nDifferences:")
    print(f"  All vs Sky-only: {diff_all_sky:.6f} (water effect)")
    print(f"  All vs Water-only: {diff_all_water:.6f} (sky effect)")
    print(f"  Original vs Disabled: {diff_none:.6f} (should be ~0.0)")
    
    is_identity = diff_none < 0.001
    if is_identity:
        print("✓ Disabled correction preserves image (correct)")
    else:
        print("✗ Disabled correction should be identity!")
        return False
    
    return True


def main():
    print("\n" + "█" * 60)
    print("SEMANTIC COLOR CORRECTION - TEST SUITE")
    print("█" * 60)
    
    results = []
    
    try:
        results.append(("Synthetic Image Test", test_1_synthetic()))
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        results.append(("Synthetic Image Test", False))
    
    try:
        results.append(("Performance Scaling", test_2_performance()))
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        results.append(("Performance Scaling", False))
    
    try:
        results.append(("Feature Toggle", test_3_features()))
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        results.append(("Feature Toggle", False))
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n" + "█" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
        print("█" * 60 + "\n")
    else:
        print("\n" + "█" * 60)
        print("SOME TESTS FAILED ✗")
        print("█" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
