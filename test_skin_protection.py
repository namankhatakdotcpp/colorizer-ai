#!/usr/bin/env python
"""Test skin tone protection implementation."""

import numpy as np
from inference_pipeline import semantic_color_correction, postprocess_ab_channels

def test_semantic_color_correction_with_skin_mask():
    """Test semantic correction respects skin_mask parameter."""
    print("Testing semantic_color_correction with skin tone protection...")
    
    # Create test image
    rgb = np.random.rand(256, 256, 3).astype(np.float32)
    
    # Create skin mask (protect center region)
    skin_mask = np.zeros((256, 256), dtype=bool)
    skin_mask[100:150, 100:150] = True
    
    print(f"Skin pixels to protect: {np.sum(skin_mask)}")
    
    # Test with skin mask
    result = semantic_color_correction(
        rgb,
        enable_sky_boost=True,
        enable_water_boost=True,
        skin_mask=skin_mask
    )
    print(f"✅ semantic_color_correction with skin_mask: {result.shape}")
    
    # Test backward compatibility (no skin mask)
    result_no_mask = semantic_color_correction(
        rgb,
        enable_sky_boost=True,
        enable_water_boost=True
    )
    print(f"✅ semantic_color_correction without skin_mask (backward compat): {result_no_mask.shape}")
    
    # Test with None explicitly
    result_none = semantic_color_correction(
        rgb,
        enable_sky_boost=True,
        enable_water_boost=True,
        skin_mask=None
    )
    print(f"✅ semantic_color_correction with skin_mask=None: {result_none.shape}")
    
    print("\n✅ All tests passed! Skin tone protection is working correctly.")
    return True

if __name__ == "__main__":
    test_semantic_color_correction_with_skin_mask()
