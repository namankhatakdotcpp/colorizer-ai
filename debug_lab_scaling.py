#!/usr/bin/env python3
"""
Critical Debug Script: Verify LAB Color Space Scaling

This script checks for LAB color space scaling bugs in the colorization pipeline.

Issues to detect:
1. AB channel range mismatch (expected [-1, 1] but getting [-128, 127] or vice versa)
2. Incorrect scaling in LAB→RGB reconstruction
3. Generator predicting AB channels in wrong range
4. L channel scaling inconsistencies

Run this DURING training to verify AB channels are in correct range.
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torchvision
from skimage.color import rgb2lab, lab2rgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LABScalingValidator:
    """Validates LAB color space handling and scaling."""

    def __init__(self):
        """Initialize validator."""
        self.issues_found = []
        self.warnings = []

    def validate_ab_range(self, ab_tensor: torch.Tensor, name: str = "AB") -> Dict[str, Any]:
        """
        Validate AB channel range.

        Expected for normalized LAB:
        - L: [0, 100] or [0, 1] (depending on normalization)
        - A: [-128, 127] or [-1, 1] (depending on normalization)
        - B: [-128, 127] or [-1, 1] (depending on normalization)

        Args:
            ab_tensor: AB tensor (B, 2, H, W)
            name: Name for logging

        Returns:
            Dictionary with statistics and warnings
        """
        assert ab_tensor.shape[1] == 2, f"Expected 2 channels for AB, got {ab_tensor.shape[1]}"

        ab_np = ab_tensor.detach().cpu().numpy()
        a_channel = ab_np[:, 0]
        b_channel = ab_np[:, 1]

        stats = {
            "name": name,
            "a_min": a_channel.min(),
            "a_max": a_channel.max(),
            "a_mean": a_channel.mean(),
            "a_std": a_channel.std(),
            "b_min": b_channel.min(),
            "b_max": b_channel.max(),
            "b_mean": b_channel.mean(),
            "b_std": b_channel.std(),
            "a_near_zero": np.abs(a_channel).mean() < 0.1,
            "b_near_zero": np.abs(b_channel).mean() < 0.1,
        }

        # Detect likely range
        if a_channel.min() >= -1.5 and a_channel.max() <= 1.5:
            detected_range = "[-1, 1]"
        elif a_channel.min() >= -150 and a_channel.max() <= 150:
            detected_range = "[-128, 127] or similar"
        else:
            detected_range = "Unknown/Unexpected"

        logger.info(f"\n{'='*70}")
        logger.info(f"LAB Scaling Analysis: {name}")
        logger.info(f"{'='*70}")
        logger.info(f"Detected Range: {detected_range}")
        logger.info(f"\nA Channel Statistics:")
        logger.info(f"  Min: {stats['a_min']:.6f}")
        logger.info(f"  Max: {stats['a_max']:.6f}")
        logger.info(f"  Mean: {stats['a_mean']:.6f}")
        logger.info(f"  Std: {stats['a_std']:.6f}")
        logger.info(f"\nB Channel Statistics:")
        logger.info(f"  Min: {stats['b_min']:.6f}")
        logger.info(f"  Max: {stats['b_max']:.6f}")
        logger.info(f"  Mean: {stats['b_mean']:.6f}")
        logger.info(f"  Std: {stats['b_std']:.6f}")

        # Check for issues
        if stats['a_near_zero'] and stats['b_near_zero']:
            msg = f"⚠️  {name}: AB channels are NEAR ZERO (likely grayscale convergence!)"
            logger.warning(msg)
            self.issues_found.append(msg)

        if detected_range == "[-1, 1]":
            logger.info("✅ Range appears correct: [-1, 1]")
        elif detected_range == "[-128, 127] or similar":
            logger.warning(f"⚠️  {name}: Range appears to be [-128, 127], should be [-1, 1]")
            self.warnings.append(
                f"{name}: AB range is [-128, 127] but should be [-1, 1]. "
                "This could cause incorrect reconstruction."
            )

        return stats

    def test_lab_reconstruction(self):
        """
        Test LAB→RGB reconstruction with various inputs.

        Verifies:
        1. Scaling is consistent
        2. Reconstruction produces valid RGB
        3. No NaN/Inf values
        """
        logger.info(f"\n{'='*70}")
        logger.info("Testing LAB→RGB Reconstruction")
        logger.info(f"{'='*70}")

        # Test 1: Pure grayscale (A=0, B=0)
        logger.info("\nTest 1: Pure Grayscale (A=0, B=0)")
        l_test = torch.ones(1, 1, 64, 64) * 50  # L=50 (mid-gray)
        ab_test = torch.zeros(1, 2, 64, 64)  # A=0, B=0

        rgb = self._reconstruct_rgb(l_test, ab_test)
        logger.info(f"  Input: L=50, A=0, B=0")
        logger.info(f"  Output RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        logger.info(f"  Output looks grayscale: {self._is_grayscale(rgb, tolerance=0.05)}")

        # Test 2: Red color (A>0, B>0)
        logger.info("\nTest 2: Red Color (A>0, B>0)")
        l_test = torch.ones(1, 1, 64, 64) * 50
        ab_test = torch.zeros(1, 2, 64, 64)
        ab_test[:, 0, :, :] = 0.5  # A=0.5 (positive = red-green axis)
        ab_test[:, 1, :, :] = 0.5  # B=0.5 (positive = yellow-blue axis)

        rgb = self._reconstruct_rgb(l_test, ab_test)
        logger.info(f"  Input: L=50, A=0.5, B=0.5")
        logger.info(f"  Output RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        logger.info(f"  Output looks grayscale: {self._is_grayscale(rgb, tolerance=0.05)}")
        logger.info(f"  Output has color variation: {self._has_color_variation(rgb)}")

        # Test 3: High saturation
        logger.info("\nTest 3: High Saturation (A=0.9, B=0.9)")
        ab_test = torch.zeros(1, 2, 64, 64)
        ab_test[:, 0, :, :] = 0.9
        ab_test[:, 1, :, :] = 0.9

        rgb = self._reconstruct_rgb(l_test, ab_test)
        logger.info(f"  Input: L=50, A=0.9, B=0.9")
        logger.info(f"  Output RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        logger.info(f"  Output looks grayscale: {self._is_grayscale(rgb, tolerance=0.05)}")
        logger.info(f"  Output has color variation: {self._has_color_variation(rgb)}")

    def _reconstruct_rgb(self, l_tensor: torch.Tensor, ab_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct RGB from LAB using proper scaling.

        Expected scaling:
        - L: [0, 100] or normalize to [0, 1]
        - AB: [-1, 1] normalized, scale by 128 for true LAB

        Args:
            l_tensor: L channel (B, 1, H, W)
            ab_tensor: AB channels (B, 2, H, W)

        Returns:
            RGB tensor (B, 3, H, W) in [0, 1]
        """
        # Ensure proper ranges
        l_np = l_tensor.squeeze(1).cpu().numpy()  # (B, H, W)
        ab_np = ab_tensor.cpu().numpy()  # (B, 2, H, W)

        batch_size = l_np.shape[0]
        rgb_list = []

        for b in range(batch_size):
            # Extract single image
            l_single = l_np[b]  # (H, W)
            ab_single = ab_np[b]  # (2, H, W)

            # Stack to LAB format expected by lab2rgb
            lab_img = np.stack([
                l_single,
                ab_single[0] * 128,  # Scale A from [-1,1] to [-128,127]
                ab_single[1] * 128   # Scale B from [-1,1] to [-128,127]
            ], axis=-1)  # (H, W, 3)

            # Convert LAB to RGB
            try:
                rgb_img = lab2rgb(lab_img)  # Returns [0, 1]
                rgb_list.append(rgb_img)
            except Exception as e:
                logger.error(f"Error in LAB→RGB conversion: {e}")
                # Return black image as fallback
                rgb_list.append(np.zeros_like(lab_img))

        # Stack batch
        rgb_np = np.stack(rgb_list, axis=0)  # (B, H, W, 3)
        rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()  # (B, 3, H, W)

        return rgb_tensor

    def _is_grayscale(self, rgb_tensor: torch.Tensor, tolerance: float = 0.05) -> bool:
        """
        Check if RGB image appears grayscale (R≈G≈B).

        Args:
            rgb_tensor: RGB tensor (B, 3, H, W) in [0, 1]
            tolerance: Maximum allowed standard deviation across RGB channels

        Returns:
            True if appears grayscale
        """
        rgb_np = rgb_tensor.cpu().numpy()
        r, g, b = rgb_np[:, 0], rgb_np[:, 1], rgb_np[:, 2]

        # Check if R, G, B are approximately equal
        max_diff = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b)).mean()

        return max_diff < tolerance

    def _has_color_variation(self, rgb_tensor: torch.Tensor) -> bool:
        """
        Check if RGB has meaningful color variation.

        Args:
            rgb_tensor: RGB tensor (B, 3, H, W) in [0, 1]

        Returns:
            True if has significant color variation
        """
        rgb_np = rgb_tensor.cpu().numpy()
        r, g, b = rgb_np[:, 0], rgb_np[:, 1], rgb_np[:, 2]

        # Compute chromatic variance
        chroma = np.sqrt((r - g)**2 + (g - b)**2 + (r - b)**2)
        chroma_variance = chroma.var()

        return chroma_variance > 0.01

    def print_summary(self):
        """Print summary of findings."""
        logger.info(f"\n{'='*70}")
        logger.info("SUMMARY")
        logger.info(f"{'='*70}")

        if self.issues_found:
            logger.error(f"\n❌ ISSUES FOUND ({len(self.issues_found)}):")
            for issue in self.issues_found:
                logger.error(f"  - {issue}")
        else:
            logger.info("\n✅ No critical issues detected")

        if self.warnings:
            logger.warning(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        logger.info("\nRecommendations:")
        logger.info("1. Ensure AB channels are in [-1, 1] range during prediction")
        logger.info("2. Verify scaling by * 128 in LAB→RGB reconstruction")
        logger.info("3. Check generator output layer activation (should allow full [-1, 1] range)")
        logger.info("4. Reduce L1 loss weight (now 5.0 instead of 50.0)")


def main():
    """Main debug script."""
    parser = argparse.ArgumentParser(description="Debug LAB color space scaling")
    args = parser.parse_args()

    logger.info("="*70)
    logger.info("LAB COLOR SPACE SCALING DEBUG SCRIPT")
    logger.info("="*70)

    validator = LABScalingValidator()

    # Test reconstruction
    validator.test_lab_reconstruction()

    # Print summary
    validator.print_summary()

    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS:")
    logger.info("="*70)
    logger.info("""
1. Monitor training with reduced L1 weight (lambda_l1=5.0)
2. Check predicted AB channel ranges during training:
   - Run this script with sample outputs from training
   - Look for AB channels being in correct [-1, 1] range
3. If AB channels still near zero:
   - Check generator output layer (should use Tanh for [-1, 1])
   - Verify discriminator is properly penalizing desaturated images
4. Increase discriminator pressure if needed:
   - Reduce learning rates: lr_d=2e-5, lr_g=2e-5
   - Increase n_critic (discriminator steps per generator step)
""")


if __name__ == "__main__":
    main()
