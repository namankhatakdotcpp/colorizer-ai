#!/usr/bin/env python3
"""
Utility script for testing and validating Stage 5 GAN Refinement.

Tests:
- Model loading and inference
- Image preprocessing/postprocessing
- Batch processing
- Memory consumption
- Inference speed
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.stage5_gan_refinement import Stage5GANRefinement, create_stage5_refiner


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Stage5Tester:
    """Test and validate Stage 5 GAN refinement."""

    def __init__(self, checkpoint_path: Path = None):
        """
        Initialize tester.

        Args:
            checkpoint_path: Path to GAN checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.refiner = Stage5GANRefinement(
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
        logger.info(f"Tester initialized on {self.device}")

    def test_model_loading(self) -> bool:
        """Test model loading."""
        logger.info("Testing model loading...")
        try:
            assert self.refiner.generator is not None
            num_params = sum(p.numel() for p in self.refiner.generator.parameters())
            logger.info(f"✓ Model loaded with {num_params:,} parameters")
            return True
        except Exception as e:
            logger.error(f"✗ Model loading failed: {e}")
            return False

    def test_single_image_inference(
        self,
        image_path: Path,
        num_runs: int = 5,
    ) -> Dict:
        """
        Test single image inference.

        Args:
            image_path: Path to test image
            num_runs: Number of inference runs for timing

        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing single image inference on {image_path}...")

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return {"success": False}

            logger.info(f"Image shape: {image.shape}")

            # Warm up
            self.refiner.refine(image.copy())

            # Time inference
            times = []
            for _ in range(num_runs):
                start = time.time()
                output = self.refiner.refine(image.copy())
                times.append(time.time() - start)

            avg_time = np.mean(times)
            std_time = np.std(times)

            logger.info(f"✓ Inference successful")
            logger.info(f"  Average time: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
            logger.info(f"  FPS: {1/avg_time:.1f}")
            logger.info(f"  Output shape: {output.shape}")

            return {
                "success": True,
                "avg_time_ms": avg_time * 1000,
                "fps": 1 / avg_time,
                "output_shape": output.shape,
            }

        except Exception as e:
            logger.error(f"✗ Single image inference failed: {e}")
            return {"success": False}

    def test_batch_inference(
        self,
        image_path: Path,
        batch_size: int = 4,
    ) -> Dict:
        """
        Test batch inference.

        Args:
            image_path: Path to test image (will create batch by copying)
            batch_size: Batch size

        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing batch inference (batch_size={batch_size})...")

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return {"success": False}

            # Create batch
            batch = np.stack([image.copy() for _ in range(batch_size)])

            start = time.time()
            batch_output = self.refiner.refine_batch(batch)
            elapsed = time.time() - start

            logger.info(f"✓ Batch inference successful")
            logger.info(f"  Batch size: {batch_size}")
            logger.info(f"  Total time: {elapsed*1000:.2f}ms")
            logger.info(f"  Time per image: {elapsed*1000/batch_size:.2f}ms")
            logger.info(f"  Output shape: {batch_output.shape}")

            return {
                "success": True,
                "batch_size": batch_size,
                "total_time_ms": elapsed * 1000,
                "time_per_image_ms": elapsed * 1000 / batch_size,
                "output_shape": batch_output.shape,
            }

        except Exception as e:
            logger.error(f"✗ Batch inference failed: {e}")
            return {"success": False}

    def test_different_image_sizes(
        self,
        base_image_path: Path,
    ) -> Dict:
        """
        Test inference on different image sizes.

        Args:
            base_image_path: Path to base image

        Returns:
            Dictionary with results for different sizes
        """
        logger.info("Testing different image sizes...")

        image = cv2.imread(str(base_image_path))
        if image is None:
            logger.error(f"Failed to load image: {base_image_path}")
            return {"success": False}

        sizes = [256, 512, 768, 1024]
        results = {"success": True, "sizes": {}}

        for size in sizes:
            try:
                resized = cv2.resize(image, (size, size))
                start = time.time()
                output = self.refiner.refine(resized)
                elapsed = time.time() - start

                logger.info(f"✓ {size}x{size}: {elapsed*1000:.2f}ms")
                results["sizes"][size] = {
                    "time_ms": elapsed * 1000,
                    "output_shape": output.shape,
                }

            except Exception as e:
                logger.warning(f"✗ {size}x{size}: {e}")
                results["sizes"][size] = {"error": str(e)}

        return results

    def test_memory_usage(
        self,
        image_path: Path,
        max_batch_size: int = 16,
    ) -> Dict:
        """
        Test memory usage.

        Args:
            image_path: Path to test image
            max_batch_size: Maximum batch size to test

        Returns:
            Dictionary with memory usage results
        """
        logger.info("Testing memory usage...")

        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return {"success": False}

        results = {"success": True, "batches": {}}

        for batch_size in [1, 2, 4, 8, 16]:
            if batch_size > max_batch_size:
                break

            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                batch = np.stack([image.copy() for _ in range(batch_size)])
                output = self.refiner.refine_batch(batch)

                torch.cuda.synchronize()
                memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

                logger.info(f"✓ Batch {batch_size}: {memory_used:.0f}MB")
                results["batches"][batch_size] = {"memory_mb": memory_used}

            except RuntimeError as e:
                logger.warning(f"✗ Batch {batch_size}: {e}")
                results["batches"][batch_size] = {"error": str(e)}

        return results

    def test_output_quality(
        self,
        input_image_path: Path,
        reference_image_path: Path,
    ) -> Dict:
        """
        Test output quality (compare with reference).

        Args:
            input_image_path: Path to input colorized image
            reference_image_path: Path to reference/ground truth

        Returns:
            Dictionary with quality metrics
        """
        logger.info("Testing output quality...")

        input_image = cv2.imread(str(input_image_path))
        reference_image = cv2.imread(str(reference_image_path))

        if input_image is None or reference_image is None:
            logger.error("Failed to load images")
            return {"success": False}

        try:
            # Run refinement
            refined_image = self.refiner.refine(input_image)

            # Ensure same size
            if refined_image.shape != reference_image.shape:
                reference_image = cv2.resize(reference_image, (refined_image.shape[1], refined_image.shape[0]))

            # Compute metrics
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity

            refined_float = refined_image.astype(np.float32) / 255.0
            reference_float = reference_image.astype(np.float32) / 255.0

            psnr = peak_signal_noise_ratio(reference_float, refined_float, data_range=1.0)
            ssim = structural_similarity(reference_float, refined_float, channel_axis=2, data_range=1.0)

            logger.info(f"✓ Quality metrics computed")
            logger.info(f"  PSNR: {psnr:.2f} dB")
            logger.info(f"  SSIM: {ssim:.4f}")

            return {
                "success": True,
                "psnr": psnr,
                "ssim": ssim,
            }

        except Exception as e:
            logger.error(f"✗ Quality test failed: {e}")
            return {"success": False}

    def run_all_tests(
        self,
        test_image_path: Path,
        reference_image_path: Path = None,
    ) -> Dict:
        """
        Run all tests.

        Args:
            test_image_path: Path to test image
            reference_image_path: Path to reference image (optional)

        Returns:
            Dictionary with all test results
        """
        logger.info("="*60)
        logger.info("STAGE 5 GAN REFINEMENT - COMPREHENSIVE TEST SUITE")
        logger.info("="*60)

        all_results = {}

        # Test 1: Model loading
        all_results["model_loading"] = self.test_model_loading()

        # Test 2: Single image inference
        all_results["single_image_inference"] = self.test_single_image_inference(
            test_image_path,
            num_runs=5,
        )

        # Test 3: Batch inference
        all_results["batch_inference"] = self.test_batch_inference(
            test_image_path,
            batch_size=4,
        )

        # Test 4: Different sizes
        all_results["different_sizes"] = self.test_different_image_sizes(
            test_image_path,
        )

        # Test 5: Memory usage
        if torch.cuda.is_available():
            all_results["memory_usage"] = self.test_memory_usage(
                test_image_path,
                max_batch_size=16,
            )

        # Test 6: Quality (if reference provided)
        if reference_image_path and Path(reference_image_path).exists():
            all_results["output_quality"] = self.test_output_quality(
                test_image_path,
                reference_image_path,
            )

        logger.info("="*60)
        logger.info("TEST SUITE COMPLETED")
        logger.info("="*60)

        return all_results


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test and validate Stage 5 GAN Refinement",
    )

    parser.add_argument(
        "--test-image",
        type=str,
        required=True,
        help="Path to test image",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to GAN checkpoint (optional)",
    )
    parser.add_argument(
        "--reference-image",
        type=str,
        default=None,
        help="Path to reference image for quality comparison (optional)",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default="stage5_test_output.jpg",
        help="Path to save test output image",
    )

    args = parser.parse_args()

    # Create tester
    tester = Stage5Tester(checkpoint_path=Path(args.checkpoint) if args.checkpoint else None)

    # Run tests
    results = tester.run_all_tests(
        test_image_path=Path(args.test_image),
        reference_image_path=args.reference_image,
    )

    # Save simple test output
    if Path(args.test_image).exists():
        image = cv2.imread(args.test_image)
        refined = tester.refiner.refine(image)
        cv2.imwrite(args.output_image, refined)
        logger.info(f"Test output saved to {args.output_image}")


if __name__ == "__main__":
    main()
