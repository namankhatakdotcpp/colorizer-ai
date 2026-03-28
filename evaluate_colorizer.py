#!/usr/bin/env python3
"""
Production-level evaluation script for image colorization models.

- Converts RGB images to grayscale
- Runs colorization inference
- Computes PSNR, SSIM, and FID metrics
- Generates evaluation report

Usage:
    python evaluate_colorizer.py --dataset dataset/images --checkpoints checkpoints/
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ColorizerEvaluator:
    """Evaluates colorization model performance."""

    def __init__(
        self,
        dataset_dir: Path,
        checkpoints_dir: Path,
        output_dir: Path,
        resize_size: int = 256,
    ):
        """
        Initialize evaluator.

        Args:
            dataset_dir: Directory containing original RGB images
            checkpoints_dir: Directory containing model checkpoints
            output_dir: Directory to save evaluation outputs
            resize_size: Target size for image resizing (square)
        """
        self.dataset_dir = Path(dataset_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.output_dir = Path(output_dir)
        self.resize_size = resize_size

        # Create output subdirectories
        self.grayscale_dir = self.output_dir / "grayscale"
        self.generated_dir = self.output_dir / "generated"
        self.ground_truth_dir = self.output_dir / "ground_truth"

        for directory in [self.grayscale_dir, self.generated_dir, self.ground_truth_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directories created: {self.output_dir}")

    def get_image_files(self) -> list:
        """Get list of image files from dataset directory."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [
            f
            for f in self.dataset_dir.glob("*")
            if f.suffix.lower() in valid_extensions and f.is_file()
        ]
        image_files.sort()
        logger.info(f"Found {len(image_files)} images in dataset")
        return image_files

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.

        Args:
            image: Input image array

        Returns:
            Resized image array
        """
        return cv2.resize(
            image, (self.resize_size, self.resize_size), interpolation=cv2.INTER_LANCZOS4
        )

    def convert_to_grayscale(self, image_path: Path, save_path: Path) -> bool:
        """
        Convert RGB image to grayscale and save.

        Args:
            image_path: Path to input RGB image
            save_path: Path to save grayscale image

        Returns:
            True if successful, False otherwise
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to read image: {image_path}")
                return False

            # Resize image
            image = self.resize_image(image)

            # Convert BGR to grayscale
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Save grayscale image
            success = cv2.imwrite(str(save_path), grayscale)
            if not success:
                logger.warning(f"Failed to write image: {save_path}")
                return False

            return True
        except Exception as e:
            logger.error(f"Error converting to grayscale {image_path}: {e}")
            return False

    def save_ground_truth(self, image_path: Path, save_path: Path) -> bool:
        """
        Save resized ground truth image.

        Args:
            image_path: Path to input RGB image
            save_path: Path to save ground truth image

        Returns:
            True if successful, False otherwise
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to read image: {image_path}")
                return False

            # Resize image
            image = self.resize_image(image)

            # Save ground truth (convert to RGB for PIL compatibility)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            success = cv2.imwrite(str(save_path), image_rgb)
            if not success:
                logger.warning(f"Failed to write image: {save_path}")
                return False

            return True
        except Exception as e:
            logger.error(f"Error saving ground truth {image_path}: {e}")
            return False

    def run_inference(self, input_path: Path, output_path: Path) -> bool:
        """
        Run colorization inference on grayscale image.

        Args:
            input_path: Path to grayscale input image
            output_path: Path to save colorized output

        Returns:
            True if successful, False otherwise
        """
        try:
            command = [
                "python",
                "inference_pipeline.py",
                str(input_path),
                "--stages",
                "colorizer",
                "sr",
                "--checkpoints",
                str(self.checkpoints_dir),
                "--output-dir",
                str(self.generated_dir),
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                logger.warning(f"Inference failed for {input_path}: {result.stderr}")
                return False

            # The inference pipeline outputs to a specific naming convention
            # Try to find the output file
            generated_files = list(self.generated_dir.glob("*"))
            if generated_files:
                # Get the most recently created file
                generated_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest_output = generated_files[0]
                # Move/rename to expected location
                latest_output.rename(output_path)
                return True
            else:
                logger.warning(f"No output generated for {input_path}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Inference timed out for {input_path}")
            return False
        except Exception as e:
            logger.error(f"Error running inference for {input_path}: {e}")
            return False

    def compute_psnr_ssim(self, gt_path: Path, pred_path: Path) -> Tuple[float, float]:
        """
        Compute PSNR and SSIM between ground truth and prediction.

        Args:
            gt_path: Path to ground truth image
            pred_path: Path to predicted image

        Returns:
            Tuple of (PSNR, SSIM) values
        """
        try:
            # Read images
            gt = cv2.imread(str(gt_path))
            pred = cv2.imread(str(pred_path))

            if gt is None or pred is None:
                logger.warning(f"Failed to read images for metric computation")
                return 0.0, 0.0

            # Ensure same size
            if gt.shape != pred.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

            # Convert to float and normalize
            gt = gt.astype(np.float32) / 255.0
            pred = pred.astype(np.float32) / 255.0

            # Compute PSNR
            psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)

            # Compute SSIM (average across channels)
            ssim = structural_similarity(gt, pred, channel_axis=2, data_range=1.0)

            return float(psnr), float(ssim)

        except Exception as e:
            logger.error(f"Error computing PSNR/SSIM: {e}")
            return 0.0, 0.0

    def compute_fid_score(self) -> float:
        """
        Compute Fréchet Inception Distance (FID) between real and generated images.

        Args:
            None (uses ground_truth_dir and generated_dir)

        Returns:
            FID score, or -1.0 if computation fails
        """
        try:
            command = [
                "python",
                "-m",
                "pytorch_fid",
                str(self.ground_truth_dir),
                str(self.generated_dir),
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                logger.warning(f"FID computation failed: {result.stderr}")
                return -1.0

            # Parse FID score from output
            output_lines = result.stdout.strip().split("\n")
            for line in output_lines:
                if "FID" in line:
                    # Extract float value
                    parts = line.split()
                    for i, part in enumerate(parts):
                        try:
                            fid_score = float(part)
                            logger.info(f"FID Score: {fid_score:.4f}")
                            return fid_score
                        except ValueError:
                            continue

            logger.warning("Could not parse FID score from output")
            return -1.0

        except subprocess.TimeoutExpired:
            logger.error("FID computation timed out")
            return -1.0
        except FileNotFoundError:
            logger.warning(
                "pytorch-fid not installed. Install with: pip install pytorch-fid"
            )
            return -1.0
        except Exception as e:
            logger.error(f"Error computing FID: {e}")
            return -1.0

    def evaluate(self) -> Dict:
        """
        Run full evaluation pipeline.

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Starting colorization evaluation pipeline...")

        image_files = self.get_image_files()
        if not image_files:
            logger.error("No images found in dataset directory")
            return {}

        psnr_scores = []
        ssim_scores = []
        successful_evaluations = 0

        # Process each image
        with tqdm(total=len(image_files), desc="Evaluating images") as pbar:
            for image_path in image_files:
                pbar.set_description(f"Processing {image_path.name}")

                # Generate output filenames
                image_stem = image_path.stem
                grayscale_path = self.grayscale_dir / f"{image_stem}_gray.jpg"
                ground_truth_path = self.ground_truth_dir / f"{image_stem}_gt.jpg"
                generated_path = self.generated_dir / f"{image_stem}_colorized.jpg"

                # Step 1: Save ground truth
                if not self.save_ground_truth(image_path, ground_truth_path):
                    logger.warning(f"Skipping {image_path.name}: failed to save ground truth")
                    pbar.update(1)
                    continue

                # Step 2: Convert to grayscale
                if not self.convert_to_grayscale(image_path, grayscale_path):
                    logger.warning(f"Skipping {image_path.name}: failed to convert to grayscale")
                    pbar.update(1)
                    continue

                # Step 3: Run inference
                if not self.run_inference(grayscale_path, generated_path):
                    logger.warning(f"Skipping {image_path.name}: inference failed")
                    pbar.update(1)
                    continue

                # Step 4: Compute metrics
                psnr, ssim = self.compute_psnr_ssim(ground_truth_path, generated_path)
                if psnr > 0 and ssim > 0:
                    psnr_scores.append(psnr)
                    ssim_scores.append(ssim)
                    successful_evaluations += 1
                    logger.info(
                        f"{image_path.name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}"
                    )
                else:
                    logger.warning(f"Skipping {image_path.name}: failed to compute metrics")

                pbar.update(1)

        # Compute aggregates
        logger.info(f"Successfully evaluated {successful_evaluations}/{len(image_files)} images")

        metrics = {
            "psnr": float(np.mean(psnr_scores)) if psnr_scores else 0.0,
            "ssim": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
            "fid": self.compute_fid_score(),
            "num_images": len(image_files),
            "successful_evaluations": successful_evaluations,
        }

        return metrics

    def save_results(self, metrics: Dict) -> None:
        """
        Save evaluation results to JSON file.

        Args:
            metrics: Dictionary containing evaluation metrics
        """
        results_file = self.output_dir / "results.json"
        try:
            with open(results_file, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate colorization model on image dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_colorizer.py --dataset dataset/images --checkpoints checkpoints/
  python evaluate_colorizer.py --dataset data/ --checkpoints ckpt/ --output eval_results/
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory containing RGB images",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        required=True,
        help="Path to directory containing model checkpoints",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_outputs",
        help="Output directory for evaluation results (default: eval_outputs)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        help="Target size for image resizing (default: 256)",
    )

    args = parser.parse_args()

    # Validate input paths
    dataset_path = Path(args.dataset)
    checkpoints_path = Path(args.checkpoints)

    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        sys.exit(1)

    if not checkpoints_path.exists():
        logger.error(f"Checkpoints directory not found: {checkpoints_path}")
        sys.exit(1)

    # Create evaluator
    output_path = Path(args.output)
    evaluator = ColorizerEvaluator(
        dataset_dir=dataset_path,
        checkpoints_dir=checkpoints_path,
        output_dir=output_path,
        resize_size=args.resize,
    )

    # Run evaluation
    metrics = evaluator.evaluate()

    # Save results
    if metrics:
        evaluator.save_results(metrics)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Average PSNR:  {metrics['psnr']:.2f} dB")
        logger.info(f"Average SSIM:  {metrics['ssim']:.4f}")
        logger.info(f"FID Score:     {metrics['fid']:.4f}" if metrics['fid'] > 0 else "FID Score:     N/A")
        logger.info(f"Images evaluated: {metrics['successful_evaluations']}/{metrics['num_images']}")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {output_path / 'results.json'}")
    else:
        logger.error("Evaluation failed: no metrics generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
