#!/usr/bin/env python3
"""
Production-level dataset preparation script for GAN refinement training.

Prepares dataset by:
1. Organizing images into ground_truth and colorized folders
2. Installing required dependencies
3. Running inference pipeline on images (or fallback to grayscale)
4. Generating summary statistics

Usage:
    python setup_dataset.py --input-dir dataset/images
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("setup_log.txt"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class DatasetPreparator:
    """Prepares dataset for GAN training."""

    def __init__(
        self,
        input_dir: Path,
        output_base_dir: Path = Path("data"),
        resize_size: int = 256,
    ):
        """
        Initialize dataset preparator.

        Args:
            input_dir: Directory containing original RGB images
            output_base_dir: Base directory for output (will create colorized/ and ground_truth/)
            resize_size: Target size for image resizing (square)
        """
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        self.colorized_dir = self.output_base_dir / "colorized"
        self.ground_truth_dir = self.output_base_dir / "ground_truth"
        self.resize_size = resize_size

        # Statistics
        self.stats = {
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "inference_success": 0,
            "inference_fallback": 0,
            "corrupted_images": [],
        }

        logger.info(f"Initialized DatasetPreparator")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output base directory: {self.output_base_dir}")
        logger.info(f"Resize size: {self.resize_size}x{self.resize_size}")

    def validate_input_dir(self) -> bool:
        """
        Validate input directory exists and contains images.

        Returns:
            True if valid, False otherwise
        """
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return False

        if not self.input_dir.is_dir():
            logger.error(f"Input path is not a directory: {self.input_dir}")
            return False

        image_files = self.get_image_files()
        if not image_files:
            logger.error(f"No image files found in {self.input_dir}")
            return False

        logger.info(f"Found {len(image_files)} images in input directory")
        return True

    def get_image_files(self) -> List[Path]:
        """
        Get list of image files from input directory.

        Returns:
            List of image file paths
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [
            f
            for f in self.input_dir.glob("*")
            if f.suffix.lower() in valid_extensions and f.is_file()
        ]
        image_files.sort()
        return image_files

    def prepare_directories(self) -> bool:
        """
        Create output directory structure.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.output_base_dir.mkdir(parents=True, exist_ok=True)
            self.colorized_dir.mkdir(parents=True, exist_ok=True)
            self.ground_truth_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Created directories:")
            logger.info(f"  - {self.output_base_dir}")
            logger.info(f"  - {self.colorized_dir}")
            logger.info(f"  - {self.ground_truth_dir}")

            return True
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.

        Args:
            image: Input image array

        Returns:
            Resized image array
        """
        return cv2.resize(
            image,
            (self.resize_size, self.resize_size),
            interpolation=cv2.INTER_LANCZOS4,
        )

    def save_ground_truth(self, image_path: Path, output_path: Path) -> bool:
        """
        Read image, resize, and save as ground truth.

        Args:
            image_path: Path to input image
            output_path: Path to save ground truth

        Returns:
            True if successful, False otherwise
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to read image: {image_path}")
                self.stats["corrupted_images"].append(str(image_path))
                return False

            # Resize
            image_resized = self.resize_image(image)

            # Save
            success = cv2.imwrite(str(output_path), image_resized)
            if not success:
                logger.warning(f"Failed to write image: {output_path}")
                return False

            return True
        except Exception as e:
            logger.error(f"Error saving ground truth for {image_path}: {e}")
            return False

    def convert_to_grayscale(self, image_path: Path) -> np.ndarray:
        """
        Convert image to grayscale.

        Args:
            image_path: Path to image

        Returns:
            Grayscale image array, or None if failed
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to read image: {image_path}")
                return None

            # Resize
            image_resized = self.resize_image(image)

            # Convert to grayscale
            grayscale = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

            # Convert back to 3-channel (simulate colorized input)
            grayscale_3ch = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

            return grayscale_3ch
        except Exception as e:
            logger.error(f"Error converting to grayscale {image_path}: {e}")
            return None

    def run_inference(self, input_path: Path, output_path: Path) -> bool:
        """
        Run inference pipeline on image.

        Args:
            input_path: Path to input grayscale image
            output_path: Path to save colorized output

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if inference_pipeline.py exists
            inference_script = Path("inference_pipeline.py")
            if not inference_script.exists():
                logger.debug("inference_pipeline.py not found, will use fallback")
                return False

            command = [
                "python",
                str(inference_script),
                str(input_path),
                "--stages",
                "colorizer",
                "sr",
                "--checkpoints",
                "checkpoints/",
                "--output-dir",
                str(output_path.parent),
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                logger.debug(f"Inference failed for {input_path}: {result.stderr[:100]}")
                return False

            # Try to find output file
            output_files = list(output_path.parent.glob("*"))
            if output_files:
                output_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest_output = output_files[0]
                latest_output.rename(output_path)
                return True
            else:
                logger.debug(f"No output files generated for {input_path}")
                return False

        except subprocess.TimeoutExpired:
            logger.debug(f"Inference timed out for {input_path}")
            return False
        except FileNotFoundError:
            logger.debug("inference_pipeline.py not found")
            return False
        except Exception as e:
            logger.debug(f"Error running inference: {e}")
            return False

    def process_image(self, image_path: Path) -> bool:
        """
        Process single image: save ground truth and colorized version.

        Args:
            image_path: Path to input image

        Returns:
            True if successful, False otherwise
        """
        try:
            image_stem = image_path.stem

            # Paths
            ground_truth_path = self.ground_truth_dir / f"{image_stem}_gt.jpg"
            colorized_path = self.colorized_dir / f"{image_stem}_colorized.jpg"

            # Step 1: Save ground truth
            if not self.save_ground_truth(image_path, ground_truth_path):
                logger.warning(f"Skipping {image_path.name}: failed to save ground truth")
                self.stats["failed_images"] += 1
                return False

            # Step 2: Create grayscale input
            grayscale_image = self.convert_to_grayscale(image_path)
            if grayscale_image is None:
                logger.warning(f"Skipping {image_path.name}: failed to convert to grayscale")
                self.stats["failed_images"] += 1
                return False

            # Step 3: Save temporary grayscale for inference
            temp_grayscale_path = self.colorized_dir / f"{image_stem}_temp_gray.jpg"
            cv2.imwrite(str(temp_grayscale_path), grayscale_image)

            # Step 4: Try to run inference
            inference_success = self.run_inference(temp_grayscale_path, colorized_path)

            if inference_success:
                self.stats["inference_success"] += 1
                # Clean up temp file
                try:
                    temp_grayscale_path.unlink()
                except:
                    pass
            else:
                # Fallback: use grayscale as colorized
                logger.debug(f"Using fallback (grayscale) for {image_path.name}")
                success = cv2.imwrite(str(colorized_path), grayscale_image)
                if not success:
                    logger.warning(f"Fallback failed for {image_path.name}")
                    self.stats["failed_images"] += 1
                    return False
                self.stats["inference_fallback"] += 1
                # Clean up temp file
                try:
                    temp_grayscale_path.unlink()
                except:
                    pass

            self.stats["processed_images"] += 1
            logger.info(f"✓ Processed: {image_path.name}")
            return True

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            self.stats["failed_images"] += 1
            return False

    def prepare_dataset(self) -> bool:
        """
        Prepare full dataset.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("STARTING DATASET PREPARATION")
        logger.info("=" * 60)

        # Validate input
        if not self.validate_input_dir():
            logger.error("Input validation failed")
            return False

        # Create directories
        if not self.prepare_directories():
            logger.error("Failed to create directories")
            return False

        # Get image files
        image_files = self.get_image_files()
        self.stats["total_images"] = len(image_files)

        logger.info(f"Processing {len(image_files)} images...")

        # Process images
        with tqdm(total=len(image_files), desc="Preparing dataset") as pbar:
            for image_path in image_files:
                pbar.set_description(f"Processing {image_path.name}")
                self.process_image(image_path)
                pbar.update(1)

        # Verify file counts match
        self.verify_dataset()

        return True

    def verify_dataset(self) -> None:
        """Verify dataset integrity."""
        colorized_files = list(self.colorized_dir.glob("*.jpg"))
        ground_truth_files = list(self.ground_truth_dir.glob("*.jpg"))

        logger.info(f"Colorized folder: {len(colorized_files)} files")
        logger.info(f"Ground truth folder: {len(ground_truth_files)} files")

        if len(colorized_files) != len(ground_truth_files):
            logger.warning(
                f"File count mismatch: colorized={len(colorized_files)}, "
                f"ground_truth={len(ground_truth_files)}"
            )

    def print_summary(self) -> None:
        """Print summary statistics."""
        logger.info("=" * 60)
        logger.info("DATASET PREPARATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total images processed: {self.stats['total_images']}")
        logger.info(f"Successfully processed: {self.stats['processed_images']}")
        logger.info(f"Failed images: {self.stats['failed_images']}")
        logger.info(f"Inference success: {self.stats['inference_success']}")
        logger.info(f"Inference fallback (grayscale): {self.stats['inference_fallback']}")

        if self.stats["corrupted_images"]:
            logger.warning(f"Corrupted images ({len(self.stats['corrupted_images'])}):")
            for image in self.stats["corrupted_images"][:5]:
                logger.warning(f"  - {image}")
            if len(self.stats["corrupted_images"]) > 5:
                logger.warning(f"  ... and {len(self.stats['corrupted_images']) - 5} more")

        logger.info("=" * 60)
        logger.info(f"Output directory: {self.output_base_dir}")
        logger.info(f"Colorized dir: {self.colorized_dir}")
        logger.info(f"Ground truth dir: {self.ground_truth_dir}")
        logger.info("=" * 60)


def install_dependencies() -> bool:
    """
    Install required dependencies.

    Returns:
        True if successful, False otherwise
    """
    logger.info("Installing required dependencies...")

    packages = [
        "torch",
        "torchvision",
        "tqdm",
        "opencv-python",
        "pytorch-fid",
        "scikit-image",
    ]

    try:
        for package in packages:
            logger.info(f"Installing {package}...")
            result = subprocess.run(
                ["pip", "install", package, "-q"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                logger.warning(f"Installation of {package} had warnings (may be already installed)")
            else:
                logger.info(f"✓ {package} installed")

        logger.info("Dependencies installation complete")
        return True

    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset for GAN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_dataset.py --input-dir dataset/images
  python setup_dataset.py --input-dir ~/Downloads/colorized_images --output-dir my_data
  python setup_dataset.py --input-dir data/raw --no-install
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing original RGB images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Base output directory (default: data)",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        default=False,
        help="Skip dependency installation",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        default=256,
        help="Target size for image resizing (default: 256)",
    )

    args = parser.parse_args()

    # Install dependencies if requested
    if not args.no_install:
        if not install_dependencies():
            logger.warning("Some dependencies failed to install, continuing anyway...")

    # Create preparator
    try:
        preparator = DatasetPreparator(
            input_dir=args.input_dir,
            output_base_dir=Path(args.output_dir),
            resize_size=args.resize_size,
        )
    except Exception as e:
        logger.error(f"Failed to initialize preparator: {e}")
        sys.exit(1)

    # Prepare dataset
    success = preparator.prepare_dataset()

    # Print summary
    preparator.print_summary()

    if success:
        logger.info("Dataset preparation completed successfully!")
        sys.exit(0)
    else:
        logger.error("Dataset preparation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
