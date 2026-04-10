#!/usr/bin/env python3
"""
Production-ready dataset builder for GAN colorization.

Combines COCO, DIV2K, and Flickr2K into paired input/target dataset.
Generates grayscale inputs from color targets with robust error handling.

Usage:
    python datasets/dataset_builder.py --mode grayscale --num-coco 20000
    python datasets/dataset_builder.py --mode stage4 --num-coco 25000
    python datasets/dataset_builder.py --validate-only
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import random
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import json
from datetime import datetime

# ============================================================================
# CONFIG & LOGGING
# ============================================================================

@dataclass
class Config:
    """Dataset builder configuration."""
    project_root: Path = Path(__file__).parent.parent
    datasets_dir: Path = Path(__file__).parent
    data_output_dir: Path = None
    
    # Image settings
    target_size: Tuple[int, int] = (256, 256)
    jpeg_quality: int = 95
    
    # Processing
    num_workers: int = 0  # auto-detect in __post_init__
    batch_size: int = 100
    
    # Dataset sources
    coco_dir: Path = None
    div2k_dir: Path = None
    flickr2k_dir: Path = None
    stage4_dir: Path = None
    
    # Counts
    num_coco_samples: int = 20000
    
    def __post_init__(self):
        """Initialize paths and defaults."""
        if self.data_output_dir is None:
            self.data_output_dir = self.project_root / "data"
        
        if self.coco_dir is None:
            self.coco_dir = self.datasets_dir / "coco"
        if self.div2k_dir is None:
            self.div2k_dir = self.datasets_dir / "div2k"
        if self.flickr2k_dir is None:
            self.flickr2k_dir = self.datasets_dir / "flickr2k"
        if self.stage4_dir is None:
            self.stage4_dir = self.project_root / "checkpoints" / "stage4_outputs"
        
        if self.num_workers == 0:
            self.num_workers = max(1, cpu_count() - 2)


def setup_logging() -> logging.Logger:
    """Configure logging with timestamps and colors."""
    logger = logging.getLogger("DatasetBuilder")
    logger.setLevel(logging.DEBUG)
    
    # Console handler with formatting
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = setup_logging()

# ============================================================================
# IMAGE UTILITIES
# ============================================================================

class ImageValidator:
    """Validate and preprocess images."""
    
    @staticmethod
    def is_valid_image(image_path: Path) -> bool:
        """Check if image file is valid and readable."""
        if not image_path.exists():
            return False
        
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            if img.size == 0:
                return False
            # Check minimum dimensions
            if img.shape[0] < 64 or img.shape[1] < 64:
                return False
            return True
        except Exception:
            return False
    
    @staticmethod
    def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size with aspect ratio preservation."""
        h, w = img.shape[:2]
        target_h, target_w = size
        
        # Calculate scaling factor to fill target size
        scale = max(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Crop center to target size
        start_h = (new_h - target_h) // 2
        start_w = (new_w - target_w) // 2
        cropped = resized[start_h:start_h+target_h, start_w:start_w+target_w]
        
        return cropped
    
    @staticmethod
    def load_and_preprocess(image_path: Path, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load and preprocess image."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convert BGR to RGB for consistency
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = ImageValidator.resize_image(img, target_size)
            
            return img
        except Exception as e:
            logger.debug(f"Failed to load {image_path.name}: {e}")
            return None


def convert_to_grayscale(color_img: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale."""
    # RGB to LAB, keep L channel only, convert back to RGB
    lab = cv2.cvtColor(color_img, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    
    # Create RGB from L channel (all channels equal = grayscale appearance)
    gray_rgb = np.stack([l_channel, l_channel, l_channel], axis=-1)
    return gray_rgb


def save_image(image_path: Path, image: np.ndarray, quality: int = 95) -> bool:
    """Save image to disk."""
    try:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        success = cv2.imwrite(
            str(image_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return success
    except Exception as e:
        logger.debug(f"Failed to save {image_path.name}: {e}")
        return False

# ============================================================================
# DATASET COLLECTION
# ============================================================================

class DatasetCollector:
    """Collect images from all dataset sources."""
    
    @staticmethod
    def get_all_images(directory: Path) -> List[Path]:
        """Get all valid image files from directory."""
        if not directory.exists():
            return []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = []
        
        for ext in valid_extensions:
            images.extend(directory.rglob(f"*{ext}"))
            images.extend(directory.rglob(f"*{ext.upper()}"))
        
        return sorted(set(images))  # Remove duplicates
    
    @staticmethod
    def collect_dataset(config: Config) -> Tuple[List[Path], dict]:
        """Collect images from all sources with sampling."""
        logger.info("=" * 70)
        logger.info("COLLECTING DATASET SOURCES")
        logger.info("=" * 70)
        
        sources = {}
        all_images = []
        
        # COCO
        logger.info(f"📁 Scanning COCO: {config.coco_dir}")
        coco_images = DatasetCollector.get_all_images(config.coco_dir)
        logger.info(f"   Found: {len(coco_images)} images")
        
        if len(coco_images) > config.num_coco_samples:
            coco_images = random.sample(coco_images, config.num_coco_samples)
            logger.info(f"   Sampled: {len(coco_images)} images randomly")
        
        sources['coco'] = len(coco_images)
        all_images.extend(coco_images)
        
        # DIV2K
        logger.info(f"📁 Scanning DIV2K: {config.div2k_dir}")
        div2k_images = DatasetCollector.get_all_images(config.div2k_dir)
        logger.info(f"   Found: {len(div2k_images)} images")
        sources['div2k'] = len(div2k_images)
        all_images.extend(div2k_images)
        
        # Flickr2K
        logger.info(f"📁 Scanning Flickr2K: {config.flickr2k_dir}")
        flickr2k_images = DatasetCollector.get_all_images(config.flickr2k_dir)
        logger.info(f"   Found: {len(flickr2k_images)} images")
        sources['flickr2k'] = len(flickr2k_images)
        all_images.extend(flickr2k_images)
        
        logger.info("-" * 70)
        logger.info(f"✅ TOTAL: {len(all_images)} images collected")
        logger.info(f"   Sources: {sources}")
        logger.info("=" * 70)
        
        return all_images, sources

# ============================================================================
# DATA PROCESSING PIPELINE
# ============================================================================

class ProcessingPipeline:
    """Process images: validate, resize, generate pairs."""
    
    def __init__(self, config: Config, mode: str = "grayscale"):
        """Initialize pipeline.
        
        Args:
            config: Configuration object
            mode: "grayscale" or "stage4" (use stage4 outputs as input)
        """
        self.config = config
        self.mode = mode
        self.validator = ImageValidator()
        self.stats = {
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'total': 0,
        }
    
    def process_single_image(self, image_path: Path) -> Optional[Tuple[Path, Path]]:
        """Process one image: load, validate, save as input/target pair.
        
        Returns:
            (input_path, target_path) on success, None on failure
        """
        self.stats['total'] += 1
        
        # Load and preprocess
        color_img = self.validator.load_and_preprocess(image_path, self.config.target_size)
        if color_img is None:
            self.stats['failed'] += 1
            return None
        
        # Generate unique filename
        filename = f"{self.stats['total']:06d}.jpg"
        target_path = self.config.data_output_dir / "target" / filename
        
        # Save target (original color image)
        if not save_image(target_path, color_img, quality=self.config.jpeg_quality):
            self.stats['failed'] += 1
            return None
        
        # Generate and save input
        if self.mode == "grayscale":
            input_img = convert_to_grayscale(color_img)
        elif self.mode == "stage4":
            # Try to load stage4 output, fallback to grayscale
            stage4_path = self.config.stage4_dir / filename
            if stage4_path.exists():
                input_img = self.validator.load_and_preprocess(stage4_path, self.config.target_size)
                if input_img is None:
                    input_img = convert_to_grayscale(color_img)
            else:
                input_img = convert_to_grayscale(color_img)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        input_path = self.config.data_output_dir / "input" / filename
        if not save_image(input_path, input_img, quality=self.config.jpeg_quality):
            self.stats['failed'] += 1
            return None
        
        self.stats['processed'] += 1
        return (input_path, target_path)
    
    def process_batch(self, image_paths: List[Path], worker_id: int = 0) -> int:
        """Process a batch of images."""
        success_count = 0
        
        for idx, image_path in enumerate(image_paths):
            result = self.process_single_image(image_path)
            if result is not None:
                success_count += 1
            
            if (idx + 1) % self.config.batch_size == 0:
                logger.info(f"Worker {worker_id}: Processed {idx + 1}/{len(image_paths)} images")
        
        return success_count


def worker_process_batch(args: Tuple) -> dict:
    """Multiprocessing worker function."""
    image_paths, config, mode, worker_id = args
    
    pipeline = ProcessingPipeline(config, mode=mode)
    pipeline.process_batch(image_paths, worker_id=worker_id)
    
    return {
        'worker_id': worker_id,
        'processed': pipeline.stats['processed'],
        'failed': pipeline.stats['failed'],
    }

# ============================================================================
# DATASET VALIDATION & GENERATION
# ============================================================================

class DatasetValidator:
    """Validate and report on generated dataset."""
    
    @staticmethod
    def validate_pair(input_path: Path, target_path: Path) -> bool:
        """Check if input/target pair is valid."""
        return input_path.exists() and target_path.exists()
    
    @staticmethod
    def validate_dataset(config: Config) -> dict:
        """Validate entire dataset and report statistics."""
        logger.info("=" * 70)
        logger.info("VALIDATING DATASET")
        logger.info("=" * 70)
        
        input_dir = config.data_output_dir / "input"
        target_dir = config.data_output_dir / "target"
        
        if not input_dir.exists() or not target_dir.exists():
            logger.error("❌ Input or target directory not found!")
            return {}
        
        input_files = sorted(input_dir.glob("*.jpg"))
        target_files = sorted(target_dir.glob("*.jpg"))
        
        logger.info(f"📊 Input directory: {len(input_files)} files")
        logger.info(f"📊 Target directory: {len(target_files)} files")
        
        # Check consistency
        if len(input_files) != len(target_files):
            logger.warning(f"⚠️  File count mismatch! {len(input_files)} input vs {len(target_files)} target")
        
        # Sample validation
        sample_size = min(100, len(input_files))
        sample_indices = random.sample(range(len(input_files)), sample_size)
        
        valid_pairs = 0
        invalid_pairs = []
        
        for idx in sample_indices:
            input_file = input_files[idx]
            target_file = target_files[idx]
            
            if not DatasetValidator.validate_pair(input_file, target_file):
                invalid_pairs.append((input_file.name, target_file.name))
            else:
                valid_pairs += 1
        
        validation_rate = (valid_pairs / sample_size * 100) if sample_size > 0 else 0
        logger.info(f"✅ Sample validation (n={sample_size}): {valid_pairs}/{sample_size} valid ({validation_rate:.1f}%)")
        
        if invalid_pairs:
            logger.warning(f"⚠️  Invalid pairs: {invalid_pairs[:5]}")
        
        # Dataset statistics
        stats = {
            'total_pairs': len(input_files),
            'input_files': len(input_files),
            'target_files': len(target_files),
            'validation_rate': validation_rate,
            'input_dir': str(input_dir),
            'target_dir': str(target_dir),
            'timestamp': datetime.now().isoformat(),
        }
        
        logger.info("-" * 70)
        logger.info(f"📈 DATASET STATISTICS")
        logger.info(f"   Total pairs: {stats['total_pairs']}")
        logger.info(f"   Input directory: {input_dir.name}/")
        logger.info(f"   Target directory: {target_dir.name}/")
        logger.info("=" * 70)
        
        return stats

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def build_dataset(config: Config, mode: str = "grayscale") -> dict:
    """Build complete dataset pipeline."""
    logger.info("\n")
    logger.info("🚀 STARTING DATASET BUILD PIPELINE")
    logger.info(f"   Mode: {mode}")
    logger.info(f"   Workers: {config.num_workers}")
    logger.info(f"   Output: {config.data_output_dir}")
    logger.info("\n")
    
    # Step 1: Collect images
    all_images, sources = DatasetCollector.collect_dataset(config)
    
    if not all_images:
        logger.error("❌ No images found! Check dataset directories.")
        return {}
    
    # Step 2: Create output directories
    (config.data_output_dir / "input").mkdir(parents=True, exist_ok=True)
    (config.data_output_dir / "target").mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Output directories created")
    
    # Step 3: Process images
    logger.info("=" * 70)
    logger.info(f"PROCESSING {len(all_images)} IMAGES")
    logger.info("=" * 70)
    
    # Option A: Single-process (for debugging)
    if config.num_workers == 1:
        pipeline = ProcessingPipeline(config, mode=mode)
        for image_path in all_images:
            pipeline.process_single_image(image_path)
        
        logger.info(f"\n✅ Processed: {pipeline.stats['processed']}")
        logger.info(f"❌ Failed: {pipeline.stats['failed']}")
    
    # Option B: Multiprocessing
    else:
        chunk_size = len(all_images) // config.num_workers
        chunks = [all_images[i:i+chunk_size] for i in range(0, len(all_images), chunk_size)]
        
        worker_args = [
            (chunk, config, mode, i) for i, chunk in enumerate(chunks)
        ]
        
        with Pool(config.num_workers) as pool:
            results = pool.map(worker_process_batch, worker_args)
        
        total_processed = sum(r['processed'] for r in results)
        total_failed = sum(r['failed'] for r in results)
        
        logger.info(f"\n✅ Total processed: {total_processed}")
        logger.info(f"❌ Total failed: {total_failed}")
    
    # Step 4: Validate dataset
    stats = DatasetValidator.validate_dataset(config)
    stats['sources'] = sources
    stats['mode'] = mode
    
    # Step 5: Save metadata
    metadata_path = config.data_output_dir / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✅ Metadata saved: {metadata_path}")
    
    logger.info("\n✅ DATASET BUILD COMPLETE\n")
    return stats


def validate_only(config: Config):
    """Run validation only without rebuilding."""
    logger.info("\n🔍 VALIDATION MODE (no processing)\n")
    stats = DatasetValidator.validate_dataset(config)
    
    metadata_path = config.data_output_dir / "dataset_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            prev_stats = json.load(f)
        logger.info(f"📝 Previous metadata found: {metadata_path}")
    
    return stats


# ============================================================================
# CLI & ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build paired dataset for GAN colorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python datasets/dataset_builder.py --mode grayscale --num-coco 20000
  python datasets/dataset_builder.py --mode stage4 --num-coco 25000
  python datasets/dataset_builder.py --validate-only
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["grayscale", "stage4"],
        default="grayscale",
        help="Input generation mode (default: grayscale)"
    )
    parser.add_argument(
        "--num-coco",
        type=int,
        default=20000,
        help="Number of COCO images to sample (default: 20000)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: PROJECT_ROOT/data)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=256,
        help="Target image size in pixels (default: 256)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers (0=auto, 1=single-process) (default: auto)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing dataset without processing"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        data_output_dir=args.output_dir,
        num_coco_samples=args.num_coco,
        target_size=(args.target_size, args.target_size),
        num_workers=args.num_workers,
    )
    
    # Validate source directories
    missing_dirs = []
    for name, path in [
        ("COCO", config.coco_dir),
        ("DIV2K", config.div2k_dir),
        ("Flickr2K", config.flickr2k_dir),
    ]:
        if not path.exists():
            missing_dirs.append(f"{name}: {path}")
    
    if missing_dirs:
        logger.error("❌ Missing dataset directories:")
        for missing in missing_dirs:
            logger.error(f"   {missing}")
        logger.error("\nPlease ensure datasets are in the correct locations.")
        return 1
    
    # Run pipeline
    try:
        if args.validate_only:
            validate_only(config)
        else:
            build_dataset(config, mode=args.mode)
        return 0
    except KeyboardInterrupt:
        logger.info("\n⚠️  Dataset build interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
