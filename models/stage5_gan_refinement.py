"""
Inference module for Stage 5: GAN Refinement.

Takes colorized images from Stage 4 and refines them using
the trained GAN for improved perceptual realism.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from models.gan_generator import GANGenerator


logger = logging.getLogger(__name__)


class Stage5GANRefinement:
    """Stage 5: GAN-based image refinement."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Stage 5 GAN refinement.

        Args:
            checkpoint_path: Path to trained generator checkpoint
            device: Device to use (cuda or cpu)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize generator
        self.generator = GANGenerator(
            in_channels=3,
            out_channels=3,
            base_filters=64,
            num_residual_blocks=4,
        )

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)
            logger.info(f"Loaded GAN generator from {checkpoint_path}")
        else:
            logger.warning("No checkpoint provided for Stage 5, using untrained model")

        self.generator = self.generator.to(self.device)
        self.generator.eval()

        logger.info(f"Stage 5 (GAN Refinement) initialized on {self.device}")

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load generator weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle both full checkpoint (with history) and state_dict only
        if isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
        else:
            self.generator.load_state_dict(checkpoint)

        logger.info(f"Generator loaded from {checkpoint_path}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.

        Args:
            image: Input image (H, W, 3) in BGR format, uint8

        Returns:
            Tensor (1, 3, H, W) normalized to [-1, 1]
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert uint8 to float and normalize to [-1, 1]
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 127.5 - 1.0

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Postprocess tensor to image.

        Args:
            tensor: Output tensor (1, 3, H, W) in [-1, 1]

        Returns:
            Image array (H, W, 3) in BGR format, uint8
        """
        # Remove batch dimension and move to CPU
        image = tensor.squeeze(0).cpu().detach()

        # Convert from [-1, 1] to [0, 255]
        image = (image + 1) / 2 * 255.0

        # Clamp to valid range
        image = torch.clamp(image, 0, 255)

        # Convert to numpy and permute
        image_np = image.permute(1, 2, 0).numpy().astype(np.uint8)

        # Convert RGB to BGR
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        return image_bgr

    def refine(self, image: np.ndarray) -> np.ndarray:
        """
        Refine colorized image using GAN.

        Args:
            image: Colorized image (H, W, 3) in BGR format, uint8

        Returns:
            Refined image (H, W, 3) in BGR format, uint8
        """
        with torch.no_grad():
            # Preprocess
            image_tensor = self.preprocess(image)

            # Forward pass
            refined_tensor = self.generator(image_tensor)

            # Postprocess
            refined_image = self.postprocess(refined_tensor)

        return refined_image

    def refine_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Refine a batch of colorized images.

        Args:
            images: Batch of images (B, H, W, 3) in BGR format, uint8

        Returns:
            Batch of refined images (B, H, W, 3) in BGR format, uint8
        """
        batch_size = images.shape[0]
        refined_images = []

        with torch.no_grad():
            # Preprocess batch
            batch_tensor = torch.stack([
                torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                .permute(2, 0, 1).float() / 127.5 - 1.0
                for img in images
            ])

            batch_tensor = batch_tensor.to(self.device)

            # Forward pass
            refined_batch = self.generator(batch_tensor)

            # Postprocess batch
            for i in range(batch_size):
                refined_img = self.postprocess(refined_batch[i:i+1])
                refined_images.append(refined_img)

        return np.stack(refined_images)


def create_stage5_refiner(
    checkpoint_dir: Path = None,
) -> Optional[Stage5GANRefinement]:
    """
    Factory function to create Stage 5 refiner.

    Args:
        checkpoint_dir: Directory containing GAN checkpoints

    Returns:
        Stage5GANRefinement instance, or None if no checkpoint found
    """
    if checkpoint_dir is None:
        return None

    checkpoint_dir = Path(checkpoint_dir)

    # Look for latest checkpoint
    checkpoints = sorted(checkpoint_dir.glob("gan_refinement_*.pth"))
    if not checkpoints:
        logger.warning(f"No GAN checkpoints found in {checkpoint_dir}")
        return None

    latest_checkpoint = checkpoints[-1]
    logger.info(f"Using GAN checkpoint: {latest_checkpoint}")

    try:
        refiner = Stage5GANRefinement(checkpoint_path=latest_checkpoint)
        return refiner
    except Exception as e:
        logger.error(f"Failed to load Stage 5 refiner: {e}")
        return None
