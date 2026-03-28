"""
Training script for GAN-based image refinement stage (Stage 5).

Combines adversarial loss, L1 loss, and perceptual loss for stable,
high-quality image refinement.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2
import numpy as np

from models.gan_generator import GANGenerator
from models.gan_discriminator import PatchGANDiscriminator, MultiscaleDiscriminator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss for feature matching."""

    def __init__(self, layer: str = "relu5_1", device: torch.device = None):
        """
        Initialize VGG perceptual loss.

        Args:
            layer: Which VGG layer to use for perceptual features
            device: Device to use (cuda or cpu)
        """
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained VGG16
        vgg16 = torchvision.models.vgg16(pretrained=True).to(self.device)
        vgg16.eval()
        for param in vgg16.parameters():
            param.requires_grad = False

        # Extract features up to specified layer
        if layer == "relu5_1":
            max_layer = 31
        elif layer == "relu4_1":
            max_layer = 24
        elif layer == "relu3_1":
            max_layer = 17
        elif layer == "relu2_1":
            max_layer = 10
        else:
            max_layer = 31

        self.feature_extractor = vgg16.features[:max_layer]

        # Normalization constants for ImageNet
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            x: Generated image
            y: Target/real image

        Returns:
            Perceptual loss value
        """
        # Normalize
        x_norm = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
        y_norm = (y + 1) / 2

        x_norm = (x_norm - self.mean) / self.std
        y_norm = (y_norm - self.mean) / self.std

        # Extract features
        x_features = self.feature_extractor(x_norm)
        y_features = self.feature_extractor(y_norm)

        # Compute L1 distance between features
        loss = torch.nn.functional.l1_loss(x_features, y_features)

        return loss


class ImageRefinementDataset(Dataset):
    """Dataset for reading colorized and target images."""

    def __init__(self, colorized_dir: Path, target_dir: Path, image_size: int = 256):
        """
        Initialize dataset.

        Args:
            colorized_dir: Directory with colorized images from pipeline
            target_dir: Directory with target/ground truth images
            image_size: Size to resize images to
        """
        self.colorized_dir = Path(colorized_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size

        # Get list of images
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_files = [
            f for f in self.colorized_dir.glob("*")
            if f.suffix.lower() in valid_extensions
        ]
        self.image_files.sort()

        if not self.image_files:
            raise ValueError(f"No images found in {colorized_dir}")

        logger.info(f"Dataset loaded: {len(self)} images")

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.

        Args:
            idx: Index of item

        Returns:
            Dictionary with colorized and target tensors
        """
        colorized_path = self.image_files[idx]
        target_path = self.target_dir / colorized_path.name

        # Read images
        colorized = cv2.imread(str(colorized_path))
        target = cv2.imread(str(target_path))

        if colorized is None or target is None:
            # Return random tensors if read fails
            logger.warning(f"Failed to read image: {colorized_path}")
            colorized = np.random.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
            target = np.random.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)

        # Resize
        if colorized.shape[:2] != (self.image_size, self.image_size):
            colorized = cv2.resize(colorized, (self.image_size, self.image_size))
        if target.shape[:2] != (self.image_size, self.image_size):
            target = cv2.resize(target, (self.image_size, self.image_size))

        # Convert BGR to RGB
        colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize to [-1, 1]
        colorized_tensor = torch.from_numpy(colorized).permute(2, 0, 1).float() / 127.5 - 1.0
        target_tensor = torch.from_numpy(target).permute(2, 0, 1).float() / 127.5 - 1.0

        return {
            "colorized": colorized_tensor,
            "target": target_tensor,
            "path": str(colorized_path),
        }


class GANRefinementTrainer:
    """Trainer for GAN refinement stage."""

    def __init__(
        self,
        device: torch.device = None,
        learning_rate_g: float = 0.0002,
        learning_rate_d: float = 0.0002,
        lambda_l1: float = 100.0,
        lambda_perceptual: float = 10.0,
        lambda_adversarial: float = 1.0,
    ):
        """
        Initialize trainer.

        Args:
            device: Device to use (cuda or cpu)
            learning_rate_g: Learning rate for generator
            learning_rate_d: Learning rate for discriminator
            lambda_l1: Weight for L1 loss
            lambda_perceptual: Weight for perceptual loss
            lambda_adversarial: Weight for adversarial loss
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d

        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adversarial = lambda_adversarial

        logger.info(f"Using device: {self.device}")

        # Build models
        self.generator = GANGenerator(
            in_channels=3,
            out_channels=3,
            base_filters=64,
            num_residual_blocks=4,
        ).to(self.device)

        self.discriminator = MultiscaleDiscriminator(
            in_channels=3,
            base_filters=64,
            num_scales=3,
        ).to(self.device)

        logger.info(f"Generator parameters: {self.generator.get_num_parameters():,}")
        logger.info(f"Discriminator parameters: {self.discriminator.get_num_parameters():,}")

        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

        try:
            import torchvision
            self.perceptual_loss = VGGPerceptualLoss(device=self.device)
            self.use_perceptual = True
        except ImportError:
            logger.warning("torchvision not available, skipping perceptual loss")
            self.perceptual_loss = None
            self.use_perceptual = False

        # Optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=learning_rate_g,
            betas=(0.5, 0.999),
        )

        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate_d,
            betas=(0.5, 0.999),
        )

        self.training_history = {
            "epoch": [],
            "loss_g": [],
            "loss_d": [],
            "loss_l1": [],
            "loss_perceptual": [],
            "loss_adversarial": [],
        }

    def train_step(
        self,
        colorized: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one training step (G and D updates).

        Args:
            colorized: Colorized images (B, 3, H, W)
            target: Target/ground truth images (B, 3, H, W)

        Returns:
            Dictionary with loss values
        """
        batch_size = colorized.size(0)

        # ============== Generator Step ==============
        self.optimizer_g.zero_grad()

        # Generate refined images
        refined = self.generator(colorized)

        # Discriminator output on refined images
        disc_refined = self.discriminator(refined)

        # Adversarial loss: make discriminator think refined is real
        loss_adv_g = 0.0
        for disc_out in disc_refined:
            # Target: discriminator outputs should be close to 1 (real)
            target_ones = torch.ones_like(disc_out)
            loss_adv_g += self.adversarial_loss(disc_out, target_ones)
        loss_adv_g = loss_adv_g / len(disc_refined)

        # L1 loss: preserve content
        loss_l1 = self.l1_loss(refined, target)

        # Perceptual loss: feature matching
        loss_perceptual = torch.tensor(0.0, device=self.device)
        if self.use_perceptual and self.perceptual_loss is not None:
            loss_perceptual = self.perceptual_loss(refined, target)

        # Total generator loss
        loss_g = (
            self.lambda_adversarial * loss_adv_g +
            self.lambda_l1 * loss_l1 +
            self.lambda_perceptual * loss_perceptual
        )

        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.optimizer_g.step()

        # ============== Discriminator Step ==============
        self.optimizer_d.zero_grad()

        # Discriminator on real images
        disc_real = self.discriminator(target)

        # Discriminator on refined images (detached)
        disc_refined = self.discriminator(refined.detach())

        # Adversarial loss for discriminator
        loss_d = 0.0
        for disc_real_out, disc_refined_out in zip(disc_real, disc_refined):
            # Real should output 1, refined should output 0
            target_ones = torch.ones_like(disc_real_out)
            target_zeros = torch.zeros_like(disc_refined_out)

            loss_d += self.adversarial_loss(disc_real_out, target_ones)
            loss_d += self.adversarial_loss(disc_refined_out, target_zeros)
        loss_d = loss_d / (len(disc_real) * 2)

        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.optimizer_d.step()

        return {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item(),
            "loss_l1": loss_l1.item(),
            "loss_perceptual": loss_perceptual.item() if isinstance(loss_perceptual, torch.Tensor) else 0.0,
            "loss_adversarial": loss_adv_g.item(),
        }

    def train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            data_loader: DataLoader for training data

        Returns:
            Dictionary with average loss values
        """
        self.generator.train()
        self.discriminator.train()

        epoch_losses = {
            "loss_g": [],
            "loss_d": [],
            "loss_l1": [],
            "loss_perceptual": [],
            "loss_adversarial": [],
        }

        with tqdm(total=len(data_loader), desc="Training") as pbar:
            for batch in data_loader:
                colorized = batch["colorized"].to(self.device)
                target = batch["target"].to(self.device)

                losses = self.train_step(colorized, target)

                for key, value in losses.items():
                    epoch_losses[key].append(value)

                pbar.set_postfix({
                    "G": f"{losses['loss_g']:.4f}",
                    "D": f"{losses['loss_d']:.4f}",
                })
                pbar.update(1)

        # Average losses
        avg_losses = {
            key: np.mean(values) for key, values in epoch_losses.items()
        }

        return avg_losses

    def validate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Validate on a dataset.

        Args:
            data_loader: DataLoader for validation data

        Returns:
            Dictionary with average loss values
        """
        self.generator.eval()
        self.discriminator.eval()

        val_losses = {
            "loss_l1": [],
            "loss_perceptual": [],
        }

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Validating"):
                colorized = batch["colorized"].to(self.device)
                target = batch["target"].to(self.device)

                refined = self.generator(colorized)

                loss_l1 = self.l1_loss(refined, target)
                val_losses["loss_l1"].append(loss_l1.item())

                if self.use_perceptual and self.perceptual_loss is not None:
                    loss_perceptual = self.perceptual_loss(refined, target)
                    val_losses["loss_perceptual"].append(loss_perceptual.item())

        avg_val_losses = {
            key: np.mean(values) for key, values in val_losses.items()
        }

        return avg_val_losses

    def save_checkpoint(self, save_dir: Path, epoch: int) -> None:
        """
        Save model checkpoint.

        Args:
            save_dir: Directory to save checkpoint
            epoch: Current epoch number
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "training_history": self.training_history,
        }

        torch.save(
            checkpoint,
            save_dir / f"gan_refinement_epoch_{epoch}.pth",
        )
        logger.info(f"Checkpoint saved: {save_dir}/gan_refinement_epoch_{epoch}.pth")

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        self.training_history = checkpoint.get("training_history", {})

        epoch = checkpoint["epoch"]
        logger.info(f"Checkpoint loaded from epoch {epoch}")

        return epoch


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(
        description="Train GAN refinement stage for colorization",
    )

    parser.add_argument(
        "--colorized-dir",
        type=str,
        required=True,
        help="Directory with colorized images",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="Directory with target images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/stage5_gan",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate-g",
        type=float,
        default=0.0002,
        help="Learning rate for generator",
    )
    parser.add_argument(
        "--learning-rate-d",
        type=float,
        default=0.0002,
        help="Learning rate for discriminator",
    )
    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=100.0,
        help="Weight for L1 loss",
    )
    parser.add_argument(
        "--lambda-perceptual",
        type=float,
        default=10.0,
        help="Weight for perceptual loss",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size for training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and dataloader
    dataset = ImageRefinementDataset(
        colorized_dir=args.colorized_dir,
        target_dir=args.target_dir,
        image_size=args.image_size,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Create trainer
    trainer = GANRefinementTrainer(
        device=device,
        learning_rate_g=args.learning_rate_g,
        learning_rate_d=args.learning_rate_d,
        lambda_l1=args.lambda_l1,
        lambda_perceptual=args.lambda_perceptual,
    )

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(Path(args.resume))

    # Training loop
    logger.info(f"Starting training for {args.num_epochs} epochs...")

    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"{'='*60}")

        losses = trainer.train_epoch(data_loader)

        trainer.training_history["epoch"].append(epoch + 1)
        for key, value in losses.items():
            if key not in trainer.training_history:
                trainer.training_history[key] = []
            trainer.training_history[key].append(value)

        logger.info(f"Loss G: {losses['loss_g']:.4f}")
        logger.info(f"Loss D: {losses['loss_d']:.4f}")
        logger.info(f"Loss L1: {losses['loss_l1']:.4f}")
        logger.info(f"Loss Perceptual: {losses['loss_perceptual']:.4f}")
        logger.info(f"Loss Adversarial: {losses['loss_adversarial']:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(Path(args.output_dir), epoch + 1)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
