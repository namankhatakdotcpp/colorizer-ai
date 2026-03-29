"""
Training script for GAN-based image refinement stage (Stage 5).

Combines adversarial loss, L1 loss, and perceptual loss for stable,
high-quality image refinement.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import cv2
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
from copy import deepcopy

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


def feature_matching_loss(real_features: List[torch.Tensor], fake_features: List[torch.Tensor]) -> torch.Tensor:
    """
    Feature matching loss - encourages generator to match discriminator features.

    Args:
        real_features: List of real image features from discriminator
        fake_features: List of fake image features from discriminator

    Returns:
        Feature matching loss value
    """
    loss = torch.tensor(0.0, device=real_features[0].device)
    for rf, ff in zip(real_features, fake_features):
        loss = loss + torch.mean(torch.abs(rf - ff))
    return loss / len(real_features) if real_features else loss


def r1_penalty(real_pred: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
    """
    R1 Gradient Penalty for discriminator regularization.
    
    Prevents discriminator from becoming too sharp by penalizing
    large gradients with respect to real images.

    Args:
        real_pred: Discriminator output for real images (requires grad)
        real_img: Real images (requires grad)

    Returns:
        R1 penalty value
    """
    grad_real = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_img,
        create_graph=True,
        retain_graph=False,
    )[0]
    return torch.mean(grad_real ** 2)


def fft_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """
    FFT-based frequency domain loss (SAFE VERSION).
    
    Uses rfft2 (real FFT) to avoid ringing artifacts.
    Optimizes frequency distribution which FID is sensitive to.

    Args:
        fake: Generated image
        real: Real image

    Returns:
        FFT loss value
    """
    # Use rfft2 (real FFT) - safer than fft2, avoids ringing artifacts
    if fake.shape[1] == 3:  # RGB image
        # Process each channel separately to avoid complex number issues
        fake_fft = torch.fft.rfft2(fake[:, 0:1])  # L channel representation
        real_fft = torch.fft.rfft2(real[:, 0:1])
    else:
        fake_fft = torch.fft.rfft2(fake)
        real_fft = torch.fft.rfft2(real)
    
    # Compare magnitude spectrum (real-valued, safe)
    fake_mag = torch.abs(fake_fft)
    real_mag = torch.abs(real_fft)
    
    return torch.mean(torch.abs(fake_mag - real_mag))


class ImageRefinementDataset(Dataset):
    """Dataset for reading colorized and target images."""

    def __init__(self, colorized_dir: Path, target_dir: Path, image_size: int = 256, augment: bool = True):
        """
        Initialize dataset.

        Args:
            colorized_dir: Directory with colorized images from pipeline
            target_dir: Directory with target/ground truth images
            image_size: Size to resize images to
            augment: Enable data augmentation (random flip, color jitter)
        """
        self.colorized_dir = Path(colorized_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size
        self.augment = augment

        # DiffAugment: Simple, effective augmentation for GAN training
        # ONLY horizontal flip to preserve color distribution
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
        ]) if augment else None

        # Get list of images
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_files = [
            f for f in self.colorized_dir.glob("*")
            if f.suffix.lower() in valid_extensions
        ]
        self.image_files.sort()

        if not self.image_files:
            raise ValueError(f"No images found in {colorized_dir}")

        logger.info(f"Dataset loaded: {len(self)} images (augmentation={'enabled' if augment else 'disabled'})")

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

        # Apply data augmentation if enabled (same to both images)
        if self.augment and self.augmentation is not None:
            colorized_pil = transforms.ToPILImage()(colorized)
            target_pil = transforms.ToPILImage()(target)
            colorized = np.array(self.augmentation(colorized_pil))
            target = np.array(self.augmentation(target_pil))

        # Convert to tensor and normalize to [-1, 1]
        colorized_tensor = torch.from_numpy(colorized).permute(2, 0, 1).float() / 127.5 - 1.0
        target_tensor = torch.from_numpy(target).permute(2, 0, 1).float() / 127.5 - 1.0

        # Color stability fix: clamp and remove NaNs
        colorized_tensor = torch.clamp(colorized_tensor, -1.0, 1.0)
        target_tensor = torch.clamp(target_tensor, -1.0, 1.0)
        colorized_tensor = torch.nan_to_num(colorized_tensor, 0.0)
        target_tensor = torch.nan_to_num(target_tensor, 0.0)

        # compute LAB conditioning for discriminator (UPGRADE: better than grayscale alone)
        # Convert RGB [-1, 1] to [0, 255] range for LAB conversion
        target_rgb_255 = ((target_tensor + 1.0) / 2.0 * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        colorized_rgb_255 = ((colorized_tensor + 1.0) / 2.0 * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        
        # Convert to LAB
        target_lab = cv2.cvtColor(target_rgb_255, cv2.COLOR_RGB2LAB).astype(np.float32)
        colorized_lab = cv2.cvtColor(colorized_rgb_255, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Extract L and ab channels
        L_channel = torch.from_numpy(target_lab[:, :, 0:1]).permute(2, 0, 1) / 100.0  # Normalize L to [0, 1]
        ab_channels = torch.from_numpy(colorized_lab[:, :, 1:3]).permute(2, 0, 1) / 127.0  # Normalize ab to [-1, 1]
        
        # Clamp and clean
        L_channel = torch.clamp(L_channel, 0.0, 1.0).squeeze(0).unsqueeze(0)  # (1, H, W)
        ab_channels = torch.clamp(ab_channels, -1.0, 1.0)  # (2, H, W)
        
        # Original grayscale (for backward compatibility)
        grayscale = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1) * target_tensor.unsqueeze(0)
        grayscale = grayscale.sum(dim=1, keepdim=True).squeeze(0)
        grayscale = torch.clamp(grayscale, -1.0, 1.0)

        return {
            "colorized": colorized_tensor,
            "target": target_tensor,
            "grayscale": grayscale,
            "L_channel": L_channel,      # NEW: L channel for LAB conditioning
            "ab_channels": ab_channels,  # NEW: ab channels for LAB conditioning
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
            num_residual_blocks=8,  # UPGRADED: 4 → 8 for better texture detail + self-attention
        ).to(self.device)

        self.discriminator = MultiscaleDiscriminator(
            in_channels=4,  # UPDATED: 4 channels for conditional GAN (grayscale + RGB)
            base_filters=64,
            num_scales=3,  # Multi-scale: original, ÷2, ÷4
        ).to(self.device)
        
        # EMA (Exponential Moving Average) of generator for improved inference
        self.generator_ema = deepcopy(self.generator)
        self.generator_ema.eval()
        for param in self.generator_ema.parameters():
            param.requires_grad = False
        self.ema_decay = 0.9995  # UPGRADED: 0.999 → 0.9995 for smoother outputs

        logger.info(f"Generator parameters: {self.generator.get_num_parameters():,}")
        logger.info(f"Discriminator parameters: {self.discriminator.get_num_parameters():,}")
        logger.info(f"Generator EMA initialized (decay: {self.ema_decay})")

        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
        # Mixed Precision Training (AMP)
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

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
            "loss_identity": [],
            "loss_r1": [],
            "loss_path_length": [],  # NEW
            "loss_fft": [],           # NEW
            "loss_style": [],         # NEW
        }
        
        self.d_steps = 0  # Track discriminator steps for R1 penalty

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        train_d_steps: int = 2,
    ) -> Dict[str, float]:
        """
        Perform one training step (G and D updates).

        Args:
            batch: Dictionary with 'colorized', 'target', 'grayscale' tensors
            train_d_steps: Number of discriminator training steps per generator step (2:1 ratio)

        Returns:
            Dictionary with loss values
        """
        colorized = batch["colorized"].to(self.device)
        target = batch["target"].to(self.device)
        L_channel = batch["L_channel"].to(self.device)      # NEW: LAB conditioning
        ab_channels = batch["ab_channels"].to(self.device)  # NEW: LAB conditioning
        
        batch_size = colorized.size(0)
        device = colorized.device

        losses_dict = {
            "loss_g": [],
            "loss_d": [],
            "loss_l1": [],
            "loss_perceptual": [],
            "loss_adversarial": [],
            "loss_feature_matching": [],
            "loss_identity": [],
            "loss_r1": [],
            "loss_fft": [],
        }

        # ============== Discriminator Step (multiple times) ==============
        for d_iter in range(train_d_steps):
            self.optimizer_d.zero_grad()
            self.d_steps += 1

            # 🔥 GENERATE REFINED FRESH FOR EACH D STEP
            # Critical: Each D step gets its own generator forward pass
            # This prevents computation graph conflicts between D iterations
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                refined_d = self.generator(colorized)  # Fresh forward for D step
                refined_detached = refined_d.detach()  # Immediately detach

            # Add small noise to real images for stability
            real_noisy = target + 0.05 * torch.randn_like(target)
            real_noisy = torch.clamp(real_noisy, -1.0, 1.0)
            
            # Real images need gradients for R1 penalty
            real_noisy_for_grad = real_noisy.detach().clone().requires_grad_(True)
            
            # Add noise to fake images
            fake_noisy = refined_detached + 0.05 * torch.randn_like(refined_detached)
            fake_noisy = torch.clamp(fake_noisy, -1.0, 1.0)

            # UPGRADE: LAB Conditioning (SIMPLE & CLEAN: [L, RGB] = 4 channels)
            # L provides brightness context from target
            # RGB provides the refined color image
            L_expanded = L_channel.expand(batch_size, -1, -1, -1) if L_channel.dim() == 3 else L_channel
            
            # Concatenate [L (1ch) + RGB (3ch)] = 4 channels (discriminator input channels)
            real_conditional = torch.cat([L_expanded, real_noisy_for_grad], dim=1)  # 4 channels: 1 (L) + 3 (RGB)
            fake_conditional = torch.cat([L_expanded, fake_noisy], dim=1)  # 4 channels: 1 (L) + 3 (RGB)

            # Discriminator outputs (multi-scale)
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                disc_real = self.discriminator(real_conditional)
                disc_fake = self.discriminator(fake_conditional)

            # Hinge loss for discriminator (more stable than BCE)
            loss_d_real = torch.mean(nn.ReLU()(1.0 - disc_real[0]))
            loss_d_fake = torch.mean(nn.ReLU()(1.0 + disc_fake[0]))
            loss_d = loss_d_real + loss_d_fake
            
            # R1 Gradient Penalty (lazy regularization every 16 steps)
            loss_r1 = torch.tensor(0.0, device=device)
            if self.d_steps % 16 == 0:
                loss_r1 = r1_penalty(disc_real[0], real_conditional)
                loss_d = loss_d + 10.0 * loss_r1

            # 🔥 BACKWARD FOR D STEP
            if self.scaler is not None:
                self.scaler.scale(loss_d).backward()
                self.scaler.unscale_(self.optimizer_d)
            else:
                loss_d.backward()
                
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            
            # 🔥 CRITICAL: After optimizer step, explicitly zero gradients
            # This prevents old gradients from interfering with G step
            if self.scaler is not None:
                self.scaler.step(self.optimizer_d)
                self.scaler.update()
            else:
                self.optimizer_d.step()
            
            # 🔥 EXPLICIT GRADIENT ZEROING
            # Ensures D parameters are completely clean before G step
            self.optimizer_d.zero_grad()

            losses_dict["loss_d"].append(loss_d.item())
            losses_dict["loss_r1"].append(loss_r1.item() if isinstance(loss_r1, torch.Tensor) else 0.0)

        # ============== Generator Step ==============
        # 🔥 CRITICAL: Freeze discriminator to prevent graph overlap during G step
        for p in self.discriminator.parameters():
            p.requires_grad = False
        
        self.optimizer_g.zero_grad()

        # 🔥 HARD SAFETY: Detach AND clone ALL inputs
        target_g = target.detach().clone()
        colorized_g = colorized.detach().clone()
        L_channel_g = L_channel.detach().clone()

        # 🔥 CRITICAL: All G computations in one autocast block for consistency
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            # ===== FRESH GENERATOR FORWARD FOR G STEP =====
            # Critical: NEVER reuse refined tensor from D step
            refined_g = self.generator(colorized_g)

            # L1 loss: preserve content
            loss_l1 = self.l1_loss(refined_g, target_g)

            # Identity loss: prevent unwanted color changes
            loss_identity = self.l1_loss(refined_g, colorized_g)

            # Perceptual loss: VGG features
            loss_perceptual = torch.tensor(0.0, device=self.device)
            if self.use_perceptual and self.perceptual_loss is not None:
                loss_perceptual = self.perceptual_loss(refined_g, target_g)

            # FFT Loss: Frequency domain matching
            fft_fake = torch.fft.rfft2(refined_g)
            fft_real = torch.fft.rfft2(target_g)
            loss_fft = torch.mean(torch.abs(fft_fake - fft_real).real)

            # ===== DISCRIMINATOR FRESH FORWARD: ADVERSARIAL LOSS =====
            L_expanded_g = L_channel_g.expand(batch_size, -1, -1, -1) if L_channel_g.dim() == 3 else L_channel_g
            refined_conditional_adv = torch.cat([L_expanded_g, refined_g], dim=1)
            
            disc_fake_adv = self.discriminator(refined_conditional_adv)
            loss_adv_g = -torch.mean(disc_fake_adv[0])

            # ===== DISCRIMINATOR FRESH FORWARD: FEATURE MATCHING LOSS =====
            # Create refined_conditional fresh for fm forward
            refined_conditional_fm = torch.cat([L_expanded_g, refined_g], dim=1)
            disc_fake_fm = self.discriminator(refined_conditional_fm)
            
            # Feature matching loss
            loss_fm = torch.tensor(0.0, device=device)
            if isinstance(disc_fake_fm, (list, tuple)) and len(disc_fake_fm) > 1:
                fake_features = disc_fake_fm[1:]  # All but first (logits)
                
                # Get real features from fresh discriminator call (inside no_grad)
                with torch.no_grad():
                    real_noisy_g = target_g + 0.05 * torch.randn_like(target_g)
                    real_noisy_g = torch.clamp(real_noisy_g, -1.0, 1.0)
                    real_conditional_g = torch.cat([L_expanded_g, real_noisy_g], dim=1)
                    
                    disc_real_fresh = self.discriminator(real_conditional_g)
                    real_features_fresh = disc_real_fresh[1:]
                
                # Detach real features
                real_features = [f.detach() for f in real_features_fresh]
                
                # Compute feature matching loss
                fm_loss_total = torch.tensor(0.0, device=device)
                for real_feat, fake_feat in zip(real_features, fake_features):
                    fm_loss_total += torch.mean((fake_feat - real_feat) ** 2)
                loss_fm = fm_loss_total / max(len(real_features), 1)
            
            # ===== COMBINE ALL LOSSES INTO SINGLE SCALAR =====
            # CRITICAL: All losses combined BEFORE backward
            loss_g = (
                self.lambda_adversarial * loss_adv_g +
                self.lambda_l1 * loss_l1 +
                self.lambda_perceptual * loss_perceptual +
                5.0 * loss_fm +
                2.0 * loss_identity +
                0.05 * loss_fft
            )

        # 🔥 DEBUG: Print gradient states AFTER computation but BEFORE backward
        print("---- BEFORE G BACKWARD ----")
        print(f"refined_g.requires_grad: {refined_g.requires_grad}")
        print(f"target_g.requires_grad: {target_g.requires_grad}")
        print(f"colorized_g.requires_grad: {colorized_g.requires_grad}")
        print(f"loss_g.requires_grad: {loss_g.requires_grad}")
        print(f"D parameters frozen: {not self.discriminator.weight_1x1.requires_grad}")
        print("---- END DEBUG ----\n")

        # 🔥 BACKWARD FOR G STEP
        # This is the ONLY backward on loss_g for this train step
        if self.scaler is not None:
            self.scaler.scale(loss_g).backward()
            self.scaler.unscale_(self.optimizer_g)
        else:
            loss_g.backward()
            
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        
        if self.scaler is not None:
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            self.optimizer_g.step()
        
        # 🔥 EXPLICIT: Clear G gradients after step
        self.optimizer_g.zero_grad()
        
        # 🔥 CRITICAL: Restore discriminator gradients AFTER G step is complete
        for p in self.discriminator.parameters():
            p.requires_grad = True
        
        # UPDATE EMA GENERATOR (important for eval) 
        with torch.no_grad():
            for param, ema_param in zip(self.generator.parameters(), self.generator_ema.parameters()):
                ema_param.data = self.ema_decay * ema_param.data + (1 - self.ema_decay) * param.data

        losses_dict["loss_g"].append(loss_g.item())
        losses_dict["loss_l1"].append(loss_l1.item())
        losses_dict["loss_perceptual"].append(loss_perceptual.item() if isinstance(loss_perceptual, torch.Tensor) else 0.0)
        losses_dict["loss_adversarial"].append(loss_adv_g.item())
        losses_dict["loss_feature_matching"].append(loss_fm.item() if isinstance(loss_fm, torch.Tensor) else 0.0)
        losses_dict["loss_identity"].append(loss_identity.item())
        losses_dict["loss_fft"].append(loss_fft.item() if isinstance(loss_fft, torch.Tensor) else 0.0)

        # Average all losses
        return {
            k: np.mean(v) if v else 0.0 for k, v in losses_dict.items()
        }

    def train_epoch(self, data_loader: DataLoader) -> Tuple[Dict[str, float], Optional[torch.Tensor]]:
        """
        Train for one epoch.

        Args:
            data_loader: DataLoader for training data

        Returns:
            Tuple of (average losses dict, first batch sample images)
        """
        self.generator.train()
        self.discriminator.train()
        self.generator_ema.train()

        epoch_losses = {
            "loss_g": [],
            "loss_d": [],
            "loss_l1": [],
            "loss_perceptual": [],
            "loss_adversarial": [],
            "loss_feature_matching": [],
            "loss_identity": [],
            "loss_r1": [],
            "loss_fft": [],
        }

        sample_images = None

        with tqdm(total=len(data_loader), desc="Training") as pbar:
            for batch_idx, batch in enumerate(data_loader):
                # Move full batch to device (train_step will handle this too)
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # Train discriminator more than generator (helps stability) - D:G = 2:1
                losses = self.train_step(batch, train_d_steps=2)

                # Capture sample images from first batch for visualization (use EMA generator)
                if batch_idx == 0 and sample_images is None:
                    with torch.no_grad():
                        colorized = batch["colorized"]
                        sample_images = self.generator_ema(colorized[:4] if colorized.size(0) >= 4 else colorized)

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

        return avg_losses, sample_images

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
            "generator_ema_state_dict": self.generator_ema.state_dict(),  # NEW: Save EMA generator
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
        if "generator_ema_state_dict" in checkpoint:
            self.generator_ema.load_state_dict(checkpoint["generator_ema_state_dict"])
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

    # Create dataset and dataloader (with augmentation enabled)
    dataset = ImageRefinementDataset(
        colorized_dir=args.colorized_dir,
        target_dir=args.target_dir,
        image_size=args.image_size,
        augment=True,  # Enable data augmentation for better generalization
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

    # Create output directory for sample images
    samples_dir = Path(args.output_dir).parent / "outputs" / "gan_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Sample images will be saved to: {samples_dir}")

    # Training loop
    logger.info(f"Starting training for {args.num_epochs} epochs...")

    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"{'='*60}")

        losses, sample_images = trainer.train_epoch(data_loader)

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

        # Save sample images every epoch
        if sample_images is not None:
            try:
                sample_path = samples_dir / f"epoch_{epoch + 1}.png"
                save_image(
                    sample_images,
                    str(sample_path),
                    normalize=True,
                    nrow=2,
                )
                logger.info(f"Sample images saved to {sample_path}")
            except Exception as e:
                logger.warning(f"Failed to save sample images: {e}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(Path(args.output_dir), epoch + 1)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
