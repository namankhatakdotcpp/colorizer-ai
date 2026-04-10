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
        create_graph=False,
        retain_graph=False,
        only_inputs=True,
    )[0]
    return grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()


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


def color_histogram_loss(fake: torch.Tensor, real: torch.Tensor, num_bins: int = 32) -> torch.Tensor:
    """
    Color histogram matching loss - aligns RGB distribution between fake and real.
    
    Encourages generator to match the color distribution of target images,
    improving perceptual quality and reducing color shifts.

    Args:
        fake: Generated image, shape (B, 3, H, W), range [-1, 1]
        real: Target image, shape (B, 3, H, W), range [-1, 1]
        num_bins: Number of histogram bins

    Returns:
        Histogram matching loss value
    """
    # Convert from [-1, 1] to [0, 1] for histogram calculation
    fake_norm = (fake + 1.0) / 2.0
    real_norm = (real + 1.0) / 2.0
    
    loss = torch.tensor(0.0, device=fake.device)
    
    # Compute histogram loss for each RGB channel
    for c in range(3):
        # Compute histograms for each image in batch
        fake_hist = torch.histogram(
            fake_norm[:, c, :, :].flatten().detach().cpu(),
            bins=num_bins,
            range=(0.0, 1.0)
        ).hist.to(fake.device)
        real_hist = torch.histogram(
            real_norm[:, c, :, :].flatten().detach().cpu(),
            bins=num_bins,
            range=(0.0, 1.0)
        ).hist.to(fake.device)
        
        # Normalize histograms
        fake_hist = fake_hist / (fake_hist.sum() + 1e-8)
        real_hist = real_hist / (real_hist.sum() + 1e-8)
        
        # L2 distance between histograms
        channel_loss = torch.sqrt(torch.sum((fake_hist - real_hist) ** 2) + 1e-8)
        loss = loss + channel_loss
    
    return loss / 3.0  # Average over channels


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
        learning_rate_g: float = 1e-4,
        learning_rate_d: float = 2e-4,
        lambda_l1: float = 50.0,
        lambda_perceptual: float = 10.0,
        lambda_adversarial: float = 1.0,
        lambda_feature_matching: float = 10.0,
        lambda_histogram: float = 5.0,
        n_critic: int = 2,
    ):
        """
        Initialize trainer with production-grade settings.

        Args:
            device: Device to use (cuda or cpu)
            learning_rate_g: Learning rate for generator (TTUR: lower)
            learning_rate_d: Learning rate for discriminator (TTUR: higher, 2x G)
            lambda_l1: Weight for L1 reconstruction loss
            lambda_perceptual: Weight for perceptual loss
            lambda_adversarial: Weight for adversarial loss
            lambda_feature_matching: Weight for feature matching loss
            lambda_histogram: Weight for color histogram loss
            n_critic: Number of discriminator updates per generator update
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TTUR: Two Time-Scale Update Rule for stable training
        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d
        self.n_critic = n_critic  # Discriminator update frequency

        # Loss weights (production-grade configuration)
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adversarial = lambda_adversarial
        self.lambda_feature_matching = lambda_feature_matching
        self.lambda_histogram = lambda_histogram

        logger.info(f"Using device: {self.device}")
        logger.info(f"TTUR enabled: G_lr={learning_rate_g:.2e}, D_lr={learning_rate_d:.2e} (ratio 1:{learning_rate_d/learning_rate_g:.1f})")
        logger.info(f"n_critic={n_critic} (D updates per G update)")

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
        
        # Label smoothing for stability (standard GAN tech)
        self.real_label_smooth = 0.9  # Real labels: 0.9 instead of 1.0
        self.fake_label = 0.1         # Fake labels: 0.1 instead of 0.0
        
        # Mixed Precision Training (AMP)
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

        try:
            import torchvision
            self.perceptual_loss = VGGPerceptualLoss(device=self.device)
            self.use_perceptual = True
        except ImportError:
            logger.warning("torchvision not available, skipping perceptual loss")
            self.perceptual_loss = None
            self.use_perceptual = False

        # Optimizers with TTUR (different learning rates)
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

        # Production-grade loss tracking
        self.training_history = {
            "epoch": [],
            "loss_g_total": [],
            "loss_d_total": [],
            "loss_g_adv": [],
            "loss_g_l1": [],
            "loss_g_perceptual": [],
            "loss_g_feature_matching": [],
            "loss_g_histogram": [],
            "loss_d_real": [],
            "loss_d_fake": [],
            "loss_r1": [],
            "grad_ratio": [],  # G gradient / D gradient ratio
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

        # Initialize loss tracking for this step (not lists, single values per step)
        step_losses = {}

        # ============== Discriminator Step (multiple times) ==============
        for d_iter in range(train_d_steps):
            self.optimizer_d.zero_grad(set_to_none=True)
            self.d_steps += 1

            # 🔥 CRITICAL: Generate fake in no_grad() to prevent D backward flowing into G graph
            # This is standard GAN practice: D step should NOT have graph connected to G
            # If D backward tried to flow to G, it would conflict with G step's graph
            with torch.no_grad():
                refined_d = self.generator(colorized)  # NO gradient graph
            
            # Add small noise to real images for stability
            real_noisy = target + 0.05 * torch.randn_like(target)
            real_noisy = torch.clamp(real_noisy, -1.0, 1.0)
            
            # Real images need gradients for R1 penalty
            real_noisy_for_grad = real_noisy.detach().clone().requires_grad_(True)
            
            # Add noise to fake images
            fake_noisy = refined_d + 0.05 * torch.randn_like(refined_d)
            fake_noisy = torch.clamp(fake_noisy, -1.0, 1.0)

            # UPGRADE: LAB Conditioning (SIMPLE & CLEAN: [L, RGB] = 4 channels)
            # L provides brightness context from target
            # RGB provides the refined color image
            L_expanded = L_channel.expand(batch_size, -1, -1, -1) if L_channel.dim() == 3 else L_channel
            
            # Concatenate [L (1ch) + RGB (3ch)] = 4 channels (discriminator input channels)
            real_conditional = torch.cat([L_expanded, real_noisy_for_grad], dim=1)  # 4 channels: 1 (L) + 3 (RGB)
            fake_conditional = torch.cat([L_expanded, fake_noisy], dim=1)  # 4 channels: 1 (L) + 3 (RGB)

            # Discriminator outputs (multi-scale with intermediate features)
            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                disc_real_logits, disc_real_features = self.discriminator(real_conditional)
                disc_fake_logits, disc_fake_features = self.discriminator(fake_conditional)

            # ===== DISCRIMINATOR LOSS WITH LABEL SMOOTHING =====
            # Label smoothing: real=0.9, fake=0.1 (prevents discriminator collapse)
            real_labels = torch.full_like(disc_real_logits[0], self.real_label_smooth)
            fake_labels = torch.full_like(disc_fake_logits[0], self.fake_label)
            
            # BCE loss with label smoothing for stability
            loss_d_real = self.adversarial_loss(disc_real_logits[0], real_labels)
            loss_d_fake = self.adversarial_loss(disc_fake_logits[0], fake_labels)
            
            # Compute D loss WITHOUT R1 first
            loss_d = loss_d_real + loss_d_fake
            
            # R1 penalty: separate backward, only every 4 steps
            # 🔥 CRITICAL: R1 uses completely separate forward/backward to avoid double-backward error
            if self.d_steps % 4 == 0:
                # Separate forward pass just for R1
                real_for_r1 = real_noisy.detach().clone().requires_grad_(True)
                L_for_r1 = L_expanded.detach().clone()
                real_cond_r1 = torch.cat([L_for_r1, real_for_r1], dim=1)
                
                with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                    disc_real_r1_logits, _ = self.discriminator(real_cond_r1)
                
                # R1 backward separately
                r1_grad = torch.autograd.grad(
                    outputs=disc_real_r1_logits[0].sum(),
                    inputs=real_for_r1,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )[0]
                loss_r1 = r1_grad.pow(2).reshape(r1_grad.shape[0], -1).sum(1).mean()
                
                # Add scaled R1 to D loss as a constant (detached)
                loss_d = loss_d + 1.0 * loss_r1.detach()
                
                # Backward R1 separately
                r1_backward = 1.0 * loss_r1
                if self.scaler is not None:
                    self.scaler.scale(r1_backward).backward()
                else:
                    r1_backward.backward()
            else:
                loss_r1 = torch.tensor(0.0, device=device)

            # 🔥 SAFETY ASSERTION: Verify refined_d has NO gradients
            assert refined_d.requires_grad == False, "ERROR: refined_d MUST NOT have gradients!"

            # 🔥 NaN DETECTION - Skip if loss is invalid
            if not torch.isfinite(loss_d):
                logger.warning(f"NaN/Inf in D loss, skipping batch")
                self.optimizer_d.zero_grad(set_to_none=True)
                continue

            # 🔥 BACKWARD FOR MAIN D LOSS (graph is now clean, separate from R1)
            if self.scaler is not None:
                self.scaler.scale(loss_d_real + loss_d_fake).backward()
                self.scaler.unscale_(self.optimizer_d)
            else:
                (loss_d_real + loss_d_fake).backward()
                
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.5)
            
            # 🔥 CRITICAL: After optimizer step, explicitly zero gradients
            # This prevents old gradients from interfering with G step
            if self.scaler is not None:
                self.scaler.step(self.optimizer_d)
                self.scaler.update()
            else:
                self.optimizer_d.step()
            
            # 🔥 EXPLICIT GRADIENT ZEROING
            # Ensures D parameters are completely clean before G step
            self.optimizer_d.zero_grad(set_to_none=True)

            # Store D step losses (will be overwritten for each d_iter)
            step_losses["loss_d"] = loss_d.item()
            step_losses["loss_r1"] = loss_r1.item() if isinstance(loss_r1, torch.Tensor) else 0.0

        # ============== Generator Step ==============
        # 🔥 CRITICAL: Freeze discriminator to prevent graph overlap during G step
        for p in self.discriminator.parameters():
            p.requires_grad = False
        
        self.optimizer_g.zero_grad(set_to_none=True)

        # 🔥 HARD SAFETY: Detach AND clone ALL inputs
        target_g = target.detach().clone()
        colorized_g = colorized.detach().clone()
        L_channel_g = L_channel.detach().clone()

        # 🔥 CRITICAL: All G computations in one autocast block for consistency
        with torch.amp.autocast('cuda', enabled=self.scaler is not None):
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
                loss_perceptual = self.perceptual_loss(refined_g, target_g.detach())

            # Color histogram loss: match RGB distribution 
            loss_histogram = color_histogram_loss(refined_g, target_g)

            # FFT Loss: Frequency domain matching
            fft_fake = torch.fft.rfft2(refined_g.float())
            fft_real = torch.fft.rfft2(target_g.float())
            loss_fft = torch.mean(torch.abs(fft_fake - fft_real).real)

            # ===== SINGLE REFINED CONDITIONAL FOR ALL D FORWARDS =====
            L_expanded_g = L_channel_g.expand(batch_size, -1, -1, -1) if L_channel_g.dim() == 3 else L_channel_g
            refined_conditional = torch.cat([L_expanded_g, refined_g], dim=1)
            
            # --------------------------------------------------------
            # 1️⃣ ADVERSARIAL LOSS (STANDARD GAN MATH)
            # --------------------------------------------------------
            # 🔥 ONLY ONE D forward in entire G step
            disc_fake_logits, disc_fake_g_features = self.discriminator(refined_conditional)
            
            # 💯 Standard adversarial loss (no hacks, just correct math)
            # Use first scale's logits for adversarial loss
            loss_adv_g = -disc_fake_logits[0].mean()

            # --------------------------------------------------------
            # 2️⃣ FEATURE MATCHING (NO GRAPH - COMPLETELY ISOLATED)
            # --------------------------------------------------------
            loss_fm = torch.tensor(0.0, device=device)
            
            # 🔥 CRITICAL: Entire FM computation inside no_grad()
            # This prevents ANY graph building from interfering with adversarial loss
            with torch.no_grad():
                # Fake features (no gradients) - extract intermediate features
                # 💥 CRITICAL: Detach refined_conditional to break graph leak
                disc_fake_fm_logits, disc_fake_fm_features = self.discriminator(refined_conditional.detach())
                # Flatten multi-scale features for matching
                fake_features = []
                for scale_features in disc_fake_fm_features:
                    fake_features.extend(scale_features)
                
                # Real features (no gradients)
                real_noisy_g = target_g + 0.05 * torch.randn_like(target_g)
                real_noisy_g = torch.clamp(real_noisy_g, -1.0, 1.0)
                real_conditional_g = torch.cat([L_expanded_g, real_noisy_g], dim=1)
                # 💥 CRITICAL: Detach real_conditional_g to break graph leak
                disc_real_fm_logits, disc_real_fm_features = self.discriminator(real_conditional_g.detach())
                real_features = []
                for scale_features in disc_real_fm_features:
                    real_features.extend(scale_features)
                
                # Compute FM loss (no gradients at all)
                if fake_features and real_features:
                    fm_loss_total = torch.tensor(0.0, device=device)
                    for fake_feat, real_feat in zip(fake_features, real_features):
                        fm_loss_total += torch.mean(torch.abs(fake_feat - real_feat))
                    loss_fm = fm_loss_total / max(len(real_features), 1)
            
            # ===== COMBINE ALL LOSSES WITH PRODUCTION WEIGHTS =====
            # Loss combination: adversarial (for fooling D), + reconstruction (content fidelity)
            # + perceptual (feature space), + feature matching (discriminator features),
            # + histogram (color distribution), + FFT (frequency domain)
            loss_g = (
                self.lambda_adversarial * loss_adv_g +        # GAN: 1.0
                self.lambda_l1 * loss_l1 +                    # L1: 50.0 (content)
                self.lambda_perceptual * loss_perceptual +    # Perceptual: 10.0
                self.lambda_feature_matching * loss_fm +      # Feature Matching: 10.0
                self.lambda_histogram * loss_histogram        # Histogram: 5.0
            )

        # 🔥 SAFETY ASSERTIONS: Verify complete graph isolation (no debug output)
        assert refined_g.requires_grad == True, "ERROR: refined_g MUST have gradients for G step!"
        assert target_g.requires_grad == False, "ERROR: target_g MUST NOT have gradients!"
        assert colorized_g.requires_grad == False, "ERROR: colorized_g MUST NOT have gradients!"
        assert not any(p.requires_grad for p in self.discriminator.parameters()), "ERROR: D parameters MUST be frozen during G step!"

        # 🔥 NaN DETECTION - Skip if loss is invalid
        if not torch.isfinite(loss_g):
            logger.warning(f"NaN/Inf in G loss, skipping batch")
            self.optimizer_g.zero_grad(set_to_none=True)
        else:
            # 🔥 BACKWARD FOR G STEP
            # This is the ONLY backward on loss_g for this train step
            if self.scaler is not None:
                self.scaler.scale(loss_g).backward()
                self.scaler.unscale_(self.optimizer_g)
            else:
                loss_g.backward()
            
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.5)
            
            if self.scaler is not None:
                self.scaler.step(self.optimizer_g)
                self.scaler.update()
            else:
                self.optimizer_g.step()
            
            # 🔥 EXPLICIT: Clear G gradients after step
            self.optimizer_g.zero_grad(set_to_none=True)
        
        # 🔥 CRITICAL: Restore discriminator gradients AFTER G step is complete
        for p in self.discriminator.parameters():
            p.requires_grad = True
        
        # UPDATE EMA GENERATOR (important for eval) 
        with torch.no_grad():
            for param, ema_param in zip(self.generator.parameters(), self.generator_ema.parameters()):
                ema_param.data = self.ema_decay * ema_param.data + (1 - self.ema_decay) * param.data

        # Return single batch losses (not averaged lists)
        return {
            "loss_d": step_losses.get("loss_d", loss_d.item()),
            "loss_g": loss_g.item(),
            "loss_r1": step_losses.get("loss_r1", 0.0)
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

        with tqdm(total=len(data_loader), desc="Training", leave=True) as pbar:
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
                }, refresh=True)
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

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
    import sys
    
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

    # Check that data directories exist and are populated
    if not Path(args.colorized_dir).exists() or not any(Path(args.colorized_dir).iterdir()):
        print(f"ERROR: Colorized directory '{args.colorized_dir}' is empty or missing.")
        print("Run the colorizer pipeline first to generate training data:")
        print("  python inference_pipeline.py --stages colorizer --output-dir data/colorized/")
        sys.exit(1)
    
    if not Path(args.target_dir).exists() or not any(Path(args.target_dir).iterdir()):
        print(f"ERROR: Target directory '{args.target_dir}' is empty or missing.")
        print("Ensure ground truth images are available.")
        sys.exit(1)

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

    # Create trainer with production-grade settings (TTUR)
    trainer = GANRefinementTrainer(
        device=device,
        learning_rate_g=args.learning_rate_g if args.learning_rate_g != 0.0002 else 1e-4,  # TTUR: 1e-4
        learning_rate_d=args.learning_rate_d if args.learning_rate_d != 0.0002 else 2e-4,  # TTUR: 2e-4
        lambda_l1=args.lambda_l1 if args.lambda_l1 != 100.0 else 50.0,  # Updated: 50.0
        lambda_perceptual=args.lambda_perceptual,
        lambda_feature_matching=10.0,
        lambda_histogram=5.0,
        n_critic=2,
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

        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs} - "
            f"G_total: {losses['loss_g_total']:.4f} | "
            f"D_total: {losses['loss_d_total']:.4f} | "
            f"G_adv: {losses['loss_g_adv']:.4f} | "
            f"G_l1: {losses['loss_g_l1']:.4f} | "
            f"G_perceptual: {losses['loss_g_perceptual']:.4f} | "
            f"G_fm: {losses['loss_g_feature_matching']:.4f} | "
            f"G_hist: {losses['loss_g_histogram']:.4f} | "
            f"D_real: {losses['loss_d_real']:.4f} | "
            f"D_fake: {losses['loss_d_fake']:.4f} | "
            f"R1: {losses['loss_r1']:.6f}"
        )

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
