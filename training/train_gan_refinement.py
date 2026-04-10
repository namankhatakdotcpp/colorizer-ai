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

# 🔥 ENABLE ANOMALY DETECTION FOR GRADIENT FLOW DEBUGGING
torch.autograd.set_detect_anomaly(True)


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
        learning_rate_g: float = 2e-5,  # TUNED: 5e-5 → 2e-5 (lower for stability)
        learning_rate_d: float = 1e-5,  # FIX 2: CRITICAL - 5e-5 → 1e-5 (prevent D overpowering G)
        lambda_l1: float = 5.0,         # TUNED: 50.0 → 5.0 (reduce reconstruction aggression)
        lambda_perceptual: float = 0.5, # TUNED: 10.0 → 0.5 (gentler perceptual loss)
        lambda_adversarial: float = 1.0,
        lambda_feature_matching: float = 10.0,
        lambda_histogram: float = 5.0,
        n_critic: int = 1,             # FIX 3: CRITICAL - 2 → 1 (less aggressive D updates)
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
        # FIX 7: ADD LOGIT STABILITY - pos_weight prevents numerical collapse
        self.adversarial_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(self.device))
        self.l1_loss = nn.L1Loss()
        
        # Label smoothing for stability (standard GAN tech)
        # FIX 4: STRONGER smoothing prevents D from being overconfident
        self.real_label_smooth = 0.85  # FIX 4: 0.9 → 0.85 (more uncertainty)
        self.fake_label = 0.15        # FIX 4: 0.1 → 0.15 (more uncertainty)
        
        # Mixed Precision Training (AMP)
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        # STEP 6: TEMPORARY DEBUG MODE - Disable AMP to run in full FP32 for testing
        # Set to True to diagnose AMP-related crashes (runs slower but more stable)
        self.debug_amp_disabled = False

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
        # FIX 1: ENSURE D PARAMETERS ARE TRAINABLE AT START
        # If a previous step crashed before restoring D params, this fixes it
        for p in self.discriminator.parameters():
            if not p.requires_grad:
                logger.warning("⚠️ Discriminator parameters were frozen! Restoring trainability...")
                p.requires_grad = True
        
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
            
            # FIX 1: EXPLICIT DETACH for safety
            refined_d = refined_d.detach()
            
            # Add small noise to real images for stability (label smoothing + noise)
            real_noisy = target + 0.01 * torch.randn_like(target)  # REDUCED: 0.05 → 0.01 for stability
            real_noisy = torch.clamp(real_noisy, -1.0, 1.0)
            
            # Real images need gradients for R1 penalty
            real_noisy_for_grad = real_noisy.detach().clone().requires_grad_(True)
            
            # Add noise to fake images (same as real for consistency)
            fake_noisy = refined_d + 0.01 * torch.randn_like(refined_d)  # REDUCED: 0.05 → 0.01
            fake_noisy = torch.clamp(fake_noisy, -1.0, 1.0)
            
            # 🔥 MIXUP TRAINING: Mix real and fake for discriminator
            # This makes discriminator stronger by training on intermediate samples
            # alpha = 0.2 creates 80% real + 20% fake blends
            alpha = 0.2
            mixed = alpha * fake_noisy + (1 - alpha) * real_noisy
            mixed = torch.clamp(mixed, -1.0, 1.0)

            # UPGRADE: LAB Conditioning (SIMPLE & CLEAN: [L, RGB] = 4 channels)
            # L provides brightness context from target
            # RGB provides the refined color image
            L_expanded = L_channel.expand(batch_size, -1, -1, -1) if L_channel.dim() == 3 else L_channel
            
            # Concatenate [L (1ch) + RGB (3ch)] = 4 channels (discriminator input channels)
            real_conditional = torch.cat([L_expanded, real_noisy_for_grad], dim=1)  # 4 channels: 1 (L) + 3 (RGB)
            fake_conditional = torch.cat([L_expanded, fake_noisy], dim=1)  # 4 channels: 1 (L) + 3 (RGB)
            mixed_conditional = torch.cat([L_expanded, mixed], dim=1)  # 4 channels: 1 (L) + 3 (RGB) - for mixup

            # FIX 2: FORCE INPUT CLAMP before D (safety)
            fake_conditional = torch.clamp(fake_conditional, -1.0, 1.0)
            real_conditional = torch.clamp(real_conditional, -1.0, 1.0)
            
            # FIX 6: SAFE CONV INPUT CHECK - ensure no NaN/Inf before discriminator
            if not torch.isfinite(fake_conditional).all():
                logger.warning("⚠️ Bad input to discriminator (fake_conditional), skipping batch")
                self.optimizer_d.zero_grad(set_to_none=True)
                continue
            if not torch.isfinite(real_conditional).all():
                logger.warning("⚠️ Bad input to discriminator (real_conditional), skipping batch")
                self.optimizer_d.zero_grad(set_to_none=True)
                continue

            # Discriminator outputs (multi-scale with intermediate features)
            # FIX 2: VERIFY REQUIRES_GRAD BEFORE FORWARD
            # FIX 3: DISABLE AMP FOR DISCRIMINATOR to prevent NaN in convolution
            
            # Diagnostic: Check that discriminator parameters require grad
            d_params_require_grad = any(p.requires_grad for p in self.discriminator.parameters())
            if not d_params_require_grad:
                logger.error("🔴 CRITICAL: Discriminator parameters do NOT require grad!")
                raise RuntimeError("Discriminator parameters are frozen! Cannot compute gradients.")
            
            # STEP 1: Use new torch.amp.autocast API (not deprecated torch.cuda.amp.autocast)
            with torch.amp.autocast(device_type='cuda', enabled=False):
                disc_real_logits, disc_real_features = self.discriminator(real_conditional)
                disc_fake_logits, disc_fake_features = self.discriminator(fake_conditional)
                disc_mixed_logits, _ = self.discriminator(mixed_conditional)  # 🔥 MIXUP: Train on mixed samples
            
            # 🔥 VERIFY PATCH DISCRIMINATOR OUTPUTS (shape check)
            # Expected: list of [B, 1, H, W] tensors (patch maps)
            # If output is [B, 1], discriminator is NOT patch-based (issue!)
            if isinstance(disc_real_logits, (list, tuple)):
                for scale_idx, logit in enumerate(disc_real_logits):
                    if logit.dim() == 2:
                        logger.warning(f"⚠️ Patch Discriminator Output Issue: Scale {scale_idx} has shape {logit.shape} (expected [B, 1, H, W])")
                        # Reshape if it's [B, 1] to ensure proper patch discrimination
                        if logit.shape[1] == 1 and logit.dim() == 2:
                            disc_real_logits[scale_idx] = logit.unsqueeze(-1).unsqueeze(-1)
                            disc_fake_logits[scale_idx] = disc_fake_logits[scale_idx].unsqueeze(-1).unsqueeze(-1)
                            disc_mixed_logits[scale_idx] = disc_mixed_logits[scale_idx].unsqueeze(-1).unsqueeze(-1)

            # FIX 1: VERIFY DISCRIMINATOR PARAMETERS REQUIRE GRAD
            # If this fails, D parameters are frozen (likely from previous step error)
            assert any(p.requires_grad for p in self.discriminator.parameters()), \
                "CRITICAL: Discriminator parameters do NOT require grad! Parameters are frozen."
            
            # FIX 3: VERIFY DISCRIMINATOR OUTPUTS REQUIRE GRAD
            # If this fails, gradient flow is broken in discriminator
            assert disc_real_logits[0].requires_grad, \
                f"ERROR: disc_real_logits[0] requires_grad={disc_real_logits[0].requires_grad}, must be True!"
            assert disc_fake_logits[0].requires_grad, \
                f"ERROR: disc_fake_logits[0] requires_grad={disc_fake_logits[0].requires_grad}, must be True!"

            # FIX 4: CLAMP DISCRIMINATOR OUTPUT before loss (prevents extreme logits)
            # Clamp to [-10, 10] to prevent numerical explosion in BCE sigmoid
            disc_real_logits = [torch.clamp(logit, -10, 10) for logit in disc_real_logits]
            disc_fake_logits = [torch.clamp(logit, -10, 10) for logit in disc_fake_logits]
            
            # FIX 4: VERIFY CLAMPING PRESERVES REQUIRES_GRAD
            # torch.clamp preserves requires_grad, but verify it
            assert disc_real_logits[0].requires_grad, "ERROR: Clamping broke requires_grad for real logits!"
            assert disc_fake_logits[0].requires_grad, "ERROR: Clamping broke requires_grad for fake logits!"

            # ===== DISCRIMINATOR LOSS WITH LABEL SMOOTHING & MIXUP =====
            # Label smoothing: real=0.9, fake=0.1 (prevents discriminator collapse)
            real_labels = torch.full_like(disc_real_logits[0], self.real_label_smooth)
            fake_labels = torch.full_like(disc_fake_logits[0], self.fake_label)
            mixed_labels = torch.full_like(disc_mixed_logits[0], self.real_label_smooth * 0.9 + self.fake_label * 0.1)  # 🔥 MIXUP: Mixed label = 0.765
            
            # BCE loss with label smoothing for stability
            loss_d_real = self.adversarial_loss(disc_real_logits[0], real_labels)
            loss_d_fake = self.adversarial_loss(disc_fake_logits[0], fake_labels)
            loss_d_mixed = self.adversarial_loss(disc_mixed_logits[0], mixed_labels)  # 🔥 MIXUP: Add mixed loss (1/3 weight)
            
            # FIX 4: VERIFY LOSS REQUIRES GRAD (CRITICAL)
            # This must be True for backward() to work
            assert loss_d_real.requires_grad, \
                f"ERROR: loss_d_real requires_grad={loss_d_real.requires_grad}, must be True!"
            assert loss_d_fake.requires_grad, \
                f"ERROR: loss_d_fake requires_grad={loss_d_fake.requires_grad}, must be True!"
            
            # Compute D loss WITHOUT R1 first
            # 🔥 MIXUP: Add mixed loss with 1/3 weight to balance with real+fake
            loss_d = loss_d_real + loss_d_fake + 0.33 * loss_d_mixed
            
            # ===== R1 GRADIENT PENALTY (STEP 1-6: CORRECT IMPLEMENTATION) =====
            # STEP 1: R1 requires real images with requires_grad=True (CRITICAL)
            # STEP 2: Compute R1 gradients separately (independent graph)
            # STEP 3: Scale R1 loss and ADD to total D loss
            # STEP 4: Do ONE backward on total_d_loss (not separate backward)
            if self.d_steps % 4 == 0:
                # Real images MUST have requires_grad=True for R1
                # STEP 1: ENABLE GRAD ON REAL IMAGES
                real_for_r1 = real_noisy.detach().clone()
                real_for_r1.requires_grad_(True)  # 💥 CRITICAL - this must be set BEFORE discriminator forward
                
                L_for_r1 = L_expanded.detach().clone()
                real_cond_r1 = torch.cat([L_for_r1, real_for_r1], dim=1)
                
                # STEP 2: COMPUTE R1 CORRECTLY with torch.autograd.grad
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    disc_real_r1_logits, _ = self.discriminator(real_cond_r1)
                
                # Compute gradient of discriminator output w.r.t. real images
                # STEP 2: Verify real_for_r1 requires grad before grad computation
                assert real_for_r1.requires_grad == True, "ERROR: real_for_r1 must require grad for R1 penalty!"
                
                try:
                    r1_grads = torch.autograd.grad(
                        outputs=disc_real_r1_logits[0].sum(),
                        inputs=real_for_r1,
                        create_graph=True,  # 💥 CRITICAL: create_graph=True so gradients can flow to D params
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    
                    # STEP 3: Compute R1 penalty
                    r1_penalty = r1_grads.pow(2).reshape(r1_grads.shape[0], -1).sum(1).mean()
                    loss_r1 = (10.0 / 2) * r1_penalty  # gamma=10 (standard for image-to-image tasks)
                    
                    # STEP 3: Verify R1 loss requires grad
                    assert loss_r1.requires_grad == True, "ERROR: loss_r1 must require grad!"
                    
                except RuntimeError as e:
                    logger.error(f"🔴 R1 gradient computation failed: {e}")
                    logger.error(f"real_for_r1.requires_grad={real_for_r1.requires_grad}")
                    logger.error(f"disc_real_r1_logits[0].requires_grad={disc_real_r1_logits[0].requires_grad}")
                    loss_r1 = torch.tensor(0.0, device=device)
            else:
                loss_r1 = torch.tensor(0.0, device=device)
            
            # STEP 4: COMBINE ALL LOSSES - Add R1 to total D loss
            # 🔥 CRITICAL: ONE backward on total loss (not separate backward)
            total_d_loss = loss_d + loss_r1  # loss_d already has requires_grad, and loss_r1 has graph
            
            # STEP 6: DEBUG CHECK - Verify both components have grad
            assert loss_d.requires_grad == True, "ERROR: loss_d must require grad!"
            assert loss_r1.requires_grad == True or (self.d_steps % 4 != 0), "ERROR: loss_r1 must require grad when computed!"

            # 🔥 SAFETY ASSERTION: Verify refined_d has NO gradients
            assert refined_d.requires_grad == False, "ERROR: refined_d MUST NOT have gradients!"

            # 🔥 NaN DETECTION - Skip if loss is invalid
            if not torch.isfinite(loss_d):
                logger.warning(f"NaN/Inf in D loss, skipping batch")
                self.optimizer_d.zero_grad(set_to_none=True)
                continue
            
            # FIX 5: GRADIENT NAN GUARD (CRITICAL) - More robust check before backward
            if torch.isnan(loss_d) or torch.isinf(loss_d):
                logger.warning(f"⚠️ Skipping D backward due to NaN/Inf in loss_d")
                self.optimizer_d.zero_grad(set_to_none=True)
                continue
            if torch.isnan(loss_d_real) or torch.isinf(loss_d_real):
                logger.warning(f"⚠️ Skipping D backward due to NaN/Inf in loss_d_real")
                self.optimizer_d.zero_grad(set_to_none=True)
                continue
            if torch.isnan(loss_d_fake) or torch.isinf(loss_d_fake):
                logger.warning(f"⚠️ Skipping D backward due to NaN/Inf in loss_d_fake")
                self.optimizer_d.zero_grad(set_to_none=True)
                continue

            # 🔥 BACKWARD FOR MAIN D LOSS (graph is now clean, separate from R1)
            # STEP 5: BACKWARD ON TOTAL LOSS (includes R1 if computed)
            # Debug output before backward
            logger.debug(f"D Step {self.d_steps}: loss_d_real.requires_grad={loss_d_real.requires_grad}, loss_d_fake.requires_grad={loss_d_fake.requires_grad}")
            logger.debug(f"D Step {self.d_steps}: total_d_loss.requires_grad={total_d_loss.requires_grad}, total_d_loss={total_d_loss.item():.6f}")
            
            if self.scaler is not None:
                self.scaler.scale(total_d_loss).backward()
                self.scaler.unscale_(self.optimizer_d)
            else:
                total_d_loss.backward()
            
            # Gradient clipping for stability
            # FIX 1: GRADIENT CLIP (MOST IMPORTANT) - prevents exploding gradients (loss scaling fix #1)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            
            # 🔥 ADDITIONAL: Verify no NaN gradients after clipping
            d_has_nan_grads = False
            for param in self.discriminator.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    d_has_nan_grads = True
                    logger.warning(f"⚠️ NaN detected in D gradients after clipping, zeroing")
                    param.grad.zero_()
            
            if d_has_nan_grads:
                logger.warning(f"⚠️ D step {self.d_steps}: NaN gradients detected, skipping optimizer step")
                self.optimizer_d.zero_grad(set_to_none=True)
                continue
            
            # 🔥 CRITICAL: After optimizer step, explicitly zero gradients
            # This prevents old gradients from interfering with G step
            if self.scaler is not None:
                self.scaler.step(self.optimizer_d)
                self.scaler.update()
            else:
                self.optimizer_d.step()
            
            # FIX 5: OPTIONAL WGAN-STYLE WEIGHT CLIPPING (pro mode)
            # Helps stabilize discriminator by clipping weights to small range
            # This is experimental and often helps with convergence
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
            
            # 🔥 EXPLICIT GRADIENT ZEROING
            # Ensures D parameters are completely clean before G step
            self.optimizer_d.zero_grad(set_to_none=True)

            # Store D step losses (will be overwritten for each d_iter, final value kept)
            step_losses["loss_d"] = total_d_loss.item()  # Total D loss (real + fake + R1)
            step_losses["loss_d_real"] = loss_d_real.item()
            step_losses["loss_d_fake"] = loss_d_fake.item()
            step_losses["loss_r1"] = loss_r1.item() if isinstance(loss_r1, torch.Tensor) and loss_r1.requires_grad else 0.0

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
        # STEP 6: Use debug_amp_disabled to run in full FP32 for testing stability
        with torch.amp.autocast(device_type='cuda', enabled=self.scaler is not None and not self.debug_amp_disabled):
            # ===== FRESH GENERATOR FORWARD FOR G STEP =====
            # Critical: NEVER reuse refined tensor from D step
            refined_g = self.generator(colorized_g)
            
            # 🔥 FIX 1: CLAMP OUTPUT TO VALID RANGE (MANDATORY)
            # Prevents output from drifting outside [-1, 1] range
            refined_g = torch.clamp(refined_g, -1.0, 1.0)

            # L1 loss: preserve content
            loss_l1 = self.l1_loss(refined_g, target_g)

            # Identity loss: prevent unwanted color changes
            loss_identity = self.l1_loss(refined_g, colorized_g)

            # Perceptual loss: VGG features
            loss_perceptual = torch.tensor(0.0, device=self.device)
            if self.use_perceptual and self.perceptual_loss is not None:
                loss_perceptual = self.perceptual_loss(refined_g, target_g.detach())

            # Color histogram loss: match RGB distribution (NO GRAD - histogram doesn't support autograd)
            # WRAPPED IN NO_GRAD: histogram operations detach tensors, so we compute separately
            with torch.no_grad():
                loss_histogram = color_histogram_loss(refined_g, target_g)
                
                # 🔥 FIX: HISTOGRAM LOSS INSTABILITY - Handle NaN case (prevents gradient explosion)
                # If histogram loss is NaN, replace with 0.0 to skip monitoring
                if torch.isnan(loss_histogram) or torch.isinf(loss_histogram):
                    logger.warning(f"⚠️ Histogram loss is NaN/Inf, replacing with 0.0 for safety")
                    loss_histogram = torch.tensor(0.0, device=device)
            
            # ⚠️ NOTE: loss_histogram is NOT part of backprop - it's monitoring only

            # FFT Loss: Frequency domain matching (monitoring loss, no gradient flow)  
            # WRAPPED IN NO_GRAD: FFT-based loss can have gradient flow issues
            with torch.no_grad():
                fft_fake = torch.fft.rfft2(refined_g.float())
                fft_real = torch.fft.rfft2(target_g.float())
                loss_fft = torch.mean(torch.abs(fft_fake - fft_real).real)
                
                # 🔥 FIX: FFT LOSS INSTABILITY - Handle NaN case
                if torch.isnan(loss_fft) or torch.isinf(loss_fft):
                    logger.warning(f"⚠️ FFT loss is NaN/Inf, replacing with 0.0 for safety")
                    loss_fft = torch.tensor(0.0, device=device)

            # ===== SINGLE REFINED CONDITIONAL FOR ALL D FORWARDS =====
            L_expanded_g = L_channel_g.expand(batch_size, -1, -1, -1) if L_channel_g.dim() == 3 else L_channel_g
            
            # FIX 2 (CRITICAL): RECOMPUTE FOR G STEP - Create fresh conditional WITH gradients for adversarial loss
            # This is separate from the D step forward and MUST have gradients flowing back to generator
            refined_conditional_with_grad = torch.cat([L_expanded_g, refined_g], dim=1)  # NO detach for adversarial loss
            
            # Keep a detached version for FM block (monitoring only)
            refined_conditional_detached = torch.cat([L_expanded_g, refined_g.detach()], dim=1)  # Detached for FM
            
            # FIX 2: FORCE CLAMP before D
            refined_conditional_with_grad = torch.clamp(refined_conditional_with_grad, -1.0, 1.0)
            refined_conditional_detached = torch.clamp(refined_conditional_detached, -1.0, 1.0)
            
            # FIX 6: SAFE CONV INPUT CHECK - ensure no NaN/Inf before discriminator
            if torch.isnan(refined_conditional_with_grad).any() or torch.isinf(refined_conditional_with_grad).any():
                logger.warning(f"⚠️ Skipping batch (NaN/Inf in refined_conditional_with_grad)")
                self.optimizer_g.zero_grad(set_to_none=True)
                for p in self.discriminator.parameters():
                    p.requires_grad = True
                return {
                    "loss_d": step_losses.get("loss_d", 0.0),
                    "loss_g": 0.0,
                    "loss_l1": 0.0,
                    "loss_identity": 0.0,
                    "loss_perceptual": 0.0,
                    "loss_histogram": 0.0,
                    "loss_fft": 0.0,
                    "loss_fm": 0.0,
                    "loss_r1": step_losses.get("loss_r1", 0.0),
                    "loss_adv_g": 0.0,
                }
            
            # --------------------------------------------------------
            # 1️⃣ ADVERSARIAL LOSS (STANDARD GAN MATH)
            # --------------------------------------------------------
            # 🔥 ONLY ONE D forward in entire G step
            # FIX 3: DISABLE AMP FOR DISCRIMINATOR (disable inside autocast context)
            # FIX 1: Use NON-DETACHED conditional to allow gradients to flow back to generator
            with torch.amp.autocast(device_type='cuda', enabled=False):
                disc_fake_logits, disc_fake_g_features = self.discriminator(refined_conditional_with_grad)
            
            # FIX 4: CLAMP DISCRIMINATOR OUTPUT before loss (prevents extreme logits - CRITICAL for loss scaling)
            disc_real_logits = [torch.clamp(logit, -10, 10) for logit in disc_real_logits]
            disc_fake_logits = [torch.clamp(logit, -10, 10) for logit in disc_fake_logits]
            disc_mixed_logits = [torch.clamp(logit, -10, 10) for logit in disc_mixed_logits]  # 🔥 MIXUP: Clamp mixed logits
            
            # 💯 Standard adversarial loss (BCE - same as discriminator)
            # Use first scale's logits for adversarial loss
            # Generator tries to fool D by making fake images look "real" (label=0.9)
            real_labels = torch.full_like(disc_fake_logits[0], 0.9)  # Label smoothing: 0.9 instead of 1.0
            loss_adv_g = self.adversarial_loss(disc_fake_logits[0], real_labels)

            # --------------------------------------------------------
            # 2️⃣ FEATURE MATCHING (NO GRAPH - COMPLETELY ISOLATED)
            # --------------------------------------------------------
            loss_fm = torch.tensor(0.0, device=device)
            
            # 🔥 CRITICAL: Entire FM computation inside no_grad()
            # This prevents ANY graph building from interfering with adversarial loss
            with torch.no_grad():
                # Fake features (no gradients) - extract intermediate features
                # FIX 3: DISABLE AMP FOR FM discriminator calls
                # FIX 1: Use detached conditional for FM (monitoring only, no gradient impact)
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    disc_fake_fm_logits, disc_fake_fm_features = self.discriminator(refined_conditional_detached)
                # FIX 4: CLAMP for consistency (monitoring only, but prevent extreme values)
                disc_fake_fm_logits = [torch.clamp(logit, -10, 10) for logit in disc_fake_fm_logits]
                # Flatten multi-scale features for matching
                fake_features = []
                for scale_features in disc_fake_fm_features:
                    fake_features.extend(scale_features)
                
                # Real features (no gradients)
                real_noisy_g = target_g + 0.01 * torch.randn_like(target_g)  # FIXED: 0.05 → 0.01 for consistency
                real_noisy_g = torch.clamp(real_noisy_g, -1.0, 1.0)
                real_conditional_g = torch.cat([L_expanded_g, real_noisy_g], dim=1)
                # 💥 CRITICAL: Detach real_conditional_g to break graph leak
                # FIX 3: DISABLE AMP FOR FM discriminator calls
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    disc_real_fm_logits, disc_real_fm_features = self.discriminator(real_conditional_g.detach())
                # FIX 4: CLAMP for consistency
                disc_real_fm_logits = [torch.clamp(logit, -10, 10) for logit in disc_real_fm_logits]
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
            # 🔥 CRITICAL: ONLY include losses with active gradients
            # - loss_adv_g: HAS gradients (backprop from disc_fake_logits)
            # - loss_l1: HAS gradients (refined_g vs target_g)
            # - loss_perceptual: HAS gradients (VGG features from refined_g)
            # - loss_identity: HAS gradients (refined_g vs colorized_g)
            # - loss_fm: NO gradients (computed in torch.no_grad())
            # - loss_histogram: NO gradients (wrapped in torch.no_grad())
            # - loss_fft: NO gradients (wrapped in torch.no_grad())
            
            # Build loss with ONLY gradient-active components
            loss_g = (
                self.lambda_adversarial * loss_adv_g +        # GAN loss: 1.0 (fool discriminator)
                self.lambda_l1 * loss_l1 +                    # L1 loss: 50.0 (content preservation)
                self.lambda_l1 * 5.0 * loss_identity +        # Identity loss: 250.0 (color stability)
                self.lambda_perceptual * loss_perceptual      # Perceptual loss: 10.0 (VGG features)
                # REMOVED: loss_fm and loss_histogram - no gradients, use for monitoring only
            )
            
            # SAFETY: Verify loss requires grad before backward
            assert loss_g.requires_grad == True, f"ERROR: loss_g requires_grad={loss_g.requires_grad}, must be True!"

        # 🔥 SAFETY ASSERTIONS: Verify complete graph isolation (no debug output)
        assert refined_g.requires_grad == True, "ERROR: refined_g MUST have gradients for G step!"
        assert target_g.requires_grad == False, "ERROR: target_g MUST NOT have gradients!"
        assert colorized_g.requires_grad == False, "ERROR: colorized_g MUST NOT have gradients!"
        assert not any(p.requires_grad for p in self.discriminator.parameters()), "ERROR: D parameters MUST be frozen during G step!"

        # 🔥 VERIFY LOSS REQUIRES GRADIENTS BEFORE BACKWARD (SAFETY CHECK)
        if not loss_g.requires_grad:
            logger.error(f"CRITICAL: loss_g does not require gradients! grad_fn={loss_g.grad_fn}")
            # Return without updating - skip this bad batch
            self.optimizer_g.zero_grad(set_to_none=True)
            for p in self.discriminator.parameters():
                p.requires_grad = True
            return {
                "loss_d": step_losses.get("loss_d", 0.0),
                "loss_g": 0.0,
                "loss_l1": 0.0,
                "loss_identity": 0.0,
                "loss_perceptual": 0.0,
                "loss_histogram": 0.0,
                "loss_fft": 0.0,
                "loss_fm": 0.0,
                "loss_r1": step_losses.get("loss_r1", 0.0),
                "loss_adv_g": 0.0,
            }

        # 🔥 NaN DETECTION - Skip if loss is invalid (SAFETY CHECK)
        if not torch.isfinite(loss_g):
            logger.warning(f"⚠️ NaN/Inf in G loss (loss={loss_g.item()}), skipping batch")
            self.optimizer_g.zero_grad(set_to_none=True)
            for p in self.discriminator.parameters():
                p.requires_grad = True
            return {
                "loss_d": step_losses.get("loss_d", 0.0),
                "loss_g": 0.0,
                "loss_l1": 0.0,
                "loss_identity": 0.0,
                "loss_perceptual": 0.0,
                "loss_histogram": 0.0,
                "loss_fft": 0.0,
                "loss_fm": 0.0,
                "loss_r1": step_losses.get("loss_r1", 0.0),
                "loss_adv_g": 0.0,
            }
        # 🔥 BACKWARD FOR G STEP (ONLY REACHED if loss is valid)
        # This is the ONLY backward on loss_g for this train step
        
        # FIX 5: GRADIENT NAN GUARD (CRITICAL) - Verify loss is safe before backward
        logger.debug(f"G Loss Details: loss_g.requires_grad={loss_g.requires_grad}, loss_g={loss_g.item():.6f}")
        logger.debug(f"G Loss Breakdown: adv={loss_adv_g.item():.6f}, l1={loss_l1.item():.6f}, identity={loss_identity.item():.6f}, perceptual={loss_perceptual.item():.6f}")
        
        # STEP 5: ADD DEBUG OUTPUT - Confirm loss requires grad before backward
        if loss_g.requires_grad:
            logger.debug("✅ G loss READY for backward (requires_grad=True)")
        else:
            logger.error(f"🔴 G loss CANNOT backward (requires_grad=False, grad_fn={loss_g.grad_fn})")
            raise RuntimeError("Loss has no grad!")
        
        if torch.isnan(loss_g) or torch.isinf(loss_g):
            logger.warning(f"⚠️ Skipping G backward due to NaN/Inf in loss_g")
            self.optimizer_g.zero_grad(set_to_none=True)
            for p in self.discriminator.parameters():
                p.requires_grad = True
            return {
                "loss_d": step_losses.get("loss_d", 0.0),
                "loss_g": 0.0,
                "loss_l1": loss_l1.item() if loss_l1 > 0 else 0.0,
                "loss_identity": loss_identity.item() if loss_identity > 0 else 0.0,
                "loss_perceptual": loss_perceptual.item() if loss_perceptual > 0 else 0.0,
                "loss_histogram": loss_histogram.item(),
                "loss_fft": loss_fft.item(),
                "loss_fm": loss_fm.item(),
                "loss_r1": step_losses.get("loss_r1", 0.0),
                "loss_adv_g": 0.0,
            }
        
        # FIX 4: VERIFY LOSS HAS GRAD (CRITICAL) - Strong assertion before backward
        assert loss_g.requires_grad == True, f"CRITICAL: loss_g does not require gradients! grad_fn={loss_g.grad_fn}"
        
        if self.scaler is not None:
            self.scaler.scale(loss_g).backward()
            self.scaler.unscale_(self.optimizer_g)
        else:
            loss_g.backward()
        
# ✅ Gradient clipping for stability (prevents explosion - loss scaling fix #2)
            # FIX 1: GRADIENT CLIP (MOST IMPORTANT) - prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            
            # 🔥 ADDITIONAL: Verify no NaN gradients in G after clipping
            g_has_nan_grads = False
            for param in self.generator.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    g_has_nan_grads = True
                    logger.warning(f"⚠️ NaN detected in G gradients after clipping, zeroing")
                    param.grad.zero_()
            
            if g_has_nan_grads:
                logger.warning(f"⚠️ G step: NaN gradients detected, skipping optimizer step but continuing training")
                self.optimizer_g.zero_grad(set_to_none=True)
                for p in self.discriminator.parameters():
                    p.requires_grad = True
                # Return previous losses instead of crashing
                return step_losses if step_losses else {
                    "loss_d": 0.0,
                    "loss_g": 0.0,
                    "loss_l1": 0.0,
                    "loss_identity": 0.0,
                    "loss_perceptual": 0.0,
                    "loss_histogram": 0.0,
                    "loss_fft": 0.0,
                    "loss_fm": 0.0,
                    "loss_r1": 0.0,
                    "loss_adv_g": 0.0,
                }
        
        # ✅ Step and clear optimizer
        if self.scaler is not None:
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            self.optimizer_g.step()
        
        # ✅ EXPLICIT: Clear G gradients after step
        self.optimizer_g.zero_grad(set_to_none=True)
        
        # 🔥 CRITICAL: Restore discriminator gradients AFTER G step is complete
        for p in self.discriminator.parameters():
            p.requires_grad = True
        
        # UPDATE EMA GENERATOR (important for eval) 
        with torch.no_grad():
            for param, ema_param in zip(self.generator.parameters(), self.generator_ema.parameters()):
                ema_param.data = self.ema_decay * ema_param.data + (1 - self.ema_decay) * param.data

        # Return single batch losses with ALL components for monitoring
        # Includes monitoring losses (histogram, fft, fm) that don't participate in backprop
        # Calculate totals for logging
        loss_g_total = (
            self.lambda_adversarial * loss_adv_g +
            self.lambda_l1 * loss_l1 +
            self.lambda_l1 * 5.0 * loss_identity +
            self.lambda_perceptual * loss_perceptual
        )
        loss_d_total = step_losses.get("loss_d", 0.0)
        
        return {
            "loss_d": step_losses.get("loss_d", 0.0),
            "loss_d_total": loss_d_total,
            "loss_d_real": step_losses.get("loss_d_real", 0.0),
            "loss_d_fake": step_losses.get("loss_d_fake", 0.0),
            "loss_g": loss_g.item(),
            "loss_g_total": loss_g_total.item() if isinstance(loss_g_total, torch.Tensor) else loss_g_total,
            "loss_g_adv": loss_adv_g.item(),
            "loss_g_l1": loss_l1.item(),
            "loss_g_identity": loss_identity.item(),
            "loss_g_perceptual": loss_perceptual.item() if loss_perceptual > 0 else 0.0,
            "loss_g_feature_matching": loss_fm.item(),
            "loss_g_histogram": loss_histogram.item(),  # Monitoring only
            "loss_g_fft": loss_fft.item(),              # Monitoring only
            "loss_histogram": loss_histogram.item(),    # Legacy key for compatibility
            "loss_fft": loss_fft.item(),                # Legacy key for compatibility
            "loss_fm": loss_fm.item(),                  # Legacy key for compatibility
            "loss_r1": step_losses.get("loss_r1", 0.0),
            "loss_adv_g": loss_adv_g.item(),
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
            "loss_g_total": [],
            "loss_g_adv": [],
            "loss_g_l1": [],
            "loss_g_identity": [],
            "loss_g_perceptual": [],
            "loss_g_feature_matching": [],
            "loss_g_histogram": [],
            "loss_g_fft": [],
            "loss_d": [],
            "loss_d_total": [],
            "loss_d_real": [],
            "loss_d_fake": [],
            "loss_l1": [],
            "loss_identity": [],
            "loss_perceptual": [],
            "loss_adversarial": [],
            "loss_histogram": [],      # Monitoring loss (legacy key)
            "loss_fft": [],            # Monitoring loss (legacy key)
            "loss_fm": [],             # Monitoring loss (legacy key)
            "loss_r1": [],
            "loss_adv_g": [],
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
                    # FIX 2: SAFE APPEND - Check if key exists before appending (defensive)
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    epoch_losses[key].append(value)

                pbar.set_postfix({
                    "G": f"{losses['loss_g']:.4f}",
                    "D": f"{losses['loss_d']:.4f}",
                }, refresh=True)
                pbar.update(1)

        # Average losses
        # FIX: Handle empty lists (prevent RuntimeWarning: Mean of empty slice)
        avg_losses = {}
        for key, values in epoch_losses.items():
            if len(values) > 0:
                avg_losses[key] = np.mean(values)
            else:
                avg_losses[key] = 0.0

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

        avg_val_losses = {}
        for key, values in val_losses.items():
            if len(values) > 0:
                avg_val_losses[key] = np.mean(values)
            else:
                avg_val_losses[key] = 0.0

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
        
        # Save scaler state if available (for AMP resume)
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

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
            Next epoch number to start from
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        if "generator_ema_state_dict" in checkpoint:
            self.generator_ema.load_state_dict(checkpoint["generator_ema_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        self.training_history = checkpoint.get("training_history", {})
        
        # Load scaler state if available (for AMP resume)
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.info("✅ GradScaler state restored (AMP)")

        saved_epoch = checkpoint["epoch"]
        next_epoch = saved_epoch + 1
        logger.info(f"🔄 Checkpoint loaded: epoch {saved_epoch + 1} completed")
        logger.info(f"🔄 Resuming training from epoch {next_epoch + 1}...")

        return next_epoch


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
        default=8,  # FIX 5: LOWER BATCH SIZE - 16 → 8 for stability
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
        default=2e-5,  # TUNED: 5e-5 → 2e-5 (lower for stability)
        help="Learning rate for generator",
    )
    parser.add_argument(
        "--learning-rate-d",
        type=float,
        default=1e-5,  # FIX 2: CRITICAL - 5e-5 → 1e-5 (prevent D overpowering G)
        help="Learning rate for discriminator",
    )
    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=5.0,  # TUNED: 100.0 → 5.0 (reduce reconstruction aggression)
        help="Weight for L1 loss",
    )
    parser.add_argument(
        "--lambda-perceptual",
        type=float,
        default=0.5,  # TUNED: 10.0 → 0.5 (gentler perceptual loss)
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
    
    # 🚀 OPTIMIZATION: Enable cudnn auto-tuning for faster GPU training
    torch.backends.cudnn.benchmark = True

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
        learning_rate_g=args.learning_rate_g,  # TUNED: 2e-5 (lower for stability)
        learning_rate_d=args.learning_rate_d,  # TUNED: 5e-5 (stronger D, TTUR 1:2.5)
        lambda_l1=args.lambda_l1,              # TUNED: 5.0 (reduce aggression)
        lambda_perceptual=args.lambda_perceptual,  # TUNED: 0.5 (gentler)
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
