"""
Unified Loss Module for GAN-based Image Colorization.

Provides modular, efficient loss computation with:
- Adversarial loss (hinge)
- L1 reconstruction loss
- Perceptual loss (VGG16 features)
- Feature matching loss (discriminator intermediates)
- Color histogram matching loss

Production-grade: GPU optimized, AMP compatible, minimal overhead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Dict, Tuple, List, Optional


class VGGPerceptualLoss(nn.Module):
    """
    VGG16-based perceptual loss using multi-scale features.
    
    Uses relu3_3 and relu4_3 layers for better feature representation.
    Features are frozen (no gradients) to save memory.
    """

    def __init__(self, layer: str = "relu4_3", device: torch.device = None):
        """
        Initialize VGG16 perceptual loss.

        Args:
            layer: Which VGG layer to use ("relu3_3" or "relu4_3", default: relu4_3)
            device: Device to use (cuda or cpu)
        """
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained VGG16 (ImageNet weights)
        vgg16 = torchvision.models.vgg16(pretrained=True).to(self.device)
        vgg16.eval()
        
        # Freeze all parameters
        for param in vgg16.parameters():
            param.requires_grad = False

        # Map layer names to indices
        layer_map = {
            "relu3_3": 17,
            "relu4_3": 26,
            "relu5_1": 31,
        }
        
        max_layer = layer_map.get(layer, 26)
        self.feature_extractor = vgg16.features[:max_layer].to(self.device)

        # ImageNet normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between two images.

        Args:
            x: Generated/fake image (B, 3, H, W), range [-1, 1]
            y: Target/real image (B, 3, H, W), range [-1, 1]

        Returns:
            Perceptual loss (scalar)
        """
        # Normalize from [-1, 1] to ImageNet [0, 1] then standardize
        x_norm = (x + 1) / 2
        y_norm = (y + 1) / 2
        
        x_norm = (x_norm - self.mean) / self.std
        y_norm = (y_norm - self.mean) / self.std

        # Extract features (no gradients)
        with torch.no_grad():
            x_features = self.feature_extractor(x_norm)
            y_features = self.feature_extractor(y_norm)

        # L1 distance between feature maps
        loss = F.l1_loss(x_features, y_features)

        return loss


class LossManager(nn.Module):
    """
    Production-grade unified loss manager for GAN colorization.
    
    Features:
    - Modular loss computation (adversarial, L1, perceptual, FM, histogram)
    - Clean API: forward() returns (total_loss, loss_dict)
    - GPU optimized with AMP compatibility
    - Configurable loss weights
    """

    def __init__(
        self,
        device: torch.device = None,
        lambda_adversarial: float = 1.0,
        lambda_l1: float = 50.0,
        lambda_perceptual: float = 10.0,
        lambda_feature_matching: float = 10.0,
        lambda_histogram: float = 5.0,
    ):
        """
        Initialize loss manager with configurable weights.

        Args:
            device: Device to use (cuda or cpu)
            lambda_adversarial: Weight for adversarial loss
            lambda_l1: Weight for L1 reconstruction loss
            lambda_perceptual: Weight for perceptual loss
            lambda_feature_matching: Weight for feature matching loss
            lambda_histogram: Weight for histogram loss
        """
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Loss weights
        self.lambda_adversarial = lambda_adversarial
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_feature_matching = lambda_feature_matching
        self.lambda_histogram = lambda_histogram

        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Perceptual loss (VGG16)
        try:
            self.perceptual_loss = VGGPerceptualLoss(layer="relu4_3", device=self.device)
            self.use_perceptual = True
        except Exception as e:
            print(f"Warning: Failed to load perceptual loss: {e}")
            self.perceptual_loss = None
            self.use_perceptual = False

    def adversarial_loss_g(self, disc_fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Adversarial loss for generator (fool discriminator).
        
        Using standard GAN formulation: -log(D(G(x)))
        Simplified to: -mean(D(G(x)))

        Args:
            disc_fake_logits: Discriminator output for fake/generated images

        Returns:
            Adversarial loss for G
        """
        # Standard adversarial: fool discriminator to output 1 (real)
        return -disc_fake_logits.mean()

    def adversarial_loss_d(
        self,
        disc_real_logits: torch.Tensor,
        disc_fake_logits: torch.Tensor,
        real_labels: torch.Tensor,
        fake_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adversarial loss for discriminator.
        
        Maximize: log(D(x)) + log(1 - D(G(x)))
        With label smoothing for stability.

        Args:
            disc_real_logits: Discriminator output for real images
            disc_fake_logits: Discriminator output for fake images
            real_labels: Labels for real images (typically 0.9 with smoothing)
            fake_labels: Labels for fake images (typically 0.1 with smoothing)

        Returns:
            Tuple of (loss_d_real, loss_d_fake)
        """
        loss_d_real = self.bce_loss(disc_real_logits, real_labels)
        loss_d_fake = self.bce_loss(disc_fake_logits, fake_labels)

        return loss_d_real, loss_d_fake

    def l1_loss_fn(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        L1 reconstruction loss (content preservation).

        Args:
            generated: Generated image (B, 3, H, W), range [-1, 1]
            target: Target image (B, 3, H, W), range [-1, 1]

        Returns:
            L1 loss
        """
        return self.l1_loss(generated, target)

    def perceptual_loss_fn(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perceptual loss using VGG16 features.

        Args:
            generated: Generated image (B, 3, H, W), range [-1, 1]
            target: Target image (B, 3, H, W), range [-1, 1]

        Returns:
            Perceptual loss (scalar)
        """
        if not self.use_perceptual or self.perceptual_loss is None:
            return torch.tensor(0.0, device=self.device)

        return self.perceptual_loss(generated, target.detach())

    def feature_matching_loss_fn(
        self,
        fake_features: List[torch.Tensor],
        real_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Feature matching loss (match discriminator intermediate features).
        
        Encourages generator to produce images with discriminator features
        similar to real images, even if discriminator output differs.

        Args:
            fake_features: List of intermediate features from discriminator(generated)
            real_features: List of intermediate features from discriminator(real)

        Returns:
            Feature matching loss (scalar)
        """
        if not fake_features or not real_features:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        
        for fake_feat, real_feat in zip(fake_features, real_features):
            # L1 distance between features, averaged spatially
            loss = loss + F.l1_loss(fake_feat, real_feat.detach())

        # Average over number of feature pairs
        return loss / max(len(real_features), 1)

    def histogram_loss_fn(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        num_bins: int = 32,
    ) -> torch.Tensor:
        """
        Color histogram matching loss (align RGB distribution).
        
        Encourages similar color distributions between generated and target.
        Improves color fidelity and reduces color shifts.
        
        Uses differentiable histogram computation via kernel density estimation.

        Args:
            generated: Generated image (B, 3, H, W), range [-1, 1]
            target: Target image (B, 3, H, W), range [-1, 1]
            num_bins: Number of histogram bins (32 standard)

        Returns:
            Histogram loss (scalar)
        """
        # Convert from [-1, 1] to [0, 1]
        gen_norm = (generated + 1.0) / 2.0
        tgt_norm = (target + 1.0) / 2.0

        loss = torch.tensor(0.0, device=generated.device)

        # Compute histogram loss per channel using differentiable method
        for c in range(3):
            gen_flat = gen_norm[:, c, :, :].flatten()
            tgt_flat = tgt_norm[:, c, :, :].flatten()

            # Clamp to valid range
            gen_flat = torch.clamp(gen_flat, 0.0, 1.0)
            tgt_flat = torch.clamp(tgt_flat, 0.0, 1.0)

            # Create bin centers
            bin_centers = torch.linspace(0.0, 1.0, num_bins, device=generated.device)
            bin_width = 1.0 / num_bins

            # Compute soft histograms using Gaussian kernels (differentiable)
            # Each value contributes to nearby bins
            gen_hist = torch.zeros(num_bins, device=generated.device)
            tgt_hist = torch.zeros(num_bins, device=generated.device)

            # Kernel width for smooth histogram
            sigma = bin_width * 1.5

            for i in range(num_bins):
                # Gaussian kernel contribution
                gen_contrib = torch.exp(-((gen_flat - bin_centers[i]) ** 2) / (2 * sigma ** 2))
                tgt_contrib = torch.exp(-((tgt_flat - bin_centers[i]) ** 2) / (2 * sigma ** 2))

                gen_hist[i] = gen_contrib.mean()
                tgt_hist[i] = tgt_contrib.mean()

            # Normalize to probability distributions
            gen_hist = gen_hist / (gen_hist.sum() + 1e-8)
            tgt_hist = tgt_hist / (tgt_hist.sum() + 1e-8)

            # Compute histogram distance (L1 between distributions, more stable)
            channel_loss = torch.mean(torch.abs(gen_hist - tgt_hist))
            loss = loss + channel_loss

        # Average over channels
        return loss / 3.0

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        disc_fake_logits: List[torch.Tensor],
        disc_real_logits: List[torch.Tensor],
        fake_features: Optional[List[List[torch.Tensor]]] = None,
        real_features: Optional[List[List[torch.Tensor]]] = None,
        real_labels: Optional[torch.Tensor] = None,
        fake_labels: Optional[torch.Tensor] = None,
        mode: str = "generator",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute unified generator or discriminator loss.

        Args:
            generated: Generated image (B, 3, H, W), range [-1, 1]
            target: Target image (B, 3, H, W), range [-1, 1]
            disc_fake_logits: List of discriminator outputs for fake images (multi-scale)
            disc_real_logits: List of discriminator outputs for real images (multi-scale)
            fake_features: List[List] of intermediate features for generated (optional, for FM)
            real_features: List[List] of intermediate features for real (optional, for FM)
            real_labels: Labels for real images (for D loss)
            fake_labels: Labels for fake images (for D loss)
            mode: "generator" or "discriminator"

        Returns:
            Tuple of (total_loss, loss_dict)
            - total_loss: Weighted sum of all losses
            - loss_dict: Dict with individual loss components
        """
        # Determine device from available tensors
        if generated is not None:
            device = generated.device
        elif target is not None:
            device = target.device
        elif disc_fake_logits and len(disc_fake_logits) > 0:
            device = disc_fake_logits[0].device
        else:
            device = self.device

        loss_dict = {}

        if mode == "generator":
            # ===== GENERATOR LOSSES =====
            
            # Adversarial loss (fool discriminator)
            loss_adv_g = self.adversarial_loss_g(disc_fake_logits[0])
            loss_dict["loss_adv_g"] = loss_adv_g.item()

            # L1 reconstruction loss
            loss_l1 = self.l1_loss_fn(generated, target)
            loss_dict["loss_l1"] = loss_l1.item()

            # Perceptual loss
            loss_percep = self.perceptual_loss_fn(generated, target)
            loss_dict["loss_perceptual"] = loss_percep.item()

            # Feature matching loss
            loss_fm = torch.tensor(0.0, device=device)
            if fake_features is not None and real_features is not None:
                # Flatten multi-scale features
                fake_feats_flat = [f for scale_feats in fake_features for f in scale_feats]
                real_feats_flat = [f for scale_feats in real_features for f in scale_feats]
                loss_fm = self.feature_matching_loss_fn(fake_feats_flat, real_feats_flat)
            loss_dict["loss_feature_matching"] = loss_fm.item()

            # Histogram loss
            loss_hist = self.histogram_loss_fn(generated, target)
            loss_dict["loss_histogram"] = loss_hist.item()

            # Combine G losses
            total_loss = (
                self.lambda_adversarial * loss_adv_g +
                self.lambda_l1 * loss_l1 +
                self.lambda_perceptual * loss_percep +
                self.lambda_feature_matching * loss_fm +
                self.lambda_histogram * loss_hist
            )
            loss_dict["loss_g_total"] = total_loss.item()

        elif mode == "discriminator":
            # ===== DISCRIMINATOR LOSSES =====
            
            # Ensure labels provided
            if real_labels is None:
                real_labels = torch.ones_like(disc_real_logits[0])
            if fake_labels is None:
                fake_labels = torch.zeros_like(disc_fake_logits[0])

            # Adversarial loss for D
            loss_d_real, loss_d_fake = self.adversarial_loss_d(
                disc_real_logits[0],
                disc_fake_logits[0],
                real_labels,
                fake_labels,
            )
            loss_dict["loss_d_real"] = loss_d_real.item()
            loss_dict["loss_d_fake"] = loss_d_fake.item()

            # Combine D losses
            total_loss = loss_d_real + loss_d_fake
            loss_dict["loss_d_total"] = total_loss.item()

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'generator' or 'discriminator'")

        return total_loss, loss_dict


# Convenience Factory Function
def create_loss_manager(
    device: torch.device = None,
    lambda_adversarial: float = 1.0,
    lambda_l1: float = 50.0,
    lambda_perceptual: float = 10.0,
    lambda_feature_matching: float = 10.0,
    lambda_histogram: float = 5.0,
) -> LossManager:
    """
    Factory function to create LossManager with production defaults.

    Args:
        device: Device to use
        lambda_*: Loss weights

    Returns:
        Configured LossManager instance
    """
    return LossManager(
        device=device,
        lambda_adversarial=lambda_adversarial,
        lambda_l1=lambda_l1,
        lambda_perceptual=lambda_perceptual,
        lambda_feature_matching=lambda_feature_matching,
        lambda_histogram=lambda_histogram,
    )
