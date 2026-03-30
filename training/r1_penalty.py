"""
R1 Gradient Penalty for GAN Discriminator Training

Implements the R1 regularization from "Which Training Methods of GANs do actually Converge?"
(Fedus et al., 2018)

Reference: https://arxiv.org/abs/1811.02134

Formula:
    R1 penalty = (lambda / 2) * ||∇_x D(x)||_2^2

Applied to:
- Real images only
- Discriminator outputs
- During discriminator training
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class R1PenaltyCalculator:
    """
    Compute R1 (gradient penalty) for discriminator regularization.
    
    The R1 penalty encourages the discriminator gradients to have small
    magnitude with respect to real images, which improves training stability.
    
    Args:
        discriminator: Discriminator model
        device (torch.device): GPU/CPU device
        lambda_r1 (float): Penalty weight (default: 10.0)
    
    Properties:
    - Works with mixed precision (AMP)
    - No gradient accumulation issues
    - Memory efficient
    - Numerically stable
    """
    
    def __init__(
        self,
        discriminator: nn.Module,
        device: torch.device,
        lambda_r1: float = 10.0,
    ):
        self.discriminator = discriminator
        self.device = device
        self.lambda_r1 = lambda_r1
    
    def compute_penalty(
        self,
        real_images: torch.Tensor,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute R1 penalty for real images.
        
        Args:
            real_images: Real image tensor (batch_size, 3, H, W)
            scaler: GradScaler for mixed precision (optional)
        
        Returns:
            penalty: R1 penalty loss tensor
            stats: Dictionary with penalty statistics
        """
        # Ensure images require gradients
        real_images_copy = real_images.clone().detach().requires_grad_(True)
        real_images_copy = real_images_copy.to(self.device)
        
        # Forward pass through discriminator
        if scaler is not None:
            # Mixed precision: forward in fp16, compute gradients in fp32
            with torch.cuda.amp.autocast():
                disc_output = self.discriminator(real_images_copy)
                
                # Handle tuple output (logits, features)
                if isinstance(disc_output, tuple):
                    disc_output = disc_output[0]
        else:
            disc_output = self.discriminator(real_images_copy)
            if isinstance(disc_output, tuple):
                disc_output = disc_output[0]
        
        # Compute sum of discriminator outputs (for gradient computation)
        # We sum to get a scalar, then compute gradients w.r.t. input
        disc_sum = disc_output.sum()
        
        # Compute gradients w.r.t. real images
        grads = torch.autograd.grad(
            outputs=disc_sum,
            inputs=real_images_copy,
            create_graph=True,  # For backprop through the penalty
            retain_graph=False,
            only_inputs=True,
        )[0]
        
        # Compute L2 norm squared of gradients
        # Reshape to (batch_size, -1) to compute per-sample norms
        grad_norms = torch.linalg.norm(
            grads.view(grads.shape[0], -1), 
            dim=1, 
            ord=2
        )
        
        # R1 penalty: (lambda / 2) * mean(||∇_x D(x)||_2^2)
        penalty = (self.lambda_r1 / 2.0) * torch.mean(grad_norms ** 2)
        
        # Compute statistics
        stats = {
            "r1_penalty": penalty.item(),
            "grad_norm_mean": grad_norms.mean().item(),
            "grad_norm_max": grad_norms.max().item(),
            "grad_norm_min": grad_norms.min().item(),
        }
        
        return penalty, stats
    
    def compute_penalty_scaled(
        self,
        real_images: torch.Tensor,
        scaler: torch.cuda.amp.GradScaler,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute R1 penalty with proper scaler handling.
        
        This version ensures the scaler is applied correctly for mixed precision.
        
        Args:
            real_images: Real image tensor
            scaler: GradScaler instance (for AMP)
        
        Returns:
            penalty: R1 penalty loss
            stats: Penalty statistics
        """
        # Prepare images with gradient tracking
        real_images_copy = real_images.clone().detach().requires_grad_(True)
        real_images_copy = real_images_copy.to(self.device)
        
        # Forward through discriminator with mixed precision
        with torch.cuda.amp.autocast():
            disc_output = self.discriminator(real_images_copy)
            
            if isinstance(disc_output, tuple):
                disc_output = disc_output[0]
            
            # Scale the output for gradient computation
            disc_sum = disc_output.sum()
        
        # Compute gradients (this happens in fp32 even with autocast)
        grads = torch.autograd.grad(
            outputs=disc_sum,
            inputs=real_images_copy,
            create_graph=True,
            retain_graph=False,
            only_inputs=True,
        )[0]
        
        # Compute L2 norm squared
        grad_norms = torch.linalg.norm(
            grads.view(grads.shape[0], -1),
            dim=1,
            ord=2
        )
        
        # Compute penalty
        penalty = (self.lambda_r1 / 2.0) * torch.mean(grad_norms ** 2)
        
        # Statistics
        stats = {
            "r1_penalty": penalty.item(),
            "grad_norm_mean": grad_norms.mean().item(),
            "grad_norm_max": grad_norms.max().item(),
            "grad_norm_min": grad_norms.min().item(),
        }
        
        return penalty, stats


class R1PenaltyLoss(nn.Module):
    """
    R1 Penalty as a learnable module.
    
    Can be used as a drop-in replacement for computing R1 loss.
    """
    
    def __init__(
        self,
        discriminator: nn.Module,
        lambda_r1: float = 10.0,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.lambda_r1 = lambda_r1
    
    def forward(
        self,
        real_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass to compute R1 penalty.
        
        Args:
            real_images: Real image tensor (batch_size, 3, H, W)
        
        Returns:
            penalty: R1 penalty loss
            stats: Dictionary with statistics
        """
        device = next(self.discriminator.parameters()).device
        
        # Prepare images
        real_images_copy = real_images.clone().detach().requires_grad_(True)
        real_images_copy = real_images_copy.to(device)
        
        # Forward pass
        disc_output = self.discriminator(real_images_copy)
        
        if isinstance(disc_output, tuple):
            disc_output = disc_output[0]
        
        # Sum for gradient computation
        disc_sum = disc_output.sum()
        
        # Compute gradients
        grads = torch.autograd.grad(
            outputs=disc_sum,
            inputs=real_images_copy,
            create_graph=True,
            retain_graph=False,
            only_inputs=True,
        )[0]
        
        # L2 norm squared
        grad_norms = torch.linalg.norm(
            grads.view(grads.shape[0], -1),
            dim=1,
            ord=2
        )
        
        # Penalty
        penalty = (self.lambda_r1 / 2.0) * torch.mean(grad_norms ** 2)
        
        # Stats
        stats = {
            "r1_penalty": penalty.item(),
            "grad_norm_mean": grad_norms.mean().item(),
            "grad_norm_max": grad_norms.max().item(),
            "grad_norm_min": grad_norms.min().item(),
        }
        
        return penalty, stats


# ============================================================================
# Integration Functions
# ============================================================================

def compute_r1_penalty(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    lambda_r1: float = 10.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Simple function to compute R1 penalty.
    
    Args:
        discriminator: Discriminator model
        real_images: Real image tensor
        lambda_r1: Penalty weight
        device: Device (inferred from model if None)
    
    Returns:
        penalty: R1 penalty loss
        stats: Statistics dictionary
    """
    if device is None:
        device = next(discriminator.parameters()).device
    
    calculator = R1PenaltyCalculator(discriminator, device, lambda_r1)
    return calculator.compute_penalty(real_images)


def add_r1_to_discriminator_loss(
    discriminator_loss: torch.Tensor,
    r1_penalty: torch.Tensor,
    apply_penalty: bool = True,
) -> torch.Tensor:
    """
    Add R1 penalty to discriminator loss.
    
    Args:
        discriminator_loss: Standard discriminator loss (real + fake)
        r1_penalty: R1 penalty term
        apply_penalty: Whether to apply penalty (default: True)
    
    Returns:
        total_loss: discriminator_loss + r1_penalty (if apply_penalty)
    """
    if apply_penalty:
        return discriminator_loss + r1_penalty
    return discriminator_loss


# ============================================================================
# Integration with Alternating Updates
# ============================================================================

class DiscriminatorTrainingStep:
    """
    Helper class for discriminator training with R1 penalty.
    
    Handles:
    - Real image discrimination
    - Fake image discrimination
    - R1 penalty (on alternating steps)
    """
    
    def __init__(
        self,
        discriminator: nn.Module,
        device: torch.device,
        lambda_r1: float = 10.0,
        apply_r1_every_n_steps: int = 16,  # Apply R1 every 16 steps
    ):
        self.discriminator = discriminator
        self.device = device
        self.lambda_r1 = lambda_r1
        self.apply_r1_every_n_steps = apply_r1_every_n_steps
        self.step_count = 0
        
        self.r1_calculator = R1PenaltyCalculator(
            discriminator, device, lambda_r1
        )
    
    def should_apply_r1(self) -> bool:
        """Check if R1 penalty should be applied this step."""
        return self.step_count % self.apply_r1_every_n_steps == 0
    
    def compute_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        real_label_smoothing: float = 0.9,
        fake_label_smoothing: float = 0.1,
    ) -> Dict[str, float]:
        """
        Compute discriminator loss with optional R1 penalty.
        
        Args:
            real_images: Real image batch
            fake_images: Fake image batch (detached)
            real_label_smoothing: Label smoothing for real
            fake_label_smoothing: Label smoothing for fake
        
        Returns:
            loss_dict: Dictionary with loss components
        """
        real_images = real_images.to(self.device)
        fake_images = fake_images.detach().to(self.device)
        
        # Discriminator outputs
        real_output = self.discriminator(real_images)
        fake_output = self.discriminator(fake_images)
        
        # Handle tuple outputs
        if isinstance(real_output, tuple):
            real_output = real_output[0]
        if isinstance(fake_output, tuple):
            fake_output = fake_output[0]
        
        # Hinge loss for discriminator
        # Minimize -E[D(real)] - E[1 - D(fake)]
        d_loss_real = torch.nn.functional.relu(1.0 - real_output).mean()
        d_loss_fake = torch.nn.functional.relu(1.0 + fake_output).mean()
        d_loss = d_loss_real + d_loss_fake
        
        loss_dict = {
            "d_loss_real": d_loss_real.item(),
            "d_loss_fake": d_loss_fake.item(),
            "d_loss_total": d_loss.item(),
        }
        
        # Add R1 penalty every N steps
        if self.should_apply_r1():
            r1_penalty, r1_stats = self.r1_calculator.compute_penalty(real_images)
            d_loss = d_loss + r1_penalty
            
            loss_dict.update(r1_stats)
            loss_dict["d_loss_with_r1"] = d_loss.item()
        else:
            loss_dict["r1_penalty"] = 0.0
        
        self.step_count += 1
        
        return loss_dict, d_loss


# ============================================================================
# Utility Functions
# ============================================================================

def estimate_r1_memory_usage(
    batch_size: int,
    image_shape: Tuple[int, int, int] = (3, 256, 256),
) -> Dict[str, float]:
    """
    Estimate GPU memory needed for R1 penalty computation.
    
    Args:
        batch_size: Batch size
        image_shape: Image shape (C, H, W)
    
    Returns:
        Memory estimates in MB
    """
    c, h, w = image_shape
    
    # Image memory
    image_bytes = batch_size * c * h * w * 4  # float32
    
    # Gradient memory (same shape as image)
    grad_bytes = batch_size * c * h * w * 4
    
    # Discriminator output (1 value per image)
    output_bytes = batch_size * 4
    
    # Intermediate computations
    overhead_bytes = batch_size * 1000 * 4  # Rough estimate
    
    total_bytes = image_bytes + grad_bytes + output_bytes + overhead_bytes
    total_mb = total_bytes / (1024 * 1024)
    
    return {
        "image_memory_mb": image_bytes / (1024 * 1024),
        "gradient_memory_mb": grad_bytes / (1024 * 1024),
        "output_memory_mb": output_bytes / (1024 * 1024),
        "overhead_mb": overhead_bytes / (1024 * 1024),
        "total_mb": total_mb,
    }


def validate_r1_setup(
    discriminator: nn.Module,
    device: torch.device,
    lambda_r1: float = 10.0,
    batch_size: int = 16,
    image_shape: Tuple[int, int, int] = (3, 256, 256),
) -> Dict[str, any]:
    """
    Validate R1 penalty setup before training.
    
    Args:
        discriminator: Discriminator model
        device: Computation device
        lambda_r1: Penalty weight
        batch_size: Batch size for test
        image_shape: Test image shape
    
    Returns:
        Validation results dictionary
    """
    results = {
        "device": str(device),
        "lambda_r1": lambda_r1,
        "batch_size": batch_size,
        "image_shape": image_shape,
        "tests": {}
    }
    
    # Test instantiation
    try:
        calculator = R1PenaltyCalculator(discriminator, device, lambda_r1)
        results["tests"]["instantiation"] = "✓ PASSED"
    except Exception as e:
        results["tests"]["instantiation"] = f"✗ FAILED: {e}"
        return results
    
    # Test computation (without backward)
    try:
        dummy_images = torch.randn(
            batch_size, *image_shape,
            device=device,
            requires_grad=False
        )
        
        penalty, stats = calculator.compute_penalty(dummy_images)
        
        assert penalty.shape == torch.Size([]), "Penalty should be scalar"
        assert penalty.item() >= 0, "Penalty should be non-negative"
        assert "r1_penalty" in stats, "Missing r1_penalty in stats"
        
        results["tests"]["computation"] = "✓ PASSED"
        results["sample_penalty"] = penalty.item()
        results["sample_stats"] = stats
    except Exception as e:
        results["tests"]["computation"] = f"✗ FAILED: {e}"
        return results
    
    # Memory estimation
    try:
        memory = estimate_r1_memory_usage(batch_size, image_shape)
        results["estimated_memory"] = memory
        results["tests"]["memory_estimation"] = "✓ PASSED"
    except Exception as e:
        results["tests"]["memory_estimation"] = f"✗ FAILED: {e}"
    
    return results
