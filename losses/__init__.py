"""
Unified loss module for GAN-based image colorization.

Exports:
- LossManager: Main loss computation class
- VGGPerceptualLoss: VGG16-based perceptual loss
- create_loss_manager: Factory function for creating LossManager
"""

from .loss_manager import (
    LossManager,
    VGGPerceptualLoss,
    create_loss_manager,
)

__all__ = [
    "LossManager",
    "VGGPerceptualLoss",
    "create_loss_manager",
]
