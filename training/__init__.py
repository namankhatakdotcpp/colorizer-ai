"""
Training module for colorizer-ai project.

This module contains:
- gan_training_refactored: Production-grade GAN training implementation
- production_training_script: Complete training script with all features
- FID evaluation and integration
- R1 penalty implementation
"""

from .gan_training_refactored import (
    RefactoredGANTrainer,
    FIDCheckpointManager,
    TrainingConfig,
    ExponentialMovingAverage,
)

__all__ = [
    "RefactoredGANTrainer",
    "FIDCheckpointManager",
    "TrainingConfig",
    "ExponentialMovingAverage",
]
