"""
Training utilities for production GAN training.

Includes:
- Mixed precision helpers
- Model weight management
- FID computation utilities
- Performance profiling
- Gradient diagnostics
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MixedPrecisionHelper:
    """
    Utilities for mixed precision training management.
    """

    @staticmethod
    def create_scalers(device: torch.device = None) -> Tuple[GradScaler, GradScaler]:
        """
        Create gradient scalers for AMP training.

        Returns:
            Tuple of (scaler_g, scaler_d) - gradient scalers for G and D
        """
        scaler_g = GradScaler()
        scaler_d = GradScaler()
        logger.info("✅ Created gradient scalers for mixed precision training")
        return scaler_g, scaler_d

    @staticmethod
    def scale_and_backward(loss: torch.Tensor, scaler: GradScaler, optimizer: torch.optim.Optimizer) -> None:
        """
        Backward pass with gradient scaling.

        Args:
            loss: Loss tensor
            scaler: Gradient scaler
            optimizer: Optimizer (needed for unscaling)
        """
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

    @staticmethod
    def optimizer_step_with_scaling(scaler: GradScaler, optimizer: torch.optim.Optimizer) -> None:
        """
        Optimizer step with gradient scaling.

        Args:
            scaler: Gradient scaler
            optimizer: Optimizer to step
        """
        scaler.step(optimizer)
        scaler.update()


class ModelWeightManager:
    """
    Utilities for managing model weights and checkpoints.
    """

    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        Count trainable and total parameters.

        Args:
            model: Model to count

        Returns:
            Dict with 'trainable' and 'total' counts
        """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        return {"trainable": trainable, "total": total}

    @staticmethod
    def get_model_size_mb(model: nn.Module) -> float:
        """
        Get model size in MB.

        Args:
            model: Model

        Returns:
            Model size in MB
        """
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        return (param_size + buffer_size) / 1024 / 1024

    @staticmethod
    def log_model_info(model: nn.Module, name: str = "Model") -> None:
        """
        Log model information.

        Args:
            model: Model
            name: Model name for logging
        """
        params = ModelWeightManager.count_parameters(model)
        size_mb = ModelWeightManager.get_model_size_mb(model)

        logger.info(f"{name} Parameters:")
        logger.info(f"  Trainable: {params['trainable']:,}")
        logger.info(f"  Total: {params['total']:,}")
        logger.info(f"  Size: {size_mb:.2f} MB")


class GradientDiagnostics:
    """
    Tools for diagnosing gradient flow during training.
    """

    @staticmethod
    def check_gradient_health(
        model: nn.Module,
        name: str = "Model",
        warn_threshold: float = 1.0,
    ) -> Dict[str, float]:
        """
        Check gradient statistics.

        Args:
            model: Model to check
            name: Model name for logging
            warn_threshold: Warning threshold for large gradients

        Returns:
            Dict with gradient statistics
        """
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.abs())

        if not grads:
            logger.warning(f"⚠️  {name}: No gradients found!")
            return {}

        grads = torch.cat([g.flatten() for g in grads])

        stats = {
            "max": grads.max().item(),
            "min": grads.min().item(),
            "mean": grads.mean().item(),
            "std": grads.std().item(),
        }

        if stats["max"] > warn_threshold:
            logger.warning(f"⚠️  {name}: Large gradients detected (max={stats['max']:.4f})")

        if stats["max"] == 0:
            logger.warning(f"⚠️  {name}: All gradients are zero!")

        return stats

    @staticmethod
    def check_nan_inf(model: nn.Module, name: str = "Model") -> bool:
        """
        Check for NaN or Inf in gradients.

        Args:
            model: Model to check
            name: Model name for logging

        Returns:
            True if any NaN/Inf found
        """
        has_nan_inf = False

        for param_name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    logger.error(f"🔴 {name} {param_name}: NaN in gradients!")
                    has_nan_inf = True
                if torch.isinf(param.grad).any():
                    logger.error(f"🔴 {name} {param_name}: Inf in gradients!")
                    has_nan_inf = True

        return has_nan_inf


class PerformanceProfiler:
    """
    Tools for profiling training performance.
    """

    def __init__(self):
        """Initialize profiler."""
        self.timings = {}

    def start(self, key: str) -> None:
        """Start timing a section."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timings[key] = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    def end(self, key: str) -> Optional[float]:
        """
        End timing a section.

        Returns:
            Elapsed time in milliseconds (or None if CPU)
        """
        if not torch.cuda.is_available():
            return None

        event = torch.cuda.Event(enable_timing=True)
        event.record()
        if self.timings[key] is not None:
            return self.timings[key].elapsed_time(event)

        return None

    @staticmethod
    def get_memory_usage(device: torch.device = None) -> Dict[str, float]:
        """
        Get current GPU memory usage.

        Returns:
            Dict with allocated and reserved memory in MB
        """
        if not torch.cuda.is_available():
            return {}

        if device is None:
            device = torch.device("cuda")

        allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device) / 1024 / 1024

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
        }

    @staticmethod
    def log_memory_usage(device: torch.device = None) -> None:
        """Log GPU memory usage."""
        memory = PerformanceProfiler.get_memory_usage(device)
        if memory:
            logger.info(f"GPU Memory:")
            logger.info(f"  Allocated: {memory['allocated_mb']:.2f} MB")
            logger.info(f"  Reserved: {memory['reserved_mb']:.2f} MB")
            logger.info(f"  Max: {memory['max_allocated_mb']:.2f} MB")


class DisableBatchNormMomentum:
    """
    Context manager to temporarily disable batch norm momentum updates.
    
    Useful for evaluation with frozen batch norm statistics.
    """

    def __init__(self, model: nn.Module):
        """
        Initialize context manager.

        Args:
            model: Model containing batch norm layers
        """
        self.model = model
        self.momentum_backup = {}

    def __enter__(self):
        """Disable batch norm momentum."""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self.momentum_backup[module] = module.momentum
                module.momentum = 0

    def __exit__(self, *args):
        """Restore batch norm momentum."""
        for module, momentum in self.momentum_backup.items():
            module.momentum = momentum


def validate_training_setup(
    generator: nn.Module,
    discriminator: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> bool:
    """
    Validate training setup before starting.

    Args:
        generator: Generator model
        discriminator: Discriminator model
        device: Training device
        use_amp: Whether using mixed precision

    Returns:
        True if setup is valid
    """
    logger.info("🔍 Validating training setup...")

    # Check device
    if device.type == "cuda":
        if not torch.cuda.is_available():
            logger.error("❌ CUDA requested but not available!")
            return False
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")

    # Check models
    if not generator.training:
        logger.warning("⚠️  Generator not in training mode")
    if not discriminator.training:
        logger.warning("⚠️  Discriminator not in training mode")

    # Check AMP support
    if use_amp:
        if device.type == "cuda":
            compute_capability = torch.cuda.get_device_capability(0)
            if compute_capability[0] < 7:
                logger.warning(f"⚠️  GPU compute capability {compute_capability[0]}.{compute_capability[1]} - AMP may be less efficient")
            else:
                logger.info(f"✅ GPU supports AMP efficiently (compute capability {compute_capability[0]}.{compute_capability[1]})")

    # Check model sizes
    ModelWeightManager.log_model_info(generator, "Generator")
    ModelWeightManager.log_model_info(discriminator, "Discriminator")

    logger.info("✅ Training setup validation complete")
    return True


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    # Test parameter counting
    from models.gan_generator import GANGenerator
    from models.gan_discriminator import MultiscaleDiscriminator

    gen = GANGenerator()
    dis = MultiscaleDiscriminator()

    ModelWeightManager.log_model_info(gen, "Generator")
    ModelWeightManager.log_model_info(dis, "Discriminator")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_training_setup(gen, dis, device)
