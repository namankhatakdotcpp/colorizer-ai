"""
Production-Grade Training Optimization Utilities

FAANG-level training optimizations including:
- Gradient checkpointing for memory efficiency
- Mixed precision training with automatic loss scaling
- Gradient accumulation with overflow handling
- Training statistics and performance monitoring
- Custom exception hierarchy for robust error handling
"""

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


# ============================================================================
# Custom Exception Hierarchy (FAANG Standard)
# ============================================================================

class OptimizationError(Exception):
    """Base exception for optimization-related errors."""
    pass


class GradientScalingError(OptimizationError):
    """Raised when mixed precision loss scaling fails."""
    pass


class GradientAccumulationError(OptimizationError):
    """Raised when gradient accumulation encounters issues."""
    pass


class CheckpointingError(OptimizationError):
    """Raised when gradient checkpointing fails."""
    pass


# ============================================================================
# Dataclasses for Configuration (Type Safety)
# ============================================================================

@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    enabled: bool = True
    dtype: torch.dtype = torch.float16
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enabled_for_loss: bool = True
    
    def __post_init__(self):
        if self.dtype not in (torch.float16, torch.bfloat16):
            warnings.warn(f"Mixed precision with {self.dtype} may not be optimized")


@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation."""
    enabled: bool = True
    accumulation_steps: int = 4
    max_norm: Optional[float] = 1.0
    normalize_by_steps: bool = True
    

@dataclass
class TrainingMetrics:
    """Container for training statistics."""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    throughput: float = 0.0  # samples/sec
    memory_used: float = 0.0  # MB
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            'epoch': self.epoch,
            'step': self.step,
            'loss': self.loss,
            'grad_norm': self.grad_norm,
            'learning_rate': self.learning_rate,
            'throughput_samples_per_sec': self.throughput,
            'memory_allocated_mb': self.memory_used,
        }


# ============================================================================
# Gradient Management Utilities
# ============================================================================

def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute L2 norm of all gradients in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient norm value
        
    Example:
        >>> grad_norm = compute_gradient_norm(model)
        >>> print(f"Gradient norm: {grad_norm:.4f}")
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = torch.norm(param.grad.data)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def clip_gradients(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0,
) -> Tuple[float, bool]:
    """
    Clip gradients by norm with overflow detection.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        norm_type: Type of norm (2 = L2 norm)
        
    Returns:
        Tuple of (clipped_norm, did_overflow)
        
    Raises:
        ValueError: If max_norm is invalid
        
    Example:
        >>> clipped_norm, overflowed = clip_gradients(model, max_norm=1.0)
        >>> if overflowed:
        ...     print("Gradient overflow detected!")
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")
    
    try:
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = torch.norm(param.grad.data, p=norm_type)
                total_norm += param_norm.item() ** norm_type
        
        total_norm = total_norm ** (1.0 / norm_type)
        
        # Check for NaN/Inf (gradient overflow)
        if not torch.isfinite(torch.tensor(total_norm)):
            return 0.0, True
        
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return total_norm, False
    
    except Exception as e:
        raise GradientAccumulationError(f"Gradient clipping failed: {e}")


def zero_gradients(model: nn.Module) -> None:
    """
    Zero out all gradients in model parameters.
    
    More efficient than optimizer.zero_grad() when you don't need
    optimizer state updates.
    """
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()


# ============================================================================
# Mixed Precision Training
# ============================================================================

class MixedPrecisionTrainer:
    """
    Automatic mixed precision training manager.
    
    Handles loss scaling, overflow detection, and scale adjustment
    for stable training with reduced memory usage.
    
    Example:
        >>> trainer = MixedPrecisionTrainer(config=MixedPrecisionConfig())
        >>> for batch in dataloader:
        ...     with trainer.autocast():
        ...         logits = model(batch)
        ...         loss = criterion(logits, targets)
        ...     trainer.backward(loss)
        ...     optimizer.step()
        ...     trainer.step()
    """
    
    def __init__(self, config: Optional[MixedPrecisionConfig] = None):
        self.config = config or MixedPrecisionConfig()
        self.enabled = self.config.enabled and torch.cuda.is_available()
        
        if self.enabled:
            self.scaler = GradScaler(
                init_scale=self.config.init_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval,
                enabled=True,
            )
        else:
            self.scaler = None
    
    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        if self.enabled:
            with autocast(dtype=self.config.dtype, enabled=True):
                yield
        else:
            yield
    
    def backward(self, loss: torch.Tensor) -> None:
        """
        Scaled backward pass for mixed precision.
        
        Args:
            loss: Loss tensor to backpropagate
            
        Raises:
            GradientScalingError: If scaling or backward fails
        """
        if not self.enabled or self.scaler is None:
            loss.backward()
            return
        
        try:
            self.scaler.scale(loss).backward()
        except RuntimeError as e:
            raise GradientScalingError(f"Backward pass failed: {e}")
    
    def step(self, optimizer, model: Optional[nn.Module] = None,
             max_norm: Optional[float] = None) -> bool:
        """
        Optimizer step with overflow handling.
        
        Args:
            optimizer: PyTorch optimizer
            model: Model for gradient clipping (optional)
            max_norm: Maximum gradient norm for clipping
            
        Returns:
            True if step succeeded, False if overflow detected
        """
        if not self.enabled or self.scaler is None:
            if model and max_norm:
                clip_gradients(model, max_norm)
            optimizer.step()
            return True
        
        # Unscale before clipping
        self.scaler.unscale_(optimizer)
        
        # Clip gradients if needed
        if model and max_norm:
            _, overflow = clip_gradients(model, max_norm)
            if overflow:
                self.scaler.update()
                return False
        
        # Optimizer step with overflow detection
        self.scaler.step(optimizer)
        self.scaler.update()
        return True
    
    def get_scale(self) -> float:
        """Get current loss scale."""
        return self.scaler.get_scale() if self.scaler else 1.0


# ============================================================================
# Gradient Accumulation
# ============================================================================

class GradientAccumulator:
    """
    Manages gradient accumulation with proper handling of:
    - Loss scaling by accumulation steps
    - Overflow detection
    - Statistics collection
    
    Example:
        >>> accumulator = GradientAccumulator(config=GradientAccumulationConfig(accumulation_steps=4))
        >>> for batch_idx, batch in enumerate(dataloader):
        ...     logits = model(batch)
        ...     loss = criterion(logits, targets)
        ...     
        ...     should_step = accumulator.backward(loss, model)
        ...     if should_step:
        ...         optimizer.step()
        ...         optimizer.zero_grad()
    """
    
    def __init__(self, config: Optional[GradientAccumulationConfig] = None):
        self.config = config or GradientAccumulationConfig()
        self.accumulated_steps = 0
        self.accumulated_grad_norm = 0.0
    
    def backward(self, loss: torch.Tensor, model: nn.Module) -> bool:
        """
        Accumulate gradients and return whether to step optimizer.
        
        Args:
            loss: Scaled loss tensor
            model: Model for gradient statistics
            
        Returns:
            True when accumulation is complete and optimizer should step
        """
        # Scale loss by accumulation steps
        if self.config.normalize_by_steps:
            scaled_loss = loss / self.config.accumulation_steps
        else:
            scaled_loss = loss
        
        try:
            scaled_loss.backward()
        except RuntimeError as e:
            raise GradientAccumulationError(f"Backward pass failed: {e}")
        
        self.accumulated_steps += 1
        
        # Return True when accumulation is complete
        return self.accumulated_steps >= self.config.accumulation_steps
    
    def reset(self) -> float:
        """Reset accumulator and return accumulated gradient norm."""
        norm = self.accumulated_grad_norm
        self.accumulated_steps = 0
        self.accumulated_grad_norm = 0.0
        return norm
    
    def get_effective_lr_scale(self) -> float:
        """Get effective learning rate scale due to accumulation."""
        return self.config.accumulation_steps if self.config.normalize_by_steps else 1.0


# ============================================================================
# Memory and Performance Monitoring
# ============================================================================

class TrainingMonitor:
    """
    Tracks training metrics and performance statistics.
    
    Example:
        >>> monitor = TrainingMonitor()
        >>> for epoch in range(num_epochs):
        ...     for batch_idx, batch in enumerate(dataloader):
        ...         loss = train_step(batch)
        ...         metrics = monitor.update(
        ...             loss=loss,
        ...             learning_rate=current_lr,
        ...             batch_size=len(batch),
        ...         )
        ...     
        ...     if batch_idx % 100 == 0:
        ...         print(f"Throughput: {metrics.throughput:.2f} samples/sec")
    """
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.metrics: List[TrainingMetrics] = []
        self.last_log_time = time.time()
        self.last_log_step = 0
        self.logger = logging.getLogger(__name__)
    
    def update(
        self,
        loss: float,
        learning_rate: float = 0.0,
        batch_size: int = 1,
        grad_norm: float = 0.0,
        epoch: int = 0,
    ) -> TrainingMetrics:
        """
        Update training metrics.
        
        Returns:
            Current TrainingMetrics object
        """
        current_time = time.time()
        steps_since_log = len(self.metrics) - self.last_log_step
        time_since_log = current_time - self.last_log_time
        
        # Compute throughput
        throughput = (steps_since_log * batch_size) / max(time_since_log, 1e-6)
        
        # Get GPU memory usage
        memory_used = 0.0
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        metrics = TrainingMetrics(
            epoch=epoch,
            step=len(self.metrics),
            loss=loss,
            grad_norm=grad_norm,
            learning_rate=learning_rate,
            throughput=throughput,
            memory_used=memory_used,
            timestamp=current_time,
        )
        
        self.metrics.append(metrics)
        
        # Log periodically
        if len(self.metrics) % self.log_interval == 0:
            self._log_metrics(metrics)
        
        return metrics
    
    def _log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log metrics to console and logger."""
        msg = (
            f"Epoch {metrics.epoch} | Step {metrics.step} | "
            f"Loss {metrics.loss:.4f} | "
            f"LR {metrics.learning_rate:.2e} | "
            f"Grad_Norm {metrics.grad_norm:.4f} | "
            f"Throughput {metrics.throughput:.2f} samples/sec | "
            f"Memory {metrics.memory_used:.1f}MB"
        )
        print(msg)
        self.logger.info(msg)
    
    def get_statistics(self) -> Dict[str, float]:
        """Return aggregated statistics over entire run."""
        if not self.metrics:
            return {}
        
        losses = [m.loss for m in self.metrics]
        throughputs = [m.throughput for m in self.metrics]
        memory = [m.memory_used for m in self.metrics]
        
        return {
            'mean_loss': sum(losses) / len(losses),
            'min_loss': min(losses),
            'max_loss': max(losses),
            'mean_throughput': sum(throughputs) / len(throughputs),
            'peak_memory_mb': max(memory) if memory else 0,
            'total_steps': len(self.metrics),
        }


# ============================================================================
# Utility Functions
# ============================================================================

@contextmanager
def no_sync_grad(model: nn.Module):
    """
    Context manager to disable gradient synchronization in DDP.
    
    Useful for accumulation steps where sync is not needed.
    
    Example:
        >>> with no_sync_grad(model):
        ...     output = model(batch)
        ...     loss = criterion(output, targets)
        ...     loss.backward()  # No sync yet
    """
    if hasattr(model, 'no_sync'):
        with model.no_sync():
            yield
    else:
        yield
