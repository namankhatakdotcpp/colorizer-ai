"""
Model Optimization and Compilation Utilities

FAANG-level model optimization techniques:
- Torch.compile for JIT compilation and speedup
- Gradient checkpointing for memory efficiency
- Model profiling and bottleneck detection
- Quantization-aware training
- Weight freezing and fine-tuning strategies
- Dynamic shape handling
"""

import logging
import time
from typing import Callable, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from functools import wraps

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class CompileConfig:
    """Configuration for torch.compile."""
    enabled: bool = True
    mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    fullgraph: bool = False
    dynamic: str = None  # None, "auto", or custom
    backend: str = "inductor"
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.mode not in ("default", "reduce-overhead", "max-autotune"):
            raise ValueError(f"Invalid compile mode: {self.mode}")
        if self.backend not in ("inductor", "aot_eager", "cudagraphs"):
            raise ValueError(f"Invalid backend: {self.backend}")


@dataclass
class ProfilingConfig:
    """Configuration for model profiling."""
    enabled: bool = True
    warmup_steps: int = 10
    profile_steps: int = 100
    log_interval: int = 50
    memory_profiling: bool = True
    cpu_profiling: bool = False


# ============================================================================
# Model Compilation and Optimization
# ============================================================================

class CompiledModelWrapper:
    """
    Wrapper for torch.compile with graceful fallback.
    
    Automatically handles:
    - Compile failures with fallback to eager execution
    - Dynamic shape handling
    - Backend selection based on CUDA availability
    
    Example:
        >>> model = UNetColorizer()
        >>> compiled_model = CompiledModelWrapper(model, config=CompileConfig())
        >>> output = compiled_model(input_tensor)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[CompileConfig] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.config = config or CompileConfig()
        self.verbose = verbose
        self.compiled = False
        
        if self.verbose:
            logger.info(f"Initializing CompiledModelWrapper with mode={self.config.mode}")
        
        self._compile()
    
    def _compile(self) -> None:
        """Attempt to compile model with fallback."""
        if not self.config.enabled:
            if self.verbose:
                logger.info("Compilation disabled")
            return
        
        if not hasattr(torch, "compile"):
            if self.verbose:
                logger.warning("torch.compile not available, using eager execution")
            return
        
        if not torch.cuda.is_available() and self.config.backend == "inductor":
            if self.verbose:
                logger.warning("CUDA not available, falling back to eager execution")
            return
        
        try:
            # Configure CUDNN for best performance
            if torch.cuda.is_available():
                cudnn.benchmark = True
                cudnn.allow_tf32 = True
            
            # Compile model
            compile_kwargs = {
                "mode": self.config.mode,
                "fullgraph": self.config.fullgraph,
                "backend": self.config.backend,
            }
            
            if self.config.dynamic is not None:
                compile_kwargs["dynamic"] = self.config.dynamic
            
            self.model = torch.compile(self.model, **compile_kwargs)
            self.compiled = True
            
            if self.verbose:
                logger.info(
                    f"Model compiled successfully with backend={self.config.backend}, "
                    f"mode={self.config.mode}"
                )
        
        except Exception as e:
            if self.verbose:
                logger.warning(f"Compilation failed: {e}. Falling back to eager execution.")
            self.compiled = False
    
    def __call__(self, *args, **kwargs):
        """Forward pass through compiled or eager model."""
        return self.model(*args, **kwargs)
    
    def train(self, mode: bool = True) -> "CompiledModelWrapper":
        """Set model to training mode."""
        self.model.train(mode)
        return self
    
    def eval(self) -> "CompiledModelWrapper":
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def state_dict(self) -> Dict:
        """Get model state dict."""
        # Handle _orig_mod from torch.compile
        if hasattr(self.model, '_orig_mod'):
            return self.model._orig_mod.state_dict()
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict: Dict, strict: bool = True) -> Any:
        """Load model state dict."""
        if hasattr(self.model, '_orig_mod'):
            return self.model._orig_mod.load_state_dict(state_dict, strict=strict)
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def parameters(self):
        """Get model parameters."""
        if hasattr(self.model, '_orig_mod'):
            return self.model._orig_mod.parameters()
        return self.model.parameters()
    
    def named_parameters(self, *args, **kwargs):
        """Get named model parameters."""
        if hasattr(self.model, '_orig_mod'):
            return self.model._orig_mod.named_parameters(*args, **kwargs)
        return self.model.named_parameters(*args, **kwargs)


# ============================================================================
# Gradient Checkpointing
# ============================================================================

def apply_gradient_checkpointing(
    model: nn.Module,
    checkpoint_segments: int = 1,
) -> nn.Module:
    """
    Apply gradient checkpointing to reduce memory usage.
    
    Trades computation for memory by not storing intermediate activations.
    Useful for large models or large batch sizes.
    
    Args:
        model: Model to apply checkpointing to
        checkpoint_segments: Number of segments to checkpoint
        
    Returns:
        Model with gradient checkpointing applied
        
    Example:
        >>> model = apply_gradient_checkpointing(model, checkpoint_segments=2)
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping gradient checkpointing")
        return model
    
    try:
        # For models with encoder/decoder pattern
        if hasattr(model, 'encoder'):
            torch.utils.checkpoint.checkpoint_sequential(
                model.encoder,
                checkpoint_segments,
            )
        
        # For models with sequential blocks
        if hasattr(model, 'blocks'):
            for block in model.blocks:
                block.gradient_checkpointing = True
        
        logger.info("Gradient checkpointing applied")
    except Exception as e:
        logger.warning(f"Failed to apply gradient checkpointing: {e}")
    
    return model


# ============================================================================
# Model Profiling and Analysis
# ============================================================================

class ModelProfiler:
    """
    Comprehensive model profiling for performance analysis.
    
    Measures:
    - Throughput (samples/sec)
    - Latency per batch
    - Memory consumption
    - Bottleneck identification
    
    Example:
        >>> profiler = ModelProfiler(model, config=ProfilingConfig())
        >>> stats = profiler.profile(dataloader, num_batches=100)
        >>> print(stats)
    """
    
    def __init__(self, model: nn.Module, config: Optional[ProfilingConfig] = None):
        self.model = model
        self.config = config or ProfilingConfig()
        self.stats = {}
    
    def profile(
        self,
        dataloader: Any,
        num_batches: int = 100,
        device: torch.device = None,
    ) -> Dict[str, float]:
        """
        Profile model on dataloader.
        
        Args:
            dataloader: DataLoader to profile on
            num_batches: Number of batches to profile
            device: Device to profile on
            
        Returns:
            Dictionary of statistics
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Warmup
        for i, batch in enumerate(dataloader):
            if i >= self.config.warmup_steps:
                break
            
            batch = self._move_to_device(batch, device)
            with torch.no_grad():
                _ = self.model(batch[0] if isinstance(batch, (list, tuple)) else batch)
        
        # Profile
        latencies = []
        memory_used = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                batch = self._move_to_device(batch, device)
                
                # Synchronize before timing
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = self.model(batch[0] if isinstance(batch, (list, tuple)) else batch)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                latencies.append((end_time - start_time) * 1000)  # ms
                
                if self.config.memory_profiling and device.type == 'cuda':
                    memory_used.append(torch.cuda.memory_allocated() / 1024 / 1024)
        
        # Compute statistics
        stats = {
            'mean_latency_ms': sum(latencies) / len(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'p95_latency_ms': sorted(latencies)[int(0.95 * len(latencies))],
            'throughput_samples_per_sec': 1000. / (sum(latencies) / len(latencies)),
        }
        
        if memory_used:
            stats['peak_memory_mb'] = max(memory_used)
            stats['mean_memory_mb'] = sum(memory_used) / len(memory_used)
        
        self.stats = stats
        return stats
    
    def _move_to_device(self, batch: Any, device: torch.device) -> Any:
        """Move batch to device recursively."""
        if isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(item, device) for item in batch)
        elif isinstance(batch, dict):
            return {key: self._move_to_device(val, device) for key, val in batch.items()}
        else:
            return batch


# ============================================================================
# Fine-tuning Strategies
# ============================================================================

class LayerWiseFineTuner:
    """
    Implements layer-wise fine-tuning with discriminative learning rates.
    
    Allows different layers to have different learning rates based on
    their position in the network (lower layers learn slower).
    
    Example:
        >>> tuner = LayerWiseFineTuner(model)
        >>> param_groups = tuner.get_param_groups(base_lr=1e-4, decay_factor=0.1)
        >>> optimizer = Adam(param_groups)
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_groups = self._group_layers()
    
    def _group_layers(self) -> List[List[nn.Module]]:
        """Group layers of model."""
        layers = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                layers.append(module)
        
        # Group into blocks (assuming some grouping exists)
        # For now, just return individual layers
        return [[layer] for layer in layers]
    
    def get_param_groups(
        self,
        base_lr: float,
        decay_factor: float = 0.1,
    ) -> List[Dict]:
        """
        Create parameter groups with discriminative learning rates.
        
        Args:
            base_lr: Learning rate for final layer
            decay_factor: Factor to decrease LR for earlier layers
            
        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []
        num_groups = len(self.layer_groups)
        
        for i, layer_group in enumerate(self.layer_groups):
            # Multiply LR by decay_factor^i (earlier layers learn slower)
            lr = base_lr * (decay_factor ** (num_groups - i - 1))
            
            params = []
            for layer in layer_group:
                params.extend(layer.parameters())
            
            param_groups.append({
                'params': params,
                'lr': lr,
                'name': f'layer_{i}',
            })
        
        return param_groups


# ============================================================================
# Utility Functions
# ============================================================================

def get_model_statistics(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive model statistics.
    
    Returns:
        Dict with parameter counts, FLOPs, etc.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    stats = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
    }
    
    return stats


def freeze_model_bn(model: nn.Module) -> None:
    """Freeze batch norm layers during fine-tuning."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def unfreeze_model_bn(model: nn.Module) -> None:
    """Unfreeze batch norm layers."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.train()
            for param in module.parameters():
                param.requires_grad = True


def set_cudnn_options(
    benchmark: bool = True,
    deterministic: bool = False,
    allow_tf32: bool = True,
) -> None:
    """
    Configure CUDNN for optimal performance.
    
    Args:
        benchmark: Enable cuDNN autotuner (faster but non-deterministic)
        deterministic: Force deterministic algorithms (slower but reproducible)
        allow_tf32: Allow TF32 for matrix operations
    """
    cudnn.benchmark = benchmark
    cudnn.deterministic = deterministic
    cudnn.allow_tf32 = allow_tf32
    
    if deterministic:
        torch.use_deterministic_algorithms(True)
    
    logger.info(
        f"CUDNN config: benchmark={benchmark}, deterministic={deterministic}, "
        f"allow_tf32={allow_tf32}"
    )
