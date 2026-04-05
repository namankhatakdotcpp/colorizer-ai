"""
Advanced Learning Rate Schedulers for Production Training

FAANG-level implementations of modern LR scheduling strategies:
- Warm-up Learning Rate Scheduler
- Cosine Annealing with Warm Restarts
- Polynomial Decay Scheduler
- Exponential Decay with Plateau Detection
- 1Cycle Scheduler

These schedulers significantly improve training stability and convergence speed.
"""

import math
from typing import List, Optional
from abc import ABC, abstractmethod


class BaseScheduler(ABC):
    """Base class for all learning rate schedulers."""

    def __init__(self, optimizer, last_epoch: int = -1, verbose: bool = False):
        """
        Args:
            optimizer: PyTorch optimizer
            last_epoch: Last epoch number (for resuming training)
            verbose: Print debug information
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(f"param 'initial_lr' is not specified in param_groups[{i}]")
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.step()

    @abstractmethod
    def get_lr(self) -> List[float]:
        """Returns a list of learning rates for each parameter group."""
        pass

    def step(self, epoch: Optional[int] = None) -> None:
        """Update learning rate."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
        if self.verbose:
            print(f"Epoch {epoch}: LR = {lrs}")


class WarmupScheduler(BaseScheduler):
    """
    Linear warmup followed by constant or scheduled learning rate.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for linear warmup
        base_scheduler: Optional scheduler to apply after warmup
        total_epochs: Total training epochs (for normalization)
    
    Example:
        >>> scheduler = WarmupScheduler(optimizer, warmup_epochs=5, total_epochs=100)
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 5,
        base_scheduler: Optional[BaseScheduler] = None,
        total_epochs: int = 100,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            progress = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * progress for base_lr in self.base_lrs]
        else:
            # Use base scheduler if provided
            if self.base_scheduler is not None:
                self.base_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
                return self.base_scheduler.get_lr()
            # Otherwise maintain constant LR
            return self.base_lrs


class CosineAnnealingWarmRestarts(BaseScheduler):
    """
    Cosine annealing with warm restarts (SGDR).
    Learning rate follows cosine curve with periodic restarts.
    
    Paper: https://arxiv.org/abs/1608.03983
    
    Args:
        optimizer: PyTorch optimizer
        T_0: Period of restart in epochs
        T_mult: Multiplier for restart period (T_i = T_0 * T_mult^i)
        eta_min: Minimum learning rate
        last_epoch: Last epoch number
    
    Example:
        >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    """

    def __init__(
        self,
        optimizer,
        T_0: int = 10,
        T_mult: float = 1.0,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Cosine annealing with warm restarts."""
        if self.last_epoch == 0:
            self.T_cur = 0
        else:
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur = 0
                self.T_i = int(self.T_i * self.T_mult)

        lrs = []
        for base_lr in self.base_lrs:
            # Cosine annealing formula
            lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            lrs.append(lr)
        return lrs


class PolynomialDecayScheduler(BaseScheduler):
    """
    Polynomial decay learning rate scheduler.
    
    LR = base_lr * (1 - epoch / total_epochs)^power
    
    Args:
        optimizer: PyTorch optimizer
        total_epochs: Total training epochs
        power: Decay power (2 = quadratic, 1 = linear)
        last_epoch: Last epoch number
    
    Example:
        >>> scheduler = PolynomialDecayScheduler(optimizer, total_epochs=100, power=2)
    """

    def __init__(
        self,
        optimizer,
        total_epochs: int,
        power: float = 2.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.total_epochs = total_epochs
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        progress = min(1.0, self.last_epoch / max(1, self.total_epochs))
        return [base_lr * ((1 - progress) ** self.power) for base_lr in self.base_lrs]


class ExponentialDecayScheduler(BaseScheduler):
    """
    Exponential decay with plateau detection.
    
    LR = base_lr * (gamma ** epoch)
    
    Args:
        optimizer: PyTorch optimizer
        gamma: Multiplicative factor of decay per epoch
        min_lr: Minimum learning rate (plateau detection)
        last_epoch: Last epoch number
    
    Example:
        >>> scheduler = ExponentialDecayScheduler(optimizer, gamma=0.95)
    """

    def __init__(
        self,
        optimizer,
        gamma: float = 0.95,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        lrs = []
        for base_lr in self.base_lrs:
            lr = max(self.min_lr, base_lr * (self.gamma ** self.last_epoch))
            lrs.append(lr)
        return lrs


class OneCycleScheduler(BaseScheduler):
    """
    1Cycle Learning Rate Scheduler (super-convergence).
    
    Paper: https://arxiv.org/abs/1803.09820
    
    Three phases:
    1. Linear increase from initial_lr to max_lr
    2. Linear decrease from max_lr to min_lr  
    3. Further decay towards min_lr
    
    Args:
        optimizer: PyTorch optimizer
        max_lr: Maximum learning rate
        total_epochs: Total training epochs
        pct_start: Percentage of cycle for phase 1 (default: 0.3)
        anneal_strategy: 'cos' or 'linear' for phase 2
        last_epoch: Last epoch number
    
    Example:
        >>> scheduler = OneCycleScheduler(optimizer, max_lr=0.1, total_epochs=100)
    """

    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_epochs: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy

        if anneal_strategy not in ("cos", "linear"):
            raise ValueError(f"anneal_strategy must be 'cos' or 'linear', got {anneal_strategy}")

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        step_size = self.total_epochs
        cycle = self.last_epoch / step_size

        lrs = []
        for base_lr in self.base_lrs:
            # Min LR is 1/10 of max LR (common heuristic)
            min_lr = self.max_lr / 10

            if cycle < self.pct_start:
                # Phase 1: Linear increase
                progress = cycle / self.pct_start
                lr = base_lr + progress * (self.max_lr - base_lr)
            else:
                # Phase 2 & 3: Decay towards min_lr
                progress = (cycle - self.pct_start) / (1 - self.pct_start)
                if self.anneal_strategy == "cos":
                    lr = min_lr + (self.max_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2
                else:  # linear
                    lr = self.max_lr - progress * (self.max_lr - min_lr)

            lrs.append(lr)
        return lrs


class ReduceLROnPlateau:
    """
    Reduce learning rate when validation metric plateaus.
    
    Compatible with metric-based evaluation (e.g., FID score).
    
    Args:
        optimizer: PyTorch optimizer
        mode: 'min' (for loss) or 'max' (for accuracy-like metrics)
        factor: Factor to multiply LR by (default: 0.5)
        patience: Number of epochs with no improvement before reducing LR
        threshold: Threshold for metric improvement
        min_lr: Minimum learning rate
        verbose: Print debug information
    
    Example:
        >>> scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
        >>> for epoch in range(num_epochs):
        ...     train()
        ...     val_loss = validate()
        ...     scheduler.step(val_loss)
    """

    def __init__(
        self,
        optimizer,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 5,
        threshold: float = 1e-4,
        min_lr: float = 1e-6,
        verbose: bool = False,
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose

        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.last_epoch = 0

    def step(self, metric: float) -> bool:
        """
        Update learning rate based on metric.
        
        Returns:
            True if learning rate was reduced, False otherwise
        """
        is_improvement = False

        if self.mode == "min":
            is_improvement = metric < (self.best_metric - self.threshold)
        else:  # max
            is_improvement = metric > (self.best_metric + self.threshold)

        if is_improvement:
            self.best_metric = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return self._reduce_lr()

        self.last_epoch += 1
        return False

    def _reduce_lr(self) -> bool:
        """Reduce learning rate and return True if successful."""
        new_lrs = []
        for param_group in self.optimizer.param_groups:
            old_lr = param_group["lr"]
            new_lr = max(self.min_lr, old_lr * self.factor)
            param_group["lr"] = new_lr
            new_lrs.append(new_lr)

        if self.verbose:
            print(f"Reducing LR: {[f'{lr:.2e}' for lr in new_lrs]}")

        self.num_bad_epochs = 0
        return True
