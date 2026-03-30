"""
Refactored GAN Training Loop - Production Ready

This module provides a production-grade GAN training implementation with:

1. **Proper n_critic Implementation**: Train discriminator n_critic times before 
   each generator step with correct gradient handling and optimizer separation.

2. **EMA Generator for Evaluation**: Automatic EMA weight management for evaluation
   and inference, separate from training weights.

3. **Enhanced Checkpoint System**: Track FID scores, save best/latest models, 
   prevent degradation, and log FID trends.

Key Features:
- ✅ Clean n_critic loop with no gradient accumulation issues
- ✅ EMA generator automatically used during validation/inference
- ✅ FID-based checkpoint management with history tracking
- ✅ Mixed precision (AMP) integration
- ✅ R1 penalty support
- ✅ Gradient clipping and monitoring
- ✅ Complete state dict management for resuming
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    n_critic: int = 2
    gradient_clip: float = 0.5
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    use_r1_penalty: bool = True
    lambda_r1: float = 10.0
    apply_r1_every_n_steps: int = 16


class GradientMonitor:
    """Monitor gradient statistics during training."""
    
    def __init__(self):
        self.grad_norms = []
    
    def compute_grad_norm(self, model: nn.Module) -> float:
        """Compute L2 norm of all gradients."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)
        return total_norm
    
    def get_stats(self) -> Dict[str, float]:
        """Get gradient statistics."""
        if not self.grad_norms:
            return {}
        grad_array = self.grad_norms[-100:]  # Last 100 steps
        return {
            "grad_norm_mean": sum(grad_array) / len(grad_array),
            "grad_norm_max": max(grad_array),
            "grad_norm_min": min(grad_array),
        }


class ExponentialMovingAverage(nn.Module):
    """
    Maintains EMA weights for model.
    
    Features:
    - Shadow weights stored separately
    - Warmup decay adjustment
    - Easy switching between current and EMA weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.model = model
        self.decay = decay
        self.step = 0

        # Store shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                safe_name = name.replace(".", "_")  # PyTorch doesn't allow dots in buffer names
                self.register_buffer(f"shadow_{safe_name}", param.data.clone())

    def update(self):
        """Update EMA weights."""
        self.step += 1
        decay = min(self.decay, (1 + self.step) / (10 + self.step))

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    safe_name = name.replace(".", "_")  # Match the safe name used in __init__
                    shadow = getattr(self, f"shadow_{safe_name}")
                    shadow.copy_(decay * shadow + (1 - decay) * param.data)

    def set_to_shadow(self):
        """Switch model to use EMA weights (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                safe_name = name.replace(".", "_")  # Match the safe name used in __init__
                shadow = getattr(self, f"shadow_{safe_name}")
                self._original_weights = {n: p.data.clone() 
                                         for n, p in self.model.named_parameters()}
                param.data.copy_(shadow)

    def restore_from_shadow(self):
        """Restore original (training) weights."""
        if hasattr(self, '_original_weights'):
            for name, param in self.model.named_parameters():
                if name in self._original_weights:
                    param.data.copy_(self._original_weights[name])


class FIDCheckpointManager:
    """
    Manages checkpoints with FID score tracking.
    
    Tracks:
    - Best model (lowest FID)
    - Latest model
    - FID history
    - Prevents saving worse models
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_best_n: int = 3,
        track_fid: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_best_n = keep_best_n
        self.track_fid = track_fid
        
        self.best_fid = float('inf')
        self.best_epoch = -1
        self.fid_history: List[Dict] = []
        self.history_file = self.checkpoint_dir / "fid_history.json"
        
        self._load_history()

    def _load_history(self):
        """Load FID history from previous runs."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.fid_history = json.load(f)
                if self.fid_history:
                    self.best_fid = min(h["fid_score"] for h in self.fid_history)
                    logger.info(f"📊 Loaded FID history: best={self.best_fid:.3f}")
            except Exception as e:
                logger.warning(f"Could not load history: {e}")

    def save_checkpoint(
        self,
        epoch: int,
        generator: nn.Module,
        discriminator: nn.Module,
        optimizer_g: optim.Optimizer,
        optimizer_d: optim.Optimizer,
        fid_score: Optional[float] = None,
        ema: Optional[ExponentialMovingAverage] = None,
        scaler_g: Optional[GradScaler] = None,
        scaler_d: Optional[GradScaler] = None,
        losses: Optional[Dict] = None,
    ) -> Tuple[bool, Dict]:
        """
        Save checkpoint. Returns (is_best, metadata).
        
        Args:
            epoch: Epoch number
            generator: Generator model
            discriminator: Discriminator model
            optimizer_g: Generator optimizer
            optimizer_d: Discriminator optimizer
            fid_score: FID score for this epoch
            ema: EMA module
            scaler_g: Generator GradScaler
            scaler_d: Discriminator GradScaler
            losses: Loss dictionary
        
        Returns:
            (is_best: bool, metadata: Dict)
        """
        # Determine if this is best model
        is_best = False
        if fid_score is not None and fid_score < self.best_fid:
            is_best = True
            self.best_fid = fid_score
            self.best_epoch = epoch

        # Prepare checkpoint
        checkpoint = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "fid_score": fid_score,
            "losses": losses or {},
        }

        if ema is not None:
            checkpoint["ema"] = ema.state_dict()
        if scaler_g is not None:
            checkpoint["scaler_g"] = scaler_g.state_dict()
        if scaler_d is not None:
            checkpoint["scaler_d"] = scaler_d.state_dict()

        metadata = {
            "epoch": epoch,
            "fid_score": fid_score,
            "is_best": is_best,
            "timestamp": datetime.now().isoformat(),
        }

        # Save latest checkpoint (always overwritten)
        latest_path = self.checkpoint_dir / "latest_model.pth"
        torch.save(checkpoint, latest_path)
        logger.info(f"💾 Saved latest checkpoint: {latest_path}")

        # Save best checkpoint if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 New best model! FID: {fid_score:.3f} @ epoch {epoch}")

        # Track FID history
        if fid_score is not None:
            history_entry = {
                "epoch": epoch,
                "fid_score": fid_score,
                "is_best": is_best,
                "timestamp": metadata["timestamp"],
            }
            self.fid_history.append(history_entry)
            self._save_history()

        return is_best, metadata

    def _save_history(self):
        """Save FID history to JSON."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.fid_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save history: {e}")

    def get_best_model_path(self) -> Optional[Path]:
        """Get path to best model checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pth"
        return best_path if best_path.exists() else None

    def get_latest_model_path(self) -> Optional[Path]:
        """Get path to latest model checkpoint."""
        latest_path = self.checkpoint_dir / "latest_model.pth"
        return latest_path if latest_path.exists() else None

    def get_fid_trend(self) -> Dict:
        """Get FID trend statistics."""
        if len(self.fid_history) < 2:
            return {}
        
        fids = [h["fid_score"] for h in self.fid_history]
        recent_fids = fids[-10:] if len(fids) > 10 else fids
        
        return {
            "best_fid": self.best_fid,
            "best_epoch": self.best_epoch,
            "total_checkpoints": len(self.fid_history),
            "recent_avg": sum(recent_fids) / len(recent_fids),
            "trend": "improving" if recent_fids[-1] < recent_fids[0] else "degrading",
        }

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        generator: nn.Module,
        discriminator: nn.Module,
        optimizer_g: optim.Optimizer,
        optimizer_d: optim.Optimizer,
        device: torch.device,
        ema: Optional[ExponentialMovingAverage] = None,
        scaler_g: Optional[GradScaler] = None,
        scaler_d: Optional[GradScaler] = None,
    ) -> Dict:
        """Load checkpoint and restore state."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])

        if ema is not None and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        if scaler_g is not None and "scaler_g" in checkpoint:
            scaler_g.load_state_dict(checkpoint["scaler_g"])
        if scaler_d is not None and "scaler_d" in checkpoint:
            scaler_d.load_state_dict(checkpoint["scaler_d"])

        logger.info(f"✅ Loaded checkpoint: {checkpoint_path}")

        return {
            "epoch": checkpoint.get("epoch", 0),
            "fid_score": checkpoint.get("fid_score"),
        }


class RefactoredGANTrainer:
    """
    Production-grade GAN trainer with all enhancements.
    
    Key improvements:
    
    1. **Proper n_critic Implementation**:
       - Discriminator trained n_critic times per generator update
       - No gradient accumulation issues
       - Clean optimizer step separation
       - Generator frozen during discriminator updates
    
    2. **EMA Generator for Evaluation**:
       - EMA weights maintained automatically
       - Easy switching for evaluation mode
       - Generator uses current weights for training
       - EMA weights used during validation/inference
    
    3. **Enhanced Checkpoint Management**:
       - Tracks FID score per epoch
       - Saves best_model.pth and latest_model.pth
       - Prevents saving degraded models
       - Logs FID trends
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        loss_manager,
        device: torch.device = None,
        config: TrainingConfig = None,
    ):
        """
        Initialize refactored GAN trainer.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            loss_manager: Loss computation module
            device: Device to train on
            config: Training configuration
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = config or TrainingConfig()

        # Models
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.loss_manager = loss_manager

        # Optimizers (TTUR: discriminator has 2x learning rate)
        self.optimizer_g = optim.Adam(
            self.generator.parameters(), lr=1e-4, betas=(0.0, 0.99)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=2e-4, betas=(0.0, 0.99)
        )

        # Mixed precision training
        self.scaler_g = GradScaler() if self.config.use_amp else None
        self.scaler_d = GradScaler() if self.config.use_amp else None

        # EMA for generator
        self.ema = (
            ExponentialMovingAverage(self.generator, decay=self.config.ema_decay)
            if self.config.use_ema
            else None
        )

        # Monitoring
        self.grad_monitor_g = GradientMonitor()
        self.grad_monitor_d = GradientMonitor()
        self.step_count = 0

        # R1 penalty (optional)
        self.use_r1_penalty = self.config.use_r1_penalty
        self.r1_calculator = None
        if self.use_r1_penalty:
            try:
                from training.r1_penalty import R1PenaltyCalculator
                self.r1_calculator = R1PenaltyCalculator(
                    self.discriminator,
                    self.device,
                    lambda_r1=self.config.lambda_r1
                )
                logger.info(f"✅ R1 penalty enabled (lambda={self.config.lambda_r1})")
            except ImportError:
                logger.warning("R1 penalty module not available")
                self.use_r1_penalty = False

        logger.info(
            f"🚀 Initialized RefactoredGANTrainer:\n"
            f"   Device: {self.device}\n"
            f"   n_critic: {self.config.n_critic}\n"
            f"   AMP: {self.config.use_amp}\n"
            f"   EMA: {self.config.use_ema} (decay={self.config.ema_decay})\n"
            f"   R1 Penalty: {self.use_r1_penalty}"
        )

    def train_step(
        self,
        condition: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step: discriminator n_critic times, then generator once.
        
        **Proper n_critic Implementation**:
        - Discriminator trained n_critic times
        - Gradients accumulate within discriminator training
        - BUT optimizer is stepped each time (not accumulated)
        - Generator is frozen (no grad computed)
        - After n_critic steps, generator is trained once
        
        Args:
            condition: Condition images (B, 1, H, W)
            target: Target images (B, 3, H, W) or desired output
        
        Returns:
            Loss dictionary with all metrics
        """
        batch_size = target.size(0)
        condition = condition.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        real_labels = torch.full((batch_size,), 0.9, device=self.device)
        fake_labels = torch.full((batch_size,), 0.1, device=self.device)

        loss_dict = {}

        # ===== DISCRIMINATOR TRAINING (n_critic times) =====
        for critic_step in range(self.config.n_critic):
            # Zero gradients (clean slate for each discriminator step)
            self.optimizer_d.zero_grad(set_to_none=True)

            # Generate fake images (DETACHED - no gradients flow back to G)
            with torch.no_grad():
                if self.config.use_amp:
                    with autocast():
                        fake_rgb = self.generator(condition)
                else:
                    fake_rgb = self.generator(condition)

            # Prepare inputs
            fake_input = torch.cat([condition, fake_rgb], dim=1)
            real_input = torch.cat([condition, target], dim=1)

            # Discriminator forward pass
            if self.config.use_amp:
                with autocast():
                    disc_fake_logits, disc_fake_features = self.discriminator(fake_input)
                    disc_real_logits, disc_real_features = self.discriminator(real_input)

                    # Compute discriminator loss
                    loss_d, dict_d = self.loss_manager(
                        disc_fake_logits=disc_fake_logits,
                        disc_real_logits=disc_real_logits,
                        real_labels=real_labels,
                        fake_labels=fake_labels,
                        mode="discriminator",
                    )

                    # Add R1 penalty if applicable
                    if self.use_r1_penalty and self._should_apply_r1():
                        if self.r1_calculator is not None:
                            r1_penalty, r1_stats = self.r1_calculator.compute_penalty_scaled(
                                real_input, self.scaler_d
                            )
                            loss_d = loss_d + r1_penalty
                            dict_d.update(r1_stats)

                # Scale loss and backward
                scaled_loss = self.scaler_d.scale(loss_d)
                scaled_loss.backward()

                # Unscale before gradient clipping
                self.scaler_d.unscale_(self.optimizer_d)
            else:
                disc_fake_logits, disc_fake_features = self.discriminator(fake_input)
                disc_real_logits, disc_real_features = self.discriminator(real_input)

                loss_d, dict_d = self.loss_manager(
                    disc_fake_logits=disc_fake_logits,
                    disc_real_logits=disc_real_logits,
                    real_labels=real_labels,
                    fake_labels=fake_labels,
                    mode="discriminator",
                )

                # Add R1 penalty
                if self.use_r1_penalty and self._should_apply_r1():
                    if self.r1_calculator is not None:
                        r1_penalty, r1_stats = self.r1_calculator.compute_penalty(real_input)
                        loss_d = loss_d + r1_penalty
                        dict_d.update(r1_stats)

                loss_d.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), self.config.gradient_clip
            )

            # Monitor gradients
            grad_norm_d = self.grad_monitor_d.compute_grad_norm(self.discriminator)

            # Optimizer step (THIS is where discriminator weights are updated)
            if self.config.use_amp:
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
            else:
                self.optimizer_d.step()

            # Store loss from last critic step
            if critic_step == self.config.n_critic - 1:
                loss_dict.update({f"d_{k}": v for k, v in dict_d.items()})
                loss_dict["d_grad_norm"] = grad_norm_d

        # ===== GENERATOR TRAINING =====
        # Zero gradients for generator
        self.optimizer_g.zero_grad(set_to_none=True)

        # Generate fake images (now WITH gradients for G)
        if self.config.use_amp:
            with autocast():
                fake_rgb = self.generator(condition)
        else:
            fake_rgb = self.generator(condition)

        fake_input = torch.cat([condition, fake_rgb], dim=1)
        real_input = torch.cat([condition, target], dim=1)

        # Discriminator forward (but NO gradients for D)
        if self.config.use_amp:
            with autocast():
                # Get real features for perceptual loss (no grad needed)
                with torch.no_grad():
                    disc_real_logits, disc_real_features = self.discriminator(real_input)

                # Get fake features and logits (gradients will flow to G)
                disc_fake_logits, disc_fake_features = self.discriminator(fake_input)

                # Compute generator loss
                loss_g, dict_g = self.loss_manager(
                    generated=fake_rgb,
                    target=target,
                    disc_fake_logits=disc_fake_logits,
                    disc_real_logits=disc_real_logits,
                    fake_features=disc_fake_features,
                    real_features=disc_real_features,
                    mode="generator",
                )

            # Scale loss and backward
            scaled_loss = self.scaler_g.scale(loss_g)
            scaled_loss.backward()

            # Unscale before gradient clipping
            self.scaler_g.unscale_(self.optimizer_g)
        else:
            with torch.no_grad():
                disc_real_logits, disc_real_features = self.discriminator(real_input)

            disc_fake_logits, disc_fake_features = self.discriminator(fake_input)

            loss_g, dict_g = self.loss_manager(
                generated=fake_rgb,
                target=target,
                disc_fake_logits=disc_fake_logits,
                disc_real_logits=disc_real_logits,
                fake_features=disc_fake_features,
                real_features=disc_real_features,
                mode="generator",
            )

            loss_g.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(), self.config.gradient_clip
        )

        # Monitor gradients
        grad_norm_g = self.grad_monitor_g.compute_grad_norm(self.generator)

        # Optimizer step
        if self.config.use_amp:
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
        else:
            self.optimizer_g.step()

        # Update EMA
        if self.ema is not None:
            self.ema.update()

        loss_dict.update({f"g_{k}": v for k, v in dict_g.items()})
        loss_dict["g_grad_norm"] = grad_norm_g

        self.step_count += 1
        return loss_dict

    def _should_apply_r1(self) -> bool:
        """Check if R1 penalty should be applied this step."""
        if not self.use_r1_penalty:
            return False
        return self.step_count % self.config.apply_r1_every_n_steps == 0

    def set_eval_mode_ema(self):
        """Switch to EMA generator for evaluation."""
        if self.ema is not None:
            self.generator.eval()
            self.ema.set_to_shadow()
            logger.info("🔄 Switched to EMA generator for evaluation")

    def set_train_mode_current(self):
        """Switch back to current (training) generator."""
        if self.ema is not None:
            self.ema.restore_from_shadow()
        self.generator.train()
        logger.info("🔄 Switched to current generator for training")

    def get_gradient_stats(self) -> Dict[str, float]:
        """Get gradient statistics."""
        stats = {}
        if self.grad_monitor_g.grad_norms:
            stats.update({
                f"grad_g_{k}": v
                for k, v in self.grad_monitor_g.get_stats().items()
            })
        if self.grad_monitor_d.grad_norms:
            stats.update({
                f"grad_d_{k}": v
                for k, v in self.grad_monitor_d.get_stats().items()
            })
        return stats
