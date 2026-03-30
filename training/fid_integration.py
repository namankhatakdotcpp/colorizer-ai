"""
FID Integration Package
- Checkpoint management based on FID scores
- Training loop integration examples
- Evaluation callbacks for GANTrainer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Callable
import json
import logging
from datetime import datetime

from training.train_gan_production import GANTrainer, CheckpointManager
from training.fid_evaluator import FIDEvaluator, GeneratedImageProxy

logger = logging.getLogger(__name__)


class FIDCheckpointManager(CheckpointManager):
    """
    Enhanced checkpoint manager that tracks FID scores.
    
    Extends CheckpointManager with:
    - FID score tracking
    - Best FID-based checkpoint selection
    - FID history logging
    
    Args:
        checkpoint_dir (str): Directory for checkpoints
        generator: Generator model
        discriminator: Discriminator model
        device (torch.device): GPU/CPU device
        fid_evaluator (FIDEvaluator): FID evaluation instance
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        generator: nn.Module,
        discriminator: nn.Module,
        device: torch.device,
        fid_evaluator: FIDEvaluator,
        keep_last_n: int = 3,
    ):
        super().__init__(checkpoint_dir, generator, discriminator, device, keep_last_n)
        self.fid_evaluator = fid_evaluator
        self.fid_history = []
        self.history_file = Path(checkpoint_dir) / "fid_history.json"
    
    def save_checkpoint_with_fid(
        self,
        epoch: int,
        fid_score: float,
        optimizers: Optional[Dict] = None,
        ema_state: Optional[Dict] = None,
        scaler_states: Optional[Dict] = None,
    ) -> bool:
        """
        Save checkpoint and track FID score.
        
        Returns True if this is a best FID checkpoint.
        """
        is_best = fid_score < self.fid_evaluator.best_fid
        
        # Save checkpoint
        self.save_checkpoint(
            epoch, optimizers, ema_state, scaler_states,
            "best_fid" if is_best else "latest"
        )
        
        # Track FID
        fid_entry = {
            "epoch": epoch,
            "fid_score": fid_score,
            "is_best": is_best,
            "timestamp": datetime.now().isoformat()
        }
        self.fid_history.append(fid_entry)
        
        # Save history
        with open(self.history_file, 'w') as f:
            json.dump(self.fid_history, f, indent=2)
        
        if is_best:
            logger.info(f"🏆 New best FID at epoch {epoch}: {fid_score:.4f}")
            self.fid_evaluator.best_fid = fid_score
        
        return is_best
    
    def load_history(self) -> list:
        """Load FID history from file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.fid_history = json.load(f)
        return self.fid_history


class FIDEvaluationCallback:
    """
    Evaluation callback for training loops.
    
    Periodically evaluates FID score and manages checkpoints.
    
    Args:
        fid_evaluator (FIDEvaluator): FID calculator
        checkpoint_manager (FIDCheckpointManager): Checkpoint handler
        eval_frequency (int): Evaluate every N epochs (default: 10)
        num_eval_samples (int): Samples for FID (default: None = all)
    """
    
    def __init__(
        self,
        fid_evaluator: FIDEvaluator,
        checkpoint_manager: FIDCheckpointManager,
        eval_frequency: int = 10,
        num_eval_samples: Optional[int] = None,
    ):
        self.fid_evaluator = fid_evaluator
        self.checkpoint_manager = checkpoint_manager
        self.eval_frequency = eval_frequency
        self.num_eval_samples = num_eval_samples
        self.eval_count = 0
    
    def on_epoch_end(
        self,
        epoch: int,
        generator: nn.Module,
        real_dataloader: DataLoader,
        condition_dataloader: DataLoader,
        device: torch.device,
        optimizers: Optional[Dict] = None,
        ema_state: Optional[Dict] = None,
        scaler_states: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Evaluate FID at end of epoch (if frequency matches).
        
        Returns:
            Metrics dictionary or empty dict if not evaluation epoch
        """
        if (epoch + 1) % self.eval_frequency != 0:
            return {}
        
        self.eval_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"FID Evaluation #{self.eval_count} at Epoch {epoch + 1}")
        logger.info(f"{'='*60}\n")
        
        generator.eval()
        
        # Create generated image proxy
        gen_proxy = GeneratedImageProxy(generator, condition_dataloader, device)
        
        # Evaluate FID
        metrics = self.fid_evaluator.evaluate(
            real_dataloader,
            gen_proxy,
            num_samples=self.num_eval_samples,
        )
        
        fid_score = metrics["fid"]
        
        # Save checkpoint with FID
        self.checkpoint_manager.save_checkpoint_with_fid(
            epoch, fid_score, optimizers, ema_state, scaler_states
        )
        
        generator.train()
        
        return metrics


# ============================================================================
# Training Loop Integration
# ============================================================================

def train_gan_with_fid(
    generator: nn.Module,
    discriminator: nn.Module,
    loss_manager,  # LossManager instance
    train_dataloader: DataLoader,
    real_eval_dataloader: DataLoader,
    condition_eval_dataloader: DataLoader,
    device: torch.device,
    num_epochs: int = 100,
    checkpoint_dir: str = "checkpoints/fid_tracking",
    eval_frequency: int = 10,
    num_eval_samples: Optional[int] = None,
    use_amp: bool = True,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    n_critic: int = 2,
) -> Dict:
    """
    Complete training loop with FID evaluation.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        loss_manager: LossManager instance
        train_dataloader: Training data
        real_eval_dataloader: Real images for FID evaluation
        condition_eval_dataloader: Conditions for FID evaluation
        device: torch.device
        num_epochs: Training epochs
        checkpoint_dir: Checkpoint directory
        eval_frequency: Evaluate every N epochs
        num_eval_samples: FID eval samples
        use_amp: Mixed precision (default: True)
        use_ema: Exponential Moving Average (default: True)
        ema_decay: EMA decay coefficient
        n_critic: Discriminator updates per generator update
    
    Returns:
        Training history dictionary
    """
    
    # Initialize FID evaluator
    fid_evaluator = FIDEvaluator(device, checkpoint_dir)
    
    # Initialize checkpoint manager
    checkpoint_manager = FIDCheckpointManager(
        checkpoint_dir, generator, discriminator, device, fid_evaluator
    )
    
    # Initialize training
    trainer = GANTrainer(
        generator, discriminator, device,
        use_amp=use_amp, use_ema=use_ema, ema_decay=ema_decay,
        n_critic=n_critic
    )
    
    # Initialize evaluation callback
    eval_callback = FIDEvaluationCallback(
        fid_evaluator, checkpoint_manager,
        eval_frequency, num_eval_samples
    )
    
    # Training history
    history = {
        "train_loss": [],
        "fid_scores": [],
        "best_fid": float('inf'),
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting GAN Training with FID Evaluation")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"FID Evaluation: Every {eval_frequency} epochs")
    logger.info(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Handle data format
            if isinstance(batch, (tuple, list)):
                condition, target = batch[0], batch[1]
            else:
                condition = target = batch
            
            # Move to device
            condition = condition.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Training step
            loss_dict = trainer.train_step(condition, target)
            
            # Track loss
            epoch_loss += loss_dict.get("g_total", 0.0)
            num_batches += 1
            
            # Log batch progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / num_batches
                logger.info(
                    f"Epoch [{epoch+1:3d}/{num_epochs}] "
                    f"Batch [{batch_idx+1:4d}] "
                    f"Loss: {avg_loss:.4f}"
                )
        
        # Average epoch loss
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        history["train_loss"].append(avg_epoch_loss)
        
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")
        
        # FID Evaluation
        fid_metrics = eval_callback.on_epoch_end(
            epoch, generator,
            real_eval_dataloader, condition_eval_dataloader,
            device,
            trainer.optimizers if hasattr(trainer, 'optimizers') else None,
            trainer.ema_state if hasattr(trainer, 'ema_state') else None,
            trainer.scaler_states if hasattr(trainer, 'scaler_states') else None,
        )
        
        if fid_metrics:
            fid_score = fid_metrics["fid"]
            history["fid_scores"].append(fid_score)
            
            if fid_score < history["best_fid"]:
                history["best_fid"] = fid_score
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Complete!")
    logger.info(f"Best FID: {history['best_fid']:.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    logger.info(f"{'='*60}\n")
    
    return history


# ============================================================================
# Quick Evaluation Functions
# ============================================================================

def quick_fid_eval(
    generator: nn.Module,
    real_dataloader: DataLoader,
    condition_dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 50,
) -> float:
    """
    Quick FID evaluation (50 samples by default).
    
    Returns:
        FID score (float)
    """
    fid_evaluator = FIDEvaluator(device)
    gen_proxy = GeneratedImageProxy(generator, condition_dataloader, device)
    
    metrics = fid_evaluator.evaluate(
        real_dataloader, gen_proxy, num_samples=num_samples
    )
    
    return metrics["fid"]


def compare_checkpoints_fid(
    checkpoint_dir: str,
    device: torch.device,
) -> Dict:
    """
    Compare FID scores across checkpoints in a directory.
    
    Returns:
        Dictionary with checkpoint comparisons
    """
    history_file = Path(checkpoint_dir) / "fid_history.json"
    
    if not history_file.exists():
        logger.warning(f"No FID history found in {checkpoint_dir}")
        return {}
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Summarize
    summary = {
        "total_evals": len(history),
        "best_fid": min(h["fid_score"] for h in history),
        "worst_fid": max(h["fid_score"] for h in history),
        "mean_fid": sum(h["fid_score"] for h in history) / len(history),
        "history": history,
    }
    
    logger.info(f"\nFID Summary for {checkpoint_dir}")
    logger.info(f"  Total Evaluations: {summary['total_evals']}")
    logger.info(f"  Best FID: {summary['best_fid']:.4f}")
    logger.info(f"  Worst FID: {summary['worst_fid']:.4f}")
    logger.info(f"  Mean FID: {summary['mean_fid']:.4f}")
    
    return summary
