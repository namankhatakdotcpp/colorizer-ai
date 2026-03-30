"""
Complete Training Example: Refactored GAN Trainer
================================================

Shows how to use the three major enhancements together:
1. Proper n_critic implementation
2. EMA generator for evaluation
3. Enhanced checkpoint system with FID tracking

This is a fully working example that you can adapt to your codebase.
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the refactored trainer
from training.gan_training_refactored import (
    RefactoredGANTrainer,
    FIDCheckpointManager,
    TrainingConfig,
)


def example_1_basic_training_loop():
    """
    Example 1: Basic training loop with proper n_critic.
    
    This demonstrates:
    - n_critic=2: Train discriminator twice per generator update
    - Proper gradient management
    - Clean optimizer step separation
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Training Loop with Proper n_critic")
    print("="*80 + "\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Assume you have these
    # generator = GANGenerator().to(device)
    # discriminator = MultiscaleDiscriminator().to(device)
    # loss_manager = create_loss_manager(device)
    # train_loader = DataLoader(...)
    
    # Create trainer with n_critic=2
    config = TrainingConfig(
        n_critic=2,  # Train discriminator 2x per generator update
        gradient_clip=0.5,
        use_amp=True,
        use_ema=True,
    )
    
    trainer = RefactoredGANTrainer(
        generator=generator,
        discriminator=discriminator,
        loss_manager=loss_manager,
        device=device,
        config=config,
    )
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = {}
        
        for batch_idx, (condition, target) in enumerate(tqdm(train_loader)):
            # Single train_step handles everything:
            # - Discriminator trained 2 times (n_critic=2)
            # - Generator trained 1 time
            # - All gradients properly managed
            # - No accumulation issues
            loss_dict = trainer.train_step(condition, target)
            
            # Accumulate losses
            for k, v in loss_dict.items():
                epoch_loss.setdefault(k, 0)
                epoch_loss[k] += v
        
        # Log epoch losses
        logger.info(f"Epoch {epoch}: " + " | ".join(
            f"{k}={v/len(train_loader):.4f}"
            for k, v in epoch_loss.items()
        ))


def example_2_ema_evaluation():
    """
    Example 2: Using EMA generator for evaluation.
    
    This demonstrates:
    - Training with current generator
    - Evaluation with EMA generator (smoother, better quality)
    - Easy switching between modes
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: EMA Generator for Evaluation")
    print("="*80 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = TrainingConfig(
        use_ema=True,
        ema_decay=0.999,  # Standard value
    )
    
    trainer = RefactoredGANTrainer(
        generator=generator,
        discriminator=discriminator,
        loss_manager=loss_manager,
        device=device,
        config=config,
    )
    
    # Training phase (uses current weights)
    for epoch in range(num_epochs):
        trainer.generator.train()
        for condition, target in train_loader:
            loss_dict = trainer.train_step(condition, target)
        
        # Evaluation phase (switch to EMA weights)
        # ✅ This is the key enhancement: automatic EMA switching
        trainer.set_eval_mode_ema()  # Switch to EMA generator
        
        # Now inference uses EMA weights (smoother, better quality)
        with torch.no_grad():
            for condition, target in val_loader:
                fake_rgb = trainer.generator(condition)
                # Use fake_rgb for metric computation (FID, LPIPS, etc.)
                fid_score = compute_fid(fake_rgb, target)
        
        # Switch back to training
        trainer.set_train_mode_current()  # Back to current generator
        
        logger.info(f"Epoch {epoch}: FID={fid_score:.3f}")


def example_3_fid_checkpoint_management():
    """
    Example 3: Enhanced checkpoint system with FID tracking.
    
    This demonstrates:
    - FID score tracking per epoch
    - Save best_model.pth (lowest FID)
    - Save latest_model.pth
    - Prevent saving worse models
    - Log FID trends
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: FID Checkpoint Management")
    print("="*80 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create trainer
    trainer = RefactoredGANTrainer(
        generator=generator,
        discriminator=discriminator,
        loss_manager=loss_manager,
        device=device,
    )
    
    # Create checkpoint manager
    checkpoint_dir = Path("checkpoints/gan_training")
    checkpoint_manager = FIDCheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        keep_best_n=3,
        track_fid=True,
    )
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        for condition, target in train_loader:
            loss_dict = trainer.train_step(condition, target)
        
        # Evaluation
        trainer.set_eval_mode_ema()
        fid_score = 0.0
        with torch.no_grad():
            for condition, target in val_loader:
                fake_rgb = trainer.generator(condition)
                fid_score += compute_fid(fake_rgb, target)
        fid_score /= len(val_loader)
        trainer.set_train_mode_current()
        
        # ✅ Save checkpoint with FID tracking
        is_best, metadata = checkpoint_manager.save_checkpoint(
            epoch=epoch,
            generator=trainer.generator,
            discriminator=trainer.discriminator,
            optimizer_g=trainer.optimizer_g,
            optimizer_d=trainer.optimizer_d,
            fid_score=fid_score,  # ← FID score tracking
            ema=trainer.ema,
            scaler_g=trainer.scaler_g,
            scaler_d=trainer.scaler_d,
        )
        
        # Log FID trend
        trend = checkpoint_manager.get_fid_trend()
        logger.info(f"Epoch {epoch}: FID={fid_score:.3f} | "
                   f"Best={trend['best_fid']:.3f} @ epoch {trend['best_epoch']} | "
                   f"Trend={trend['trend']}")
        
        if is_best:
            logger.info("🏆 New best model saved!")


def example_4_complete_training_script():
    """
    Example 4: Complete training script with all three enhancements.
    
    This is production-ready code that uses:
    1. Proper n_critic implementation
    2. EMA generator for evaluation
    3. Enhanced checkpoint management with FID
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Complete Training Script")
    print("="*80 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    config = TrainingConfig(
        n_critic=2,
        gradient_clip=0.5,
        use_amp=True,
        use_ema=True,
        ema_decay=0.999,
        use_r1_penalty=False,  # Can enable if desired
    )
    
    # Create models (replace with your models)
    # generator = GANGenerator().to(device)
    # discriminator = MultiscaleDiscriminator().to(device)
    # loss_manager = create_loss_manager(device)
    
    # Create trainer
    trainer = RefactoredGANTrainer(
        generator=generator,
        discriminator=discriminator,
        loss_manager=loss_manager,
        device=device,
        config=config,
    )
    
    # Create checkpoint manager
    checkpoint_dir = Path("checkpoints/gan_final")
    checkpoint_manager = FIDCheckpointManager(checkpoint_dir=str(checkpoint_dir))
    
    # Load checkpoint if resuming
    best_model_path = checkpoint_manager.get_best_model_path()
    if best_model_path:
        logger.info(f"Resuming from: {best_model_path}")
        checkpoint_manager.load_checkpoint(
            str(best_model_path),
            generator=trainer.generator,
            discriminator=trainer.discriminator,
            optimizer_g=trainer.optimizer_g,
            optimizer_d=trainer.optimizer_d,
            device=device,
            ema=trainer.ema,
            scaler_g=trainer.scaler_g,
            scaler_d=trainer.scaler_d,
        )
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        # TRAINING PHASE
        trainer.generator.train()
        trainer.discriminator.train()
        
        train_loss = {}
        for batch_idx, (condition, target) in enumerate(tqdm(train_loader, desc="Training")):
            # Single train_step with all enhancements
            loss_dict = trainer.train_step(condition, target)
            
            for k, v in loss_dict.items():
                train_loss.setdefault(k, 0)
                train_loss[k] += v
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"[Batch {batch_idx+1}] " + 
                           " | ".join(f"{k}={v:.4f}" for k, v in loss_dict.items()))
        
        # Average training losses
        for k in train_loss:
            train_loss[k] /= len(train_loader)
        
        # VALIDATION PHASE
        trainer.set_eval_mode_ema()  # ← Switch to EMA for evaluation
        
        val_loss = {}
        fid_scores = []
        
        with torch.no_grad():
            for condition, target in tqdm(val_loader, desc="Validation"):
                # Generate with EMA generator
                fake_rgb = trainer.generator(condition)
                
                # Compute FID
                fid = compute_fid(fake_rgb, target)
                fid_scores.append(fid)
        
        avg_fid = sum(fid_scores) / len(fid_scores)
        trainer.set_train_mode_current()  # ← Back to current for training
        
        # CHECKPOINT MANAGEMENT
        is_best, metadata = checkpoint_manager.save_checkpoint(
            epoch=epoch,
            generator=trainer.generator,
            discriminator=trainer.discriminator,
            optimizer_g=trainer.optimizer_g,
            optimizer_d=trainer.optimizer_d,
            fid_score=avg_fid,  # ← FID tracking
            ema=trainer.ema,
            scaler_g=trainer.scaler_g,
            scaler_d=trainer.scaler_d,
        )
        
        # Logging
        trend = checkpoint_manager.get_fid_trend()
        logger.info(f"\nEpoch Results:")
        logger.info(f"  Train Loss: {sum(v for v in train_loss.values()):.4f}")
        logger.info(f"  Val FID: {avg_fid:.3f}")
        logger.info(f"  Best FID: {trend['best_fid']:.3f} (epoch {trend['best_epoch']})")
        logger.info(f"  Trend: {trend['trend']}")
        
        if is_best:
            logger.info("  🏆 NEW BEST MODEL!")


def example_5_inference_with_ema():
    """
    Example 5: Inference using the EMA generator.
    
    Shows how to load and use the best model for inference.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Inference with EMA Generator")
    print("="*80 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    generator = GANGenerator().to(device)
    discriminator = MultiscaleDiscriminator().to(device)
    loss_manager = create_loss_manager(device)
    
    trainer = RefactoredGANTrainer(
        generator=generator,
        discriminator=discriminator,
        loss_manager=loss_manager,
        device=device,
    )
    
    # Load best checkpoint
    checkpoint_dir = Path("checkpoints/gan_final")
    checkpoint_manager = FIDCheckpointManager(checkpoint_dir=str(checkpoint_dir))
    
    best_model_path = checkpoint_manager.get_best_model_path()
    if best_model_path:
        checkpoint_manager.load_checkpoint(
            str(best_model_path),
            generator=trainer.generator,
            discriminator=trainer.discriminator,
            optimizer_g=trainer.optimizer_g,
            optimizer_d=trainer.optimizer_d,
            device=device,
            ema=trainer.ema,
        )
    
    # Switch to EMA generator for inference
    trainer.set_eval_mode_ema()
    
    # Inference
    with torch.no_grad():
        for condition in test_loader:
            condition = condition.to(device)
            
            # Generate with EMA weights (best quality)
            output = trainer.generator(condition)
            
            # Use output for downstream tasks (visualization, metrics, etc.)
            save_results(output)
    
    logger.info("✅ Inference complete with EMA generator")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     REFACTORED GAN TRAINING - COMPLETE EXAMPLES               ║
    ║                                                                ║
    │ This file shows how to use:                                   ║
    │  1. Proper n_critic implementation                            ║
    │  2. EMA generator for evaluation                              ║
    │  3. Enhanced checkpoint system with FID tracking              ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Note: These examples are pseudocode showing the structure
    # You'll need to provide:
    # - generator, discriminator models
    # - loss_manager
    # - train_loader, val_loader, test_loader
    # - compute_fid() function
    # - save_results() function
    
    print("\nTo run these examples replace the placeholders with your actual objects")
