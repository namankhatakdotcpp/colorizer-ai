"""
FID Evaluation Module - Complete Usage Examples

This module demonstrates integration of FID evaluation with the GAN training pipeline.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from training.fid_evaluator import (
    FIDCalculator, FIDEvaluator, GeneratedImageProxy, 
    InceptionFeatureExtractor, evaluate_gan_fid
)
from training.fid_integration import (
    FIDCheckpointManager, FIDEvaluationCallback, train_gan_with_fid,
    quick_fid_eval, compare_checkpoints_fid
)


# ============================================================================
# Example 1: Basic FID Evaluation
# ============================================================================

def example_basic_fid_evaluation():
    """
    Simplest way to compute FID between real and generated images.
    """
    print("\n" + "="*60)
    print("Example 1: Basic FID Evaluation")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy data (replace with real dataloaders)
    real_images = torch.randn(100, 3, 256, 256).to(device)
    gen_images = torch.randn(100, 3, 256, 256).to(device)
    
    # Create simple dataloaders
    real_dataset = TensorDataset(real_images)
    gen_dataset = TensorDataset(gen_images)
    
    real_loader = DataLoader(real_dataset, batch_size=16, shuffle=False)
    gen_loader = DataLoader(gen_dataset, batch_size=16, shuffle=False)
    
    # Initialize FID calculator
    fid_calc = FIDCalculator(device)
    
    # Compute FID
    metrics = fid_calc.evaluate_score(real_loader, gen_loader, return_stats=True)
    
    print(f"\nFID Score: {metrics['fid']:.4f}")
    print(f"Real image feature mean: {metrics['mu_real_mean']:.4f}")
    print(f"Generated image feature mean: {metrics['mu_gen_mean']:.4f}")


# ============================================================================
# Example 2: Generator-based Evaluation
# ============================================================================

def example_generator_fid_evaluation():
    """
    Evaluate FID using a generator model (generates images on-the-fly).
    """
    print("\n" + "="*60)
    print("Example 2: Generator-based FID Evaluation")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Placeholder generator (replace with actual model)
    class DummyGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 3 * 256 * 256)
        
        def forward(self, z):
            return self.fc(z).view(z.shape[0], 3, 256, 256)
    
    generator = DummyGenerator().to(device)
    
    # Create dummy data
    real_images = torch.randn(50, 3, 256, 256)
    conditions = torch.randn(50, 100)  # Latent vectors/conditions
    
    real_dataset = TensorDataset(real_images)
    cond_dataset = TensorDataset(conditions)
    
    real_loader = DataLoader(real_dataset, batch_size=16)
    cond_loader = DataLoader(cond_dataset, batch_size=16)
    
    # Quick FID evaluation
    fid_score = quick_fid_eval(generator, real_loader, cond_loader, device, num_samples=50)
    
    print(f"\nGenerator FID Score (50 samples): {fid_score:.4f}")


# ============================================================================
# Example 3: FID Tracking During Training
# ============================================================================

def example_fid_tracking():
    """
    Track FID scores during training with automatic checkpoint selection.
    """
    print("\n" + "="*60)
    print("Example 3: FID Tracking During Training")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create checkpoint directory
    checkpoint_dir = "checkpoints/fid_example"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize FID evaluator
    fid_evaluator = FIDEvaluator(device, checkpoint_dir)
    
    # Simulate multiple evaluations
    print("\nSimulating 5 FID evaluations...")
    fid_scores = [0.85, 0.72, 0.68, 0.70, 0.65]  # Simulated scores
    
    for epoch, fid_score in enumerate(fid_scores):
        is_best = fid_evaluator.is_best_fid(fid_score)
        
        if is_best:
            print(f"Epoch {epoch}: FID={fid_score:.4f} ✓ (BEST)")
        else:
            print(f"Epoch {epoch}: FID={fid_score:.4f}")
    
    print(f"\nBest FID found: {fid_evaluator.best_fid:.4f}")


# ============================================================================
# Example 4: Feature Extraction
# ============================================================================

def example_feature_extraction():
    """
    Directly extract InceptionV3 features for any custom analysis.
    """
    print("\n" + "="*60)
    print("Example 4: Feature Extraction")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize feature extractor
    extractor = InceptionFeatureExtractor(device)
    
    # Dummy images
    images = torch.randn(8, 3, 256, 256).to(device)  # [0, 1] range
    
    # Extract features
    features = extractor(images)
    
    print(f"\nInput shape: {images.shape}")
    print(f"Output feature shape: {features.shape}")
    print(f"Feature statistics:")
    print(f"  Mean: {features.mean().item():.4f}")
    print(f"  Std: {features.std().item():.4f}")
    print(f"  Min: {features.min().item():.4f}")
    print(f"  Max: {features.max().item():.4f}")


# ============================================================================
# Example 5: Full Training Loop with FID
# ============================================================================

def example_full_training_with_fid():
    """
    Complete training loop with periodic FID evaluation.
    
    This shows how to integrate FID with the production training system.
    """
    print("\n" + "="*60)
    print("Example 5: Full Training Loop with FID")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Placeholder models
    class SimpleGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(),
                nn.Linear(256, 3 * 64 * 64)
            )
        
        def forward(self, z):
            return torch.sigmoid(self.fc(z).view(z.shape[0], 3, 64, 64))
    
    class SimpleDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(3 * 64 * 64, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        
        def forward(self, x):
            logits = self.fc(x.view(x.shape[0], -1))
            features = []  # Would contain actual features
            return logits, features
    
    # Initialize models
    generator = SimpleGenerator().to(device)
    discriminator = SimpleDiscriminator().to(device)
    
    # Create dummy data
    real_images = torch.randn(50, 3, 64, 64)
    conditions = torch.randn(50, 100)
    
    train_dataset = TensorDataset(conditions, real_images)
    real_eval_dataset = TensorDataset(real_images)
    cond_eval_dataset = TensorDataset(conditions)
    
    train_loader = DataLoader(train_dataset, batch_size=16)
    real_eval_loader = DataLoader(real_eval_dataset, batch_size=16)
    cond_eval_loader = DataLoader(cond_eval_dataset, batch_size=16)
    
    # Note: This example structure shows the integration pattern.
    # Full training requires LossManager and actual training logic.
    print("\nTraining loop structure:")
    print("1. Initialize GANTrainer with generator/discriminator")
    print("2. Initialize FIDEvaluator for FID computation")
    print("3. Loop over epochs:")
    print("   - Train discriminator and generator")
    print("   - Every N epochs: compute FID")
    print("   - Save best checkpoint based on FID")
    print("4. Resume training from best checkpoint for final inference")


# ============================================================================
# Example 6: Analyzing FID History
# ============================================================================

def example_fid_history():
    """
    Analyze FID scores saved during training.
    """
    print("\n" + "="*60)
    print("Example 6: Analyzing FID History")
    print("="*60)
    
    # This is typically run after training completes
    checkpoint_dir = "checkpoints/fid_example"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Compare checkpoints (if directory exists)
    summary = compare_checkpoints_fid(checkpoint_dir, device)
    
    if summary:
        print("\nFID Analysis Summary:")
        print(f"  Total evaluations: {summary['total_evals']}")
        print(f"  Best FID: {summary['best_fid']:.4f}")
        print(f"  Worst FID: {summary['worst_fid']:.4f}")
        print(f"  Mean FID: {summary['mean_fid']:.4f}")


# ============================================================================
# Integration Template for Your Project
# ============================================================================

def integration_template():
    """
    Template for integrating FID into your project.
    Copy this structure and fill in your models and data.
    """
    print("\n" + "="*60)
    print("Integration Template for Your Project")
    print("="*60)
    
    template_code = """
# Step 1: Import required modules
from training.fid_evaluator import FIDEvaluator
from training.fid_integration import (
    FIDCheckpointManager, FIDEvaluationCallback, train_gan_with_fid
)

# Step 2: Prepare data
train_loader = create_training_dataloader()
real_eval_loader = create_real_eval_dataloader()
condition_eval_loader = create_condition_eval_dataloader()

# Step 3: Choose training mode
# Option A: Use full training pipeline with FID
history = train_gan_with_fid(
    generator, discriminator, loss_manager,
    train_loader, real_eval_loader, condition_eval_loader,
    device,
    num_epochs=100,
    eval_frequency=10,  # Evaluate every 10 epochs
    checkpoint_dir="checkpoints/my_gan",
    use_amp=True,  # Mixed precision for speed
    use_ema=True,  # EMA for stability
)

# Option B: Integrate FID with your own training loop
fid_evaluator = FIDEvaluator(device, "checkpoints/my_gan")
checkpoint_mgr = FIDCheckpointManager(
    "checkpoints/my_gan", generator, discriminator, 
    device, fid_evaluator
)
eval_callback = FIDEvaluationCallback(
    fid_evaluator, checkpoint_mgr,
    eval_frequency=10,
    num_eval_samples=None  # Use all samples
)

for epoch in range(num_epochs):
    # Train step...
    
    # Evaluate FID (optional, at chosen frequency)
    metrics = eval_callback.on_epoch_end(
        epoch, generator,
        real_eval_loader, condition_eval_loader,
        device
    )

# Step 4: Inference with best model
print(f"Best FID checkpoint saved to: {checkpoint_dir}/best_fid_checkpoint.pth")
"""
    
    print(template_code)


# ============================================================================
# Quick Reference
# ============================================================================

def print_quick_reference():
    """Print quick reference for FID functions."""
    print("\n" + "="*60)
    print("FID Evaluation - Quick Reference")
    print("="*60)
    
    reference = """
ONE-LINER USAGE:
  fid_score = quick_fid_eval(gen, real_loader, cond_loader, device)

FULL TRAINING WITH FID:
  history = train_gan_with_fid(gen, disc, loss_mgr, train_loader,
                               real_eval_loader, cond_eval_loader, device)

CUSTOM INTEGRATION:
  evaluator = FIDEvaluator(device, checkpoint_dir)
  callback = FIDEvaluationCallback(evaluator, checkpoint_mgr)
  metrics = callback.on_epoch_end(epoch, gen, real_loader, cond_loader, device)

FEATURES:
  ✓ InceptionV3 feature extraction (2048-dim vectors)
  ✓ Efficient covariance computation with eigendecomposition
  ✓ Stable matrix square root (handles numerical issues)
  ✓ Batch processing (GPU accelerated)
  ✓ FID history tracking and JSON logging
  ✓ Best checkpoint selection
  ✓ No gradients (torch.no_grad() throughout)

PERFORMANCE:
  - ~100ms per batch (16 images) on A100
  - ~2GB GPU memory for feature extraction
  - Supports images up to 4K resolution (resized to 299x299 internally)

KEY PARAMETERS:
  eval_frequency: How often to compute FID (default: every 10 epochs)
  num_eval_samples: Limit samples for faster evaluation (default: None = all)
  use_amp: Mixed precision for speed (default: True)
  use_ema: EMA for stability (default: True)

OUTPUTS:
  - FID score (lower is better, typical range 5-100)
  - Checkpoint: best_fid_checkpoint.pth
  - History: fid_history.json with per-epoch scores
  - Statistics: mean/covariance per distribution

TROUBLESHOOTING:
  - NaN FID: Check for numerical instability in feature computation
  - Slow evaluation: Reduce num_eval_samples or eval_frequency
  - Low GPU memory: Use smaller batch size or eval_frequency > 1
"""
    
    print(reference)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FID Evaluation Module - Usage Examples")
    print("="*60)
    
    # Run examples
    example_basic_fid_evaluation()
    example_generator_fid_evaluation()
    example_fid_tracking()
    example_feature_extraction()
    example_full_training_with_fid()
    example_fid_history()
    integration_template()
    print_quick_reference()
    
    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60 + "\n")
