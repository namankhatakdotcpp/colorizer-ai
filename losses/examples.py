"""
Example usage and documentation for the unified LossManager.

Shows how to integrate LossManager into training pipeline.
"""

import torch
from losses import LossManager, create_loss_manager


def example_basic_usage():
    """Basic usage example showing generator and discriminator losses."""
    
    print("=" * 80)
    print("EXAMPLE 1: Basic LossManager Usage")
    print("=" * 80)
    
    # Initialize loss manager
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_manager = create_loss_manager(
        device=device,
        lambda_adversarial=1.0,
        lambda_l1=50.0,
        lambda_perceptual=10.0,
        lambda_feature_matching=10.0,
        lambda_histogram=5.0,
    )
    
    # Dummy inputs (B=2, H=256, W=256)
    generated = torch.randn(2, 3, 256, 256, device=device) * 0.5
    target = torch.randn(2, 3, 256, 256, device=device) * 0.5
    
    # Discriminator outputs (multi-scale: 3 scales)
    disc_fake_logits = [
        torch.randn(2, 1, 64, 64, device=device),   # Scale 1
        torch.randn(2, 1, 32, 32, device=device),   # Scale 2
        torch.randn(2, 1, 16, 16, device=device),   # Scale 3
    ]
    
    disc_real_logits = [
        torch.randn(2, 1, 64, 64, device=device),
        torch.randn(2, 1, 32, 32, device=device),
        torch.randn(2, 1, 16, 16, device=device),
    ]
    
    # Discriminator intermediate features (multi-scale, 4 layers per scale)
    fake_features = [
        [torch.randn(2, 64, 128, 128, device=device) for _ in range(4)],   # Scale 1
        [torch.randn(2, 128, 64, 64, device=device) for _ in range(4)],    # Scale 2
        [torch.randn(2, 256, 32, 32, device=device) for _ in range(4)],    # Scale 3
    ]
    
    real_features = [
        [torch.randn(2, 64, 128, 128, device=device) for _ in range(4)],
        [torch.randn(2, 128, 64, 64, device=device) for _ in range(4)],
        [torch.randn(2, 256, 32, 32, device=device) for _ in range(4)],
    ]
    
    # ===== GENERATOR LOSS =====
    loss_g, loss_dict_g = loss_manager(
        generated=generated,
        target=target,
        disc_fake_logits=disc_fake_logits,
        disc_real_logits=disc_real_logits,
        fake_features=fake_features,
        real_features=real_features,
        mode="generator",
    )
    
    print(f"\nGenerator Loss: {loss_g.item():.4f}")
    print("Loss Components:")
    for key, value in loss_dict_g.items():
        print(f"  {key:30s}: {value:8.4f}")
    
    # ===== DISCRIMINATOR LOSS =====
    real_labels = torch.full_like(disc_real_logits[0], 0.9)  # Label smoothing
    fake_labels = torch.full_like(disc_fake_logits[0], 0.1)
    
    loss_d, loss_dict_d = loss_manager(
        generated=generated,
        target=target,
        disc_fake_logits=disc_fake_logits,
        disc_real_logits=disc_real_logits,
        real_labels=real_labels,
        fake_labels=fake_labels,
        mode="discriminator",
    )
    
    print(f"\nDiscriminator Loss: {loss_d.item():.4f}")
    print("Loss Components:")
    for key, value in loss_dict_d.items():
        print(f"  {key:30s}: {value:8.4f}")


def example_custom_weights():
    """Example showing custom loss weights."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Loss Weights")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create with different weights (emphasize L1 for sharp outputs)
    loss_manager = LossManager(
        device=device,
        lambda_adversarial=0.5,      # Reduce adversarial for stability
        lambda_l1=100.0,              # Increase L1 for detail preservation
        lambda_perceptual=5.0,        # Reduce perceptual
        lambda_feature_matching=20.0,  # Increase FM for better structure
        lambda_histogram=10.0,         # Increase histogram for color accuracy
    )
    
    print("\nConfigured with weights:")
    print(f"  Adversarial:       {loss_manager.lambda_adversarial}")
    print(f"  L1:                {loss_manager.lambda_l1}")
    print(f"  Perceptual:        {loss_manager.lambda_perceptual}")
    print(f"  Feature Matching:  {loss_manager.lambda_feature_matching}")
    print(f"  Histogram:         {loss_manager.lambda_histogram}")


def example_training_integration():
    """Example showing integration into training loop."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Training Loop Integration")
    print("=" * 80)
    
    code = '''
# In your training loop:

from losses import create_loss_manager

# Initialize once
loss_manager = create_loss_manager(device=device)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4)

for epoch in range(num_epochs):
    for batch_idx, (condition, target) in enumerate(dataloader):
        batch_size = condition.size(0)
        real_labels = torch.full((batch_size,), 0.9, device=device)
        fake_labels = torch.full((batch_size,), 0.1, device=device)
        
        # ===== DISCRIMINATOR STEP =====
        optimizer_d.zero_grad()
        
        # Generate fake images
        with torch.no_grad():
            fake_images = generator(condition)
        
        # Get discriminator outputs
        disc_fake_logits, disc_fake_features = discriminator(
            torch.cat([condition, fake_images], dim=1)
        )
        disc_real_logits, disc_real_features = discriminator(
            torch.cat([condition, target], dim=1)
        )
        
        # Compute D loss
        loss_d, loss_dict_d = loss_manager(
            generated=None,
            target=None,
            disc_fake_logits=disc_fake_logits,
            disc_real_logits=disc_real_logits,
            real_labels=real_labels,
            fake_labels=fake_labels,
            mode="discriminator",
        )
        
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.5)
        optimizer_d.step()
        
        # ===== GENERATOR STEP =====
        optimizer_g.zero_grad()
        
        fake_images = generator(condition)
        disc_fake_logits, disc_fake_features = discriminator(
            torch.cat([condition, fake_images], dim=1)
        )
        disc_real_logits, disc_real_features = discriminator(
            torch.cat([condition, target], dim=1)
        )
        
        # Compute G loss
        loss_g, loss_dict_g = loss_manager(
            generated=fake_images,
            target=target,
            disc_fake_logits=disc_fake_logits,
            disc_real_logits=disc_real_logits,
            fake_features=disc_fake_features,
            real_features=disc_real_features,
            mode="generator",
        )
        
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)
        optimizer_g.step()
        
        # Log losses every N batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}: "
                  f"D_loss={loss_dict_d['loss_d_total']:.4f} "
                  f"G_loss={loss_dict_g['loss_g_total']:.4f}")
            
            # Log individual components
            for key, value in loss_dict_g.items():
                wandb.log({f"train/{key}": value})
'''
    
    print("\nExample training loop:")
    print(code)


def example_loss_components():
    """Document what each loss does."""
    
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Loss Components Explanation")
    print("=" * 80)
    
    docs = """
1. ADVERSARIAL LOSS (lambda=1.0)
   ├─ Purpose: Fool discriminator into classifying generated images as real
   ├─ Formula: -log(D(G(z)))
   ├─ For D: Maximize log(D(x)) + log(1-D(G(z)))
   └─ Typical weight: 1.0 (prevents D from dominating via TTUR)

2. L1 RECONSTRUCTION LOSS (lambda=50.0)
   ├─ Purpose: Preserve overall image structure and content
   ├─ Formula: ||G(z) - x||_1
   ├─ Per-pixel penalty encourages sharp, detailed outputs
   └─ Typical weight: 50.0 (strong content preservation)

3. PERCEPTUAL LOSS (lambda=10.0)
   ├─ Purpose: Match high-level feature representations (VGG16)
   ├─ Layers: relu4_3 (catches semantic content)
   ├─ Formula: ||VGG(G(z)) - VGG(x)||_1
   └─ Typical weight: 10.0 (ensures perceptual quality)

4. FEATURE MATCHING LOSS (lambda=10.0)
   ├─ Purpose: Match discriminator intermediate features
   ├─ Flattens: 3 scales × 4 layers = 12 feature pairs
   ├─ Formula: Σ ||D_feat(G(z)) - D_feat(x)||_1
   ├─ Steadies training by providing gradient signal from D
   └─ Typical weight: 10.0 (strong structural guidance)

5. HISTOGRAM LOSS (lambda=5.0)
   ├─ Purpose: Align RGB color distributions
   ├─ Bins: 32 per channel
   ├─ Formula: Σ √(Σ(hist_G - hist_x)²)
   ├─ Prevents color artifacts and shifts
   └─ Typical weight: 5.0 (fine-tuning color accuracy)

TOTAL GENERATOR LOSS:
  L_G = λ_adv * loss_adv
      + λ_l1 * loss_l1
      + λ_percep * loss_percep
      + λ_fm * loss_fm
      + λ_hist * loss_hist

TOTAL DISCRIMINATOR LOSS:
  L_D = BCE(D(x), 1) + BCE(D(G(z)), 0)
      with label smoothing: real_label=0.9, fake_label=0.1
"""
    
    print(docs)


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_custom_weights()
    example_training_integration()
    example_loss_components()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
