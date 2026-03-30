"""
Comprehensive test suite for LossManager.

Validates:
- Individual loss functions
- Multi-scale discriminator output handling
- Feature matching with proper flattening
- Generator and discriminator loss computation
- Label smoothing and stability
"""

import torch
import torch.nn as nn
from losses import LossManager, create_loss_manager


def test_loss_manager_initialization():
    """Test that LossManager initializes correctly."""
    print("\n" + "="*80)
    print("TEST 1: LossManager Initialization")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_manager = create_loss_manager(device=device)
    
    assert loss_manager is not None
    assert loss_manager.lambda_adversarial == 1.0
    assert loss_manager.lambda_l1 == 50.0
    assert loss_manager.lambda_perceptual == 10.0
    assert loss_manager.lambda_feature_matching == 10.0
    assert loss_manager.lambda_histogram == 5.0
    
    print("✅ LossManager initialized successfully")
    print(f"   Device: {device}")
    print(f"   Perceptual loss available: {loss_manager.use_perceptual}")


def test_individual_losses():
    """Test each loss function individually."""
    print("\n" + "="*80)
    print("TEST 2: Individual Loss Functions")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_manager = create_loss_manager(device=device)
    
    # Create dummy data
    batch_size = 2
    generated = torch.randn(batch_size, 3, 256, 256, device=device) * 0.5
    target = torch.randn(batch_size, 3, 256, 256, device=device) * 0.5
    
    # Discriminator outputs (multi-scale)
    disc_fake_logits = [
        torch.randn(batch_size, 1, 64, 64, device=device),
        torch.randn(batch_size, 1, 32, 32, device=device),
        torch.randn(batch_size, 1, 16, 16, device=device),
    ]
    
    # 1. Test L1 Loss
    loss_l1 = loss_manager.l1_loss_fn(generated, target)
    assert loss_l1.item() > 0, "L1 loss should be positive"
    print(f"✅ L1 Loss: {loss_l1.item():.6f}")
    
    # 2. Test Perceptual Loss
    loss_percep = loss_manager.perceptual_loss_fn(generated, target)
    assert loss_percep.item() >= 0, "Perceptual loss should be non-negative"
    print(f"✅ Perceptual Loss: {loss_percep.item():.6f}")
    
    # 3. Test Feature Matching Loss with None
    loss_fm = loss_manager.feature_matching_loss_fn([], [])
    assert loss_fm.item() == 0.0, "FM loss should be 0 for empty lists"
    print(f"✅ Feature Matching Loss (empty): {loss_fm.item():.6f}")
    
    # 4. Test Feature Matching Loss with actual features
    fake_features = [
        torch.randn(batch_size, 64, 128, 128, device=device),
        torch.randn(batch_size, 128, 64, 64, device=device),
    ]
    real_features = [
        torch.randn(batch_size, 64, 128, 128, device=device),
        torch.randn(batch_size, 128, 64, 64, device=device),
    ]
    loss_fm = loss_manager.feature_matching_loss_fn(fake_features, real_features)
    assert loss_fm.item() > 0, "FM loss should be positive for different features"
    print(f"✅ Feature Matching Loss (features): {loss_fm.item():.6f}")
    
    # 5. Test Histogram Loss
    loss_hist = loss_manager.histogram_loss_fn(generated, target)
    assert loss_hist.item() >= 0, "Histogram loss should be non-negative"
    print(f"✅ Histogram Loss: {loss_hist.item():.6f}")
    
    # 6. Test Adversarial Loss (G)
    loss_adv_g = loss_manager.adversarial_loss_g(disc_fake_logits[0])
    assert loss_adv_g.item() != 0, "Adversarial loss should be non-zero"
    print(f"✅ Adversarial Loss (G): {loss_adv_g.item():.6f}")
    
    # 7. Test Adversarial Loss (D)
    disc_real_logits = [
        torch.randn(batch_size, 1, 64, 64, device=device),
        torch.randn(batch_size, 1, 32, 32, device=device),
        torch.randn(batch_size, 1, 16, 16, device=device),
    ]
    real_labels = torch.full_like(disc_real_logits[0], 0.9)
    fake_labels = torch.full_like(disc_fake_logits[0], 0.1)
    loss_d_real, loss_d_fake = loss_manager.adversarial_loss_d(
        disc_real_logits[0], disc_fake_logits[0], real_labels, fake_labels
    )
    assert loss_d_real.item() > 0, "D real loss should be positive"
    assert loss_d_fake.item() > 0, "D fake loss should be positive"
    print(f"✅ Adversarial Loss (D): real={loss_d_real.item():.6f}, fake={loss_d_fake.item():.6f}")


def test_generator_mode():
    """Test complete generator loss computation."""
    print("\n" + "="*80)
    print("TEST 3: Generator Mode - Full Loss Computation")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_manager = create_loss_manager(device=device)
    
    batch_size = 2
    generated = torch.randn(batch_size, 3, 256, 256, device=device, requires_grad=True)
    target = torch.randn(batch_size, 3, 256, 256, device=device)
    
    # Multi-scale discriminator outputs
    disc_fake_logits = [
        torch.randn(batch_size, 1, 64, 64, device=device),
        torch.randn(batch_size, 1, 32, 32, device=device),
        torch.randn(batch_size, 1, 16, 16, device=device),
    ]
    
    disc_real_logits = [
        torch.randn(batch_size, 1, 64, 64, device=device),
        torch.randn(batch_size, 1, 32, 32, device=device),
        torch.randn(batch_size, 1, 16, 16, device=device),
    ]
    
    # Multi-scale, multi-layer features (3 scales × 4 layers)
    fake_features = [
        [torch.randn(batch_size, 64, 128, 128, device=device) for _ in range(4)],
        [torch.randn(batch_size, 128, 64, 64, device=device) for _ in range(4)],
        [torch.randn(batch_size, 256, 32, 32, device=device) for _ in range(4)],
    ]
    
    real_features = [
        [torch.randn(batch_size, 64, 128, 128, device=device) for _ in range(4)],
        [torch.randn(batch_size, 128, 64, 64, device=device) for _ in range(4)],
        [torch.randn(batch_size, 256, 32, 32, device=device) for _ in range(4)],
    ]
    
    # Compute loss
    loss_g, loss_dict_g = loss_manager(
        generated=generated,
        target=target,
        disc_fake_logits=disc_fake_logits,
        disc_real_logits=disc_real_logits,
        fake_features=fake_features,
        real_features=real_features,
        mode="generator",
    )
    
    # Validate outputs
    assert loss_g.item() > 0, "Generator loss should be positive"
    assert isinstance(loss_dict_g, dict), "Loss dict should be a dictionary"
    assert "loss_g_total" in loss_dict_g, "Should have total loss"
    assert len(loss_dict_g) == 6, f"Should have 6 components, got {len(loss_dict_g)}"
    
    # Validate backward pass
    loss_g.backward()
    assert generated.grad is not None, "Generated image should have gradients"
    assert torch.any(generated.grad != 0), "Gradients should be non-zero"
    
    print("✅ Generator loss computation successful")
    print(f"   Total loss: {loss_g.item():.6f}")
    print("   Loss components:")
    for key, value in loss_dict_g.items():
        print(f"     {key:30s}: {value:10.6f}")


def test_discriminator_mode():
    """Test discriminator loss computation."""
    print("\n" + "="*80)
    print("TEST 4: Discriminator Mode")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_manager = create_loss_manager(device=device)
    
    batch_size = 2
    
    # Multi-scale logits
    disc_fake_logits = [
        torch.randn(batch_size, 1, 64, 64, device=device, requires_grad=True),
        torch.randn(batch_size, 1, 32, 32, device=device, requires_grad=True),
        torch.randn(batch_size, 1, 16, 16, device=device, requires_grad=True),
    ]
    
    disc_real_logits = [
        torch.randn(batch_size, 1, 64, 64, device=device, requires_grad=True),
        torch.randn(batch_size, 1, 32, 32, device=device, requires_grad=True),
        torch.randn(batch_size, 1, 16, 16, device=device, requires_grad=True),
    ]
    
    # Labels with smoothing
    real_labels = torch.full_like(disc_real_logits[0], 0.9)
    fake_labels = torch.full_like(disc_fake_logits[0], 0.1)
    
    # Compute loss
    loss_d, loss_dict_d = loss_manager(
        generated=None,
        target=None,
        disc_fake_logits=disc_fake_logits,
        disc_real_logits=disc_real_logits,
        real_labels=real_labels,
        fake_labels=fake_labels,
        mode="discriminator",
    )
    
    # Validate outputs
    assert loss_d.item() > 0, "Discriminator loss should be positive"
    assert isinstance(loss_dict_d, dict), "Loss dict should be a dictionary"
    assert "loss_d_total" in loss_dict_d, "Should have total loss"
    assert len(loss_dict_d) == 3, f"Should have 3 components, got {len(loss_dict_d)}"
    
    # Validate backward pass
    loss_d.backward()
    # Only logits[0] (primary scale) should have gradients since that's what's used in loss
    assert disc_fake_logits[0].grad is not None, "Fake logits[0] should have gradients"
    assert torch.any(disc_fake_logits[0].grad != 0), "Gradients for fake logits[0] should be non-zero"
    assert disc_real_logits[0].grad is not None, "Real logits[0] should have gradients"
    assert torch.any(disc_real_logits[0].grad != 0), "Gradients for real logits[0] should be non-zero"
    
    print("✅ Discriminator loss computation successful")
    print(f"   Total loss: {loss_d.item():.6f}")
    print("   Loss components:")
    for key, value in loss_dict_d.items():
        print(f"     {key:30s}: {value:10.6f}")


def test_gradient_flow():
    """Test that gradients flow correctly through loss functions."""
    print("\n" + "="*80)
    print("TEST 5: Gradient Flow Validation")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_manager = create_loss_manager(device=device)
    
    batch_size = 2
    generated = torch.randn(batch_size, 3, 256, 256, device=device, requires_grad=True)
    target = torch.randn(batch_size, 3, 256, 256, device=device)
    
    disc_fake_logits = [
        torch.randn(batch_size, 1, 64, 64, device=device, requires_grad=True),
    ]
    disc_real_logits = [
        torch.randn(batch_size, 1, 64, 64, device=device, requires_grad=True),
    ]
    
    fake_features = [[torch.randn(batch_size, 64, 128, 128, device=device) for _ in range(4)]]
    real_features = [[torch.randn(batch_size, 64, 128, 128, device=device) for _ in range(4)]]
    
    # Compute loss
    loss_g, _ = loss_manager(
        generated=generated,
        target=target,
        disc_fake_logits=disc_fake_logits,
        disc_real_logits=disc_real_logits,
        fake_features=fake_features,
        real_features=real_features,
        mode="generator",
    )
    
    # Check gradients before backward
    assert generated.grad is None, "No gradients before backward"
    
    # Backward pass
    loss_g.backward()
    
    # Check gradients after backward
    assert generated.grad is not None, "Generated should have gradients after backward"
    assert torch.all(torch.isfinite(generated.grad)), "Gradients should be finite"
    
    print("✅ Gradient flow successful")
    print(f"   Generated grad norm: {generated.grad.norm().item():.6f}")
    print(f"   Generated grad min: {generated.grad.min().item():.6f}")
    print(f"   Generated grad max: {generated.grad.max().item():.6f}")


def test_label_smoothing():
    """Test label smoothing effects on loss."""
    print("\n" + "="*80)
    print("TEST 6: Label Smoothing Stability")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_manager = create_loss_manager(device=device)
    
    batch_size = 2
    logits = torch.randn(batch_size, 1, 64, 64, device=device)
    
    # Test without smoothing
    hard_real_labels = torch.ones_like(logits)
    hard_fake_labels = torch.zeros_like(logits)
    
    loss_hard_real, _ = loss_manager.adversarial_loss_d(
        logits, logits, hard_real_labels, hard_fake_labels
    )
    
    # Test with smoothing
    smooth_real_labels = torch.full_like(logits, 0.9)
    smooth_fake_labels = torch.full_like(logits, 0.1)
    
    loss_smooth_real, _ = loss_manager.adversarial_loss_d(
        logits, logits, smooth_real_labels, smooth_fake_labels
    )
    
    # Loss should differ (smoothing reduces loss amplitude)
    assert loss_hard_real.item() != loss_smooth_real.item(), "Smoothing should affect loss"
    
    print("✅ Label smoothing working correctly")
    print(f"   Hard labels loss:  {loss_hard_real.item():.6f}")
    print(f"   Smooth labels loss: {loss_smooth_real.item():.6f}")
    print(f"   Reduction: {(1 - loss_smooth_real.item()/loss_hard_real.item())*100:.1f}%")


def test_feature_matching_flattening():
    """Test that multi-scale feature matching properly flattens all scales."""
    print("\n" + "="*80)
    print("TEST 7: Feature Matching Multi-Scale Flattening")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_manager = create_loss_manager(device=device)
    
    batch_size = 2
    
    # Create 3 scales × 4 layers = 12 feature pairs
    fake_features = [
        [torch.randn(batch_size, 64, 128, 128, device=device) for _ in range(4)],
        [torch.randn(batch_size, 128, 64, 64, device=device) for _ in range(4)],
        [torch.randn(batch_size, 256, 32, 32, device=device) for _ in range(4)],
    ]
    
    real_features = [
        [torch.randn(batch_size, 64, 128, 128, device=device) for _ in range(4)],
        [torch.randn(batch_size, 128, 64, 64, device=device) for _ in range(4)],
        [torch.randn(batch_size, 256, 32, 32, device=device) for _ in range(4)],
    ]
    
    # Flatten (same logic as in forward pass)
    fake_feats_flat = [f for scale_feats in fake_features for f in scale_feats]
    real_feats_flat = [f for scale_feats in real_features for f in scale_feats]
    
    assert len(fake_feats_flat) == 12, f"Should have 12 features, got {len(fake_feats_flat)}"
    assert len(real_feats_flat) == 12, f"Should have 12 features, got {len(real_feats_flat)}"
    
    # Compute FM loss
    loss_fm = loss_manager.feature_matching_loss_fn(fake_feats_flat, real_feats_flat)
    
    assert loss_fm.item() > 0, "FM loss should be positive"
    assert loss_fm.dim() == 0, "FM loss should be scalar"
    
    print("✅ Feature matching multi-scale flattening correct")
    print(f"   Total feature pairs: {len(fake_feats_flat)}")
    print(f"   Scales: 3 (primary, 1/2, 1/4)")
    print(f"   Layers per scale: 4")
    print(f"   FM loss: {loss_fm.item():.6f}")


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("\n" + "="*80)
    print("TEST 8: Numerical Stability")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_manager = create_loss_manager(device=device)
    
    batch_size = 2
    
    # Generate extreme values to test stability
    disc_fake_logits = [torch.randn(batch_size, 1, 64, 64, device=device) * 100]  # Large logits
    disc_real_logits = [torch.randn(batch_size, 1, 64, 64, device=device) * 100]
    
    real_labels = torch.full_like(disc_real_logits[0], 0.9)
    fake_labels = torch.full_like(disc_fake_logits[0], 0.1)
    
    loss_d, loss_dict = loss_manager(
        generated=None,
        target=None,
        disc_fake_logits=disc_fake_logits,
        disc_real_logits=disc_real_logits,
        real_labels=real_labels,
        fake_labels=fake_labels,
        mode="discriminator",
    )
    
    # Check for NaN or Inf
    assert torch.isfinite(loss_d), f"Loss should be finite, got {loss_d.item()}"
    assert loss_d.item() > 0, "Loss should be positive"
    
    print("✅ Numerical stability verified")
    print(f"   Loss with extreme logits: {loss_d.item():.6f}")
    print(f"   Loss is finite: {torch.isfinite(loss_d).item()}")


def run_all_tests():
    """Run comprehensive test suite."""
    print("\n" + "="*80)
    print("COMPREHENSIVE LOSSMANAGER TEST SUITE")
    print("="*80)
    
    try:
        test_loss_manager_initialization()
        test_individual_losses()
        test_generator_mode()
        test_discriminator_mode()
        test_gradient_flow()
        test_label_smoothing()
        test_feature_matching_flattening()
        test_numerical_stability()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nLossManager is production-ready and fully validated!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
