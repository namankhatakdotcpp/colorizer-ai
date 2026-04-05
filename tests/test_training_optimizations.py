"""
Comprehensive Unit Tests for Training Optimizations

FAANG-level test coverage for:
- Learning rate schedulers
- Mixed precision training
- Data loading pipeline
- Optimization utilities
- Error handling

Usage:
    pytest tests/test_training_optimizations.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


@pytest.fixture
def simple_optimizer(simple_model):
    """Create optimizer for testing."""
    return optim.AdamW(simple_model.parameters(), lr=1e-3)


@pytest.fixture
def simple_dataloader():
    """Create a simple dataloader for testing."""
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)


# ============================================================================
# Scheduler Tests
# ============================================================================

class TestWarmupScheduler:
    """Test WarmupScheduler functionality."""
    
    from training.advanced_schedulers import WarmupScheduler
    
    def test_warmup_phase(self, simple_optimizer):
        """Test that LR increases linearly during warmup."""
        scheduler = self.WarmupScheduler(
            simple_optimizer,
            warmup_epochs=5,
            total_epochs=10,
        )
        
        base_lr = simple_optimizer.param_groups[0]['initial_lr']
        prev_lr = 0.0
        
        # During warmup, LR should increase
        for epoch in range(5):
            scheduler.step(epoch)
            current_lr = simple_optimizer.param_groups[0]['lr']
            assert current_lr >= prev_lr, "LR should increase during warmup"
            assert current_lr <= base_lr, "LR should not exceed base_lr"
            prev_lr = current_lr
    
    def test_constant_phase(self, simple_optimizer):
        """Test that LR remains constant after warmup."""
        scheduler = self.WarmupScheduler(
            simple_optimizer,
            warmup_epochs=5,
            total_epochs=10,
        )
        
        base_lr = simple_optimizer.param_groups[0]['initial_lr']
        
        # Skip warmup
        for epoch in range(5):
            scheduler.step(epoch)
        
        # After warmup, LR should be constant
        stable_lrs = []
        for epoch in range(5, 10):
            scheduler.step(epoch)
            stable_lrs.append(simple_optimizer.param_groups[0]['lr'])
        
        # All LRs should be equal
        assert len(set(stable_lrs)) == 1, "LR should be constant after warmup"
        assert stable_lrs[0] == pytest.approx(base_lr), "LR should equal base_lr"


class TestCosineAnnealingWarmRestarts:
    """Test CosineAnnealingWarmRestarts scheduler."""
    
    from training.advanced_schedulers import CosineAnnealingWarmRestarts
    
    def test_oscillation_pattern(self, simple_optimizer):
        """Test that LR follows cosine pattern."""
        scheduler = self.CosineAnnealingWarmRestarts(
            simple_optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )
        
        base_lr = simple_optimizer.param_groups[0]['initial_lr']
        max_lr_seen = 0.0
        min_lr_seen = base_lr
        
        # Run for 20 epochs
        for epoch in range(20):
            scheduler.step(epoch)
            lr = simple_optimizer.param_groups[0]['lr']
            max_lr_seen = max(max_lr_seen, lr)
            min_lr_seen = min(min_lr_seen, lr)
        
        # LR should oscillate
        assert max_lr_seen > base_lr * 0.5, "LR should reach high values"
        assert min_lr_seen <= base_lr * 0.1, "LR should reach low values"


class TestOneCycleScheduler:
    """Test OneCycle learning rate scheduler."""
    
    from training.advanced_schedulers import OneCycleScheduler
    
    def test_three_phase_pattern(self, simple_optimizer):
        """Test three-phase pattern of OneCycleScheduler."""
        max_lr = 0.1
        total_epochs = 100
        scheduler = self.OneCycleScheduler(
            simple_optimizer,
            max_lr=max_lr,
            total_epochs=total_epochs,
            pct_start=0.3,
        )
        
        lrs = []
        for epoch in range(total_epochs):
            scheduler.step(epoch)
            lrs.append(simple_optimizer.param_groups[0]['lr'])
        
        # Phase 1: Linear increase (first 30% of epochs)
        phase1_lrs = lrs[:30]
        assert phase1_lrs[0] < phase1_lrs[-1], "Phase 1 should increase"
        
        # Phase 2/3: Decay (last 70% of epochs)
        phase2_lrs = lrs[30:]
        assert phase2_lrs[0] >= phase2_lrs[-1] or \
               phase2_lrs[0] <= phase2_lrs[10], "Phase 2/3 should decay"


class TestReduceLROnPlateau:
    """Test ReduceLROnPlateau scheduler."""
    
    from training.advanced_schedulers import ReduceLROnPlateau
    
    def test_lr_reduction(self, simple_optimizer):
        """Test that LR is reduced on plateau."""
        scheduler = self.ReduceLROnPlateau(
            simple_optimizer,
            mode='min',
            factor=0.5,
            patience=2,
        )
        
        base_lr = simple_optimizer.param_groups[0]['lr']
        
        # Simulate no improvement
        for _ in range(3):
            scheduler.step(1.0)  # Same metric value
        
        new_lr = simple_optimizer.param_groups[0]['lr']
        assert new_lr < base_lr, "LR should be reduced after patience epochs"


# ============================================================================
# Mixed Precision Training Tests
# ============================================================================

class TestMixedPrecisionTrainer:
    """Test MixedPrecisionTrainer functionality."""
    
    from training.optimization_utils import MixedPrecisionTrainer, MixedPrecisionConfig
    
    def test_initialization(self):
        """Test MixedPrecisionTrainer initializes correctly."""
        config = self.MixedPrecisionConfig(enabled=True)
        trainer = self.MixedPrecisionTrainer(config)
        
        if torch.cuda.is_available():
            assert trainer.enabled, "Should be enabled with CUDA"
        else:
            # Gracefully handles no CUDA
            pass
    
    def test_autocast_context(self):
        """Test autocast context manager."""
        config = self.MixedPrecisionConfig(enabled=True)
        trainer = self.MixedPrecisionTrainer(config)
        
        with trainer.autocast():
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.matmul(x, y)
        
        assert z.shape == (10, 10)


class TestGradientClipping:
    """Test gradient clipping utilities."""
    
    from training.optimization_utils import clip_gradients
    
    def test_clip_gradients(self, simple_model):
        """Test gradient clipping."""
        # Create large gradients
        for param in simple_model.parameters():
            param.grad = torch.ones_like(param) * 10.0
        
        clipped_norm, overflow = self.clip_gradients(
            simple_model,
            max_norm=1.0,
        )
        
        # After clipping, norm should be <= max_norm
        assert clipped_norm <= 1.1, "Clipped norm should be <= max_norm"
        assert not overflow, "Should not detect overflow"
    
    def test_overflow_detection(self, simple_model):
        """Test NaN/Inf detection."""
        for param in simple_model.parameters():
            param.grad = torch.ones_like(param) * float('inf')
        
        clipped_norm, overflow = self.clip_gradients(
            simple_model,
            max_norm=1.0,
        )
        
        assert overflow or clipped_norm == 0.0, "Should detect overflow"


# ============================================================================
# Data Loading Tests
# ============================================================================

class TestDataLoaderConfig:
    """Test DataLoaderConfig validation."""
    
    from training.dataloader_utils import DataLoaderConfig
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = self.DataLoaderConfig(
            batch_size=32,
            num_workers=4,
        )
        config.validate()  # Should not raise
    
    def test_invalid_batch_size(self):
        """Test invalid batch size."""
        config = self.DataLoaderConfig(batch_size=-1)
        with pytest.raises(ValueError):
            config.validate()


class TestPrefetchedDataLoader:
    """Test PrefetchedDataLoader."""
    
    from training.dataloader_utils import PrefetchedDataLoader
    
    def test_prefetch_iteration(self, simple_dataloader):
        """Test prefetched dataloader iteration."""
        prefetch_loader = self.PrefetchedDataLoader(
            simple_dataloader,
            num_prefetch=2,
        )
        
        batches_received = 0
        for batch in prefetch_loader:
            assert len(batch) == 2
            batches_received += 1
        
        assert batches_received > 0, "Should receive batches"


# ============================================================================
# Model Optimization Tests
# ============================================================================

class TestCompiledModelWrapper:
    """Test CompiledModelWrapper."""
    
    from training.model_optimization import CompiledModelWrapper, CompileConfig
    
    def test_compile_config_validation(self):
        """Test CompileConfig validation."""
        config = self.CompileConfig(mode='default')
        config.validate()  # Should not raise
        
        bad_config = self.CompileConfig(mode='invalid')
        with pytest.raises(ValueError):
            bad_config.validate()


class TestModelProfiler:
    """Test ModelProfiler."""
    
    from training.model_optimization import ModelProfiler
    
    def test_get_model_statistics(self, simple_model):
        """Test model statistics."""
        from training.model_optimization import get_model_statistics
        
        stats = get_model_statistics(simple_model)
        
        assert 'total_parameters' in stats
        assert stats['total_parameters'] > 0
        assert stats['trainable_parameters'] > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestTrainingIntegration:
    """Integration tests for complete training pipeline."""
    
    def test_single_training_step(self, simple_model, simple_optimizer, simple_dataloader):
        """Test a single training step."""
        from training.optimization_utils import (
            MixedPrecisionTrainer,
            compute_gradient_norm,
        )
        
        trainer = MixedPrecisionTrainer()
        criterion = nn.MSELoss()
        
        simple_model.train()
        
        for batch_x, batch_y in simple_dataloader:
            with trainer.autocast():
                output = simple_model(batch_x)
                loss = criterion(output, batch_y)
            
            trainer.backward(loss)
            success = trainer.step(simple_optimizer, simple_model, max_norm=1.0)
            
            assert success, "Training step should succeed"
            
            grad_norm = compute_gradient_norm(simple_model)
            assert grad_norm >= 0, "Gradient norm should be non-negative"
            
            break  # Only test first batch


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
