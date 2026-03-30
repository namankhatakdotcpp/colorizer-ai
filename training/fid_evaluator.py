"""
FID (Fréchet Inception Distance) Evaluation Module
- InceptionV3-based feature extraction
- Efficient covariance and FID calculation
- GPU-accelerated batch processing
- Integration with GANTrainer for checkpoint selection
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from torchvision.transforms import transforms
import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class InceptionFeatureExtractor(nn.Module):
    """
    Extract features from InceptionV3 for FID calculation.
    
    Args:
        device (torch.device): GPU/CPU device
        normalize_input (bool): If True, normalize to [-1, 1] range (default: True)
    
    Features:
    - Returns (batch_size, 2048) feature vectors
    - Frozen weights (no gradients)
    - Efficient batch processing
    """
    
    def __init__(self, device: torch.device, normalize_input: bool = True):
        super().__init__()
        self.device = device
        self.normalize_input = normalize_input
        
        # Load pretrained InceptionV3
        inception = inception_v3(pretrained=True, transform_input=False)
        
        # Remove classification layer, keep feature extraction
        self.model = nn.Sequential(*list(inception.children())[:-1])
        self.model.to(device)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Storage for dimensions
        self.feature_dim = 2048
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor (batch_size, 3, H, W) with values in [0, 1] or [-1, 1]
        
        Returns:
            features: (batch_size, 2048) feature vectors
        """
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Handle image scaling: InceptionV3 expects [0, 1] → [-1, 1]
        if self.normalize_input and x.min() >= 0:
            x = 2 * x - 1
        
        # Resize to InceptionV3 input size (299, 299)
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = torch.nn.functional.interpolate(
                x, size=(299, 299), mode='bilinear', align_corners=False
            )
        
        # Extract features
        features = self.model(x)
        
        # Flatten: (batch_size, 2048, 1, 1) → (batch_size, 2048)
        features = features.squeeze(-1).squeeze(-1)
        
        return features


class FIDCalculator:
    """
    Calculate Fréchet Inception Distance (FID) score.
    
    FID measures the distance between distributions of generated and real images
    using statistics computed from InceptionV3 features.
    
    Formula:
    FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2√(Σ_real · Σ_gen))
    
    Args:
        device (torch.device): GPU/CPU device
        eps (float): Small epsilon for numerical stability (default: 1e-6)
    """
    
    def __init__(self, device: torch.device, eps: float = 1e-6):
        self.device = device
        self.eps = eps
        self.feature_extractor = InceptionFeatureExtractor(device)
    
    def compute_statistics(
        self, 
        dataloader: DataLoader,
        num_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance of features.
        
        Args:
            dataloader: DataLoader yielding image tensors (batch_size, 3, H, W)
            num_samples: Maximum samples to process (default: None = all)
        
        Returns:
            mean: (2048,) mean feature vector
            cov: (2048, 2048) covariance matrix
        """
        features_list = []
        
        for i, batch in enumerate(dataloader):
            # Handle both single tensor and (x, y) tuple formats
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch
            
            # Extract features
            batch_features = self.feature_extractor(images)
            features_list.append(batch_features.cpu().numpy())
            
            # Track progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {(i + 1) * len(images)} images for FID")
            
            # Stop if reached max samples
            if num_samples is not None:
                total_samples = sum(f.shape[0] for f in features_list)
                if total_samples >= num_samples:
                    break
        
        # Concatenate all features
        all_features = np.concatenate(features_list, axis=0)
        
        # Trim to exact number of samples if needed
        if num_samples is not None:
            all_features = all_features[:num_samples]
        
        # Compute statistics
        mu = np.mean(all_features, axis=0)
        sigma = np.cov(all_features.T)
        
        logger.info(f"Computed statistics from {all_features.shape[0]} samples")
        logger.info(f"Mean: {mu.shape}, Covariance: {sigma.shape}")
        
        return mu, sigma
    
    @staticmethod
    def _matrix_sqrt(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        """
        Compute matrix square root using eigendecomposition.
        
        Handles numerical instability by clipping small eigenvalues.
        
        Args:
            matrix: Square matrix
            epsilon: Minimum eigenvalue threshold
        
        Returns:
            Matrix square root
        """
        eigvalues, eigvectors = np.linalg.eigh(matrix)
        
        # Clip negative eigenvalues to epsilon (numerical stability)
        eigvalues = np.maximum(eigvalues, epsilon)
        
        # Reconstitute matrix sqrt
        sqrt_matrix = eigvectors @ np.diag(np.sqrt(eigvalues)) @ eigvectors.T
        
        return np.real(sqrt_matrix)
    
    def compute_fid(
        self,
        mu_real: np.ndarray,
        sigma_real: np.ndarray,
        mu_gen: np.ndarray,
        sigma_gen: np.ndarray,
    ) -> float:
        """
        Compute FID score from feature statistics.
        
        Args:
            mu_real: Mean of real features (2048,)
            sigma_real: Covariance of real features (2048, 2048)
            mu_gen: Mean of generated features (2048,)
            sigma_gen: Covariance of generated features (2048, 2048)
        
        Returns:
            fid_score: Float FID score (lower is better)
        """
        # Mean difference
        diff = mu_real - mu_gen
        mean_term = np.sum(diff ** 2)
        
        # Covariance terms
        sqrt_covgen = self._matrix_sqrt(sigma_gen)
        product = sqrt_covgen @ sigma_real @ sqrt_covgen
        sqrt_product = self._matrix_sqrt(product)
        
        cov_term = np.trace(sigma_real + sigma_gen - 2 * sqrt_product)
        
        # FID score
        fid = mean_term + cov_term
        
        return float(np.sqrt(np.maximum(fid, 0)))
    
    def evaluate_score(
        self,
        real_dataloader: DataLoader,
        gen_dataloader: DataLoader,
        num_samples: Optional[int] = None,
        return_stats: bool = False,
    ) -> Dict[str, float]:
        """
        Compute FID score between real and generated images.
        
        Args:
            real_dataloader: DataLoader for real images
            gen_dataloader: DataLoader for generated images
            num_samples: Maximum samples per distribution (default: None = all)
            return_stats: If True, return mean/cov separately
        
        Returns:
            Dictionary with 'fid' score and optionally stats
        """
        logger.info("=" * 60)
        logger.info("Computing FID Score")
        logger.info("=" * 60)
        
        # Compute real statistics
        logger.info("Computing statistics for REAL images...")
        mu_real, sigma_real = self.compute_statistics(real_dataloader, num_samples)
        
        # Compute generated statistics
        logger.info("Computing statistics for GENERATED images...")
        mu_gen, sigma_gen = self.compute_statistics(gen_dataloader, num_samples)
        
        # Compute FID
        fid_score = self.compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)
        
        result = {"fid": fid_score}
        
        if return_stats:
            result.update({
                "mu_real_mean": float(np.mean(mu_real)),
                "mu_gen_mean": float(np.mean(mu_gen)),
                "sigma_real_trace": float(np.trace(sigma_real)),
                "sigma_gen_trace": float(np.trace(sigma_gen)),
            })
        
        logger.info(f"FID Score: {fid_score:.4f}")
        logger.info("=" * 60)
        
        return result


class FIDEvaluator:
    """
    Full evaluation pipeline for FID calculation.
    
    Integrates feature extraction, statistics computation, and FID scoring.
    Designed for use in training loops with callbacks.
    
    Args:
        device (torch.device): GPU/CPU device
        checkpoint_dir (Path): Directory to save best models
    """
    
    def __init__(
        self,
        device: torch.device,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.fid_calculator = FIDCalculator(device)
        self.best_fid = float('inf')
    
    def evaluate(
        self,
        real_dataloader: DataLoader,
        gen_dataloader: DataLoader,
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run full FID evaluation.
        
        Args:
            real_dataloader: DataLoader for real images
            gen_dataloader: DataLoader for generated images
            num_samples: Maximum samples per distribution
        
        Returns:
            Dictionary with FID metrics
        """
        return self.fid_calculator.evaluate_score(
            real_dataloader, gen_dataloader, num_samples, return_stats=True
        )
    
    def is_best_fid(self, fid_score: float) -> bool:
        """Track if this is the best FID score seen so far."""
        is_best = fid_score < self.best_fid
        if is_best:
            self.best_fid = fid_score
            logger.info(f"🏆 New best FID: {fid_score:.4f}")
        return is_best


# ============================================================================
# Batch Generation Helper
# ============================================================================

class GeneratedImageProxy:
    """
    Wrapper for generating images on-the-fly during evaluation.
    
    Used with DataLoader to generate images batch by batch without
    storing all generated images in memory.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        condition_dataloader: DataLoader,
        device: torch.device,
    ):
        self.generator = generator
        self.condition_dataloader = condition_dataloader
        self.device = device
        self.current_batch = None
        self.current_idx = 0
    
    def __iter__(self):
        """Iterate through generated images."""
        self.generator.eval()
        
        with torch.no_grad():
            for batch in self.condition_dataloader:
                # Extract condition (handle both tuple and tensor formats)
                if isinstance(batch, (tuple, list)):
                    condition = batch[0]
                else:
                    condition = batch
                
                condition = condition.to(self.device)
                
                # Generate images
                generated = self.generator(condition)
                
                # Ensure output is in [0, 1] range
                if generated.min() < 0:
                    generated = (generated + 1) / 2
                
                yield generated
    
    def __len__(self):
        return len(self.condition_dataloader)


# ============================================================================
# Integration Utilities
# ============================================================================

def create_fid_evaluator(
    device: torch.device,
    checkpoint_dir: Optional[str] = None,
) -> FIDEvaluator:
    """
    Factory function to create FIDEvaluator.
    
    Args:
        device: torch.device (cuda or cpu)
        checkpoint_dir: Directory for saving best models
    
    Returns:
        FIDEvaluator instance ready for use
    """
    return FIDEvaluator(device, checkpoint_dir)


def evaluate_gan_fid(
    generator: nn.Module,
    real_dataloader: DataLoader,
    condition_dataloader: DataLoader,
    device: torch.device,
    num_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Quick FID evaluation for a generator.
    
    Args:
        generator: Generator model
        real_dataloader: DataLoader for real images
        condition_dataloader: DataLoader for conditions (input to generator)
        device: torch.device
        num_samples: Max samples to use (None = all)
    
    Returns:
        FID metrics dictionary
    """
    evaluator = FIDEvaluator(device)
    
    # Create generated image dataloader
    gen_proxy = GeneratedImageProxy(generator, condition_dataloader, device)
    
    # Evaluate
    metrics = evaluator.evaluate(
        real_dataloader,
        gen_proxy,
        num_samples=num_samples,
    )
    
    return metrics
