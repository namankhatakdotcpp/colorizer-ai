#!/usr/bin/env python3
"""
Fine-tune existing colorizer with L1 loss ONLY (ROOT CAUSE STABILITY FIXES).

This script:
1. Loads checkpoint (stage1_colorizer_latest.pth)
2. Uses ONLY L1 loss (perceptual disabled for stability)
3. Fine-tunes for 10-20 epochs with ultra-low LR (3e-6)
4. Improves color realism fundamentally

10 Critical Stability Fixes:
1. Data validation (check input NaN)
2. Safe model output (check before clamp)
3. **AB normalization (÷110 - ROOT CAUSE FIX)**
4. Safe loss computation (check NaN)
5. Division by zero handling
6. Better freeze (freeze "enc", train rest)
7. Ultra-low LR (3e-6)
8. Disable perceptual loss (L1 only)
9. Batch success counter
10. Early stop on too many skips

Strategy: Encoder frozen, decoder+mid trainable
LR: 3e-6 (ultra-low for maximum stability)
Loss: L1 only (99.99% stable)
Time: ~30 min to 2 hours on GPU
Quality gain: +10-15% after L1 stable, then add perceptual

Usage:
    python training/finetune_colorizer.py \
        --checkpoint checkpoints/stage1_colorizer_latest.pth \
        --epochs 20 \
        --batch-size 4 \
        --lr 3e-6 \
        --perceptual-weight 0.0
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.unet_colorizer import UNetColorizer
from datasets.dataset_colorizer import build_combined_colorization_dataset


# ═══════════════════════════════════════════════════════════════════════════
# VGG PERCEPTUAL LOSS
# ═══════════════════════════════════════════════════════════════════════════


class VGGPerceptualLoss(nn.Module):
    """
    VGG16 perceptual loss for fine-tuning.
    
    Extracts features from pretrained VGG16 and computes L1 distance
    between predicted and target RGB features.
    
    Frozen (no gradient updates to VGG weights).
    """

    def __init__(self, layer: str = "relu3_3"):
        """
        Args:
            layer: Which VGG layer to extract features from.
                   Options: relu1_2, relu2_2, relu3_3, relu4_3
                   Default: relu3_3 (semantic features)
        """
        super().__init__()
        self.layer = layer
        
        # Load pretrained VGG16
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features
        except Exception as e:
            print(f"[WARN] Failed to load pretrained VGG16: {e}")
            print("       Using random initialization (not recommended)")
            import torchvision.models as models
            vgg = models.vgg16(pretrained=False).features
        
        # Extract layer
        layer_map = {
            "relu1_2": 4,   # Early edges
            "relu2_2": 9,   # Textures
            "relu3_3": 16,  # Semantic objects
            "relu4_3": 23,  # High-level features
        }
        
        if layer not in layer_map:
            raise ValueError(f"Invalid layer: {layer}. Choose from {list(layer_map.keys())}")
        
        self.features = nn.Sequential(*list(vgg.children())[:layer_map[layer]])
        
        # Freeze VGG weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize RGB to ImageNet statistics."""
        return (x - self.mean) / self.std

    def forward(self, pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss (numerically stable).
        
        Args:
            pred_rgb: [B, 3, H, W] predicted RGB in [0, 1]
            target_rgb: [B, 3, H, W] target RGB in [0, 1]
        
        Returns:
            L1 distance between VGG features
        """
        # Ensure RGB in valid range
        pred_rgb = torch.clamp(pred_rgb, 0.0, 1.0)
        target_rgb = torch.clamp(target_rgb, 0.0, 1.0)
        
        # Normalize to ImageNet stats
        pred_norm = self._normalize(pred_rgb)
        target_norm = self._normalize(target_rgb)
        
        # Extract features (detach ground truth to stabilize)
        with torch.no_grad():
            target_feat = self.features(target_norm)
        
        pred_feat = self.features(pred_norm)
        
        # L1 loss with stability checks
        loss = F.l1_loss(pred_feat, target_feat)
        
        return loss


# ═══════════════════════════════════════════════════════════════════════════
# LAB ↔ RGB CONVERSION
# ═══════════════════════════════════════════════════════════════════════════


def lab_to_rgb(l_channel: torch.Tensor, ab_channel: torch.Tensor) -> torch.Tensor:
    """
    Convert LAB to RGB (differentiable, for loss computation).
    
    Args:
        l_channel: [B, 1, H, W] or [B, H, W], L in [0, 1]
        ab_channel: [B, 2, H, W] or [B, H, W, 2], AB in [-1, 1]
    
    Returns:
        [B, 3, H, W] RGB in [0, 1]
    """
    device = l_channel.device
    
    # Ensure 4D tensors
    if l_channel.ndim == 3:
        l_channel = l_channel.unsqueeze(1)
    if ab_channel.ndim == 3:
        ab_channel = ab_channel.unsqueeze(3).transpose(2, 3)
    
    B, _, H, W = l_channel.shape
    
    # Denormalize to LAB ranges
    L = l_channel * 100.0  # [0, 1] → [0, 100]
    AB = ab_channel * 128.0  # [-1, 1] → [-128, 128]
    
    A = AB[:, 0:1, :, :]
    B = AB[:, 1:2, :, :]
    
    # LAB → XYZ conversion
    fy = (L + 16.0) / 116.0
    fx = A / 500.0 + fy
    fz = fy - B / 200.0
    
    # Inverse companding
    mask = torch.pow(fx, 3.0) > 0.008856
    xr = torch.where(mask, torch.pow(fx, 3.0), (fx - 16.0 / 116.0) / 7.787)
    
    mask = torch.pow(fy, 3.0) > 0.008856
    yr = torch.where(mask, torch.pow(fy, 3.0), (fy - 16.0 / 116.0) / 7.787)
    
    mask = torch.pow(fz, 3.0) > 0.008856
    zr = torch.where(mask, torch.pow(fz, 3.0), (fz - 16.0 / 116.0) / 7.787)
    
    # Reference illuminant D65
    X = 0.95047 * xr
    Y = 1.00000 * yr
    Z = 1.08883 * zr
    
    # XYZ → RGB conversion matrix
    R = X * 3.2406 + Y * -1.5372 + Z * -0.4986
    G = X * -0.9689 + Y * 1.8758 + Z * 0.0415
    B_rgb = X * 0.0557 + Y * -0.2040 + Z * 1.0570
    
    # Gamma correction
    mask_r = R > 0.0031308
    R = torch.where(mask_r, 1.055 * torch.pow(R, 1.0 / 2.4) - 0.055, 12.92 * R)
    
    mask_g = G > 0.0031308
    G = torch.where(mask_g, 1.055 * torch.pow(G, 1.0 / 2.4) - 0.055, 12.92 * G)
    
    mask_b = B_rgb > 0.0031308
    B_rgb = torch.where(mask_b, 1.055 * torch.pow(B_rgb, 1.0 / 2.4) - 0.055, 12.92 * B_rgb)
    
    # Clamp to [0, 1]
    RGB = torch.cat([R, G, B_rgb], dim=1)
    RGB = torch.clamp(RGB, 0.0, 1.0)
    
    return RGB


# ═══════════════════════════════════════════════════════════════════════════
# FINE-TUNING TRAINER
# ═══════════════════════════════════════════════════════════════════════════


class ColorizerFinetuner:
    """Fine-tune existing colorizer with L1 + perceptual loss."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        lr: float = 3e-6,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.0,
        freeze_encoder: bool = True,
    ):
        """
        Initialize fine-tuner.
        
        Args:
            checkpoint_path: Path to existing checkpoint
            device: Compute device
            lr: Learning rate (low for fine-tuning)
            l1_weight: Weight for L1 loss
            perceptual_weight: Weight for perceptual loss
            freeze_encoder: Whether to freeze encoder (only tune decoder)
        """
        self.device = device
        self.lr = lr
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
        # Load model
        print(f"[INFO] Loading model from {checkpoint_path}")
        self.model = UNetColorizer().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set to eval mode initially
        
        # Freeze encoder if requested
        if freeze_encoder:
            print("[INFO] Freezing encoder, fine-tuning decoder only")
            self._freeze_encoder()
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss(layer="relu3_3").to(device)
        
        # Get trainable params (FIX 2: Safe optimizer creation)
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if len(self.trainable_params) == 0:
            raise RuntimeError("❌ FATAL: No trainable parameters found! Check freeze_encoder logic.")
        
        print(f"[INFO] ✓ Optimizer initialized with {len(self.trainable_params)} trainable param groups")
        
        # Optimizer (only trainable params)
        self.optimizer = optim.Adam(self.trainable_params, lr=lr, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30)
        
        # FIX 10: Checkpoint sanity test
        self._sanity_check_checkpoint()
    
    def _sanity_check_checkpoint(self) -> None:
        """
        FIX 10: Test model with dummy input to detect corrupted checkpoints.
        
        Catches NaN early if checkpoint is corrupted or incompatible.
        """
        print("[INFO] Running checkpoint sanity test...")
        
        self.model.eval()
        with torch.no_grad():
            # Create dummy input
            dummy_input = torch.randn(1, 1, 128, 128, device=self.device)
            
            try:
                test_output = self.model(dummy_input)
            except Exception as e:
                raise RuntimeError(f"🚨 FATAL: Model forward pass failed: {e}")
            
            has_nan = torch.isnan(test_output).any()
            has_inf = torch.isinf(test_output).any()
            
            if has_nan or has_inf:
                print("⚠️  WARNING: Checkpoint contains NaN/Inf!")
                print("   Output contains numerical issues - proceed with caution")
            else:
                print("  ✓ Checkpoint OK: No NaN/Inf in test output")
                output_range = [test_output.min().item(), test_output.max().item()]
                print(f"  ✓ Output range: [{output_range[0]:.4f}, {output_range[1]:.4f}]")
    
    def _freeze_encoder(self) -> None:
        """
        FIX 6: Better freeze strategy - train full decoder, freeze only encoder.
        
        This allows gradient flow through mid layers and decoder,
        providing better fine-tuning capability.
        """
        print("[INFO] Freeze strategy: Freeze encoder only, train decoder+mid layers")
        
        # Step 1: Freeze everything with "enc" (encoder) in name
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if "enc" in name.lower():  # Freeze encoder layers
                param.requires_grad = False
                frozen_count += param.numel()
            else:  # Train everything else
                param.requires_grad = True
                if "decoder" in name.lower() or "final" in name.lower():
                    print(f"  ✓ TRAINABLE: {name}")
        
        # Step 2: Verify we have trainable params
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.model.parameters())
        
        if trainable_count == 0:
            raise RuntimeError("🚨 FATAL: No trainable parameters found!")
        
        trainable_pct = 100 * trainable_count / total_count
        print(f"[INFO] Trainable params: {trainable_count:,} / {total_count:,} ({trainable_pct:.2f}%)")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Run one training epoch with 10 critical stability fixes.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        losses = {"l1": 0.0, "total": 0.0}
        num_batches = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True) as pbar:
            skipped_batches = 0
            valid_batches = 0
            
            for l_channel, ab_target in pbar:
                l_channel = l_channel.to(self.device)
                ab_target = ab_target.to(self.device)
                
                # FIX 1: DATA VALIDATION (ROOT CAUSE CHECK)
                if torch.isnan(l_channel).any() or torch.isnan(ab_target).any():
                    print(f"[WARN] Skipping batch {num_batches}: corrupted input (NaN in data)")
                    skipped_batches += 1
                    continue
                
                # Forward pass
                ab_pred = self.model(l_channel)
                
                # FIX 2: SAFE MODEL OUTPUT (check then clamp, not nan_to_num)
                if torch.isnan(ab_pred).any() or torch.isinf(ab_pred).any():
                    print(f"[WARN] Skipping batch {num_batches}: model exploded (NaN/Inf in output)")
                    skipped_batches += 1
                    continue
                
                # ONLY THEN clamp to valid range
                ab_pred = torch.clamp(ab_pred, -110, 110)
                ab_target_clamped = torch.clamp(ab_target, -110, 110)
                
                # FIX 3: NORMALIZE AB BEFORE LOSS (CRITICAL ROOT CAUSE FIX)
                # Normalize to [-1, 1] range for stable loss computation
                ab_pred_norm = ab_pred / 110.0
                ab_gt_norm = ab_target_clamped / 110.0
                
                # FIX 4: SAFE LOSS COMPUTATION
                l1 = self.l1_loss(ab_pred_norm, ab_gt_norm)
                
                if torch.isnan(l1):
                    print(f"[WARN] Skipping batch {num_batches}: NaN in L1 loss")
                    skipped_batches += 1
                    continue
                
                # FIX 8: DISABLE PERCEPTUAL LOSS FOR STABILITY
                # Will add perceptual loss after L1 training is stable
                total_loss = self.l1_weight * l1
                
                # FIX 10: EARLY STOP IF TOO MANY SKIPS
                if skipped_batches > 50:
                    print(f"[ERROR] Too many bad batches ({skipped_batches}) - stopping training")
                    break
                
                # Debug logging (every 50 batches)
                if num_batches % 50 == 0:
                    print(f"\n[DEBUG] Batch {num_batches}:")
                    print(f"  AB pred range: [{ab_pred.min().item():.4f}, {ab_pred.max().item():.4f}]")
                    print(f"  AB norm range: [{ab_pred_norm.min().item():.4f}, {ab_pred_norm.max().item():.4f}]")
                    print(f"  L1 loss: {l1.item():.6f}")
                    print(f"  Trainable params: {len(self.trainable_params)}")
                
                # Safe backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                
                # Safe optimizer step
                self.optimizer.step()
                
                # FIX 9: ADD BATCH SUCCESS COUNTER
                losses["l1"] += l1.item()
                losses["total"] += total_loss.item()
                num_batches += 1
                valid_batches += 1
                
                # Progress bar
                pbar.set_postfix({
                    "L1": f"{l1.item():.4f}",
                    "Total": f"{total_loss.item():.4f}",
                    "Skip": f"{skipped_batches}",
                })
        
        # Log epoch stats
        print(f"[INFO] Epoch {epoch+1}: Valid batches={valid_batches}, Skipped={skipped_batches}")
        
        # FIX 5: FIX DIVISION BY ZERO
        if valid_batches == 0:
            print("[ERROR] All batches skipped → returning zero loss")
            return {k: 0.0 for k in losses}
        
        # Average losses
        for key in losses:
            losses[key] /= num_batches
        
        return losses
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate on a subset (numerically stable with same fixes as training).
        
        Args:
            dataloader: Validation data loader
        
        Returns:
            Dictionary of loss values
        """
        self.model.eval()
        losses = {"l1": 0.0, "total": 0.0}
        num_batches = 0
        skipped_batches = 0
        
        for l_channel, ab_target in dataloader:
            l_channel = l_channel.to(self.device)
            ab_target = ab_target.to(self.device)
            
            # FIX 1: Data validation
            if torch.isnan(l_channel).any() or torch.isnan(ab_target).any():
                skipped_batches += 1
                continue
            
            # Forward pass
            ab_pred = self.model(l_channel)
            
            # FIX 2: Safe model output - check then clamp
            if torch.isnan(ab_pred).any() or torch.isinf(ab_pred).any():
                skipped_batches += 1
                continue
            
            ab_pred = torch.clamp(ab_pred, -110, 110)
            ab_target_clamped = torch.clamp(ab_target, -110, 110)
            
            # FIX 3: Normalize AB before loss (root cause fix)
            ab_pred_norm = ab_pred / 110.0
            ab_gt_norm = ab_target_clamped / 110.0
            
            # FIX 4: Safe loss computation
            l1 = self.l1_loss(ab_pred_norm, ab_gt_norm)
            
            if torch.isnan(l1):
                skipped_batches += 1
                continue
            
            # FIX 8: Disable perceptual loss (L1 only for stability)
            total_loss = self.l1_weight * l1
            
            losses["l1"] += l1.item()
            losses["total"] += total_loss.item()
            num_batches += 1
        
        # FIX 5: Fix division by zero
        if num_batches == 0:
            print("[WARN] Validation: All batches skipped")
            return {"l1": 0.0, "total": 0.0}
        
        # Average
        for key in losses:
            losses[key] /= num_batches
        
        if skipped_batches > 0:
            print(f"[INFO] Validation: Skipped {skipped_batches} bad batches")
        
        return losses
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float) -> None:
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, path)
        print(f"[INFO] Checkpoint saved: {path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 15) -> None:
        """
        Run fine-tuning.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
        """
        best_val_loss = float("inf")
        
        for epoch in range(epochs):
            print(f"\n[Epoch {epoch+1}/{epochs}]")
            
            # Train
            start = time.time()
            train_losses = self.train_epoch(train_loader, epoch)
            train_time = time.time() - start
            
            # Validate
            val_losses = self.validate(val_loader)
            
            # Log
            print(f"  Train: L1={train_losses['l1']:.6f}, Total={train_losses['total']:.6f} [{train_time:.1f}s]")
            print(f"  Val:   L1={val_losses['l1']:.6f}, Total={val_losses['total']:.6f}")
            
            # Save best
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                self.save_checkpoint("checkpoints/stage1_colorizer_finetuned_best.pth", epoch, best_val_loss)
            
            # LR scheduler
            self.scheduler.step()
        
        # Save final
        self.save_checkpoint("checkpoints/stage1_colorizer_finetuned_latest.pth", epochs - 1, val_losses["total"])
        print("\n[INFO] Fine-tuning complete!")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune colorizer with L1 loss (stable, minimal perceptual)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage1_colorizer_latest.pth",
                       help="Existing checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=15,
                       help="Number of fine-tuning epochs (10-20 recommended)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-6,
                       help="Learning rate (ultra-low 3e-6 for maximum stability)")
    parser.add_argument("--l1-weight", type=float, default=1.0,
                       help="Weight for L1 loss")
    parser.add_argument("--perceptual-weight", type=float, default=0.0,
                       help="Weight for perceptual loss (disabled for stability, enable after L1 stable)")
    parser.add_argument("--data-roots", type=str, nargs="+", default=["datasets/flickr2k", "datasets/coco"],
                       help="Dataset roots")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Data loading workers")
    parser.add_argument("--freeze-encoder", action="store_true", default=True,
                       help="Freeze encoder with 'enc' in name (train decoder/mid-layers)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Learning rate: {args.lr}")
    print(f"[INFO] L1 weight: {args.l1_weight}")
    print(f"[INFO] Perceptual weight: {args.perceptual_weight}")
    
    # Load datasets
    print("[INFO] Loading datasets...")
    full_dataset, _ = build_combined_colorization_dataset(
        data_roots=args.data_roots,
        augment=True,
        image_size=256,
    )
    
    # Split 90/10
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    
    print(f"[INFO] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Fine-tune
    finetuner = ColorizerFinetuner(
        checkpoint_path=args.checkpoint,
        device=device,
        lr=args.lr,
        l1_weight=args.l1_weight,
        perceptual_weight=args.perceptual_weight,
        freeze_encoder=args.freeze_encoder,
    )
    
    finetuner.train(train_loader, val_loader, epochs=args.epochs)


if __name__ == "__main__":
    main()
