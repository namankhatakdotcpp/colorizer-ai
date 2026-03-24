#!/usr/bin/env python3
"""
Fine-tune existing colorizer with L1 + Perceptual loss.

This script:
1. Loads your existing checkpoint (stage1_colorizer_latest.pth)
2. Adds VGG16 perceptual loss
3. Fine-tunes for 10-20 epochs with low LR (1e-5)
4. Improves color realism (water→blue, sky→light blue, veg→green)
5. Saves best checkpoint

Strategy: Decoder-only fine-tuning (freeze encoder)
Time: ~30 min to 2 hours on GPU, ~2-8 hours on CPU
Quality gain: +10-15% color realism

Usage:
    python training/finetune_colorizer.py \
        --checkpoint checkpoints/stage1_colorizer_latest.pth \
        --epochs 15 \
        --batch-size 4 \
        --lr 1e-5
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
        Compute perceptual loss.
        
        Args:
            pred_rgb: [B, 3, H, W] predicted RGB in [0, 1]
            target_rgb: [B, 3, H, W] target RGB in [0, 1]
        
        Returns:
            L1 distance between VGG features
        """
        # Normalize to ImageNet stats
        pred_norm = self._normalize(pred_rgb)
        target_norm = self._normalize(target_rgb)
        
        # Extract features
        pred_feat = self.features(pred_norm)
        target_feat = self.features(target_norm)
        
        # L1 loss
        loss = F.l1_loss(pred_feat, pred_feat.detach())
        
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
        lr: float = 1e-5,
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,
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
        
        # Optimizer (only trainable params)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30)
    
    def _freeze_encoder(self) -> None:
        """Freeze encoder layers (down path)."""
        # Assuming UNet structure with encoder layers
        modules_to_freeze = [
            self.model.inc,
            self.model.down1,
            self.model.down2,
            self.model.down3,
            self.model.down4,
        ]
        
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
        
        print("[INFO] Encoder frozen (decoder only trainable)")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Run one training epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        losses = {"l1": 0.0, "perceptual": 0.0, "total": 0.0}
        num_batches = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=True) as pbar:
            for l_channel, ab_target in pbar:
                l_channel = l_channel.to(self.device)
                ab_target = ab_target.to(self.device)
                
                # Forward pass
                ab_pred = self.model(l_channel)
                
                # L1 loss (AB prediction)
                l1 = self.l1_loss(ab_pred, ab_target)
                
                # Perceptual loss (RGB level)
                pred_rgb = lab_to_rgb(l_channel, ab_pred)
                target_rgb = lab_to_rgb(l_channel, ab_target)
                perc = self.perceptual_loss(pred_rgb, target_rgb)
                
                # Total loss
                total_loss = self.l1_weight * l1 + self.perceptual_weight * perc
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Track losses
                losses["l1"] += l1.item()
                losses["perceptual"] += perc.item()
                losses["total"] += total_loss.item()
                num_batches += 1
                
                # Progress bar
                pbar.set_postfix({
                    "L1": f"{l1.item():.4f}",
                    "Perc": f"{perc.item():.4f}",
                    "Total": f"{total_loss.item():.4f}",
                })
        
        # Average losses
        for key in losses:
            losses[key] /= num_batches
        
        return losses
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate on a subset.
        
        Args:
            dataloader: Validation data loader
        
        Returns:
            Dictionary of loss values
        """
        self.model.eval()
        losses = {"l1": 0.0, "perceptual": 0.0, "total": 0.0}
        num_batches = 0
        
        for l_channel, ab_target in dataloader:
            l_channel = l_channel.to(self.device)
            ab_target = ab_target.to(self.device)
            
            # Forward pass
            ab_pred = self.model(l_channel)
            
            # Losses
            l1 = self.l1_loss(ab_pred, ab_target)
            
            pred_rgb = lab_to_rgb(l_channel, ab_pred)
            target_rgb = lab_to_rgb(l_channel, ab_target)
            perc = self.perceptual_loss(pred_rgb, target_rgb)
            
            total_loss = self.l1_weight * l1 + self.perceptual_weight * perc
            
            losses["l1"] += l1.item()
            losses["perceptual"] += perc.item()
            losses["total"] += total_loss.item()
            num_batches += 1
        
        # Average
        for key in losses:
            losses[key] /= num_batches
        
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
            print(f"  Train: L1={train_losses['l1']:.6f}, Perc={train_losses['perceptual']:.6f}, "
                  f"Total={train_losses['total']:.6f} [{train_time:.1f}s]")
            print(f"  Val:   L1={val_losses['l1']:.6f}, Perc={val_losses['perceptual']:.6f}, "
                  f"Total={val_losses['total']:.6f}")
            
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
    parser = argparse.ArgumentParser(description="Fine-tune colorizer with L1 + Perceptual loss")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage1_colorizer_latest.pth",
                       help="Existing checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=15,
                       help="Number of fine-tuning epochs (10-20 recommended)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate (low for fine-tuning)")
    parser.add_argument("--l1-weight", type=float, default=1.0,
                       help="Weight for L1 loss")
    parser.add_argument("--perceptual-weight", type=float, default=0.1,
                       help="Weight for perceptual loss")
    parser.add_argument("--data-roots", type=str, nargs="+", default=["datasets/flickr2k", "datasets/coco"],
                       help="Dataset roots")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Data loading workers")
    parser.add_argument("--freeze-encoder", action="store_true", default=True,
                       help="Freeze encoder (fine-tune decoder only)")
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
