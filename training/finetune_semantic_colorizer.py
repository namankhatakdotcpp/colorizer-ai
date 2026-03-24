#!/usr/bin/env python3
"""
Fine-tune existing UNet colorizer with semantic understanding.

This script:
1. Loads existing baseline checkpoint (stage1_colorizer_latest.pth)
2. Adds semantic head and attention gates
3. Fine-tunes for 20-30 epochs with semantic supervision

Usage:
    python training/finetune_semantic_colorizer.py \
        --baseline-checkpoint checkpoints/stage1_colorizer_latest.pth \
        --epochs 25 \
        --batch-size 4 \
        --lr 5e-5 \
        --data-roots datasets/flickr2k datasets/coco
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.unet_colorizer_semantic import UNetColorizerSemantic
from datasets.dataset_colorizer import build_combined_colorization_dataset
from utils.losses import ColorizationLossV2
from training.train_colorizer import VGGPerceptualLoss


class SemanticFineTuner:
    """Fine-tune semantic colorizer from baseline checkpoint."""

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        lr: float = 5e-5,
        use_attention: bool = True,
        semantic_weight: float = 0.3,
    ):
        self.device = device
        self.lr = lr
        self.use_attention = use_attention

        # Load baseline model and convert to semantic
        print(f"[INFO] Loading baseline checkpoint: {checkpoint_path}")
        self.model = UNetColorizerSemantic.from_baseline_checkpoint(
            checkpoint_path,
            use_attention=use_attention,
            semantic_weight=semantic_weight,
        ).to(device)

        # Loss components
        self.base_loss = ColorizationLossV2(
            l1_weight=1.0,
            perceptual_weight=0.6,
            histogram_weight=0.4,
        )
        self.perceptual_loss = VGGPerceptualLoss().to(device)
        self.semantic_loss = nn.CrossEntropyLoss()

        # Optimizer (lower LR for fine-tuning)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        """Run one training epoch."""
        self.model.train()
        losses = {"total": 0.0, "base": 0.0, "perceptual": 0.0, "semantic": 0.0}
        num_batches = 0

        with tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False) as pbar:
            for batch_idx, (l_channel, ab_target) in enumerate(pbar):
                l_channel = l_channel.to(self.device)
                ab_target = ab_target.to(self.device)

                # Forward pass
                ab_pred, semantic_logits = self.model(l_channel, return_semantic=True)

                # Compute base AB loss
                with torch.no_grad():
                    base = self.base_loss(ab_pred, ab_target, epoch=epoch, total_epochs=30)

                # Perceptual loss
                perc = self.perceptual_loss(ab_pred, ab_target, l_channel)

                # Semantic loss (weight reduction over epochs)
                # Start strong, gradually reduce as AB quality improves
                sem_weight = 0.3 * (1.0 - epoch / 30.0)
                if semantic_logits is not None:
                    # Create pseudo-labels from image statistics
                    # (In practice, you'd use pre-computed labels or a separate annotation)
                    b_channel = ab_target[:, 1, :, :]  # Blue channel
                    pseudo_labels = torch.argmax(
                        torch.stack([
                            (b_channel < -50).float().mean(),  # water/sky
                            (b_channel < -30).float().mean(),  # sky
                            (ab_target[:, 0, :, :] < -30).float().mean(),  # vegetation
                            (ab_target[:, 0, :, :] > 10).float().mean(),  # skin
                            torch.ones(1).to(self.device),  # other
                        ], dim=1),
                        dim=1,
                    )
                    sem = self.semantic_loss(semantic_logits, pseudo_labels) * sem_weight
                else:
                    sem = 0.0

                # Total loss
                total_loss = base + perc + sem

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Track losses
                losses["total"] += total_loss.item()
                losses["base"] += base.item()
                losses["perceptual"] += perc.item()
                losses["semantic"] += sem.item() if isinstance(sem, torch.Tensor) else 0.0
                num_batches += 1

                pbar.update(1)

        # Average losses
        for key in losses:
            losses[key] /= num_batches
        return losses

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Validate on a subset."""
        self.model.eval()
        losses = {"total": 0.0, "base": 0.0, "perceptual": 0.0}
        num_batches = 0

        for l_channel, ab_target in dataloader:
            l_channel = l_channel.to(self.device)
            ab_target = ab_target.to(self.device)

            ab_pred, _ = self.model(l_channel, return_semantic=True)

            base = self.base_loss(ab_pred, ab_target, epoch=0, total_epochs=30)
            perc = self.perceptual_loss(ab_pred, ab_target, l_channel)
            total_loss = base + perc

            losses["total"] += total_loss.item()
            losses["base"] += base.item()
            losses["perceptual"] += perc.item()
            num_batches += 1

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
            "semantic": True,  # Flag this as semantic model
        }
        torch.save(checkpoint, path)
        print(f"[INFO] Checkpoint saved: {path}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 25) -> None:
        """Run fine-tuning."""
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
            print(
                f"  Train: Loss={train_losses['total']:.6f} "
                f"(base={train_losses['base']:.6f}, perc={train_losses['perceptual']:.6f}) "
                f"[{train_time:.1f}s]"
            )
            print(
                f"  Val:   Loss={val_losses['total']:.6f} "
                f"(base={val_losses['base']:.6f}, perc={val_losses['perceptual']:.6f})"
            )

            # Save best checkpoint
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                self.save_checkpoint(
                    "checkpoints/stage1_colorizer_semantic_best.pth",
                    epoch,
                    best_val_loss,
                )

            # Step LR scheduler
            self.scheduler.step()

        # Save final
        self.save_checkpoint(
            "checkpoints/stage1_colorizer_semantic_latest.pth",
            epochs - 1,
            val_losses["total"],
        )
        print("\n[INFO] Fine-tuning complete!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune semantic colorizer from baseline")
    parser.add_argument("--baseline-checkpoint", type=str, default="checkpoints/stage1_colorizer_latest.pth")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5, help="Lower LR for fine-tuning")
    parser.add_argument("--data-roots", type=str, nargs="+", default=["datasets/flickr2k", "datasets/coco"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-attention", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    # Load data
    print("[INFO] Loading datasets...")
    full_dataset = build_combined_colorization_dataset(
        root_dirs=args.data_roots,
        augment=True,
        image_size=256,
    )

    # Split into train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"[INFO] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Fine-tune
    finetuner = SemanticFineTuner(
        checkpoint_path=args.baseline_checkpoint,
        device=device,
        lr=args.lr,
        use_attention=args.use_attention,
    )
    finetuner.train(train_loader, val_loader, epochs=args.epochs)


if __name__ == "__main__":
    main()
