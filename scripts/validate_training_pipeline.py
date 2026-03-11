import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.dataset_colorizer import ColorizationDataset
from datasets.dataset_sr import SRDataset
from datasets.dataset_depth import DepthDataset
from models.unet_colorizer import UNetColorizer
from models.rrdb_sr import RRDBNet
from models.depth_model import DynamicFilterNetwork
from models.micro_contrast_model import MicroContrastModel
from training.train_micro_contrast import MicroContrastDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate all training stages before launching DDP")
    parser.add_argument("--colorizer-root", type=str, default="datasets/flickr2k")
    parser.add_argument("--sr-root", type=str, default="datasets/div2k")
    parser.add_argument("--depth-root", type=str, default="datasets/coco")
    parser.add_argument("--contrast-root", type=str, default="datasets/flickr2k")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def make_loader(dataset, batch_size: int, min_batches: int, stage: str) -> DataLoader:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    if len(loader) == 0:
        raise RuntimeError(f"{stage}: dataloader produced 0 batches")
    if len(loader) < min_batches:
        raise RuntimeError(f"{stage}: expected at least {min_batches} batches, got {len(loader)}")
    return loader


def main() -> None:
    args = parse_args()

    color_root = Path(args.colorizer_root)
    if not (color_root / "L").is_dir() or not (color_root / "AB").is_dir():
        raise RuntimeError(
            f"Missing preprocessed LAB folders: {color_root / 'L'} and {color_root / 'AB'}"
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Stage 1
    stage1_ds = ColorizationDataset(args.colorizer_root, augment=False, min_samples=101)
    stage1_loader = make_loader(stage1_ds, args.batch_size, min_batches=51, stage="stage1_colorizer")
    stage1_x, stage1_y_true = next(iter(stage1_loader))
    stage1_out = UNetColorizer(in_channels=1, out_channels=2).to(device)(stage1_x.to(device))
    if stage1_out.shape != stage1_y_true.to(device).shape:
        raise RuntimeError(f"stage1_colorizer: output shape mismatch {tuple(stage1_out.shape)} vs {tuple(stage1_y_true.shape)}")
    print(f"[stage1_colorizer] dataset={len(stage1_ds)} batch={args.batch_size} steps={len(stage1_loader)} out={tuple(stage1_out.shape)}")

    # Stage 2
    stage2_ds = SRDataset(args.sr_root, min_samples=101)
    stage2_loader = make_loader(stage2_ds, args.batch_size, min_batches=21, stage="stage2_sr")
    stage2_x, stage2_y_true = next(iter(stage2_loader))
    stage2_out = RRDBNet().to(device)(stage2_x.to(device))
    if stage2_out.shape != stage2_y_true.to(device).shape:
        raise RuntimeError(f"stage2_sr: output shape mismatch {tuple(stage2_out.shape)} vs {tuple(stage2_y_true.shape)}")
    print(f"[stage2_sr] dataset={len(stage2_ds)} batch={args.batch_size} steps={len(stage2_loader)} out={tuple(stage2_out.shape)}")

    # Stage 3
    stage3_ds = DepthDataset(args.depth_root, min_samples=101)
    stage3_loader = make_loader(stage3_ds, args.batch_size, min_batches=51, stage="stage3_depth")
    stage3_x, stage3_y_true = next(iter(stage3_loader))
    stage3_out = DynamicFilterNetwork(in_channels=3).to(device)(stage3_x.to(device))
    if stage3_out.shape != stage3_y_true.to(device).shape:
        raise RuntimeError(f"stage3_depth: output shape mismatch {tuple(stage3_out.shape)} vs {tuple(stage3_y_true.shape)}")
    print(f"[stage3_depth] dataset={len(stage3_ds)} batch={args.batch_size} steps={len(stage3_loader)} out={tuple(stage3_out.shape)}")

    # Stage 4
    stage4_ds = MicroContrastDataset(args.contrast_root, min_samples=101)
    stage4_loader = make_loader(stage4_ds, args.batch_size, min_batches=21, stage="stage4_contrast")
    stage4_x, stage4_y_true = next(iter(stage4_loader))
    stage4_out = MicroContrastModel(in_channels=3, out_channels=3).to(device)(stage4_x.to(device))
    if stage4_out.shape != stage4_y_true.to(device).shape:
        raise RuntimeError(f"stage4_contrast: output shape mismatch {tuple(stage4_out.shape)} vs {tuple(stage4_y_true.shape)}")
    print(f"[stage4_contrast] dataset={len(stage4_ds)} batch={args.batch_size} steps={len(stage4_loader)} out={tuple(stage4_out.shape)}")

    print("Training pipeline validation passed.")


if __name__ == "__main__":
    main()
