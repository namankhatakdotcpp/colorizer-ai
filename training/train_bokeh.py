from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms.functional as TF

from models.dfn_bokeh import BokehLoss, DFNBokehModel


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class BokehDataset(Dataset):
    """
    Expected layout:
      <root>/f8     (sharp RGB)
      <root>/f1_4   (target bokeh RGB)
      <root>/depth  (single-channel depth)
    """

    def __init__(self, root_dir: str, image_size: int = 384):
        self.root = Path(root_dir)
        self.image_size = image_size

        sharp_dir = self.root / "f8"
        bokeh_dir = self.root / "f1_4"
        depth_dir = self.root / "depth"

        self.samples = []
        if sharp_dir.is_dir() and bokeh_dir.is_dir() and depth_dir.is_dir():
            for sharp_path in sorted(sharp_dir.rglob("*")):
                if sharp_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                stem = sharp_path.stem
                bokeh_path = bokeh_dir / f"{stem}{sharp_path.suffix}"
                if not bokeh_path.exists():
                    bokeh_path = next((p for p in bokeh_dir.rglob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTENSIONS), None)
                depth_path = next((p for p in depth_dir.rglob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTENSIONS), None)
                if bokeh_path is not None and depth_path is not None:
                    self.samples.append((sharp_path, bokeh_path, depth_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No paired bokeh samples found under {self.root}. "
                "Expected f8/, f1_4/, depth/ with matching stems."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sharp_path, bokeh_path, depth_path = self.samples[idx]

        sharp = Image.open(sharp_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        bokeh = Image.open(bokeh_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        depth = Image.open(depth_path).convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        if random.random() > 0.5:
            sharp = sharp.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            bokeh = bokeh.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        sharp_t = TF.to_tensor(sharp)
        bokeh_t = TF.to_tensor(bokeh)
        depth_t = TF.to_tensor(depth)
        return sharp_t, bokeh_t, depth_t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage4 bokeh DFN")
    parser.add_argument("--data-root", type=str, default="datasets/bokeh")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--focus-threshold", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BokehDataset(root_dir=args.data_root, image_size=args.image_size)
    train_len = int(0.9 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = DFNBokehModel(kernel_size=args.kernel_size).to(device)
    criterion = BokehLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "stage4_bokeh.pth"

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for sharp, target_bokeh, depth in train_loader:
            sharp = sharp.to(device, non_blocking=True)
            target_bokeh = target_bokeh.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                pred_bokeh, _, kh, kv, _ = model(
                    sharp,
                    depth,
                    focus_threshold=args.focus_threshold,
                    return_aux=True,
                )
                loss, _, _, _ = criterion(pred_bokeh, target_bokeh, kh, kv)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sharp, target_bokeh, depth in val_loader:
                sharp = sharp.to(device, non_blocking=True)
                target_bokeh = target_bokeh.to(device, non_blocking=True)
                depth = depth.to(device, non_blocking=True)

                with autocast(enabled=(device.type == "cuda")):
                    pred_bokeh, _, kh, kv, _ = model(
                        sharp,
                        depth,
                        focus_threshold=args.focus_threshold,
                        return_aux=True,
                    )
                    loss, _, _, _ = criterion(pred_bokeh, target_bokeh, kh, kv)
                val_loss += float(loss.item())

        train_avg = train_loss / max(1, len(train_loader))
        val_avg = val_loss / max(1, len(val_loader))
        print(f"Epoch [{epoch + 1}/{args.epochs}] | train={train_avg:.6f} | val={val_avg:.6f}")

        if val_avg < best_val:
            best_val = val_avg
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_avg}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
