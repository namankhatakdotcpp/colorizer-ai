from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms.functional as TF

from models.zero_dce import ZeroDCEModel


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class ToneDataset(Dataset):
    """Unpaired image dataset for Zero-DCE-style self-supervision."""

    def __init__(self, root_dir: str, image_size: int = 384):
        self.root = Path(root_dir)
        self.image_size = image_size
        self.images = [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        self.images = sorted(self.images)

        if len(self.images) == 0:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = Image.open(self.images[idx]).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        if random.random() > 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return TF.to_tensor(img)


class LColor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_rgb = torch.mean(x, dim=[2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        drg = torch.pow(mr - mg, 2)
        drb = torch.pow(mr - mb, 2)
        dgb = torch.pow(mg - mb, 2)
        return torch.sqrt(drg + drb + dgb + 1e-6).mean()


class LSpa(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(4)

    def forward(self, org: torch.Tensor, enhance: torch.Tensor) -> torch.Tensor:
        org_pool = self.pool(org)
        enh_pool = self.pool(enhance)

        d_org_l = org_pool[:, :, :, :-1] - org_pool[:, :, :, 1:]
        d_org_r = org_pool[:, :, :, 1:] - org_pool[:, :, :, :-1]
        d_org_u = org_pool[:, :, :-1, :] - org_pool[:, :, 1:, :]
        d_org_d = org_pool[:, :, 1:, :] - org_pool[:, :, :-1, :]

        d_enh_l = enh_pool[:, :, :, :-1] - enh_pool[:, :, :, 1:]
        d_enh_r = enh_pool[:, :, :, 1:] - enh_pool[:, :, :, :-1]
        d_enh_u = enh_pool[:, :, :-1, :] - enh_pool[:, :, 1:, :]
        d_enh_d = enh_pool[:, :, 1:, :] - enh_pool[:, :, :-1, :]

        return (
            torch.pow(d_org_l - d_enh_l, 2).mean()
            + torch.pow(d_org_r - d_enh_r, 2).mean()
            + torch.pow(d_org_u - d_enh_u, 2).mean()
            + torch.pow(d_org_d - d_enh_d, 2).mean()
        )


class LExp(nn.Module):
    def __init__(self, patch_size: int = 16, mean_val: float = 0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gray = torch.mean(x, dim=1, keepdim=True)
        mean_pool = self.pool(x_gray)
        return torch.mean(torch.pow(mean_pool - self.mean_val, 2))


class LTV(nn.Module):
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        batch_size = a.shape[0]
        h_tv = torch.pow(a[:, :, 1:, :] - a[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(a[:, :, :, 1:] - a[:, :, :, :-1], 2).sum()
        total_pixels = a.shape[2] * a.shape[3]
        return (h_tv + w_tv) / (batch_size * total_pixels)


class ZeroDCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_spa = LSpa()
        self.l_exp = LExp()
        self.l_col = LColor()
        self.l_tv = LTV()

    def forward(self, org: torch.Tensor, enhance: torch.Tensor, curve: torch.Tensor):
        loss_spa = self.l_spa(org, enhance)
        loss_exp = self.l_exp(enhance)
        loss_col = self.l_col(enhance)
        loss_tv = self.l_tv(curve)
        total = loss_spa + loss_exp + loss_col + (200.0 * loss_tv)
        return total, loss_spa, loss_exp, loss_col, loss_tv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage5 tone model (Zero-DCE)")
    parser.add_argument("--data-root", type=str, default="datasets/flickr2k")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ToneDataset(root_dir=args.data_root, image_size=args.image_size)
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

    model = ZeroDCEModel(num_iterations=args.iterations, channels=args.channels).to(device)
    criterion = ZeroDCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "stage5_tone.pth"

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for img in train_loader:
            img = img.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                enhanced, curve = model(img)
                loss, _, _, _, _ = criterion(img, enhanced, curve)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img in val_loader:
                img = img.to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda")):
                    enhanced, curve = model(img)
                    loss, _, _, _, _ = criterion(img, enhanced, curve)
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
