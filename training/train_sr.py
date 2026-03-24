import argparse
import os
import time
from pathlib import Path
from typing import List

from PIL import Image
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torchvision.models as models

from models.rrdb_sr import RRDBNet


STAGE_NAME = "stage2_sr"
MIN_BATCHES_PER_EPOCH = 10
MIN_CHECKPOINT_MB = 20.0
EXPECTED_CHECKPOINT_MIN_MB = 30.0
EXPECTED_CHECKPOINT_MAX_MB = 80.0


class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss for perceptual image quality."""
    def __init__(self, layer_name: str = "relu3_4", device: torch.device = None):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        layers = list(vgg.features.children())
        
        if layer_name == "relu3_4":
            self.layers = layers[:22]
        elif layer_name == "relu2_2":
            self.layers = layers[:13]
        else:
            self.layers = layers[:22]
        
        self.features = nn.Sequential(*self.layers)
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.features.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x_feat = self.features(x)
        y_feat = self.features(y)
        return F.l1_loss(x_feat, y_feat)


class EdgeLoss(nn.Module):
    """Sobel edge detection loss for preserving sharp edges."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer("sobel_y", sobel_y.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size, channels, h, w = x.shape
        
        x_edges_x = F.conv2d(x, self.sobel_x.expand(channels, -1, -1, -1), groups=channels, padding=1)
        x_edges_y = F.conv2d(x, self.sobel_y.expand(channels, -1, -1, -1), groups=channels, padding=1)
        x_edges = torch.sqrt(x_edges_x**2 + x_edges_y**2 + 1e-8)
        
        y_edges_x = F.conv2d(y, self.sobel_x.expand(channels, -1, -1, -1), groups=channels, padding=1)
        y_edges_y = F.conv2d(y, self.sobel_y.expand(channels, -1, -1, -1), groups=channels, padding=1)
        y_edges = torch.sqrt(y_edges_x**2 + y_edges_y**2 + 1e-8)
        
        return F.l1_loss(x_edges, y_edges)


class CombinedSRLoss(nn.Module):
    """Combined loss for SR: L1 + Perceptual + Edge."""
    def __init__(self, device: torch.device):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(device=device).to(device)
        self.edge_loss = EdgeLoss().to(device)
        self.l1_weight = 0.7
        self.perceptual_weight = 0.2
        self.edge_weight = 0.1

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        edge = self.edge_loss(pred, target)
        
        total = self.l1_weight * l1 + self.perceptual_weight * perceptual + self.edge_weight * edge
        return total, {"l1": l1.item(), "perceptual": perceptual.item(), "edge": edge.item()}


class SRImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, crop_size: int = 256, scale_factor: int = 4):
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.lr_size = crop_size // scale_factor

        if not self.root_dir.exists() or not self.root_dir.is_dir():
            raise FileNotFoundError(f"SR dataset directory not found: {self.root_dir}")

        # Requirement: recursive loader with glob("**/*.jpg") and glob("**/*.png").
        image_paths = list(self.root_dir.glob("**/*.jpg")) + list(self.root_dir.glob("**/*.png"))
        # Practical extension: support uppercase variants too.
        image_paths += list(self.root_dir.glob("**/*.JPG")) + list(self.root_dir.glob("**/*.PNG"))
        self.images: List[Path] = sorted(set(image_paths))

        if len(self.images) == 0:
            raise RuntimeError(f"SR dataset is empty: {self.root_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        hr_img = Image.open(self.images[idx]).convert("RGB")
        w, h = hr_img.size

        # Safe random crop.
        if w < self.crop_size or h < self.crop_size:
            hr_img = hr_img.resize((max(w, self.crop_size), max(h, self.crop_size)), Image.Resampling.BICUBIC)
            w, h = hr_img.size

        left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
        top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
        hr_crop = hr_img.crop((left, top, left + self.crop_size, top + self.crop_size))
        lr_img = hr_crop.resize((self.lr_size, self.lr_size), Image.Resampling.BICUBIC)

        return TF.to_tensor(lr_img), TF.to_tensor(hr_crop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: SR DDP training (with Perceptual Loss)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--flickr-root", type=str, default="datasets/flickr2k")
    parser.add_argument("--div2k-root", type=str, default="datasets/div2k")
    parser.add_argument("--coco-root", type=str, default="datasets/coco/train2017")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--run-name", type=str, default=STAGE_NAME)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def save_checkpoint(path: Path, epoch: int, model: DDP, optimizer: torch.optim.Optimizer, scheduler: object, loss_value: float) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": float(loss_value),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: object, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"]) + 1, float(checkpoint.get("loss", float("inf")))


def validate_checkpoint_size(
    path: Path,
    min_mb: float = MIN_CHECKPOINT_MB,
    expected_min_mb: float = EXPECTED_CHECKPOINT_MIN_MB,
    expected_max_mb: float = EXPECTED_CHECKPOINT_MAX_MB,
) -> float:
    checkpoint_size = os.path.getsize(path) / (1024 * 1024)
    print(f"Checkpoint saved: {checkpoint_size:.2f} MB")
    if checkpoint_size < min_mb:
        raise RuntimeError("Checkpoint too small — model weights not saved correctly")
    if checkpoint_size < expected_min_mb or checkpoint_size > expected_max_mb:
        print(
            f"[WARN] {STAGE_NAME} checkpoint size {checkpoint_size:.2f} MB is outside "
            f"expected range {expected_min_mb:.0f}-{expected_max_mb:.0f} MB."
        )
    return checkpoint_size


def compute_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    count = 0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().abs().mean().item())
            count += 1
    return total if count > 0 else 0.0


def run_sanity_check(loader: DataLoader, model: DDP, device: torch.device, rank: int) -> None:
    sample_lr, _ = next(iter(loader))
    sample_lr = sample_lr.to(device, non_blocking=True)
    with torch.no_grad():
        output = model(sample_lr)
    if rank == 0:
        print(f"Sanity check output shape: {tuple(output.shape)}")


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this DDP training script (backend=nccl).")

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cudnn.benchmark = True

    try:
        flickr_dataset = SRImageFolderDataset(args.flickr_root)
        div2k_dataset = SRImageFolderDataset(args.div2k_root)
        coco_dataset = SRImageFolderDataset(args.coco_root)

        dataset = ConcatDataset([flickr_dataset, div2k_dataset, coco_dataset])
        total_images = len(dataset)

        if rank == 0:
            print("--------------------------------")
            print("SR Training Dataset Summary")
            print(f"Flickr2K: {len(flickr_dataset)}")
            print(f"DIV2K: {len(div2k_dataset)}")
            print(f"COCO: {len(coco_dataset)}")
            print(f"TOTAL: {total_images}")
            print("--------------------------------")

        if total_images == 0:
            raise RuntimeError("Dataset is empty")
        if total_images < 10000:
            raise RuntimeError("Dataset too small — loader failure")

        # In DDP, shuffling is handled by DistributedSampler(shuffle=True).
        sampler = DistributedSampler(dataset, shuffle=True)
        num_workers = max(0, min(args.num_workers, (os.cpu_count() or 1) // max(world_size, 1)))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2,
            drop_last=False,
        )

        steps_per_epoch = len(loader)
        if steps_per_epoch == 0:
            raise RuntimeError("No batches produced")
        if steps_per_epoch < MIN_BATCHES_PER_EPOCH:
            raise RuntimeError("Dataset too small")

        first_batch = next(iter(loader))
        first_batch_shape = tuple(first_batch[0].shape)
        image_resolution = f"{first_batch_shape[-2]}x{first_batch_shape[-1]}"

        if rank == 0:
            print(f"Dataset size: {len(dataset)}")
            print(f"Batch size: {args.batch_size}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"GPUs: {world_size}")
            print(f"Image resolution: {image_resolution}")
            print(f"First batch shape: {first_batch_shape}")

        model = RRDBNet().cuda(local_rank)
        total_params = sum(p.numel() for p in model.parameters())
        if total_params < 1_000_000:
            raise RuntimeError("Model architecture incorrect")
        if rank == 0:
            print(f"Model parameters: {total_params:,}")
        model = DDP(model, device_ids=[local_rank])
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
        criterion = CombinedSRLoss(device=device)
        scaler = GradScaler("cuda")

        run_sanity_check(loader, model, device, rank)

        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        latest_path = ckpt_dir / f"{STAGE_NAME}_latest.pth"
        best_path = ckpt_dir / f"{STAGE_NAME}_best.pth"

        start_epoch = 0
        best_loss = float("inf")
        if latest_path.exists():
            if rank == 0:
                print(f"Resuming from {latest_path}")
            start_epoch, best_loss = load_checkpoint(latest_path, model, optimizer, scheduler, device)

        dist.barrier()

        if start_epoch >= args.epochs:
            raise RuntimeError(
                f"Resume epoch ({start_epoch}) >= target epochs ({args.epochs}). "
                "Increase --epochs or remove checkpoint to continue training."
            )

        prev_epoch_loss = None
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            zero_grad_counter = 0
            
            if rank == 0:
                dataloader = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", 
                                 total=steps_per_epoch, unit="batch")
            else:
                dataloader = loader

            for step, (lr_img, hr_img) in enumerate(dataloader, start=1):
                lr_img = lr_img.to(device, non_blocking=True)
                hr_img = hr_img.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                
                with autocast("cuda"):
                    pred = model(lr_img)
                    loss, loss_dict = criterion(pred, hr_img)
                
                if torch.isnan(loss):
                    raise RuntimeError(f"NaN loss detected at step {step}")
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                grad_norm = compute_grad_norm(model)
                
                if grad_norm <= 1e-12:
                    zero_grad_counter += 1
                    if zero_grad_counter >= 5:
                        raise RuntimeError("Gradients are zero for 5 consecutive checks.")
                else:
                    zero_grad_counter = 0

                running_loss += float(loss.item())
                
                if rank == 0:
                    dataloader.set_postfix(loss=f"{loss.item():.4f}")

            stats = torch.tensor([running_loss], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            avg_loss = (stats.item() / world_size) / max(steps_per_epoch, 1)
            epoch_time = time.time() - epoch_start
            improved_vs_prev = prev_epoch_loss is None or avg_loss < prev_epoch_loss

            if rank == 0:
                improved_text = "✓" if improved_vs_prev else "✗"
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] | Loss: {avg_loss:.6f} "
                    f"| Time: {epoch_time:.1f}s | Improved: {improved_text}"
                )
            
            scheduler.step()
            prev_epoch_loss = avg_loss

            dist.barrier()
            if dist.get_rank() == 0:
                current_epoch = epoch
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss

                save_checkpoint(latest_path, current_epoch, model, optimizer, scheduler, avg_loss)
                validate_checkpoint_size(latest_path, MIN_CHECKPOINT_MB)

                if is_best:
                    save_checkpoint(best_path, current_epoch, model, optimizer, scheduler, avg_loss)
                    validate_checkpoint_size(best_path, MIN_CHECKPOINT_MB)
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    if "WORLD_SIZE" not in os.environ:
        print("Please run with torchrun, e.g. torchrun --nproc_per_node=4 training/train_sr.py")
    else:
        main()
