import argparse
import os
import time
from pathlib import Path
from typing import List

from PIL import Image
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.functional as TF

from models.rrdb_sr import RRDBNet


STAGE_NAME = "stage2_sr"
MIN_BATCHES_PER_EPOCH = 21
MIN_CHECKPOINT_MB = 20.0
EXPECTED_CHECKPOINT_MIN_MB = 30.0
EXPECTED_CHECKPOINT_MAX_MB = 80.0


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
    parser = argparse.ArgumentParser(description="Stage 2: SR DDP training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--flickr-root", type=str, default="datasets/flickr2k")
    parser.add_argument("--div2k-root", type=str, default="datasets/div2k")
    parser.add_argument("--coco-root", type=str, default="datasets/coco/train2017")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--run-name", type=str, default=STAGE_NAME)
    parser.add_argument("--grad-log-interval", type=int, default=20)
    return parser.parse_args()


def save_checkpoint(path: Path, epoch: int, model: DDP, optimizer: torch.optim.Optimizer, loss_value: float) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": float(loss_value),
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
        num_workers = 8
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )

        steps_per_epoch = len(loader)
        if steps_per_epoch == 0:
            raise RuntimeError("No batches produced")
        if steps_per_epoch < MIN_BATCHES_PER_EPOCH:
            raise RuntimeError("Dataset too small")

        first_batch = next(iter(loader))
        first_batch_shape = tuple(first_batch[0].shape)

        if rank == 0:
            print(f"Dataset size: {len(dataset)}")
            print(f"Batch size: {args.batch_size}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"First batch shape: {first_batch_shape}")

        model = RRDBNet().cuda(local_rank)
        total_params = sum(p.numel() for p in model.parameters())
        if total_params < 1_000_000:
            raise RuntimeError("Model architecture incorrect")
        if rank == 0:
            print(f"Model parameters: {total_params:,}")
        model = DDP(model, device_ids=[local_rank])
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        criterion = torch.nn.L1Loss().to(device)

        run_sanity_check(loader, model, device, rank)

        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        latest_path = ckpt_dir / f"{STAGE_NAME}_latest.pth"
        best_path = ckpt_dir / f"{STAGE_NAME}_best.pth"

        start_epoch = 0
        best_loss = float("inf")
        if latest_path.exists():
            if rank == 0:
                print(f"Resuming from {latest_path}")
            start_epoch, best_loss = load_checkpoint(latest_path, model, optimizer, device)

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

            for step, (lr_img, hr_img) in enumerate(loader, start=1):
                lr_img = lr_img.to(device, non_blocking=True)
                hr_img = hr_img.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred = model(lr_img)
                loss = criterion(pred, hr_img)
                if torch.isnan(loss):
                    raise RuntimeError("NaN loss detected")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                grad_norm = compute_grad_norm(model)
                if step % args.grad_log_interval == 0 and rank == 0:
                    print(f"Grad norm: {grad_norm:.6e}")

                if grad_norm <= 1e-12:
                    zero_grad_counter += 1
                    if zero_grad_counter >= 5:
                        raise RuntimeError("Gradients are zero for 5 consecutive checks. Stopping training.")
                else:
                    zero_grad_counter = 0

                optimizer.step()
                running_loss += float(loss.item())

            stats = torch.tensor([running_loss], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            avg_loss = (stats.item() / world_size) / max(steps_per_epoch, 1)
            epoch_time = time.time() - epoch_start
            improved_vs_prev = prev_epoch_loss is None or avg_loss < prev_epoch_loss

            if rank == 0:
                improved_text = "yes" if improved_vs_prev else "no"
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] | Loss: {avg_loss:.6f} "
                    f"| Steps: {steps_per_epoch} | Time: {epoch_time:.1f}s | Improved: {improved_text}"
                )
            prev_epoch_loss = avg_loss

            dist.barrier()
            if dist.get_rank() == 0:
                current_epoch = epoch
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss

                save_checkpoint(latest_path, current_epoch, model, optimizer, avg_loss)
                validate_checkpoint_size(latest_path, MIN_CHECKPOINT_MB)

                if is_best:
                    save_checkpoint(best_path, current_epoch, model, optimizer, avg_loss)
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
