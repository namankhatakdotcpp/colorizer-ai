import argparse
import os
import time
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import VGG16_Weights, vgg16

from datasets.dataset_colorizer import build_combined_colorization_dataset
from models.unet_colorizer import UNetColorizer


STAGE_NAME = "stage1_colorizer"
DEFAULT_DATA_ROOTS = ("datasets/flickr2k", "datasets/coco", "datasets/div2k")
MIN_BATCHES_PER_EPOCH = 10
MIN_CHECKPOINT_MB = 80.0
MAX_CHECKPOINT_MB = 150.0


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enabled = True
        try:
            backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.features = backbone.features[:16].eval()
            for param in self.features.parameters():
                param.requires_grad = False
        except Exception as exc:
            # Keep training stable in offline environments where VGG weights are unavailable.
            print(f"[WARN] Failed to load pretrained VGG16 for perceptual loss: {exc}")
            self.enabled = False
            self.features = torch.nn.Identity()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    @staticmethod
    def _lab_proxy_rgb(l_channel: torch.Tensor, ab_channel: torch.Tensor) -> torch.Tensor:
        # Fast proxy to map LAB tensors into 3 channels for VGG feature comparison.
        return torch.cat([l_channel, (ab_channel + 1.0) * 0.5], dim=1).clamp(0.0, 1.0)

    def forward(self, pred_ab: torch.Tensor, target_ab: torch.Tensor, l_channel: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return pred_ab.new_tensor(0.0)

        pred_rgb = self._lab_proxy_rgb(l_channel, pred_ab)
        target_rgb = self._lab_proxy_rgb(l_channel, target_ab)

        pred_feat = self.features(self._normalize(pred_rgb))
        with torch.no_grad():
            target_feat = self.features(self._normalize(target_rgb))
        return F.l1_loss(pred_feat, target_feat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: UNet colorizer DDP training")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-root", type=str, default=None, help="Legacy single-dataset override.")
    parser.add_argument("--data-roots", type=str, nargs="+", default=list(DEFAULT_DATA_ROOTS))
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--run-name", type=str, default=STAGE_NAME)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-log-interval", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--perceptual-weight", type=float, default=0.1)
    parser.add_argument("--chroma-var-weight", type=float, default=0.01)
    parser.add_argument("--disable-perceptual-loss", action="store_true")
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    loss_value: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
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
    start_epoch = int(checkpoint["epoch"]) + 1
    best_metric = float(checkpoint.get("best_metric", float("inf")))
    return start_epoch, best_metric


def validate_checkpoint_size(path: Path, min_mb: float = MIN_CHECKPOINT_MB, max_mb: float = MAX_CHECKPOINT_MB) -> float:
    checkpoint_size = path.stat().st_size / (1024 * 1024)
    if checkpoint_size < min_mb:
        raise RuntimeError(
            f"Checkpoint too small ({checkpoint_size:.2f} MB) at {path}. Model not saved correctly."
        )
    if checkpoint_size > max_mb:
        raise RuntimeError(
            f"Checkpoint too large ({checkpoint_size:.2f} MB) at {path}. Expected range is {min_mb:.0f}-{max_mb:.0f} MB."
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
    sample_l, _ = next(iter(loader))
    sample_l = sample_l.to(device, non_blocking=True)
    with torch.no_grad():
        output = model(sample_l)
    if rank == 0:
        print(f"Sanity check output shape: {tuple(output.shape)}")


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this DDP training script (backend=nccl).")

    data_roots: List[str] = [args.data_root] if args.data_root else args.data_roots
    if len(data_roots) == 0:
        raise RuntimeError("No dataset roots configured for Stage1 training.")

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cudnn.benchmark = True

    try:
        dataset, dataset_stats = build_combined_colorization_dataset(
            data_roots=data_roots,
            augment=True,
            image_size=args.image_size,
            warn_min_total_samples=50000,
        )
        if len(dataset) == 0:
            raise RuntimeError("Dataset is empty")

        sampler = DistributedSampler(dataset, shuffle=True)

        num_workers = max(0, min(args.num_workers, (os.cpu_count() or 1) // max(world_size, 1)))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
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
            print("Stage1 datasets:")
            for root, count in dataset_stats.items():
                print(f"  - {root}: {count}")
            print(f"Dataset size: {len(dataset)}")
            print(f"Batch size: {args.batch_size}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"GPUs: {world_size}")
            print(f"Image resolution: {image_resolution}")
            print(f"First batch shape: {first_batch_shape}")

        model = UNetColorizer(in_channels=1, out_channels=2).cuda(local_rank)
        total_params = sum(p.numel() for p in model.parameters())
        if total_params < 1_000_000:
            raise RuntimeError("Model architecture incorrect")
        if rank == 0:
            print(f"Model parameters: {total_params:,}")
        model = DDP(model, device_ids=[local_rank])

        criterion = torch.nn.L1Loss().to(device)
        perceptual_loss_fn = VGGPerceptualLoss().to(device)
        perceptual_loss_fn.eval()
        if args.disable_perceptual_loss:
            perceptual_loss_fn.enabled = False
        if rank == 0 and not perceptual_loss_fn.enabled:
            print("[WARN] Perceptual loss disabled.")

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.0, weight_decay=1e-5)

        run_sanity_check(loader, model, device, rank)

        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        latest_path = ckpt_dir / f"{args.run_name}_latest.pth"
        best_path = ckpt_dir / f"{args.run_name}_best.pth"

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

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            sampler.set_epoch(epoch)
            model.train()

            running_loss = 0.0
            running_l1_loss = 0.0
            running_perceptual_loss = 0.0
            running_color_var = 0.0
            zero_grad_counter = 0

            for step, (l_channel, ab_target) in enumerate(loader, start=1):
                l_channel = l_channel.to(device, non_blocking=True)
                ab_target = ab_target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                ab_pred = model(l_channel)
                l1_loss = criterion(ab_pred, ab_target)
                p_loss = perceptual_loss_fn(ab_pred, ab_target, l_channel)
                color_var = torch.var(ab_pred)
                loss = l1_loss + args.perceptual_weight * p_loss + args.chroma_var_weight * color_var
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
                running_l1_loss += float(l1_loss.item())
                running_perceptual_loss += float(p_loss.item())
                running_color_var += float(color_var.item())

            stats = torch.tensor([running_loss, running_l1_loss, running_perceptual_loss, running_color_var], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            denom = world_size * max(steps_per_epoch, 1)
            avg_loss = float(stats[0].item()) / denom
            avg_l1_loss = float(stats[1].item()) / denom
            avg_perceptual_loss = float(stats[2].item()) / denom
            avg_color_var = float(stats[3].item()) / denom
            epoch_time = time.time() - epoch_start

            if rank == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] | Loss: {avg_loss:.6f} "
                    f"(L1: {avg_l1_loss:.6f}, P: {avg_perceptual_loss:.6f}, Var: {avg_color_var:.6f}) "
                    f"| Steps: {steps_per_epoch} | Time: {epoch_time:.1f}s"
                )

            dist.barrier()
            if dist.get_rank() == 0:
                current_epoch = epoch
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss

                save_checkpoint(latest_path, current_epoch, model, optimizer, best_loss, avg_loss)
                size_mb = validate_checkpoint_size(latest_path, MIN_CHECKPOINT_MB)
                print(f"Checkpoint size: {size_mb:.2f} MB ({latest_path})")

                if is_best:
                    save_checkpoint(best_path, current_epoch, model, optimizer, best_loss, avg_loss)
                    validate_checkpoint_size(best_path, MIN_CHECKPOINT_MB)
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    if "WORLD_SIZE" not in os.environ:
        print("Please run with torchrun, e.g. torchrun --nproc_per_node=4 training/train_colorizer.py")
    else:
        main()
