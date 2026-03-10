import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.dataset_colorizer import ColorizationDataset
from models.unet_colorizer import UNetColorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: UNet colorizer DDP training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-root", type=str, default="datasets/flickr2k")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--run-name", type=str, default="stage1_colorizer")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def save_checkpoint(path: Path, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, best_metric: float) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
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
    start_epoch = int(checkpoint["epoch"])
    best_metric = float(checkpoint.get("best_metric", float("inf")))
    return start_epoch, best_metric


def _warn_checkpoint_size(rank: int, ckpt_path: Path) -> None:
    if rank != 0 or not ckpt_path.exists():
        return

    size_mb = ckpt_path.stat().st_size / (1024 * 1024)
    print(f"Checkpoint size: {size_mb:.2f} MB ({ckpt_path})")
    if size_mb < 80 or size_mb > 150:
        print("WARNING: checkpoint size outside expected 80MB-150MB range.")


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this DDP training script (backend=nccl).")

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cudnn.benchmark = True

    dataset = ColorizationDataset(root_dir=args.data_root, augment=True)
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

    model = UNetColorizer(in_channels=1, out_channels=2).to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = torch.nn.L1Loss().to(device)
    # SGD keeps optimizer state compact so checkpoint size reflects model weights.
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.0, weight_decay=1e-5)

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

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        running_loss = 0.0
        for l_channel, ab_target in loader:
            l_channel = l_channel.to(device, non_blocking=True)
            ab_target = ab_target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            ab_pred = model(l_channel)
            loss = criterion(ab_pred, ab_target)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        stats = torch.tensor([running_loss], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        avg_loss = (stats.item() / world_size) / max(len(loader), 1)

        if rank == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}] | Loss: {avg_loss:.6f}")

        # Ensure all ranks complete compute before rank-0 writes checkpoint.
        dist.barrier()
        if dist.get_rank() == 0:
            current_epoch = epoch + 1
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            save_checkpoint(latest_path, current_epoch, model, optimizer, best_loss)
            if is_best:
                save_checkpoint(best_path, current_epoch, model, optimizer, best_loss)
        dist.barrier()

    _warn_checkpoint_size(rank, latest_path)
    dist.destroy_process_group()


if __name__ == "__main__":
    if "WORLD_SIZE" not in os.environ:
        print("Please run with torchrun, e.g. torchrun --nproc_per_node=4 training/train_colorizer.py")
    else:
        main()
