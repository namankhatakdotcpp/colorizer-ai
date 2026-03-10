import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.dataset_sr import SRDataset
from models.rrdb_sr import RRDBNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: SR DDP training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--data-root", type=str, default="datasets/div2k")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--run-name", type=str, default="stage2_sr")
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
    return int(checkpoint["epoch"]), float(checkpoint.get("best_metric", float("inf")))


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

    dataset = SRDataset(root_dir=args.data_root)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)

    model = DDP(RRDBNet().to(device), device_ids=[local_rank])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = torch.nn.L1Loss().to(device)

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

        for lr_img, hr_img in loader:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(lr_img)
            loss = criterion(pred, hr_img)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        stats = torch.tensor([running_loss], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        avg_loss = (stats.item() / world_size) / max(len(loader), 1)

        if rank == 0:
            print(f"Epoch [{epoch + 1}/{args.epochs}] | Loss: {avg_loss:.6f}")

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

    dist.destroy_process_group()


if __name__ == "__main__":
    if "WORLD_SIZE" not in os.environ:
        print("Please run with torchrun, e.g. torchrun --nproc_per_node=4 training/train_sr.py")
    else:
        main()
