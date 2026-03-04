import os
import argparse
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

# Stage-Specific Imports
from models.depth_model import DynamicFilterNetwork
from datasets.dataset_depth import DepthDataset
from utils.tracker import ModelTracker

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Depth Pipeline")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # 1. Distributed Data Parallel (DDP) Matrix Initialization
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # NVIDIA Execution Optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    # 2. Stage-Specific Dataset
    train_dataset = DepthDataset(root_dir="data/depth/train")
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
    num_workers = min(8, os.cpu_count() // world_size)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )

    # 3. Model, Loss, Optimizer
    model = DynamicFilterNetwork(in_channels=3).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    criterion = torch.nn.L1Loss().to(device) # Simplified depth loss for example
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scaler = GradScaler() # Automatic Mixed Precision (AMP)

    # 4. Tracker & Synchronous Checkpoint Resumption
    tracker = None
    best_loss = float('inf')
    start_epoch = 0
    
    if rank == 0:
        tracker = ModelTracker(run_name="stage3_depth")
        
    resume_path = "checkpoints/stage3_depth_latest.pth"
    if os.path.isfile(resume_path):
        if rank == 0: print(f"Resuming stage 3 from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_metric']
        dist.barrier()

    # 5. Training Epoch Engine
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        r_loss_l1 = 0.0
        
        for batch_idx, (Images, DepthMaps) in enumerate(train_loader):
            Images, DepthMaps = Images.to(device), DepthMaps.to(device)
            optimizer.zero_grad()
            
            with autocast():
                Depth_pred = model(Images)
                total_loss = criterion(Depth_pred, DepthMaps)
                
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            r_loss_l1 += total_loss.item()
            
        stats = torch.tensor([r_loss_l1], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        avg_train_loss = (stats[0].item() / world_size) / max(1, len(train_loader))
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] | Depth Loss: {avg_train_loss:.4f}")
            tracker.log_metrics(epoch+1, train_loss=avg_train_loss, val_loss=0.0, psnr=0.0, ssim=0.0)
            
            is_best = avg_train_loss < best_loss
            if is_best:
                best_loss = avg_train_loss
            tracker.save_checkpoint(epoch+1, model, optimizer, best_loss, is_best)

    if rank == 0:
        tracker.close()
    dist.destroy_process_group()

if __name__ == '__main__':
    if "WORLD_SIZE" in os.environ:
        main()
    else:
        print("Please run this script using torchrun")
