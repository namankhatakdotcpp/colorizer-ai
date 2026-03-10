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
from models.unet_colorizer import UNetColorizer
from datasets.dataset_colorizer import ColorizationDataset
from utils.losses import HybridColorizationLoss
from utils.tracker import ModelTracker

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Colorizer Pipeline")
    parser.add_argument("--epochs", type=int, default=250)
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
    train_dataset = ColorizationDataset(root_dir="datasets/flickr2k")

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    num_workers = min(8, os.cpu_count() // world_size)
    train_loader = DataLoader(
    train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )



    # 3. Model, Loss, Optimizer
    model = UNetColorizer(in_channels=1, out_channels=2).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    criterion = HybridColorizationLoss(l1_weight=1.0, perceptual_weight=0.2, ssim_weight=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    scaler = GradScaler() # Automatic Mixed Precision (AMP)

    # 4. Tracker & Synchronous Checkpoint Resumption
    tracker = None
    best_loss = float('inf')
    start_epoch = 0
    
    if rank == 0:
        tracker = ModelTracker(run_name="stage1_colorizer")
        
    resume_path = "checkpoints/stage1_colorizer_latest.pth"
    if os.path.isfile(resume_path):
        if rank == 0: print(f"Resuming stage 1 from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict']) # Load to generic module
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_metric']
        dist.barrier() # Sync ranks before training

    # 5. Training Epoch Engine
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch) # Reshuffle data safely across cluster!
        model.train()
        r_loss_l1 = 0.0
        
        for batch_idx, (L_channel, AB_target) in enumerate(train_loader):
            L_channel, AB_target = L_channel.to(device), AB_target.to(device)
            optimizer.zero_grad()
            
            with autocast():
                AB_pred = model(L_channel)
                RGB_pred = torch.cat([L_channel, AB_pred], dim=1) # Mock differentiable concat for VGG
                RGB_target = torch.cat([L_channel, AB_target], dim=1)
                
                # Hybrid Loss
                total_loss, loss_l1, loss_vgg, loss_ssim = criterion(AB_pred, AB_target, RGB_pred, RGB_target)
                
            # Mixed Precision Stepping
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            r_loss_l1 += loss_l1.item()
            
        # Synchronize Train Loss globally
        stats = torch.tensor([r_loss_l1], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        avg_train_loss = (stats[0].item() / world_size) / max(1, len(train_loader))
        
        # Logging & Checkpointing only on master node
        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] | Colorizer L1 Loss: {avg_train_loss:.4f}")
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
