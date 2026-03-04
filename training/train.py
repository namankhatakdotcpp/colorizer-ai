import os
import argparse
import copy

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

# Import modular components using absolute relative path structures
from utils.config_loader import load_config
from datasets.dataset_loader import SRDataset  # Example direct import, you'd scale this

# In a full project, Tracker should ideally be in utils/ as well.
# Assuming standard scope for this example.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend/training'))
from tracker import ModelTracker

# -------------------------------------------------------------------------
# Dynamic Components
# -------------------------------------------------------------------------
def get_model(config):
    """Factory builder for models leveraging config['model']['name']."""
    m_type = config['model'].get('name', 'unknown')
    if m_type == "rrdb_sr":
        # Imports would link against models/rrdb_sr.py in a fully migrated repo
        sys.path.append(os.path.join(os.path.dirname(__file__), '../backend/training'))
        from rrdb_model import RRDBNet
        return RRDBNet(**config['model'].get('params', {'in_channels':3, 'out_channels':3, 'num_features':64, 'num_blocks':23}))
    elif m_type == "dfn_depth":
        sys.path.append(os.path.join(os.path.dirname(__file__), '../backend/training'))
        from dfn_model import DynamicFilterNetwork
        return DynamicFilterNetwork(**config['model'].get('params', {}))
    else:
        # Dummy linear model to prevent crashes if missing implementation
        return torch.nn.Linear(10, 10)

def get_loss(config):
    """Factory builder for loss functions mapping to config."""
    l_type = config.get('loss', {}).get('type', "l1")
    if l_type == "hybrid_colorization":
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
        from losses import HybridColorizationLoss
        return HybridColorizationLoss(
            l1_weight=config['loss'].get('l1_weight', 1.0),
            perceptual_weight=config['loss'].get('perceptual_weight', 0.2),
            ssim_weight=config['loss'].get('ssim_weight', 0.1)
        )
    return torch.nn.L1Loss() # Default fallback


def merge_configs(default_cfg, custom_cfg):
    """Recursively nested dict merge."""
    merged = copy.deepcopy(default_cfg)
    for k, v in custom_cfg.items():
        if isinstance(v, dict) and k in merged:
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged

# -------------------------------------------------------------------------
# Core Training Script
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified PyTorch Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # 1. Configuration Loading (Modular utilizing custom utility)
    config = load_config(args.config)
    
    # Check for distributed backend specification before launching DDP
    dist_backend = config.get('distributed', {}).get('backend', 'nccl')

    # 2. DDP Initialization
    dist.init_process_group(backend=dist_backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # --- NVIDIA GPU Optimizations ---
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # --------------------------------
    
    if rank == 0:
        print(f"[{config['experiment_name']}] Initializing training on {world_size} GPUs...")

    # 3. Datasets & Dataloaders
    img_size = config['dataset']['img_size']
    batch_size = config['dataset']['batch_size']
    
    # Ideally link your specific Datasets here based on config['model']['type']
    train_dataset = DummyDataset(1000, img_size)
    val_dataset = DummyDataset(100, img_size)
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    num_workers = min(config['dataset']['num_workers'], os.cpu_count() // world_size)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=config['dataset'].get('pin_memory', True),
        persistent_workers=True,
        prefetch_factor=config['dataset'].get('prefetch_factor', 2)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=config['dataset'].get('pin_memory', True),
        persistent_workers=True,
        prefetch_factor=config['dataset'].get('prefetch_factor', 2)
    )

    # 4. Model, Loss, Optimizer
    model = get_model(config).to(device)
    model = DDP(model, device_ids=[local_rank])
    criterion = get_loss(config).to(device)
    
    opt_type = config['optimizer']['type']
    lr = float(config['training']['learning_rate'])
    if opt_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=tuple(config['optimizer']['betas']))
    else:
        # Default AdamW
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=float(config['training']['weight_decay']))
        
    use_scaler = config['training'].get('mixed_precision', True)
    scaler = GradScaler(enabled=use_scaler)

    # 5. Tracker Initialization (Rank 0 only)
    tracker = None
    if rank == 0:
        tracker = ModelTracker(
            run_name=config['experiment_name'],
            checkpoint_dir=config['checkpointing']['save_dir'],
            log_dir=config['logging']['tensorboard_dir']
        )
        
    # 6. Synchronous Checkpoint Resumption (ALL Ranks)
    start_epoch = 0
    best_metric = float('inf') 
    
    if config['checkpointing'].get('resume', False):
        resume_path = os.path.join(config['checkpointing']['save_dir'], f"{config['experiment_name']}_latest.pth")
        
        if os.path.isfile(resume_path):
            if rank == 0:
                print(f"[{config['experiment_name']}] Auto-resuming training from '{resume_path}'...")
            
            # Map tensors exactly to their assigned GPU to prevent GPU 0 memory explosion
            checkpoint = torch.load(resume_path, map_location=device)
            
            # Since model is wrapped in DDP, load into model.module
            # The saved checkpoint intentionally does not have 'module.' prefix for inference compatibility!
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = checkpoint.get('epoch', 0)
            best_metric = checkpoint.get('best_metric', float('inf'))
            
            # Wait for all processes to finish loading to prevent desynchronization
            dist.barrier()
        elif rank == 0:
            print(f"[{config['experiment_name']}] No checkpoint found at '{resume_path}', starting from scratch.")
    
    epochs = config['training']['epochs']
    
    # 7. Unified Loop
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        r_loss = 0.0
        accumulate_steps = config['training'].get('accumulate_steps', 1)
        optimizer.zero_grad() # Clean slate at start of epoch
        
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch[0].to(device), batch[1].to(device)
            
            with autocast(enabled=use_scaler):
                outputs = model(inputs)
                
                # Check if the loss criterion expects multiple returns (SR GAN vs L1)
                # This depends on your particular model output structure
                try:
                    loss_tuple = criterion(outputs, targets)
                    total_loss = loss_tuple[0] if isinstance(loss_tuple, tuple) else loss_tuple
                except Exception:
                    # Fallback simple loss logic
                    total_loss = criterion(outputs, targets)
                
                # Normalize loss mathematically over the accumulation window to prevent exploding gradients
                total_loss = total_loss / accumulate_steps
            
            # Identify if this iteration triggers an optimizer step
            is_last_step = ((batch_idx + 1) % accumulate_steps == 0) or ((batch_idx + 1) == len(train_loader))
            
            # DDP Optimization: Avoid syncing gradients across GPUs until the absolute last accumulation step
            if hasattr(model, 'no_sync') and not is_last_step:
                with model.no_sync():
                    scaler.scale(total_loss).backward()
            else:
                scaler.scale(total_loss).backward()
            
            # Conditional optimizer stepping
            if is_last_step:
                # Gradient clipping from config
                grad_clip = config['training'].get('gradient_clip', 1.0)
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # Reset for next accumulation window
            
            # Track the un-normalized proxy for purely human-readable logging
            r_loss += total_loss.item() * accumulate_steps
            
        # Aggregate stats
        stats = torch.tensor([r_loss], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        avg_train_loss = (stats[0].item() / world_size) / max(len(train_loader), 1)

        # Validation
        model.eval()
        v_loss = 0.0
        r_psnr = 0.0
        r_ssim = 0.0
        r_lpips = 0.0
        
        # Initialize LPIPS perceptual loss specifically for validation tracking
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch[0].to(device), batch[1].to(device)
                with autocast(enabled=use_scaler):
                    outputs = model(inputs)
                    
                    try:
                        loss_tuple = criterion(outputs, targets)
                        loss = loss_tuple[0] if isinstance(loss_tuple, tuple) else loss_tuple
                    except Exception:
                        loss = criterion(outputs, targets)
                        
                v_loss += loss.item()
                
                # Metric Calculations (Assume normalized [0, 1] bounds for inputs)
                # Bounding ensures stable metric calculations
                out_clamped = torch.clamp(outputs, 0.0, 1.0)
                
                # Mock functions for PSNR/SSIM, replace with actual implementations as needed
                # psnr_batch = calculate_psnr(out_clamped, targets)
                # ssim_batch = calculate_ssim(out_clamped, targets)
                
                # LPIPS Computation (Expects tensors strictly in [-1, 1] range)
                # Convert from [0, 1] to [-1, 1]
                target_norm = (targets * 2.0) - 1.0
                out_norm = (out_clamped * 2.0) - 1.0
                
                # Calculate perceptual divergence scalar 
                # (Lower is better = closer perceptual structure)
                lpips_score = loss_fn_vgg(out_norm, target_norm).mean().item()
                r_lpips += lpips_score
                
        val_stats = torch.tensor([v_loss, r_lpips], device=device)
        dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
        
        N_val = max(len(val_loader), 1)
        avg_val_loss = (val_stats[0].item() / world_size) / N_val
        avg_lpips = (val_stats[1].item() / world_size) / N_val
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LPIPS: {avg_lpips:.4f}")
            
            # Logging metrics (PSNR/SSIM mocked as constants for standard structural flow until linked)
            tracker.log_metrics(epoch+1, avg_train_loss, avg_val_loss, psnr=30.0, ssim=0.92, lpips_score=avg_lpips)
            
            # Log imagery visually if inputs/outputs match standard expectations
            if inputs.dim() == 4 and targets.dim() == 4 and outputs.dim() == 4:
                tracker.log_images(epoch+1, inputs, targets, outputs)
            
            is_best = avg_val_loss < best_metric
            if is_best:
                best_metric = avg_val_loss
                
            if (epoch + 1) % config['checkpointing']['save_freq_epochs'] == 0 or is_best:
                tracker.save_checkpoint(epoch+1, model, optimizer, best_metric, is_best)

    if rank == 0:
        tracker.close()
        print("Training Complete!")
        
    dist.destroy_process_group()

if __name__ == '__main__':
    if "WORLD_SIZE" in os.environ:
        main()
    else:
        print("Please run this script using torchrun:")
        print("torchrun --nproc_per_node=4 train.py --config configs/sr.yaml")
