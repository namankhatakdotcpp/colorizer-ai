import os
import csv
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import torchvision.transforms as T
import warnings

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dfn_model import DynamicFilterNetwork
from dfn_blur_ops import BokehLoss

warnings.filterwarnings('ignore')

class BokehDataset(Dataset):
    """
    Bokeh Dataset expects paired apertured shots: 
    - f/8 (Sharp, everything in focus) -> Base Input
    - f/1.4 (Shallow Depth, DSLR Bokeh) -> Ground Truth Target
    Additionally expects a Depth Map corresponding to the scene.
    """
    def __init__(self, root_dir, patch_size=512):
        super().__init__()
        self.size = patch_size
        
        self.sharp_paths = []
        self.bokeh_paths = []
        self.depth_paths = []
        
        sharp_dir = os.path.join(root_dir, 'f8')
        bokeh_dir = os.path.join(root_dir, 'f1_4')
        depth_dir = os.path.join(root_dir, 'depth')
        
        if os.path.exists(sharp_dir) and os.path.exists(bokeh_dir) and os.path.exists(depth_dir):
            for f in sorted(os.listdir(sharp_dir)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.sharp_paths.append(os.path.join(sharp_dir, f))
                    self.bokeh_paths.append(os.path.join(bokeh_dir, f))
                    self.depth_paths.append(os.path.join(depth_dir, f))
                    
        if not self.sharp_paths:
            self.sharp_paths = ["dummy"] * 20
            self.bokeh_paths = ["dummy"] * 20
            self.depth_paths = ["dummy"] * 20
            self.is_dummy = True
        else:
            self.is_dummy = False
            
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.sharp_paths)

    def extract_patch(self, img_sharp, img_bokeh, img_depth):
        w, h = img_sharp.size
        if w < self.size or h < self.size:
            # Upscale everything to at least fit the patch physically
            scale_w = max(w, self.size)
            scale_h = max(h, self.size)
            img_sharp = img_sharp.resize((scale_w, scale_h), Image.BICUBIC)
            img_bokeh = img_bokeh.resize((scale_w, scale_h), Image.BICUBIC)
            img_depth = img_depth.resize((scale_w, scale_h), Image.NEAREST) # MUST be nearest neighbor
            
        w, h = img_sharp.size
        # Get random matching coordinate window
        x = random.randint(0, max(0, w - self.size))
        y = random.randint(0, max(0, h - self.size))
        
        box = (x, y, x + self.size, y + self.size)
        return img_sharp.crop(box), img_bokeh.crop(box), img_depth.crop(box)

    def __getitem__(self, idx):
        if self.is_dummy:
            return torch.rand(3, self.size, self.size), \
                   torch.rand(3, self.size, self.size), \
                   torch.rand(1, self.size, self.size)
            
        sharp_img = Image.open(self.sharp_paths[idx]).convert("RGB")
        bokeh_img = Image.open(self.bokeh_paths[idx]).convert("RGB")
        # Depth is assumed to be single channel grayscale [0, 255]
        depth_img = Image.open(self.depth_paths[idx]).convert("L") 
        
        sharp_patch, bokeh_patch, depth_patch = self.extract_patch(sharp_img, bokeh_img, depth_img)
        
        sharp_tensor = self.to_tensor(sharp_patch)
        bokeh_tensor = self.to_tensor(bokeh_patch)
        depth_tensor = self.to_tensor(depth_patch) # Maps to normalized [0, 1] automatically
        
        return sharp_tensor, bokeh_tensor, depth_tensor

def print_gpu_memory(device):
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print(f" -> GPU Memory | Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    elif device.type == "mps":
        try:
            allocated = torch.mps.current_allocated_memory() / (1024 ** 2)
            print(f" -> MPS Memory | Allocated: {allocated:.2f} MB")
        except AttributeError:
            pass

def main():
    # DDP Initialization
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"Initializing DSLR Dynamic Filter Network (Bokeh) with DDP on {world_size} GPUs.")
    
    # 1. Dataset Config 
    # Extract 512x512 patches
    dataset = BokehDataset("data/bokeh_images", patch_size=512)
    if dataset.is_dummy and rank == 0:
        print(f"Warning: Bokeh dataset empty in data/bokeh_images. Bootstrapping dummy data for DFN.")
    
    # Deterministic split across ranks
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_split, val_split = random_split(dataset, [train_size, val_size], generator=generator)
    
    batch_size = 4
    train_sampler = DistributedSampler(train_split, shuffle=True)
    val_sampler = DistributedSampler(val_split, shuffle=False)
    
    # Optimal DataLoader settings for high-throughput training on A6000 GPUs
    # Maximize num_workers but cap it to prevent CPU thrashing across multiple GPUs
    cpu_count = os.cpu_count()
    num_workers = min(8, cpu_count // world_size) if cpu_count is not None else 4
    
    train_loader = DataLoader(
        train_split, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_split, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # 2. Architectures & Loss
    model = DynamicFilterNetwork(kernel_size=11).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    criterion = BokehLoss().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 40
    
    use_scaler = True
    scaler = GradScaler(enabled=use_scaler)
    
    csv_file = "logs/dfn_bokeh_metrics.csv"
    
    if rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Total", "Train L1", "Train Sobel", "Train KernelTV", "Val Loss"])
            
    best_val_loss = float('inf')
    
    if rank == 0:
        print("\nStarting Edge-Aware DFN Bokeh Training Loop...")
        
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        # -- TRAIN LOOP --
        model.train()
        r_total, r_l1, r_sob, r_ktv = 0.0, 0.0, 0.0, 0.0
        
        for sharp_rgb, bokeh_rgb, depth in train_loader:
            sharp_rgb = sharp_rgb.to(device)
            bokeh_rgb = bokeh_rgb.to(device)
            depth = depth.to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=use_scaler):
                # Set subjective threshold (e.g. 0.2 focuses on near camera objects)
                pred_rgb, _, kh, kv, _ = model(sharp_rgb, depth, focus_threshold=0.2)
                
                total_loss, loss_l1, loss_focus, loss_k_tv = criterion(pred_rgb, bokeh_rgb, kh, kv)
                
            scaler.scale(total_loss).backward()
            
            # Unscale gradients to clip them and avoid NaN issues
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
                
            r_total += total_loss.item()
            r_l1 += loss_l1.item()
            r_sob += loss_focus.item()
            r_ktv += loss_k_tv.item()
            
        # Aggregate training stats over all processes
        stats = torch.tensor([r_total, r_l1, r_sob, r_ktv], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        stats /= world_size
        r_total, r_l1, r_sob, r_ktv = stats.tolist()

        N = max(len(train_loader), 1)
        a_total, a_l1, a_sob, a_ktv = r_total/N, r_l1/N, r_sob/N, r_ktv/N
        
        # -- VALIDATION LOOP --
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for sharp_rgb, bokeh_rgb, depth in val_loader:
                sharp_rgb = sharp_rgb.to(device)
                bokeh_rgb = bokeh_rgb.to(device)
                depth = depth.to(device)
                
                with autocast(enabled=use_scaler):
                    pred_rgb, _, kh, kv, _ = model(sharp_rgb, depth, focus_threshold=0.2)
                    v_tot, _, _, _ = criterion(pred_rgb, bokeh_rgb, kh, kv)
                    
                running_val_loss += v_tot.item()
                
        # Aggregate validation stat over all processes
        val_stat = torch.tensor([running_val_loss], device=device)
        dist.all_reduce(val_stat, op=dist.ReduceOp.SUM)
        running_val_loss = val_stat.item() / world_size

        avg_val_loss = running_val_loss / max(len(val_loader), 1)
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Total: {a_total:.4f} | Val Total: {avg_val_loss:.4f}")
            print_gpu_memory(device)
            
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, a_total, a_l1, a_sob, a_ktv, avg_val_loss])
                
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_save_path = "checkpoints/dfn_bokeh_v1.pth"
                
                # Unwrap model.module to get original un-DDP model dict
                torch.save(model.module.state_dict(), model_save_path)
                print(f" *** Optical Fidelity Improved. Checkpoint saved to: {model_save_path}")

    dist.destroy_process_group()

if __name__ == '__main__':
    if "WORLD_SIZE" in os.environ:
        main()
    else:
        print("Please run this script using torchrun, e.g.:")
        print("torchrun --nproc_per_node=8 train_dfn.py")
