import os
import math
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models

from rrdb_model import RRDBNet
from dataset_pipeline import SRDataset

def calculate_psnr(img1, img2):
    """
    Computes Peak Signal-to-Noise Ratio for validation.
    Measures the ratio between maximum possible pixel value and MSE.
    Images must be in bounds [0, 1].
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    # 20 * log10(MAX / sqrt(MSE)). Since images are normalized [0, 1], MAX=1.0. 
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

class VGGPerceptualLoss(nn.Module):
    """
    Computes Perceptual divergence strictly up to 'relu4_3' to enforce 
    strong texture hallucinations via RRDBNet.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # In torchvision VGG16, 'relu4_3' is at layer index 22
        self.slice = vgg[:23]
        for param in self.parameters():
            param.requires_grad = False
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        x = (pred - self.mean) / self.std
        y = (target - self.mean) / self.std
        
        return nn.functional.l1_loss(self.slice(x), self.slice(y))

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
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Initializing RRDB Super Resolution Backend on: {device}")
    
    # 1. Dataset Config 
    # Extract patches mapping 64x64 -> 4x upscale -> 256x256
    dataset = SRDataset("data/training_images", crop_size=256, scale_factor=4)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_split, val_split = random_split(dataset, [train_size, val_size])
    
    # Batch size kept low (e.g., 4) to prevent OOM when passing 512p HR tensors through 23 blocks
    batch_size = 4
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, pin_memory=(device.type != "cpu"))
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False)
    
    # 2. RRDBNet (No adversarial framework)
    model = RRDBNet(in_channels=3, out_channels=3, num_features=64, num_blocks=23).to(device)
    
    # 3. Component Loss Functions
    l1_loss = nn.L1Loss().to(device)
    vgg_loss = VGGPerceptualLoss().to(device)
    
    # 4. Adam Optimizer & Schedulers
    optimizer = optim.Adam(model.parameters(), lr=2e-4) # Fixed 2e-4
    epochs = 100
    
    # Decay learning rate by Half at exactly 50% and 75% progression
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.75)], gamma=0.5)
    
    # Mixed precision handling
    use_scaler = (device.type == "cuda")
    scaler = GradScaler(enabled=use_scaler)
    
    # 5. Logging initialization 
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    csv_file = "logs/sr_training_metrics.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val PSNR (dB)"])
        
    best_psnr = 0.0
    
    print("\nStarting Patch-based Super-Resolution (L1 + Perceptual) Training Loop...")
    for epoch in range(epochs):
        
        # -- TRAIN LOOP --
        model.train()
        running_loss = 0.0
        
        for lr_patch, hr_patch in train_loader:
            lr_patch, hr_patch = lr_patch.to(device), hr_patch.to(device)
            optimizer.zero_grad()
            
            with autocast(enabled=(device.type in ["cuda", "mps"])):
                sr_patch = model(lr_patch)
                
                # Combine Explicit Loss Strategy
                loss_pixel = l1_loss(sr_patch, hr_patch)
                loss_perc = vgg_loss(sr_patch, hr_patch)
                
                # 0.05 weighting coefficient structurally calibrates the heavy VGG scale 
                # to parity with minimal pixel gradients.
                total_loss = loss_pixel + (0.05 * loss_perc)
                
            if use_scaler:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
                
            running_loss += total_loss.item()
            
        avg_train_loss = running_loss / max(len(train_loader), 1)
        
        # -- VALIDATION LOOP --
        model.eval()
        running_psnr = 0.0
        
        with torch.no_grad():
            for lr_patch, hr_patch in val_loader:
                lr_patch, hr_patch = lr_patch.to(device), hr_patch.to(device)
                with autocast(enabled=(device.type in ["cuda", "mps"])):
                    sr_patch = model(lr_patch)
                
                # The CNN outputs might drift barely outside [0, 1] bounds without the GAN discriminator
                # Clamp manually before calculating mathematical PSNR metric
                sr_clamped = torch.clamp(sr_patch, 0.0, 1.0)
                running_psnr += calculate_psnr(sr_clamped, hr_patch)
                
        avg_psnr = running_psnr / max(len(val_loader), 1)
        
        # Step the MultiStep Scheduler at the end of every epoch
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Validation PSNR: {avg_psnr:.2f}dB")
        print_gpu_memory(device)
        
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_psnr])
            
        # 6. Quality-based Checkpointing (PSNR Higher is better)
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            model_save_path = "checkpoints/sr_rrdb_v1.pth"
            
            torch.save(model.state_dict(), model_save_path)
            print(f" *** Record Validation Metrics! New Best Super-Resolution Checkpoint Extracted to: {model_save_path}")

if __name__ == '__main__':
    main()
