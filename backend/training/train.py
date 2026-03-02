import os
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import numpy as np
from skimage.color import rgb2lab
import warnings

from model import UNetColorizer
from losses.losses import CompositeLoss

warnings.filterwarnings('ignore')

class ColorizationDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        if os.path.exists(root_dir):
            for f in os.listdir(root_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root_dir, f))
        
        # Prevent crash if user runs before downloading dataset. Uses dummy data.
        if not self.image_paths:
            print(f"Warning: Dataset empty in '{root_dir}'. Bootstrapping memory pipeline purely for script testing.")
            self.image_paths = ["dummy"] * 20
            self.is_dummy = True
        else:
            self.is_dummy = False
            
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.is_dummy:
            # Produce validly scaled synthetic dummy data bounds
            L = torch.rand(1, 256, 256) * 100.0
            ab = (torch.rand(2, 256, 256) - 0.5) * 2.0  # Range [-1, 1]
            return L.float(), ab.float()
            
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((256, 256))
        
        img_np = np.array(img).astype(np.float32) / 255.0
        lab = rgb2lab(img_np)
        
        # LAB normalizations explicitly demanded: 
        # L belongs to [0, 100]
        # AB belongs to real distribution normalized to [-1, 1] mapped internally logic
        L = lab[:, :, 0]
        ab = lab[:, :, 1:3] / 128.0
        
        L_tensor = torch.tensor(L).unsqueeze(0).float()
        ab_tensor = torch.tensor(ab).permute(2, 0, 1).float()
        
        return L_tensor, ab_tensor

def print_gpu_memory(device):
    """
    Tracks and prints GPU memory usage per epoch depending on hardware architecture backend.
    """
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print(f" -> GPU Memory | Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    elif device.type == "mps":
        try:
            allocated = torch.mps.current_allocated_memory() / (1024 ** 2)
            print(f" -> MPS Memory | Allocated: {allocated:.2f} MB")
        except AttributeError:
            print(" -> MPS Memory tracking not supported in this torch build.")

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Activating advanced UNet processing backend. Device Context: {device}")
    
    # 1. Dataset Configuration & Train/Val split
    dataset = ColorizationDataset("data/training_images")
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_split, val_split = random_split(dataset, [train_size, val_size])
    
    print(f"Data mapping: {train_size} Training | {val_size} Validation images loaded.")
    
    batch_size = 4
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, pin_memory=(device.type != "cpu"))
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False)
    
    # 2. Advanced UNet (Base filters strictly 64, exactly 4-level deep) Output is LAB
    model = UNetColorizer(base_filters=64).to(device)
    
    # 3. Composite Objective Loss
    criterion = CompositeLoss().to(device)
    
    # 4. Optimizer & Schedulers
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Adam at LR=1e-4
    epochs = 60
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # We use AMP GradScaler purely if CUDA is verified. MPS handles autocast internally without Scaler natively
    use_scaler = (device.type == "cuda")
    scaler = GradScaler(enabled=use_scaler)
    
    # 5. Logging initialization
    os.makedirs("checkpoints", exist_ok=True)
    csv_file = "logs/training_metrics.csv"
    os.makedirs("logs", exist_ok=True)
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss"])
        
    best_val_loss = float('inf')
    
    print("\nStarting Advanced Structural UNet Training Loop...")
    
    for epoch in range(epochs):
        # -- TRAIN LOOP --
        model.train()
        running_train_loss = 0.0
        
        for L, ab in train_loader:
            L, ab = L.to(device), ab.to(device)
            optimizer.zero_grad()
            
            # Autocast context for Mixed Precision Memory optimization
            with autocast(enabled=(device.type in ["cuda", "mps"])):
                pred_ab = model(L)
                total_loss, l1_loss, vgg_loss, ssim_l = criterion(pred_ab, ab, L)
                
            if use_scaler:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
                
            running_train_loss += total_loss.item()
            
        avg_train_loss = running_train_loss / max(len(train_loader), 1)

        # -- VALIDATION LOOP --
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for L, ab in val_loader:
                L, ab = L.to(device), ab.to(device)
                with autocast(enabled=(device.type in ["cuda", "mps"])):
                    pred_ab = model(L)
                    v_loss, _, _, _ = criterion(pred_ab, ab, L)
                running_val_loss += v_loss.item()
                
        avg_val_loss = running_val_loss / max(len(val_loader), 1)
        
        # Update Scheduler natively tracking steps per epoch 
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print_gpu_memory(device)
        
        # 6. CSV Export Log
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss])
            
        # 7. Optimal Checkpoint Guarding mechanism
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = "checkpoints/unet_colorizer_v1.pth"
            
            # Save strictly safely to state dict natively bypassing object refs
            torch.save(model.state_dict(), model_save_path)
            print(f" *** Improved. Optimal Checkpoint saved to: {model_save_path}")

if __name__ == '__main__':
    main()
