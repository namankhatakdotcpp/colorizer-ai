import os
import csv
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from PIL import Image, ImageFilter
import torchvision.transforms as T

from micro_contrast_model import MicroContrastEnhancementCNN, MicroContrastLoss

import warnings
warnings.filterwarnings('ignore')

class MicroContrastDataset(Dataset):
    """
    Pairs soft/blurred images to raw sharp DSLR targets.
    Input images are generated on-the-fly dynamically to simulate lens/SR softness.
    """
    def __init__(self, root_dir, final_size=1024):
        super().__init__()
        self.size = final_size
        
        self.image_paths = []
        if os.path.exists(root_dir):
            for f in os.listdir(root_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root_dir, f))
        
        if not self.image_paths:
            print(f"Warning: Dataset empty in '{root_dir}'. Bootstrapping MicroContrast dummy data.")
            self.image_paths = ["dummy"] * 20
            self.is_dummy = True
        else:
            self.is_dummy = False
            
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def extract_patch(self, img):
        """ Force image strictly into 1024x1024 randomly """
        w, h = img.size
        # Fallback scaler if physical image is too small
        if w < self.size or h < self.size:
            img = img.resize((max(w, self.size), max(h, self.size)), Image.BICUBIC)
            
        w, h = img.size
        x = random.randint(0, max(0, w - self.size))
        y = random.randint(0, max(0, h - self.size))
        
        return img.crop((x, y, x + self.size, y + self.size))

    def __getitem__(self, idx):
        if self.is_dummy:
            soft_patch = torch.rand(3, self.size, self.size)
            sharp_patch = torch.rand(3, self.size, self.size)
            return soft_patch, sharp_patch
            
        img_path = self.image_paths[idx]
        sharp_img = Image.open(img_path).convert("RGB")
        
        # Crop exactly 1024x1024
        sharp_patch = self.extract_patch(sharp_img)
        
        # Degrade to simulate slight softness/Super-Resolution interpolation drift
        radius = random.uniform(0.5, 2.5)
        soft_patch = sharp_patch.filter(ImageFilter.GaussianBlur(radius))
        
        soft_tensor = self.to_tensor(soft_patch)
        sharp_tensor = self.to_tensor(sharp_patch)
        
        return soft_tensor, sharp_tensor

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
        
    print(f"Initializing Micro-Contrast Engine on: {device}")
    
    # 1. Dataset Config (Expects massive 1024p patches)
    dataset = MicroContrastDataset("data/training_images", final_size=1024)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_split, val_split = random_split(dataset, [train_size, val_size])
    
    # Batch size severely restricted (e.g., 1 or 2) due to massive 1024x1024 dimensionality passing through 8 residual blocks
    batch_size = 1
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, pin_memory=(device.type != "cpu"))
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False)
    
    # 2. Straight-through RGB -> RGB Micro-Contrast Architecture
    model = MicroContrastEnhancementCNN(in_channels=3, base_filters=64, num_blocks=8).to(device)
    
    # 3. High-Frequency Loss
    criterion = MicroContrastLoss().to(device)
    
    # 4. Optimizer & Scaler
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Gentle LR for refinement tasks
    epochs = 40 
    
    use_scaler = (device.type == "cuda")
    scaler = GradScaler(enabled=use_scaler)
    
    # 5. Logging initialization 
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    csv_file = "logs/micro_contrast_metrics.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Total", "Train Laplacian", "Train Sobel", "Val Total"])
        
    best_val_loss = float('inf')
    
    # ----------------------------------------------------
    # Explicit Gradient Accumulation Required for 1024x1024
    # Simulate a physical batch_size of 8
    # ----------------------------------------------------
    accumulate_steps = 8
    
    print("\nStarting Micro-Contrast Enhancement Training Loop...")
    for epoch in range(epochs):
        
        # -- TRAIN LOOP --
        model.train()
        running_total_loss = 0.0
        running_lap_loss = 0.0
        running_sob_loss = 0.0
        
        optimizer.zero_grad()
        
        for step, (soft_patch, sharp_patch) in enumerate(train_loader):
            soft_patch, sharp_patch = soft_patch.to(device), sharp_patch.to(device)
            
            with autocast(enabled=(device.type in ["cuda", "mps"])):
                enhanced_patch = model(soft_patch)
                total_loss, loss_l1, loss_lap, loss_sob = criterion(enhanced_patch, sharp_patch)
                
                # Normalize loss to account for accumulated iterations mathematically
                loss_scaled = total_loss / accumulate_steps
                
            if use_scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()
                
            # Perform optimization step only after X accumulates
            if (step + 1) % accumulate_steps == 0 or (step + 1) == len(train_loader):
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
            running_total_loss += total_loss.item()
            running_lap_loss += loss_lap.item()
            running_sob_loss += loss_sob.item()
            
        avg_train_loss = running_total_loss / max(len(train_loader), 1)
        avg_lap = running_lap_loss / max(len(train_loader), 1)
        avg_sob = running_sob_loss / max(len(train_loader), 1)
        
        # -- VALIDATION LOOP --
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for soft_patch, sharp_patch in val_loader:
                soft_patch, sharp_patch = soft_patch.to(device), sharp_patch.to(device)
                with autocast(enabled=(device.type in ["cuda", "mps"])):
                    enhanced_patch = model(soft_patch)
                    val_t_loss, _, _, _ = criterion(enhanced_patch, sharp_patch)
                running_val_loss += val_t_loss.item()
                
        avg_val_loss = running_val_loss / max(len(val_loader), 1)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train (Total:{avg_train_loss:.4f} Lap:{avg_lap:.4f} Sob:{avg_sob:.4f}) | Val: {avg_val_loss:.4f}")
        print_gpu_memory(device)
        
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_lap, avg_sob, avg_val_loss])
            
        # 6. Extraction
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = "checkpoints/micro_contrast_v1.pth"
            
            torch.save(model.state_dict(), model_save_path)
            print(f" *** New Best Micro-Contrast Model! Checkpoint saved to: {model_save_path}")

if __name__ == '__main__':
    main()
