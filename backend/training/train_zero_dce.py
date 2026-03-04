import os
import csv
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import torchvision.transforms as T

from zero_dce_model import ZeroDCENet
from zero_dce_losses import ZeroDCELoss

import warnings
warnings.filterwarnings('ignore')

class ZeroReferenceDataset(Dataset):
    """
    Zero-Reference Dataset.
    Zero-DCE does not require paired input/target images! 
    It trains self-supervised via mathematical constraint losses across randomized single LDR/HDR imagery.
    """
    def __init__(self, root_dir, crop_size=512):
        super().__init__()
        self.crop_size = crop_size
        self.image_paths = []
        
        if os.path.exists(root_dir):
            for f in os.listdir(root_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root_dir, f))
                    
        if not self.image_paths:
            print(f"Warning: Dataset empty in '{root_dir}'. Bootstrapping Zero-DCE dummy data.")
            self.image_paths = ["dummy"] * 20
            self.is_dummy = True
        else:
            self.is_dummy = False
            
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def extract_patch(self, img):
        # Scale to ensure random crop works
        w, h = img.size
        if w < self.crop_size or h < self.crop_size:
            img = img.resize((max(w, self.crop_size), max(h, self.crop_size)), Image.BICUBIC)
            
        w, h = img.size
        x = random.randint(0, max(0, w - self.crop_size))
        y = random.randint(0, max(0, h - self.crop_size))
        
        return img.crop((x, y, x + self.crop_size, y + self.crop_size))

    def __getitem__(self, idx):
        if self.is_dummy:
            # Returns single un-paired image format [B, 3, H, W]
            return torch.rand(3, self.crop_size, self.crop_size)
            
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        patch = self.extract_patch(img)
        # Normalize to [0, 1] automatically via transform
        patch_tensor = self.to_tensor(patch)
        
        return patch_tensor

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
        
    print(f"Initializing Zero-DCE Tone Mapping Backend on: {device}")
    
    # 1. Dataset Config 
    dataset = ZeroReferenceDataset("data/training_images", crop_size=512)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_split, val_split = random_split(dataset, [train_size, val_size])
    
    # Batch size can be larger compared to other modules since ZeroDCE is only 7 layers
    batch_size = 8
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, pin_memory=(device.type != "cpu"))
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False)
    
    # 2. Architectures & Loss Setup
    model = ZeroDCENet().to(device)
    criterion = ZeroDCELoss().to(device)
    
    # 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4) 
    epochs = 100
    
    use_scaler = (device.type == "cuda")
    scaler = GradScaler(enabled=use_scaler)
    
    # 4. Storage Logging
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    csv_file = "logs/zero_dce_metrics.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Total", "Train Spa", "Train Exp", "Train Col", "Train TV", "Val Loss"])
        
    best_val_loss = float('inf')
    
    print("\nStarting Self-Supervised Zero-DCE HDR Training Loop...")
    for epoch in range(epochs):
        # -- TRAIN LOOP --
        model.train()
        
        r_total, r_spa, r_exp, r_col, r_tv = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for img in train_loader:
            img = img.to(device)
            optimizer.zero_grad()
            
            with autocast(enabled=(device.type in ["cuda", "mps"])):
                enhanced_img, curve_params = model(img)
                # No 'target' image provided! Purely self-supervised mathematical losses
                total_loss, l_spa, l_exp, l_col, l_tv = criterion(img, enhanced_img, curve_params)
                
            if use_scaler:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
                
            r_total += total_loss.item()
            r_spa += l_spa.item()
            r_exp += l_exp.item()
            r_col += l_col.item()
            r_tv += l_tv.item()
            
        N = max(len(train_loader), 1)
        a_total, a_spa, a_exp, a_col, a_tv = r_total/N, r_spa/N, r_exp/N, r_col/N, r_tv/N
        
        # -- VALIDATION LOOP --
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for img in val_loader:
                img = img.to(device)
                with autocast(enabled=(device.type in ["cuda", "mps"])):
                    enhanced_img, curve_params = model(img)
                    val_t_loss, _, _, _, _ = criterion(img, enhanced_img, curve_params)
                running_val_loss += val_t_loss.item()
                
        avg_val_loss = running_val_loss / max(len(val_loader), 1)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {a_total:.4f} | Val Loss: {avg_val_loss:.4f}")
        print_gpu_memory(device)
        
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, a_total, a_spa, a_exp, a_col, a_tv, avg_val_loss])
            
        # 5. Output Weights Extraction
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = "checkpoints/zero_dce_v1.pth"
            
            # Extract
            torch.save(model.state_dict(), model_save_path)
            print(f" *** Mathematical Constraints Satisfied! Checkpoint mapped to: {model_save_path}")

if __name__ == '__main__':
    main()
