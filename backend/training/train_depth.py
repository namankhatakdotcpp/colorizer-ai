import os
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import torchvision.transforms as T
import warnings

from depth_model import MiDaSResNet50
from ssil_loss import SSILoss

warnings.filterwarnings('ignore')

class DepthDataset(Dataset):
    """
    Pairs raw RGB inputs with Ground Truth Depth Maps.
    Resizes inputs natively to 384x384 (MiDaS strict spatial constraint).
    """
    def __init__(self, root_dir, size=384):
        super().__init__()
        self.size = size
        
        self.rgb_paths = []
        self.depth_paths = []
        
        rgb_dir = os.path.join(root_dir, 'rgb')
        depth_dir = os.path.join(root_dir, 'depth')
        
        if os.path.exists(rgb_dir) and os.path.exists(depth_dir):
            for f in sorted(os.listdir(rgb_dir)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.rgb_paths.append(os.path.join(rgb_dir, f))
                    # Assumes corresponding depth map has exactly the same base filename
                    self.depth_paths.append(os.path.join(depth_dir, f))
                    
        if not self.rgb_paths:
            print(f"Warning: Depth dataset empty in '{root_dir}'. Bootstrapping dummy data for MiDaS.")
            self.rgb_paths = ["dummy"] * 20
            self.depth_paths = ["dummy"] * 20
            self.is_dummy = True
        else:
            self.is_dummy = False
            
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.rgb_paths)

    def extract_patch(self, img_rgb, img_depth):
        img_rgb = img_rgb.resize((self.size, self.size), Image.BICUBIC)
        img_depth = img_depth.resize((self.size, self.size), Image.NEAREST) # MUST be nearest neighbor to prevent interpolated depth edges
        return img_rgb, img_depth

    def __getitem__(self, idx):
        if self.is_dummy:
            return torch.rand(3, self.size, self.size), torch.rand(1, self.size, self.size)
            
        rgb_img = Image.open(self.rgb_paths[idx]).convert("RGB")
        depth_img = Image.open(self.depth_paths[idx]).convert("L") # Single channel depth
        
        rgb_img, depth_img = self.extract_patch(rgb_img, depth_img)
        
        rgb_tensor = self.to_tensor(rgb_img)
        depth_tensor = self.to_tensor(depth_img)
        
        return rgb_tensor, depth_tensor

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
        
    print(f"Initializing MiDaS Depth Estimator Backend on: {device}")
    
    # 1. Dataset Config 
    dataset = DepthDataset("data/depth_images", size=384)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_split, val_split = random_split(dataset, [train_size, val_size])
    
    batch_size = 8
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, pin_memory=(device.type != "cpu"))
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False)
    
    # 2. Architectures & Loss Setup
    # Flag set natively to freeze the first two ResNet-50 stages initially 
    model = MiDaSResNet50(freeze_early_stages=True).to(device)
    criterion = SSILoss().to(device)
    
    # 3. Optimizer Configuration - Fine-tuning initialized at extremely low LR
    optimizer = optim.Adam(model.parameters(), lr=1e-5) 
    epochs = 40
    
    use_scaler = (device.type == "cuda")
    scaler = GradScaler(enabled=use_scaler)
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    csv_file = "logs/depth_midas_metrics.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train SSIL", "Val SSIL"])
        
    best_val_loss = float('inf')
    
    print("\nStarting Scale-and-Shift Invariant Depth Training Loop...")
    for epoch in range(epochs):
        
        # Unfreeze encoder explicitly halfway through training to refine high-frequency edges natively
        if epoch == epochs // 2:
            print("\n*** Unfreezing full ResNet-50 Encoder for Phase 2 fine-tuning! ***")
            for param in model.layer1.parameters():
                param.requires_grad = True
            for param in model.layer2.parameters():
                param.requires_grad = True
                
        # -- TRAIN LOOP --
        model.train()
        running_train_loss = 0.0
        
        for rgb, depth in train_loader:
            rgb, depth = rgb.to(device), depth.to(device)
            optimizer.zero_grad()
            
            with autocast(enabled=(device.type in ["cuda", "mps"])):
                pred_depth = model(rgb)
                
                # Mask out completely invalid depth areas dynamically (e.g. skies equal to 0.0)
                mask = (depth > 0.001).float()
                
                loss_ssil = criterion(pred_depth, depth, mask)
                
            if use_scaler:
                scaler.scale(loss_ssil).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_ssil.backward()
                optimizer.step()
                
            running_train_loss += loss_ssil.item()

        avg_train_loss = running_train_loss / max(len(train_loader), 1)
        
        # -- VALIDATION LOOP --
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for rgb, depth in val_loader:
                rgb, depth = rgb.to(device), depth.to(device)
                
                with autocast(enabled=(device.type in ["cuda", "mps"])):
                    pred_depth = model(rgb)
                    mask = (depth > 0.001).float()
                    val_t_loss = criterion(pred_depth, depth, mask)
                    
                running_val_loss += val_t_loss.item()
                
        avg_val_loss = running_val_loss / max(len(val_loader), 1)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train SSIL: {avg_train_loss:.4f} | Val SSIL: {avg_val_loss:.4f}")
        print_gpu_memory(device)
        
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss])
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = "checkpoints/depth_midas_v1.pth"
            
            # Save final checkpoint
            torch.save(model.state_dict(), model_save_path)
            print(f" *** Optimal Alignment Secured. Checkpoint mapped to: {model_save_path}")

if __name__ == '__main__':
    main()
