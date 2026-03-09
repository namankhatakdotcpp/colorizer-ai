import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from .transforms import get_train_transforms, get_val_transforms, RGBToLAB

class ColorizationDataset(Dataset):
    """
    PyTorch Dataset wrapper for ImageNet ILSVRC structure dedicated to Colorization.
    It expects standard PyTorch ImageFolder hierarchy:
    root_dir/
      ├─ train/
      │  ├─ n01440764/
      │  └─ ...
      └─ val/
         ├─ n01440764/
         └─ ...
    """
    def __init__(self, root_dir: str, split: str = "train", image_size: int = 256):
        self.root_dir = Path(root_dir) / split
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Image folder not found: {self.root_dir}")
            
        # ImageFolder efficiently handles massive nested directories
        self.image_folder = ImageFolder(root=str(self.root_dir))
        
        # Define transform pipeline based on mode
        if split == "train":
            self.spatial_transforms = get_train_transforms(image_size)
        else:
            self.spatial_transforms = get_val_transforms(image_size)
            
        self.rgb_to_lab = RGBToLAB()

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx: int):
        # In ImageFolder, .samples is a tuple of (path, class_idx). We ignore the class target.
        path, _ = self.image_folder.samples[idx]
        
        try:
            # Enforce RGB to safely handle RGBA, CMYK, or purely grayscale base images 
            img = Image.open(path).convert("RGB")
            
            # Step 1: Execute spatial augmentations and to_tensor mapping (returns FloatTensor in [0, 1])
            img_tensor = self.spatial_transforms(img)
            
            # Step 2: Convert strictly to LAB format
            l_channel, ab_channels = self.rgb_to_lab(img_tensor)
            
            # Return standard dictionary representation for DataLoader
            return {
                "l_channel": l_channel,    # Shape: (1, 256, 256)
                "ab_channels": ab_channels # Shape: (2, 256, 256)
            }
            
        except Exception as e:
            # ImageNet sometimes features corrupt file headers.
            # Production Datasets securely skip bad data by recursively requesting the next index.
            print(f"[Warning] Failed to load {path} - Error: {e}. Skipping to next...")
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)
