import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class SRDataset(Dataset):
    """
    Super-Resolution Dataset Generator optimized for DIV2K and Flickr2K.
    Dynamically applies RandomCrop(256) on HR images, pads if needed, and generates LR via 4x bicubic downscaling.
    """
    def __init__(self, root_dir, crop_size=256, scale_factor=4):
        super().__init__()
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.lr_size = crop_size // scale_factor
        
        self.image_paths = []
        if os.path.exists(root_dir):
            # Recursively find images to seamlessly support DIV2K & Flickr2K tree structures
            for root, _, files in os.walk(root_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(root, f))
        
        if not self.image_paths:
            print(f"Warning: Dataset empty in '{root_dir}'. Bootstrapping SR pipeline dummy data.")
            self.image_paths = ["dummy"] * 20
            self.is_dummy = True
        else:
            self.is_dummy = False
            
        # Torchvision Transforms for pipeline standardization
        self.hr_transform = T.Compose([
            T.RandomCrop(self.crop_size, pad_if_needed=True, padding_mode='reflect'),
            T.ToTensor() # implicitly normalizes to [0, 1]
        ])
        
        self.lr_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.lr_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor() # implicitly normalizes to [0, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.is_dummy:
            # Generate synthetic bounds matching exactly the HR/LR scale constraints
            hr = torch.rand(3, self.crop_size, self.crop_size)
            lr = torch.rand(3, self.lr_size, self.lr_size)
            return lr, hr
            
        img_path = self.image_paths[idx]
        hr_img = Image.open(img_path).convert("RGB")
        
        # Execute PyTorch Transform chaining: HR Crop (with padding if needed) -> LR Degrade Projection
        hr_tensor = self.hr_transform(hr_img)
        lr_tensor = self.lr_transform(hr_tensor)
        
        return lr_tensor, hr_tensor
