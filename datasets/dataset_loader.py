import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from datasets.dataset_sr import SRDataset

class ColorizationGANDataset(Dataset):
    """
    Dataset for Stage 5 GAN Refinement.
    Loads paired grayscale/stage4 (input) and original color (target) images.
    """
    def __init__(self, input_dir, target_dir, transform_input=None, transform_target=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        
        # Keep track of matching image files in input directory
        # We assume the matching target has the exact same filename
        valid_exts = ('.png', '.jpg', '.jpeg')
        self.image_files = sorted([
            f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)
        ])
        
        # Verify matching targets exist for robustness
        self.image_files = [
            f for f in self.image_files if os.path.exists(os.path.join(target_dir, f))
        ]
        
        # Transforms handling
        self.transform_input = transform_input
        self.transform_target = transform_target
        
        if self.transform_input is None:
            self.transform_input = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
        if self.transform_target is None:
            # Add slight color jitter only for target to improve robustness
            self.transform_target = T.Compose([
                T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.1, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        input_path = os.path.join(self.input_dir, img_name)
        target_path = os.path.join(self.target_dir, img_name)
        
        # RGB conversion ensures 3 channels
        try:
            input_img = Image.open(input_path).convert("RGB")
            target_img = Image.open(target_path).convert("RGB")
        except Exception as e:
            # Handle bad files gracefully by returning another random image
            return self.__getitem__(random.randint(0, len(self)-1))
        
        # Synchronized Random Horizontal Flip
        if random.random() > 0.5:
            input_img = T.functional.hflip(input_img)
            target_img = T.functional.hflip(target_img)
            
        # Apply individual transforms
        input_tensor = self.transform_input(input_img)
        target_tensor = self.transform_target(target_img)
        
        return {
            "input": input_tensor,
            "target": target_tensor,
            "path": target_path
        }

def get_gan_dataloader(input_dir, target_dir, batch_size=16, num_workers=4, shuffle=True, pin_memory=True):
    dataset = ColorizationGANDataset(input_dir, target_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Important for GAN batch norm stability
    )
    return dataloader

__all__ = ["SRDataset", "ColorizationGANDataset", "get_gan_dataloader"]
