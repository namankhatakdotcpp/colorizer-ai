import torch
import numpy as np
import torchvision.transforms as T
from skimage import color

class RGBToLAB(object):
    """
    Custom transform to convert an RGB image (Tensor in [0,1]) to LAB color space.
    Returns targeted L channel and AB channels normalized for deep learning models.
    """
    def __call__(self, img_tensor: torch.Tensor):
        # 1. T.ToTensor() outputs a FloatTensor (3, H, W) in range [0, 1].
        # Transpose back to (H, W, 3) for scikit-image
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        # 2. scikit-image rgb2lab expects float RGB in [0, 1] 
        lab = color.rgb2lab(img_np)
        
        # 3. Extract channels
        # L axis ranges from [0, 100], A and B axes range roughly from [-128, 127]
        l = lab[:, :, 0:1]
        ab = lab[:, :, 1:3]
        
        # 4. Normalize L to [-1, 1] (or [0, 1] based on your architecture preference, -1 to 1 is standard)
        l_norm = (l / 50.0) - 1.0
        
        # Normalize AB to [-1, 1] safely. Dividing by 110.0 covers ~99.5% of standard natural image colors.
        ab_norm = ab / 110.0
        
        # 5. Convert back to tensors: (H, W, C) -> (C, H, W)
        l_tensor = torch.from_numpy(l_norm).permute(2, 0, 1).float()
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1).float()
        
        return l_tensor, ab_tensor

def get_train_transforms(image_size: int = 256):
    """
    Spatial & Color augmentations for training.
    We augment the RGB image first, which produces consistent valid L + AB targets downstream.
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        # ColorJitter brightness changes L heavily without breaking the semantic relation to AB.
        # Changing contrast/saturation directly augments the target color space effectively.
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),  # Converts PIL Image to tensor [0, 1]
    ])

def get_val_transforms(image_size: int = 256):
    """
    Validation transforms (No random augmentations, strict deterministic resizing).
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
