import torch
import torch.nn as nn
import torch.nn.functional as F
from dfn_blur_ops import apply_separable_filters, get_depth_mask

class DynamicFilterNetwork(nn.Module):
    """
    Dynamic Filter Network (DFN) for Depth-Aware Bokeh.
    Predicts separable 1D kernels (Horizontal and Vertical) for every pixel.
    Input: RGB (3) + Depth (1) -> 4 channels.
    Output: DSLR-style shallow depth RGB.
    """
    def __init__(self, kernel_size=11):
        super().__init__()
        self.k = kernel_size
        
        # 1. Feature Extractor (5 Convolution Layers)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 2. Kernel Prediction Heads (Separable 1D Kernels)
        # Separable filters reduce memory footprint significantly (N^2 -> 2N)
        # We need N weights for the horizontal pass and N weights for the vertical pass
        self.kernel_h = nn.Conv2d(64, self.k, kernel_size=3, padding=1)
        self.kernel_v = nn.Conv2d(64, self.k, kernel_size=3, padding=1)
        
    def forward(self, rgb, depth, focus_threshold=0.2):
        """
        rgb: [B, 3, H, W]
        depth: [B, 1, H, W] (normalized 0 to 1, where 1 is closest)
        focus_threshold: depth threshold representing the focal plane limit.
        """
        # Concatenate inputs
        x = torch.cat([rgb, depth], dim=1) # [B, 4, H, W]
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Predict spatial kernels [B, N, H, W]
        # Softmax over the kernel dimension ensures kernel weights sum to 1 (energy preservation)
        kh = F.softmax(self.kernel_h(features), dim=1)
        kv = F.softmax(self.kernel_v(features), dim=1)
        
        # 3. Apply Separable Kernels using custom unfold logic
        blurred_rgb = apply_separable_filters(rgb, kh, kv, self.k)
        
        # 4. Perform Depth-Aware Masking
        # Mask calculation logic prevents "foreground bleed" (halos) 
        # by keeping elements closer than the threshold perfectly sharp
        mask = get_depth_mask(depth, threshold=focus_threshold)
        
        # 5. Combine filtered output with in-focus region
        # mask = 1 -> foreground (keep original RGB)
        # mask = 0 -> background (use blurred RGB)
        final_rgb = mask * rgb + (1 - mask) * blurred_rgb
        
        return final_rgb, blurred_rgb, kh, kv, mask
