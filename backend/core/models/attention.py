import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Self-Attention module, inspired by SAGAN (Self-Attention Generative Adversarial Networks).
    Calculates attention maps to capture long-range dependencies in feature maps.
    This is highly beneficial for colorizing disconnected elements (e.g. matching shirt colors, or blue skies on opposite ends of a tree).
    Placed in the bottleneck where spatial resolution is low, making O(N^2) complexity manageable.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        # 1x1 convolutions to project inputs to query, key, value spaces
        # We reduce channels by 8 to save computation (standard practice)
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable scale parameter to gradually introduce attention outputs
        # initialized to 0 so the module acts perfectly as an identity map early in training
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        x shape: (Batch_Size, Channels, Height, Width)
        """
        batch_size, channels, height, width = x.size()
        
        # Project and flatten spatial dimensions for Key/Query spaces
        # query -> (B, H*W, C//8)
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        
        # key -> (B, C//8, H*W)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        
        # Calculate attention map: (B, H*W, C//8) x (B, C//8, H*W) -> (B, H*W, H*W)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1) # softmax over the keys dimension
        
        # Project and reshape value: (B, C, H*W)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        # Apply attention map to values: (B, C, H*W) x (B, H*W, H*W)^T -> (B, C, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        
        # Fold back to spatial tensor: (B, C, H, W)
        out = out.view(batch_size, channels, height, width)
        
        # Inject attention smoothly via gamma parameter
        out = self.gamma * out + x
        return out
