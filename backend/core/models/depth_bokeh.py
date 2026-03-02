import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Bilinear upsample is preferred over ConvTranspose for depth continuity
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return F.relu(self.conv(x), inplace=True)

class DepthCNN(nn.Module):
    """
    Simplified CNN-based Depth Estimator (MiDaS v2.1 philosophy).
    Uses a minimal ResNet-style encoder and a progressive upsampling decoder.
    Focuses on Relative Scale Depth.
    """
    def __init__(self):
        super().__init__()
        # Simulated Encoder (in practice, load a pretrained ResNet-50 here)
        self.enc1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # -> 1/2
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # -> 1/4
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # -> 1/8
        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # -> 1/16

        # Decoder with Skip Connections via concatenation
        self.dec4 = UpSampleBlock(512, 256)   # -> 1/8
        self.dec3 = UpSampleBlock(512, 128)   # (256+256 skip) -> 1/4
        self.dec2 = UpSampleBlock(256, 64)    # (128+128 skip) -> 1/2
        
        self.final_up = nn.Sequential(
            UpSampleBlock(128, 64),           # (64+64 skip) -> 1/1
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # Normalizes relative depth to [0, 1]
        )

    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        e4 = F.relu(self.enc4(e3))

        d4 = self.dec4(e4)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        
        depth = self.final_up(torch.cat([d2, e1], dim=1))
        return depth


class DepthAwareBokehDFN(nn.Module):
    """
    Dynamic Filter Network for Spatially Variant Bokeh.
    Predicts a unique convolution kernel for EVERY pixel based on the RGB and Depth Map,
    allowing for edge-aware, DSLR-accurate "circles of confusion".
    """
    def __init__(self, kernel_size=9):
        super().__init__()
        self.k = kernel_size
        
        # Kernel Prediction Network (Given RGB + Depth)
        # We output k*k channels -> Represents the flattened Custom Kernel for each pixel
        self.kernel_predictor = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Output channels = kernel_size^2
            nn.Conv2d(64, self.k * self.k, kernel_size=3, padding=1)
        )

    def forward(self, rgb, depth):
        # 1. Combine inputs
        x = torch.cat([rgb, depth], dim=1) # Shape: (B, 4, H, W)
        
        # 2. Predict unique filters per pixel
        B, _, H, W = rgb.size()
        # Shape: (B, K^2, H, W)
        dynamic_filters = self.kernel_predictor(x) 
        
        # Normalize the predicted kernels so they sum to 1 (Ensures energy preservation / no arbitrary brightness shifts)
        # Softmax over the channel dimension (the K^2 kernel elements)
        dynamic_filters = F.softmax(dynamic_filters, dim=1) 
        
        # 3. Apply the dynamic filters spatially explicitly
        # Extract sliding local patches from the RGB image
        # padding ensures output size matches input size
        pad = self.k // 2
        # rgb_unfolded: (B, C * K^2, H * W)
        rgb_unfolded = F.unfold(rgb, kernel_size=self.k, padding=pad) 
        
        # Reshape for element-wise multiplication
        # rgb_unfolded: (B, C, K^2, H*W)
        rgb_unfolded = rgb_unfolded.view(B, 3, self.k * self.k, H * W)
        
        # filters: (B, 1, K^2, H*W) -> The '1' broadcasts across the 3 RGB channels
        filters = dynamic_filters.view(B, 1, self.k * self.k, H * W)
        
        # Multiply patch pixels by their respective predicted dynamic kernel weights
        # Sum across the K^2 patch dimension to produce the final filtered pixel
        # output: (B, 3, H*W)
        bokeh_output = torch.sum(rgb_unfolded * filters, dim=2) 
        
        # Fold back to image shape
        bokeh_output = bokeh_output.view(B, 3, H, W)
        return bokeh_output

