import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelFilter(nn.Module):
    """
    Fixed weights Convolutional layer to extract horizontal and vertical edges.
    Used internally to guide the Micro-Contrast enhancement branch.
    """
    def __init__(self):
        super().__init__()
        # Pre-define Sobel Kernels
        sobel_x = torch.tensor([[-1., 0., 1.], 
                                [-2., 0., 2.], 
                                [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], 
                                [0., 0., 0.], 
                                [1., 2., 1.]]).view(1, 1, 3, 3)
        
        # We duplicate the kernel for 3-channel RGB (Depth-wise convolution)
        self.register_buffer('sobel_x', sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(3, 1, 1, 1))
        
    def forward(self, x):
        # Apply depth-wise convolution to calculate gradients
        edge_x = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        edge_y = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        
        # Magnitude of gradient
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        return edge_mag

class ResidualRefinementBlock(nn.Module):
    """
    A lightweight Dual-Conv residual block to refine local geometries.
    No Pooling or Striding to strictly preserve original resolution spatial integrity.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = self.conv2(self.lrelu(self.conv1(x)))
        return x + residual

class MicroContrastEnhancer(nn.Module):
    """
    Deep CNN for DSLR-level micro-contrast extraction.
    Utilizes a dedicated high-frequency branch (Sobel) alongside a standard RGB branch
    to explicitly enhance edges without globally over-sharpening flat regions (halo prevention).
    """
    def __init__(self, in_channels=3, base_filters=64, num_residual_blocks=8):
        super().__init__()
        
        # Physical Edge Extractor (Fixed, No Gradients Required)
        self.sobel = SobelFilter()
        
        # ----------------------------------------------------
        # Branch 1: Frequency / Edge Processing (Sobel)
        # ----------------------------------------------------
        self.edge_conv_in = nn.Conv2d(in_channels, base_filters // 2, kernel_size=3, padding=1)
        self.edge_blocks = nn.Sequential(*[
            ResidualRefinementBlock(base_filters // 2) for _ in range(4)
        ])
        self.edge_conv_out = nn.Conv2d(base_filters // 2, base_filters // 2, kernel_size=3, padding=1)

        # ----------------------------------------------------
        # Branch 2: Standard Pixel Intensity (RGB)
        # ----------------------------------------------------
        self.rgb_conv_in = nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1)
        self.rgb_blocks = nn.Sequential(*[
            ResidualRefinementBlock(base_filters) for _ in range(num_residual_blocks)
        ])
        
        # Multi-scale feature extraction for RGB (capturing localized contrast patches)
        self.dilated_conv1 = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=2, dilation=2)
        self.dilated_conv2 = nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=4, dilation=4)

        # ----------------------------------------------------
        # Fusion & Refinement
        # ----------------------------------------------------
        # Fuses (RGB features 64) + (Dilated RGB features 64+64) + (Edge features 32)
        total_fused_channels = base_filters * 3 + (base_filters // 2)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_fused_channels, base_filters, kernel_size=1, bias=False), # 1x1 BottleNeck 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.final_refinement = nn.Sequential(*[
            ResidualRefinementBlock(base_filters) for _ in range(3)
        ])
        
        self.output_conv = nn.Conv2d(base_filters, in_channels, kernel_size=3, padding=1)

    def forward(self, rgb_in):
        # 1. Edge Extraction Branch
        edge_map = self.sobel(rgb_in)
        edge_features = self.edge_conv_in(edge_map)
        edge_features = self.edge_blocks(edge_features)
        edge_features = self.edge_conv_out(edge_features)

        # 2. RGB Processing Branch
        rgb_features_base = self.rgb_conv_in(rgb_in)
        rgb_features = self.rgb_blocks(rgb_features_base)
        
        # 3. Multi-scale context (helps identify local contrast anomalies)
        d1 = F.leaky_relu(self.dilated_conv1(rgb_features), 0.2)
        d2 = F.leaky_relu(self.dilated_conv2(rgb_features), 0.2)
        
        # 4. Fusion
        fused = torch.cat([rgb_features, d1, d2, edge_features], dim=1)
        fused = self.fusion_conv(fused)
        
        # 5. Output mapping and Global Residual (Ensures model only learns the "difference" or "enhancement")
        refined = self.final_refinement(fused)
        out = self.output_conv(refined)
        
        # Global residual scalar to strictly add/subtract micro-contrast from the baseline image
        return out + rgb_in

