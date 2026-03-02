import torch
import torch.nn as nn
from .blocks import ResidualBlock, DownBlock, UpBlock
from .attention import SelfAttention

class UNetColorizer(nn.Module):
    """
    Production-grade U-Net for Image Colorization in PyTorch.
    Input: L channel (1 dimension)
    Output: ab channels (2 dimensions)
    
    Architecture features:
    - Residual blocks in encoder and decoder for robust gradient propagation.
    - LeakyReLU to prevent dead neurons, which is critical if extended into a GAN framework.
    - BatchNorm2d for stable internal covariate shift.
    - Skip connections retaining high-frequency spatial details.
    - Self-Attention in the deepest bottleneck to leverage global image context.
    - Configurable base filters scaling memory size dynamically.
    """
    def __init__(self, input_channels=1, output_channels=2, min_filters=64, use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        
        # ---------------------------------------------------------
        # Encoder (Downsampling)
        # ---------------------------------------------------------
        # Initial convolution to map input shape to initial min_filters without downsampling grid
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels, min_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(min_filters),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Progressive spatial downsampling
        self.down1 = DownBlock(min_filters, min_filters * 2)        # -> 128
        self.down2 = DownBlock(min_filters * 2, min_filters * 4)    # -> 256
        self.down3 = DownBlock(min_filters * 4, min_filters * 8)    # -> 512
        
        # ---------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------
        # Deepest layer, processes highly abstract semantic features (e.g., detecting "this is grass")
        self.bottleneck_conv = ResidualBlock(min_filters * 8, min_filters * 8)
        
        if self.use_attention:
            self.attention = SelfAttention(min_filters * 8)
            
        # ---------------------------------------------------------
        # Decoder (Upsampling with Skip Connections)
        # ---------------------------------------------------------
        # Skip concatenation: UpBlock handles upconv, concats skip tensor, then processes via ResBlock
        self.up1 = UpBlock(in_channels=min_filters * 8, skip_channels=min_filters * 4, out_channels=min_filters * 4)
        self.up2 = UpBlock(in_channels=min_filters * 4, skip_channels=min_filters * 2, out_channels=min_filters * 2)
        self.up3 = UpBlock(in_channels=min_filters * 2, skip_channels=min_filters, out_channels=min_filters)
        
        # ---------------------------------------------------------
        # Output Head
        # ---------------------------------------------------------
        # Final convolution maps deep feature space strictly to [A, B] color channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(min_filters, min_filters // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(min_filters // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(min_filters // 2, output_channels, kernel_size=3, padding=1),
            nn.Tanh() # Tanh tightly restricts output strictly to [-1, 1], aligning precisely with LAB normalization
        )

        # Apply robust generative initialization
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass defining the U-Net spatial routes and skip connection concatenations.
        x: FloatTensor of Shape (Batch, 1, 256, 256) [Standard]
        """
        # --- Encoder Path ---
        skip1 = self.init_conv(x)      # Yields: (B, 64, 256, 256)
        skip2 = self.down1(skip1)      # Yields: (B, 128, 128, 128)
        skip3 = self.down2(skip2)      # Yields: (B, 256, 64, 64)
        encoded = self.down3(skip3)    # Yields: (B, 512, 32, 32)
        
        # --- Bottleneck ---
        bottleneck = self.bottleneck_conv(encoded)
        if self.use_attention:
            bottleneck = self.attention(bottleneck)
            
        # --- Decoder Path ---
        # Skips are attached in reverse order matching the descending hierarchy
        decoded = self.up1(bottleneck, skip3) # Outputs: (B, 256, 64, 64)
        decoded = self.up2(decoded, skip2)    # Outputs: (B, 128, 128, 128)
        decoded = self.up3(decoded, skip1)    # Outputs: (B, 64, 256, 256)
        
        # --- Output Head ---
        out = self.final_conv(decoded)        # Final: (B, 2, 256, 256)
        
        return out

    def _initialize_weights(self):
        """
        Applies standard normal initialization for Conv layers (established best practice for GANs/SR)
        and constants for BatchNorm to break symmetry effectively.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
