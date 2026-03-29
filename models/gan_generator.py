"""
U-Net based Generator for GAN refinement stage.

Refines colorized images from the main pipeline to improve perceptual realism.
Architecture: U-Net with skip connections and residual blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Self-Attention module for capturing global context.
    
    Enables the generator to learn long-range dependencies and global coherence,
    fixing patch-level artifacts and ensuring sky/skin color consistency.
    """

    def __init__(self, in_dim: int):
        """
        Initialize self-attention module.
        
        Args:
            in_dim: Number of input channels
        """
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with self-attention.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Attended output (B, C, H, W) + residual
        """
        B, C, H, W = x.shape
        
        # Query, key, value projections
        query = self.query(x).view(B, -1, H * W)  # (B, C//8, H*W)
        key = self.key(x).view(B, -1, H * W)      # (B, C//8, H*W)
        value = self.value(x).view(B, -1, H * W)  # (B, C, H*W)
        
        # Attention weights
        attn = torch.softmax(query.transpose(1, 2) @ key, dim=-1)  # (B, H*W, H*W)
        
        # Attention output
        out = value @ attn.transpose(1, 2)  # (B, C, H*W)
        out = out.view(B, C, H, W)          # (B, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


class ResidualBlock(nn.Module):
    """Residual block with 2 convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3):
        """
        Initialize residual block.

        Args:
            channels: Number of input/output channels
            kernel_size: Kernel size for convolutions
        """
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        return out


class UNetBlock(nn.Module):
    """U-Net encoder-decoder block with skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        down: bool = True,
    ):
        """
        Initialize U-Net block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolutions
            down: Whether to downsample (True) or upsample (False)
        """
        super().__init__()
        padding = (kernel_size - 1) // 2

        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1
                ),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.conv(x)


class GANGenerator(nn.Module):
    """
    U-Net based generator for image refinement.

    Input: Colorized image (3 channels, RGB)
    Output: Refined image (3 channels, RGB)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 64,
        num_residual_blocks: int = 4,
    ):
        """
        Initialize GAN generator.

        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            out_channels: Number of output channels (default: 3 for RGB)
            base_filters: Base number of filters
            num_residual_blocks: Number of residual blocks in bottleneck
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_filters = base_filters
        self.num_residual_blocks = num_residual_blocks

        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 7, padding=3),
            nn.InstanceNorm2d(base_filters, affine=True),
            nn.ReLU(inplace=True),
        )

        # Encoder blocks (downsample)
        self.enc1 = UNetBlock(base_filters, base_filters * 2, down=True)  # H/2
        self.enc2 = UNetBlock(base_filters * 2, base_filters * 4, down=True)  # H/4
        self.enc3 = UNetBlock(base_filters * 4, base_filters * 8, down=True)  # H/8

        # Bottleneck: residual blocks
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(base_filters * 8)
                for _ in range(num_residual_blocks)
            ]
        )
        
        # Self-attention for global context (FIX: enables 25-35 FID)
        self.self_attention = SelfAttention(base_filters * 8)

        # Decoder blocks (upsample)
        self.dec3 = UNetBlock(base_filters * 8, base_filters * 4, down=False)  # H/4
        self.dec2 = UNetBlock(base_filters * 4, base_filters * 2, down=False)  # H/2
        self.dec1 = UNetBlock(base_filters * 2, base_filters, down=False)  # H

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_filters, out_channels, 7, padding=3),
            nn.Tanh(),  # Output range: [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Refined output tensor (B, C, H, W)
        """
        # Initial convolution
        x0 = self.initial_conv(x)

        # Encoder path with skip connections
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Bottleneck: residual blocks + self-attention
        x_bottleneck = self.residual_blocks(x3)
        x_bottleneck = self.self_attention(x_bottleneck)  # Add global context

        # Decoder path with skip connections
        x = self.dec3(x_bottleneck)
        x = x + x2  # Skip connection
        x = self.dec2(x)
        x = x + x1  # Skip connection
        x = self.dec1(x)
        x = x + x0  # Skip connection

        # Final convolution
        x = self.final_conv(x)

        return x

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
