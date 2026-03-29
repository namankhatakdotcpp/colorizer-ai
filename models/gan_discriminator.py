"""
PatchGAN Discriminator for image refinement adversarial training.

Classifies patches of images as real or generated to provide feedback
to the generator during training.

Uses Spectral Normalization to stabilize training.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator.

    Classifies local patches of the image (not global) to provide
    fine-grained feedback for improved perceptual quality.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 64,
        num_layers: int = 4,
        patch_size: int = 70,
    ):
        """
        Initialize PatchGAN discriminator.

        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            base_filters: Base number of filters
            num_layers: Number of convolutional layers
            patch_size: Receptive field size (output patch size)
        """
        super().__init__()

        self.in_channels = in_channels
        self.base_filters = base_filters
        self.num_layers = num_layers
        self.patch_size = patch_size

        # Build discriminator layers with Spectral Normalization
        layers = []
        in_channels_current = in_channels

        for i in range(num_layers):
            out_channels = base_filters * min(2 ** i, 8)

            if i == 0:
                # First layer: no normalization, but with spectral norm
                layers.extend([
                    spectral_norm(nn.Conv2d(in_channels_current, out_channels, 4, stride=2, padding=1)),
                    nn.LeakyReLU(0.2, inplace=True),
                ])
            elif i == num_layers - 1:
                # Last layer: stride=1 with spectral norm
                layers.extend([
                    spectral_norm(nn.Conv2d(in_channels_current, out_channels, 4, stride=1, padding=1)),
                    nn.InstanceNorm2d(out_channels, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                ])
            else:
                # Middle layers: stride=2, normalization, spectral norm
                layers.extend([
                    spectral_norm(nn.Conv2d(in_channels_current, out_channels, 4, stride=2, padding=1)),
                    nn.InstanceNorm2d(out_channels, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                ])

            in_channels_current = out_channels

        # Final output layer: single channel with spectral norm
        layers.append(spectral_norm(nn.Conv2d(in_channels_current, 1, 4, stride=1, padding=1)))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Patch-wise classification (B, 1, H_patch, W_patch)
            Values close to 1 indicate "real", close to 0 indicate "fake"
        """
        return self.layers(x)

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


class MultiscaleDiscriminator(nn.Module):
    """
    Multi-scale conditional discriminator for improved stability and detail perception.

    Operates on multiple scales to capture both global structure and fine details.
    Accepts concatenated input: [original_color, grayscale, refined_color] or similar.
    
    FIX: Conditional input prevents unrealistic color hallucination.
    """

    def __init__(
        self,
        in_channels: int = 4,  # UPDATED: 4 channels for conditional GAN (gray + RGB)
        base_filters: int = 64,
        num_scales: int = 3,
    ):
        """
        Initialize multi-scale conditional discriminator.

        Args:
            in_channels: Number of input channels (4 for conditional: grayscale + RGB)
            base_filters: Base number of filters
            num_scales: Number of scales to process (original, ÷2, ÷4)
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_scales = num_scales

        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(in_channels, base_filters, num_layers=4)
            for _ in range(num_scales)
        ])

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass on multiple scales.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of discriminator outputs at different scales
        """
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(x))
            x = self.downsample(x)

        return outputs

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
