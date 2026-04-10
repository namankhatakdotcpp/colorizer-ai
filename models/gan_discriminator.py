"""
PatchGAN Discriminator for image refinement adversarial training.

Classifies patches of images as real or generated to provide feedback
to the generator during training.

Uses Spectral Normalization to stabilize training.
Production-grade: exposes intermediate features for feature matching loss.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import Tuple, List


class PatchGANDiscriminator(nn.Module):
    """
    Production-grade PatchGAN discriminator with intermediate feature extraction.

    Architecture:
    - Fully convolutional with progressive downsampling
    - Spectral normalization on all conv layers for training stability
    - Outputs NxN patch-wise real/fake logits (not single value)
    - Exposes intermediate feature maps for feature matching loss

    Input: Concatenated conditioning + generated/real image (4ch: L + RGB)
    Output: (patch_logits, intermediate_features_list)
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_filters: int = 64,
        num_layers: int = 4,
        patch_size: int = 70,
    ):
        """
        Initialize production-grade PatchGAN discriminator.

        Args:
            in_channels: Number of input channels (4 for L + RGB)
            base_filters: Base number of filters (64 recommended)
            num_layers: Number of convolutional layers (4-5 typical)
            patch_size: Receptive field/output patch size (70 for 256x256)
        """
        super().__init__()

        self.in_channels = in_channels
        self.base_filters = base_filters
        self.num_layers = num_layers
        self.patch_size = patch_size

        # Build discriminator blocks with Spectral Normalization
        # Store layers separately for feature extraction
        self.conv_blocks = nn.ModuleList()
        self.intermediate_features = []  # Runtime storage for intermediate features
        
        in_channels_current = in_channels

        for i in range(num_layers):
            out_channels = base_filters * min(2 ** i, 8)
            
            if i == 0:
                # First layer: stride=2 (downsample)
                conv_block = nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channels_current, out_channels, kernel_size=4, stride=2, padding=1)),
                    nn.LeakyReLU(0.2, inplace=False),  # inplace=False to preserve intermediate values
                )
            elif i == num_layers - 1:
                # Last layer: stride=1 (no further downsampling)
                conv_block = nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channels_current, out_channels, kernel_size=4, stride=1, padding=1)),
                    nn.InstanceNorm2d(out_channels, affine=True, eps=1e-5),
                    nn.LeakyReLU(0.2, inplace=False),
                )
            else:
                # Middle layers: stride=2 (progressive downsampling)
                conv_block = nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channels_current, out_channels, kernel_size=4, stride=2, padding=1)),
                    nn.InstanceNorm2d(out_channels, affine=True, eps=1e-5),
                    nn.LeakyReLU(0.2, inplace=False),
                )
            
            self.conv_blocks.append(conv_block)
            in_channels_current = out_channels

        # Final output layer: single channel logits (patch-wise classification)
        # stride=1, no normalization to preserve logit range
        self.final_conv = spectral_norm(
            nn.Conv2d(in_channels_current, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with intermediate feature extraction.

        Args:
            x: Input tensor (B, in_channels, H, W)
               Typically concatenated [L_channel (1), RGB (3)] = 4 channels

        Returns:
            Tuple of:
            - patch_logits: (B, 1, H_patch, W_patch) - patch-level real/fake scores
            - intermediate_features: List of intermediate activations for feature matching
        """
        # Clear cached features
        self.intermediate_features = []
        
        # Progressive forward pass through conv blocks
        feat = x
        for conv_block in self.conv_blocks:
            feat = conv_block(feat)
            # Store intermediate feature for feature matching loss
            self.intermediate_features.append(feat)
        
        # Final output: patch-wise logits
        patch_logits = self.final_conv(feat)
        
        # Return logits and intermediate features (excluding final layer for clarity)
        return patch_logits, self.intermediate_features

    def get_intermediate_features(self) -> List[torch.Tensor]:
        """
        Get cached intermediate features from last forward pass.
        
        Returns:
            List of intermediate feature maps for feature matching loss
        """
        return self.intermediate_features

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self):
        """Freeze all parameters (typically during G step)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters (typically during D step)."""
        for param in self.parameters():
            param.requires_grad = True


class MultiscaleDiscriminator(nn.Module):
    """
    Production-grade multi-scale conditional discriminator.

    Architecture:
    - Operates on 3 scales: original, 1/2, 1/4
    - Each scale processes independently with PatchGANDiscriminator
    - Exposes features from all scales for feature matching

    Benefits:
    - Captures both global structure and local details
    - Improved gradient flow and training stability
    - Better conditioning through multi-scale feedback
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_filters: int = 64,
        num_scales: int = 3,
    ):
        """
        Initialize multi-scale conditional discriminator.

        Args:
            in_channels: Number of input channels (4: L_channel + RGB)
            base_filters: Base number of filters (64 standard)
            num_scales: Number of scales to process (3 = original, 1/2, 1/4)
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_scales = num_scales

        # Build independent PatchGAN discriminators for each scale
        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(
                in_channels=in_channels,
                base_filters=base_filters,
                num_layers=4,
            )
            for _ in range(num_scales)
        ])

        # Downsampling for progressive scale processing
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Forward pass on multiple scales with feature extraction.

        Args:
            x: Input tensor (B, in_channels, H, W)

        Returns:
            Tuple of:
            - logits: List of patch logits at each scale
            - features: List of intermediate feature lists for each scale
        """
        all_logits = []
        all_features = []
        
        feat = x
        for discriminator in self.discriminators:
            # Forward pass returns (logits, intermediate_features)
            patch_logits, intermediate_features = discriminator(feat)
            all_logits.append(patch_logits)
            all_features.append(intermediate_features)
            
            # Downsample for next scale
            feat = self.downsample(feat)

        return all_logits, all_features

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self):
        """Freeze all parameters."""
        for discriminator in self.discriminators:
            discriminator.freeze()

    def unfreeze(self):
        """Unfreeze all parameters."""
        for discriminator in self.discriminators:
            discriminator.unfreeze()
