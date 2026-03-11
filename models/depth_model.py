import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.block(x)


class DynamicFilterNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 192, num_blocks: int = 8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.trunk = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])

        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels // 2, base_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.enc(x)
        feat = self.trunk(feat)
        return self.dec(feat)
