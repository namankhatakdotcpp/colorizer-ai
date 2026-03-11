import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2 * self.block(x)


class RRDBNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_features: int = 192, num_blocks: int = 8, scale: int = 4):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.trunk = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.conv_trunk = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_last = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        trunk = self.conv_trunk(self.trunk(feat))
        feat = feat + trunk
        out = self.up(feat)
        return self.conv_last(out)
