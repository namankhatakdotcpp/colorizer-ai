import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator for adversarial color realism.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        def block(in_c: int, out_c: int, normalize: bool = True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not normalize)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
