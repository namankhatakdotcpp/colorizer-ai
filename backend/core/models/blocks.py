import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Standard Residual Block with BatchNorm and LeakyReLU.
    Uses two 3x3 convolutions with a skip connection to allow direct gradient flow.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection for matching dimensions if stride > 1 or in_channels != out_channels
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.leaky_relu(out)
        
        return out

class DownBlock(nn.Module):
    """
    Downsampling block for the Encoder path.
    Uses a ResidualBlock with stride=2 to halve spatial dimensions.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = ResidualBlock(in_channels, out_channels, stride=2)

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    """
    Upsampling block for the Decoder path.
    Uses ConvTranspose2d followed by a ResidualBlock.
    Takes care of concatenating skip connections from the encoder.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # ConvTranspose2d spatial doubling (stride=2, kernel_size=4, padding=1 is standard for valid x2 upsample)
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1, bias=False)
        # ResidualBlock processes concatenated features: (in_channels//2 + skip_channels) -> out_channels
        self.block = ResidualBlock((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x, skip):
        """
        x: Features from previous decoder layer
        skip: Features from corresponding encoder layer
        """
        out = self.upconv(x)
        
        # Handle edge cases where feature maps might have slightly misaligned shape due to uneven pooling
        diffY = skip.size()[2] - out.size()[2]
        diffX = skip.size()[3] - out.size()[3]
        if diffY > 0 or diffX > 0:
            out = F.pad(out, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
            
        # Concatenate along channel dimension (B, C, H, W) -> Dim 1
        out = torch.cat([skip, out], dim=1)
        out = self.block(out)
        return out
