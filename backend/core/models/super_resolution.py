import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, channels=64, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_rate * 1, growth_rate, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + growth_rate * 2, growth_rate, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + growth_rate * 3, growth_rate, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + growth_rate * 4, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirical scaling factor for stability (often 0.2 in RRDB)
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, channels=64, growth_rate=32):
        super().__init__()
        self.rdb1 = DenseBlock(channels, growth_rate)
        self.rdb2 = DenseBlock(channels, growth_rate)
        self.rdb3 = DenseBlock(channels, growth_rate)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    """
    CNN-only Super Resolution Architecture (RRDBNet).
    Recommended over EDSR (too heavy) and ESPCN (too simple).
    No BatchNorm layers to prevent artifacts and preserve fine textures.
    """
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=16):
        super().__init__()
        # Initial feature extraction
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)

        # Main trunk of RRDB blocks
        self.trunk = nn.Sequential(*[RRDB(channels=num_features) for _ in range(num_blocks)])

        # Post-trunk convolution
        self.conv_trunk = nn.Conv2d(num_features, num_features, 3, 1, 1)

        # Upsampling (x4 = two successive x2 upsamples)
        self.upconv1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(0.2, True)

        # Final output layers
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)

    def forward(self, x):
        # Initial features
        feat = self.conv_first(x)
        
        # Deep features
        trunk_out = self.trunk(feat)
        trunk_out = self.conv_trunk(trunk_out)
        
        # Global residual connection
        feat = feat + trunk_out

        # Upsampling x2
        feat = self.lrelu(self.upconv1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        # Upsampling x4
        feat = self.lrelu(self.upconv2(F.interpolate(feat, scale_factor=2, mode='nearest')))

        # High-res output mapping
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
