import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    """
    Standard DenseBlock containing 5 convolutional layers.
    Combines input from all preceding convolutions within the block via concatenation.
    """
    def __init__(self, channels=64, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate * 1, growth_rate, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels + growth_rate * 2, growth_rate, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(channels + growth_rate * 3, growth_rate, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(channels + growth_rate * 4, channels, kernel_size=3, stride=1, padding=1)
        
        # LeakyReLU explicitly requested to be inplace
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        
        # Residual scaling 0.2
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block.
    Contains 3 individual Dense Blocks connected sequentially with an additional residual wrap.
    """
    def __init__(self, channels=64, growth_rate=32):
        super().__init__()
        self.rdb1 = DenseBlock(channels, growth_rate)
        self.rdb2 = DenseBlock(channels, growth_rate)
        self.rdb3 = DenseBlock(channels, growth_rate)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Residual scaling 0.2
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    CNN-only (No GAN) Super-Resolution formulation based on ESRGAN's generator.
    4x upscale via PixelShuffle explicitly.
    Constraints: No Batchnorm, LeakyReLU(inplace=True), 23 RRDB Blocks.
    """
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=23):
        super().__init__()
        
        # Initial Spatial Conv
        self.conv_first = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)

        # 23 sequentially stacked RRDB blocks
        self.body = nn.Sequential(*[RRDB(num_features) for _ in range(num_blocks)])
        
        # Post-trunk block merging processing before upsampling
        self.conv_body = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)

        # Upsampling via consecutive PixelShuffles (4x total: 2x, 2x)
        # PixelShuffle expects (channels * upscale_factor^2) input channels per shuffle logic
        self.upconv1 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        
        self.upconv2 = nn.Conv2d(num_features, num_features * 4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Output projection layers
        self.conv_hr = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(num_features, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        feat = self.conv_first(x)
        
        body_feat = self.conv_body(self.body(feat))
        
        # Global residual connection over the RRDB block trunk
        feat = feat + body_feat

        # Upsampling Stage 1 (2x)
        feat = self.lrelu(self.pixel_shuffle1(self.upconv1(feat)))
        
        # Upsampling Stage 2 (4x total)
        feat = self.lrelu(self.pixel_shuffle2(self.upconv2(feat)))

        # Final High Res Mapping
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        
        return out
