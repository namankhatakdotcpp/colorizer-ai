import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),  # bias=False when using BatchNorm
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetColorizer(nn.Module):
    """
    Encoder-decoder UNet for Colorization.
    Input: L channel (1x256x256)
    Output: AB channels (2x256x256)
    """
    def __init__(self, base_filters=64):
        super().__init__()
        
        # Encoder
        self.enc1 = UNetBlock(1, base_filters)                           # -> 256x256
        self.pool1 = nn.MaxPool2d(2)                                     # -> 128x128
        
        self.enc2 = UNetBlock(base_filters, base_filters * 2)            # -> 128x128
        self.pool2 = nn.MaxPool2d(2)                                     # -> 64x64
        
        self.enc3 = UNetBlock(base_filters * 2, base_filters * 4)        # -> 64x64
        self.pool3 = nn.MaxPool2d(2)                                     # -> 32x32
        
        self.enc4 = UNetBlock(base_filters * 4, base_filters * 8)        # -> 32x32
        self.pool4 = nn.MaxPool2d(2)                                     # -> 16x16
        
        # Bottleneck
        self.bottleneck = UNetBlock(base_filters * 8, base_filters * 16) # -> 16x16
        
        # Decoder (with skip connections)
        self.upconv4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(base_filters * 16, base_filters * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(base_filters * 8, base_filters * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(base_filters * 4, base_filters * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(base_filters * 2, base_filters)
        
        # Final output layer
        self.final_conv = nn.Conv2d(base_filters, 2, kernel_size=1)
        self.tanh = nn.Tanh() # Constrain output strictly between [-1, 1]

    def forward(self, l_input):
        """
        Forward logic with proper LAB scaling.
        Expected l_input shape: (B, 1, 256, 256) containing values in range [0, 100].
        It is scaled to [-1, 1] internally for optimal neural network processing.
        """
        # Proper scaling logic: [0, 100] -> [-1, 1]
        x = (l_input / 50.0) - 1.0
        
        # Encoding stages
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoding stages with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1) # Skip connection
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1) # Skip connection
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1) # Skip connection
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1) # Skip connection
        d1 = self.dec1(d1)
        
        # Map to AB channels
        out_ab = self.tanh(self.final_conv(d1))
        
        return out_ab
