import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicColorizationModel(nn.Module):
    """
    A foundational CNN model for image colorization using the LAB color space.
    Input: L channel (Grayscale)
    Output: a and b channels (Color)
    """
    def __init__(self):
        super(BasicColorizationModel, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)
        
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(64)

    def forward(self, x):
        # x is (B, 1, H, W) where 1 is the L channel
        
        # Encoding
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        
        # Decoding
        x = F.relu(self.bn5(self.upconv1(x4)))
        x = F.relu(self.bn6(self.upconv2(x)))
        x = F.relu(self.bn7(self.upconv3(x)))
        # Output is (B, 2, H, W) for a and b channels, activated with Tanh to bound [-1, 1] usually
        x = torch.tanh(self.upconv4(x)) 
        
        # Note: the actual target standard lab values A and B should be properly scaled.
        # usually multiplying by 110 or 128 is a common post-processing step if scaled to [-1, 1]
        out = x * 128.0
        return out
