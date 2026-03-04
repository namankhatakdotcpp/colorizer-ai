import torch
import torch.nn as nn
import torch.nn.functional as F

class ZeroDCENet(nn.Module):
    """
    Zero-Reference Deep Curve Estimation (Zero-DCE) Network.
    Designed for HDR and Low-Light image enhancement without paired datasets.
    Architecture: 7 Conv layers, 32 channels, ReLU activations.
    Output: 24-channel curve parameter maps (8 iterations x 3 color channels).
    """
    def __init__(self, scale_factor=1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        
        # 1. Feature extraction
        self.e_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True) 
        self.e_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True) 
        self.e_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True) 
        self.e_conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True) 
        
        # 2. Symmetrical feature fusion
        self.e_conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True) 
        self.e_conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True) 
        
        # 3. Output layer: predicts 24 feature maps for 8 iterations (8 * 3 channels)
        self.e_conv7 = nn.Conv2d(64, 24, kernel_size=3, stride=1, padding=1, bias=True) 

    def forward(self, x):
        # Forward pass
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        # Skip connections
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))

        # Output curve parameters scaled to [-1, 1] via Tanh
        curve_params = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        # Iterative curve application: LE(I, x) = I + A * I * (1 - I)
        enhanced_image = x
        
        for i in range(8):
            # Extract 3-channel parameter map for iteration i
            A_i = curve_params[:, i*3:(i+1)*3, :, :]
            enhanced_image = enhanced_image + A_i * (torch.pow(enhanced_image, 2) - enhanced_image)

        return enhanced_image, curve_params
