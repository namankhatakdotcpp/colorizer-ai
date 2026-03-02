import torch
import torch.nn as nn
import torch.nn.functional as F

class DCE_Net(nn.Module):
    """
    Zero-Reference Deep Curve Estimation (Zero-DCE) for HDR Tone Mapping.
    Rather than predicting arbitrary RGB output (which causes fake/glowing HDR artifacts),
    this fully convolutional network predicts high-order, pixel-wise Light Enhancement Curves.
    """
    def __init__(self, n_iterations=8):
        super().__init__()
        self.n_iterations = n_iterations
        
        # Symmetrical 7-layer structure (No pooling/downsampling!)
        # Maintains perfect spatial resolution for curve parameter generation
        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True) 
        self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True) 
        self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True) 
        self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True) 
        
        self.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1, bias=True) 
        self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1, bias=True) 
        
        # Predicts 24 curve parameter maps based on iterations (3 channels * 8 iterations)
        self.e_conv7 = nn.Conv2d(64, 24, 3, 1, 1, bias=True) 

    def forward(self, x):
        # Forward feature extraction (with symmetrical concatenation skips)
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        # Output: Spatial Curve Parameters 'A'
        # Tanh constrains the parameter map A to [-1, 1], critical for stable curves
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        # Apply the learned higher-order curves iteratively to the original image
        enhanced_image = x
        illu_maps = [] # Tracks intermediate exposures
        
        # Apply the curve (LE(I, x) = I + A * I * (1 - I)) across n_iterations
        for i in range(self.n_iterations):
            # Extract the specific 3-channel parameter map for this iteration (e.g. 0:3, 3:6...)
            A_i = x_r[:, i*3:(i+1)*3, :, :]
            enhanced_image = enhanced_image + A_i * (torch.pow(enhanced_image, 2) - enhanced_image)
            illu_maps.append(enhanced_image)

        # illu_maps provides access to intermediate enhancements for logging, 
        # x_r is returned to compute the Illumination Smoothness Total Variation (TV) loss on the curves
        return enhanced_image, x_r, illu_maps

