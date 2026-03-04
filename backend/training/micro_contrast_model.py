import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = self.conv2(self.lrelu(self.conv1(x)))
        return x + residual

class MicroContrastEnhancementCNN(nn.Module):
    """
    Micro-Contrast Enhancement CNN.
    Input: 1024x1024 RGB -> Output: 1024x1024 RGB
    Focuses on enhancing local micro-textures without hallucination using 8 Residual blocks.
    """
    def __init__(self, in_channels=3, base_filters=64, num_blocks=8):
        super().__init__()
        
        self.conv_in = nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        # 8 Residual Blocks
        self.res_blocks = nn.Sequential(*[
            ResBlock(base_filters) for _ in range(num_blocks)
        ])
        
        self.conv_out = nn.Conv2d(base_filters, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Enhance difference / details 
        features = self.lrelu(self.conv_in(x))
        residual = self.res_blocks(features)
        enhanced = self.conv_out(residual)
        
        # Global Skip Connection (forces network to learn high-frequency contrast adjustments only)
        return x + enhanced

# ==========================================
# High Frequency Loss Filters
# ==========================================

class LaplacianFilter(nn.Module):
    def __init__(self):
        super().__init__()
        # 3x3 Laplacian filter for edge detection
        kernel = torch.tensor([
            [ 0.,  1.,  0.],
            [ 1., -4.,  1.],
            [ 0.,  1.,  0.]
        ]).view(1, 1, 3, 3)
        self.register_buffer('kernel', kernel.repeat(3, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=1, groups=3)

class SobelFilter(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([
            [-1.,  0.,  1.],
            [-2.,  0.,  2.],
            [-1.,  0.,  1.]
        ]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ]).view(1, 1, 3, 3)
        
        self.register_buffer('kernel_x', kernel_x.repeat(3, 1, 1, 1))
        self.register_buffer('kernel_y', kernel_y.repeat(3, 1, 1, 1))

    def forward(self, x):
        edge_x = F.conv2d(x, self.kernel_x, padding=1, groups=3)
        edge_y = F.conv2d(x, self.kernel_y, padding=1, groups=3)
        # Magnitude
        return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)

class MicroContrastLoss(nn.Module):
    """
    Total Loss: L1 + 0.2 * Laplacian + 0.1 * Sobel
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.laplacian = LaplacianFilter()
        self.sobel = SobelFilter()

    def forward(self, pred, target):
        loss_pixel = self.l1(pred, target)
        
        pred_lap = self.laplacian(pred)
        target_lap = self.laplacian(target)
        loss_laplacian = self.l1(pred_lap, target_lap)
        
        pred_sobel = self.sobel(pred)
        target_sobel = self.sobel(target)
        loss_sobel = self.l1(pred_sobel, target_sobel)
        
        total_loss = loss_pixel + 0.2 * loss_laplacian + 0.1 * loss_sobel
        return total_loss, loss_pixel, loss_laplacian, loss_sobel
