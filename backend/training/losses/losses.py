import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def lab_to_rgb(l_channel, ab_channels):
    """
    Differentiable step-by-step conversion of LAB back to RGB.
    l_channel: (B, 1, H, W) in range [0, 100]
    ab_channels: (B, 2, H, W) in range [-128, 127]
    Returns: RGB tensor structurally mapped into [0, 1] for Perceptual & SSIM pipelines.
    """
    # 1. Split LAB into L, A, and B
    L = l_channel
    a = ab_channels[:, 0:1, :, :]
    b = ab_channels[:, 1:2, :, :]

    # 2. Convert LAB to XYZ
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    xyz = torch.cat([x, y, z], dim=1)

    # Condition
    mask = xyz > 0.2068966
    xyz_cubic = torch.pow(xyz, 3.0)
    xyz_linear = (xyz - 16.0 / 116.0) / 7.787
    
    # torch.where avoids inplace operations that could break the graph
    xyz = torch.where(mask, xyz_cubic, xyz_linear)

    # Multiply by reference Illuminant D65
    xyz_x = xyz[:, 0:1, :, :] * 0.95047
    xyz_y = xyz[:, 1:2, :, :] * 1.00000
    xyz_z = xyz[:, 2:3, :, :] * 1.08883

    # 3. Convert XYZ to RGB
    r = xyz_x *  3.2406 + xyz_y * -1.5372 + xyz_z * -0.4986
    g = xyz_x * -0.9689 + xyz_y *  1.8758 + xyz_z *  0.0415
    b_rgb = xyz_x *  0.0557 + xyz_y * -0.2040 + xyz_z *  1.0570

    rgb = torch.cat([r, g, b_rgb], dim=1)

    # 4. Apply SRGB Gamma correction
    mask_rgb = rgb > 0.0031308
    rgb_gamma = 1.055 * torch.pow(torch.clamp(rgb, min=1e-8), 1/2.4) - 0.055
    rgb_linear = rgb * 12.92
    
    rgb = torch.where(mask_rgb, rgb_gamma, rgb_linear)

    # Clamp safely to [0,1]
    return torch.clamp(rgb, 0.0, 1.0)


class VGG16PerceptualLoss(nn.Module):
    """
    Computes perceptual loss using early layers of a pretrained VGG16 network.
    Extracts up to relu3_3 mapping semantic textures over raw pixels.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # Slicing up to relu3_3 (layer index 16)
        self.slice = vgg[:16]
        
        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred_rgb, target_rgb):
        # Normalize into VGG expected bounds
        x = (pred_rgb - self.mean) / self.std
        y = (target_rgb - self.mean) / self.std
        
        x_feat = self.slice(x)
        y_feat = self.slice(y)
        
        return F.l1_loss(x_feat, y_feat)


def gaussian_window(window_size, sigma):
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.view(1, -1) * g.view(-1, 1)

def ssim_loss(img1, img2, window_size=11, sigma=1.5):
    """
    Simplified native implementation of SSIM.
    Returns structurally inverted loss (1 - SSIM).
    """
    try:
        from pytorch_msssim import ssim as msssim
        return 1 - msssim(img1, img2, data_range=1.0, size_average=True)
    except ImportError:
        # Fallback to pure PyTorch local implementation
        channel = img1.size(1)
        window = gaussian_window(window_size, sigma).expand(channel, 1, window_size, window_size).to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


class CompositeLoss(nn.Module):
    """
    Composite total loss required by the user:
    1.0 * L1 + 0.1 * Perceptual + 0.1 * SSIM.
    
    Includes proper real-time translation to RGB for robust perceptual calculations.
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = VGG16PerceptualLoss()

    def forward(self, pred_ab, target_ab, L_input):
        """
        pred_ab: Predicted AB shape (B, 2, 256, 256) inside [-1, 1]
        target_ab: Targets AB shape (B, 2, 256, 256) inside [-1, 1]
        L_input: Real L channel (B, 1, 256, 256) inside [0, 100]
        """
        # 1. Base L1 Loss (directly on AB predictions for structural color gradient)
        loss_l1 = self.l1(pred_ab, target_ab)
        
        # Re-scale back to true LAB space [-128, 127] for geometric translation to RGB
        pred_ab_real = pred_ab * 128.0
        target_ab_real = target_ab * 128.0
        
        # Convert internal LAB maps perfectly over to proper RGB spaces [0, 1]
        pred_rgb = lab_to_rgb(L_input, pred_ab_real)
        target_rgb = lab_to_rgb(L_input, target_ab_real)
        
        # 2. VGG Perceptual Loss on explicitly reconstructed RGB
        loss_vgg = self.perceptual(pred_rgb, target_rgb)
        
        # 3. Micro-Contrast Structural Similarity Loss (SSIM)
        loss_ssim = ssim_loss(pred_rgb, target_rgb)
        
        # Combine explicitly
        total_loss = 1.0 * loss_l1 + 0.1 * loss_vgg + 0.1 * loss_ssim
        
        return total_loss, loss_l1, loss_vgg, loss_ssim
