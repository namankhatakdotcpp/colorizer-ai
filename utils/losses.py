import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import exp

def gaussian(window_size, sigma):
    """Generates a 1D Gaussian kernel."""
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    """Generates a 2D Gaussian window for SSIM."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Computes the Structural Similarity Index Measure (SSIM)."""
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    """
    SSIM Loss wrapper adapted for continuous image colorization monitoring.
    Expects tensors normalized in [0, 1] or equivalent domain bounded identically.
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        # Detect channel alterations dynamically and cache weights
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
                
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        # Return the structural *distance* by negating SSIM index (1.0 = perfect match)
        return 1.0 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class VGGPerceptualLoss(nn.Module):
    """
    Computes Perceptual divergence strictly up to 'relu2_2' to enforce 
    strong texture and contextual semantic alignment.
    Works on RGB converted representation of LAB outputs.
    """
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        # VGG expects RGB format, so colorized LAB maps must be converted before passing into this loss
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # 'relu2_2' is layer index 8. It usually captures robust low-mid level colored textures and edges 
        self.slice = vgg[:9]
        for param in self.parameters():
            param.requires_grad = False
            
        # Standard ImageNet normalization parameters
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred_rgb, target_rgb):
        x = (pred_rgb - self.mean) / self.std
        y = (target_rgb - self.mean) / self.std
        
        return F.l1_loss(self.slice(x), self.slice(y))


class FrequencyColorLoss(nn.Module):
    """
    FFT-based AB-spectrum matching to encourage coherent color distribution.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_ab: torch.Tensor, target_ab: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.fft2(pred_ab, norm="ortho")
        target_fft = torch.fft.fft2(target_ab, norm="ortho")
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        return F.l1_loss(pred_mag, target_mag)


class HybridColorizationLoss(nn.Module):
    """
    Combines:
    1. L1 (Huber) absolute color error penalty
    2. VGG perceptual divergence for semantic structural plausibility
    3. Frequency-domain AB matching to discourage flat/washy chroma outputs.
    """
    def __init__(self, l1_weight=1.0, perceptual_weight=0.2, ssim_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        
        self.l1_criterion = nn.L1Loss()
        self.vgg_criterion = VGGPerceptualLoss()
        self.freq_criterion = FrequencyColorLoss()

    def forward(self, pred_ab, target_ab, pred_rgb, target_rgb):
        """
        Args:
            pred_ab (Tensor): Predicted raw [A, B] channels from network. Shape: (B, 2, H, W)
            target_ab (Tensor): Ground truth [A, B] channels. Shape: (B, 2, H, W)
            pred_rgb (Tensor): Differentiable conversion of [L, pred_a, pred_b] to RGB. 
            target_rgb (Tensor): True RGB representation.
            
        Returns:
            total_loss (Tensor): Total combined focal loss scalar.
            loss_l1 (Tensor): Individual L1
            loss_vgg (Tensor): Individual VGG
            loss_freq (Tensor): Individual frequency AB loss
        """
        # 1. L1 loss calculates the absolute distance in the 'ab' color coordinate space directly.
        loss_l1 = self.l1_criterion(pred_ab, target_ab)
        
        # 2. Perceptual compares RGB representations for texture awareness using VGG relu2_2 features
        loss_vgg = self.vgg_criterion(pred_rgb, target_rgb)
        
        # 3. Frequency loss enforces coherent color spectrum across AB predictions.
        loss_freq = self.freq_criterion(pred_ab, target_ab)
        
        # Combine using configurated weighting scales
        total_loss = (
            (0.05 * loss_l1) + 
            (self.perceptual_weight * loss_vgg) + 
            (self.ssim_weight * loss_freq)
        )
        
        return total_loss, loss_l1, loss_vgg, loss_freq

# --- Example Usage ---
if __name__ == "__main__":
    # Simulate batch size 4, 256x256 resolution outputs
    dummy_pred_ab = torch.rand(4, 2, 256, 256, requires_grad=True)
    dummy_target_ab = torch.rand(4, 2, 256, 256)
    
    # We simulate the differentiable output of LAB->RGB
    dummy_pred_rgb = torch.rand(4, 3, 256, 256, requires_grad=True)
    dummy_target_rgb = torch.rand(4, 3, 256, 256)
    
    criterion = HybridColorizationLoss()
    
    # Run forward pass
    loss, l1, vgg, freq = criterion(dummy_pred_ab, dummy_target_ab, dummy_pred_rgb, dummy_target_rgb)
    
    print(f"Total Combined Colorization Loss: {loss.item():.4f}")
    print(f" -> L1 Component: {l1.item():.4f}")
    print(f" -> VGG Perceptual Component: {vgg.item():.4f}")
    print(f" -> Frequency AB Component: {freq.item():.4f}")
    
    loss.backward()
    print("Backward pass gradients computed successfully on predicted tensors!")
