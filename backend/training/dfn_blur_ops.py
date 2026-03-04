import torch
import torch.nn as nn
import torch.nn.functional as F

def get_depth_mask(depth, threshold=0.2):
    """
    Creates a smooth differentiable mask that keeps foreground sharp.
    depth: [B, 1, H, W] normalized from [0, 1] where 1 is closest.
    threshold: Target Focal plane.
    """
    # Everything closer than threshold becomes 1.0 (Sharp).
    # Backgrounds fade to 0.0 (Blurred).
    mask = torch.sigmoid((depth - threshold) * 15.0) 
    return mask

def apply_separable_filters(rgb, kh, kv, k):
    """
    Applies 1D Separable convolutions (Horizontal then Vertical) dynamically
    per pixel based on predicted DFN kernels.
    rgb: [B, 3, H, W]
    kh: [B, k, H, W]
    kv: [B, k, H, W]
    k: scalar kernel size (e.g. 11)
    """
    B, C, H, W = rgb.shape
    pad = k // 2
    
    # ----------------------------------------------------
    # 1. Horizontal Pass [1 x K]
    # ----------------------------------------------------
    # Pad width only
    rgb_h_padded = F.pad(rgb, (pad, pad, 0, 0), mode='replicate')
    
    # Unfold into explicitly horizontal patches [B, C*K, H*W]
    # kernel size is (1, k)
    unfold_h = F.unfold(rgb_h_padded, kernel_size=(1, k))
    
    # Reshape: [B, C, k, H, W]
    unfold_h = unfold_h.view(B, C, k, H, W)
    
    # Reshape kernel: [B, 1, k, H, W] -> broadcast across 3 channels
    kh_expanded = kh.unsqueeze(1)
    
    # Multiply structurally and sum across the kernel width 'k' dimension
    # Output: [B, C, H, W]
    out_h = torch.sum(unfold_h * kh_expanded, dim=2)
    
    # ----------------------------------------------------
    # 2. Vertical Pass [K x 1] 
    # ----------------------------------------------------
    # Pad height only output of horizontal pass
    rgb_v_padded = F.pad(out_h, (0, 0, pad, pad), mode='replicate')
    
    # Unfold into explicitly vertical patches [B, C*K, H*W]
    # kernel size is (k, 1)
    unfold_v = F.unfold(rgb_v_padded, kernel_size=(k, 1))
    
    # Reshape: [B, C, k, H, W]
    unfold_v = unfold_v.view(B, C, k, H, W)
    
    # Reshape kernel: [B, 1, k, H, W]
    kv_expanded = kv.unsqueeze(1)
    
    # Multiply and sum across height 'k' dimension
    out_final = torch.sum(unfold_v * kv_expanded, dim=2)
    
    return out_final

# ==========================================
# Bokeh Rendering Losses
# ==========================================

class SobelFilter(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('kx', kernel_x.repeat(3, 1, 1, 1))
        self.register_buffer('ky', kernel_y.repeat(3, 1, 1, 1))

    def forward(self, x):
        edge_x = F.conv2d(x, self.kx, padding=1, groups=3)
        edge_y = F.conv2d(x, self.ky, padding=1, groups=3)
        return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)

class BokehLoss(nn.Module):
    """
    Combined Loss:
    - L1 loss: Exact pixel mappings.
    - Sobel Focus Loss: Ensures subject high-frequencies match perfectly (foreground remains crisp).
    - Background Blur Consistency Loss: Smooths predicted kernel matrices to avoid chaotic pixel jumps.
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.sobel = SobelFilter()

    def forward(self, pred_rgb, target_rgb, pred_kh, pred_kv):
        # 1. Base L1 mapping
        loss_pixel = self.l1(pred_rgb, target_rgb)
        
        # 2. Focus Loss (matching edges)
        pred_edge = self.sobel(pred_rgb)
        target_edge = self.sobel(target_rgb)
        loss_focus = self.l1(pred_edge, target_edge)
        
        # 3. Background Blur Consistency Loss (Total Variation on Kernel prediction)
        # Prevents kernel matrices from jumping chaotically spatially
        B, k, H, W = pred_kh.shape
        loss_kh_tv = torch.sum(torch.abs(pred_kh[:, :, :, :-1] - pred_kh[:, :, :, 1:])) + \
                     torch.sum(torch.abs(pred_kh[:, :, :-1, :] - pred_kh[:, :, 1:, :]))
        loss_kv_tv = torch.sum(torch.abs(pred_kv[:, :, :, :-1] - pred_kv[:, :, :, 1:])) + \
                     torch.sum(torch.abs(pred_kv[:, :, :-1, :] - pred_kv[:, :, 1:, :]))
                     
        # Normalize TV
        loss_kernel_tv = (loss_kh_tv + loss_kv_tv) / (B * k * H * W)
        
        # Combine
        total_loss = loss_pixel + 0.5 * loss_focus + 0.1 * loss_kernel_tv
        return total_loss, loss_pixel, loss_focus, loss_kernel_tv
