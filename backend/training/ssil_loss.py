import torch
import torch.nn as nn

class SSILoss(nn.Module):
    """
    Scale-and-Shift Invariant Loss (SSIL).
    Crucial for training robust Depth Estimators on mixed datasets (like MiDaS).
    Since different datasets measure depth in different absolute scales and shifts 
    (e.g., mm vs meters, relative vs absolute), we force the prediction and target 
    into a uniformly scaled and zero-mean distribution before penalization.
    """
    def __init__(self):
        super().__init__()

    def compute_scale_and_shift(self, prediction, target, mask):
        """
        Dynamically calculates the optimal scale (s) and shift (t) parameters 
        to precisely align the prediction with the ground truth target.
        """
        # Summation components across valid pixels only (mask)
        N = torch.sum(mask, dim=(1, 2, 3), keepdim=True) + 1e-8
        
        sum_p = torch.sum(prediction * mask, dim=(1, 2, 3), keepdim=True)
        sum_t = torch.sum(target * mask, dim=(1, 2, 3), keepdim=True)
        
        sum_p2 = torch.sum((prediction ** 2) * mask, dim=(1, 2, 3), keepdim=True)
        sum_pt = torch.sum(prediction * target * mask, dim=(1, 2, 3), keepdim=True)

        # Scale 's'
        s = (N * sum_pt - sum_p * sum_t) / (N * sum_p2 - (sum_p ** 2) + 1e-8)
        
        # Shift 't'
        t = (sum_t - s * sum_p) / N
        
        return s, t

    def forward(self, prediction, target, mask=None):
        """
        prediction: (B, 1, H, W) Output of network
        target: (B, 1, H, W) Ground truth Relative/Absolute Depth map
        mask: Optional binary tensor marking valid pixels (ignores sky/invalid sensor reads)
        """
        if mask is None:
            # If no manual mask provided, assume valid where target > 0
            mask = (target > 0).float()
            
        # Avoid division by zero if an image is completely invalid
        if mask.sum() == 0:
            return torch.tensor(0.0, device=prediction.device, requires_grad=True)

        # 1. Align prediction to target's arbitrary scale and distribution
        scale, shift = self.compute_scale_and_shift(prediction, target, mask)
        aligned_prediction = scale * prediction + shift

        # 2. Compute Mean Absolute Error (MAE) on the aligned distributions
        loss = torch.sum(torch.abs(aligned_prediction - target) * mask) / (torch.sum(mask) + 1e-8)
        
        # 3. Add Multi-Scale Gradient Matching Loss to enforce crisp object boundaries
        # This penalizes differing spatial partial derivatives matching object edges
        diff = aligned_prediction - target
        
        # Gradient differences
        grad_x = torch.abs(diff[:, :, :, :-1] - diff[:, :, :, 1:])
        grad_y = torch.abs(diff[:, :, :-1, :] - diff[:, :, 1:, :])
        
        # Apply mask to gradients
        mask_x = mask[:, :, :, :-1] * mask[:, :, :, 1:]
        mask_y = mask[:, :, :-1, :] * mask[:, :, 1:, :]
        
        loss_grad = (torch.sum(grad_x * mask_x) + torch.sum(grad_y * mask_y)) / (torch.sum(mask) + 1e-8)
        
        # Weighting: 1.0 Aligned MAE + 0.5 Boundary Gradients
        return loss + 0.5 * loss_grad
