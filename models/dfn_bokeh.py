from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_depth_mask(depth: torch.Tensor, threshold: float = 0.2, sharpness: float = 15.0) -> torch.Tensor:
    """Create a smooth foreground mask from normalized depth."""
    return torch.sigmoid((depth - threshold) * sharpness)


def apply_separable_filters(rgb: torch.Tensor, kh: torch.Tensor, kv: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply predicted per-pixel separable kernels (horizontal then vertical)."""
    b, c, h, w = rgb.shape
    pad = kernel_size // 2

    rgb_h_padded = F.pad(rgb, (pad, pad, 0, 0), mode="replicate")
    unfold_h = F.unfold(rgb_h_padded, kernel_size=(1, kernel_size)).view(b, c, kernel_size, h, w)
    out_h = torch.sum(unfold_h * kh.unsqueeze(1), dim=2)

    rgb_v_padded = F.pad(out_h, (0, 0, pad, pad), mode="replicate")
    unfold_v = F.unfold(rgb_v_padded, kernel_size=(kernel_size, 1)).view(b, c, kernel_size, h, w)
    out_v = torch.sum(unfold_v * kv.unsqueeze(1), dim=2)
    return out_v


class DFNBokehModel(nn.Module):
    """
    Depth-aware dynamic filter network for bokeh rendering.
    Input: RGB [B,3,H,W], depth [B,1,H,W]
    Output: RGB [B,3,H,W]
    """

    def __init__(self, kernel_size: int = 11):
        super().__init__()
        self.kernel_size = kernel_size

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.kernel_h = nn.Conv2d(64, kernel_size, kernel_size=3, padding=1)
        self.kernel_v = nn.Conv2d(64, kernel_size, kernel_size=3, padding=1)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        focus_threshold: float = 0.2,
        return_aux: bool = False,
    ):
        if depth.shape[-2:] != rgb.shape[-2:]:
            depth = F.interpolate(depth, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([rgb, depth], dim=1)
        features = self.feature_extractor(x)

        kh = F.softmax(self.kernel_h(features), dim=1)
        kv = F.softmax(self.kernel_v(features), dim=1)

        blurred_rgb = apply_separable_filters(rgb, kh, kv, self.kernel_size)
        mask = get_depth_mask(depth, threshold=focus_threshold)
        out = mask * rgb + (1.0 - mask) * blurred_rgb

        if return_aux:
            return out, blurred_rgb, kh, kv, mask
        return out


class SobelFilter(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kernel_x.repeat(3, 1, 1, 1))
        self.register_buffer("ky", kernel_y.repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_x = F.conv2d(x, self.kx, padding=1, groups=3)
        edge_y = F.conv2d(x, self.ky, padding=1, groups=3)
        return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)


class BokehLoss(nn.Module):
    """L1 + edge consistency + kernel smoothness."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.sobel = SobelFilter()

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        pred_kh: torch.Tensor,
        pred_kv: torch.Tensor,
    ):
        loss_pixel = self.l1(pred_rgb, target_rgb)

        pred_edge = self.sobel(pred_rgb)
        target_edge = self.sobel(target_rgb)
        loss_focus = self.l1(pred_edge, target_edge)

        b, k, h, w = pred_kh.shape
        loss_kh_tv = torch.sum(torch.abs(pred_kh[:, :, :, :-1] - pred_kh[:, :, :, 1:])) + torch.sum(
            torch.abs(pred_kh[:, :, :-1, :] - pred_kh[:, :, 1:, :])
        )
        loss_kv_tv = torch.sum(torch.abs(pred_kv[:, :, :, :-1] - pred_kv[:, :, :, 1:])) + torch.sum(
            torch.abs(pred_kv[:, :, :-1, :] - pred_kv[:, :, 1:, :])
        )
        loss_kernel_tv = (loss_kh_tv + loss_kv_tv) / (b * k * h * w)

        total = loss_pixel + 0.5 * loss_focus + 0.1 * loss_kernel_tv
        return total, loss_pixel, loss_focus, loss_kernel_tv
