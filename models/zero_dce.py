from __future__ import annotations

import torch
import torch.nn as nn


class ZeroDCEModel(nn.Module):
    """
    Lightweight Zero-DCE model.
    Input: RGB [B,3,H,W]
    Output: enhanced RGB + curve parameters.
    """

    def __init__(self, num_iterations: int = 8, channels: int = 32):
        super().__init__()
        self.num_iterations = num_iterations
        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv5 = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv6 = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv7 = nn.Conv2d(channels * 2, num_iterations * 3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x: torch.Tensor):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))

        curve = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        enhanced = x
        for i in range(self.num_iterations):
            a_i = curve[:, i * 3 : (i + 1) * 3, :, :]
            enhanced = enhanced + a_i * (enhanced * enhanced - enhanced)

        return enhanced, curve
