"""
Enhanced UNet Colorizer with Semantic Understanding & Attention Gates

Features:
- Semantic classification head (5 scene classes: water, sky, vegetation, skin, other)
- Attention gates on skip connections (focus on important regions)
- Optional auxiliary semantic supervision
- Backward compatible with baseline model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionGate(nn.Module):
    """Channel-wise attention for skip connections (Squeeze-and-Excitation block)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, max(channels // reduction, 1), kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(max(channels // reduction, 1), channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention: scale channels by importance."""
        se = self.avg_pool(x)
        se = self.fc2(self.relu(self.fc1(se)))
        return x * self.sigmoid(se)


class SpatialAttentionGate(nn.Module):
    """Spatial attention for skip connections (focus on important image regions)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention: scale spatial regions."""
        attention = self.conv(x)
        return x * attention


class AttentionGate(nn.Module):
    """Combined channel and spatial attention for skip connections."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttentionGate(channels, reduction)
        self.spatial_att = SpatialAttentionGate(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_attention: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)
        self.attention = AttentionGate(skip_channels) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle odd-sized tensors safely
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # Apply attention to skip connection
        skip = self.attention(skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetColorizerSemantic(nn.Module):
    """
    Enhanced UNet with semantic understanding.
    
    Outputs:
    - AB channels (colorization)
    - Scene type logits (5 classes: water, sky, vegetation, skin, other)
    """

    SCENE_CLASSES = ("water", "sky", "vegetation", "skin", "other")
    NUM_CLASSES = len(SCENE_CLASSES)

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        base_channels: int = 64,
        use_attention: bool = True,
        semantic_weight: float = 1.0,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.semantic_weight = semantic_weight

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        # Encoder
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        # Decoder with attention on skip connections
        self.up1 = Up(c5, c4, c4, use_attention=use_attention)
        self.up2 = Up(c4, c3, c3, use_attention=use_attention)
        self.up3 = Up(c3, c2, c2, use_attention=use_attention)
        self.up4 = Up(c2, c1, c1, use_attention=use_attention)

        # AB output (colorization)
        self.outc = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.final_act = nn.Tanh()

        # Semantic classification head (bottleneck features -> scene type)
        self.semantic_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(c5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.NUM_CLASSES),
        )

    def forward(
        self, x: torch.Tensor, return_semantic: bool = True
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input L channel [B, 1, H, W]
            return_semantic: Whether to return semantic logits

        Returns:
            If return_semantic=False: AB predictions [B, 2, H, W] in [-1, 1]
            If return_semantic=True: (AB predictions, semantic_logits) where semantic_logits is [B, 5]
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Semantic classification from bottleneck
        semantic_logits = self.semantic_head(x5) if return_semantic else None

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # AB output
        ab = self.final_act(self.outc(x))

        if return_semantic:
            return ab, semantic_logits
        return ab

    @classmethod
    def from_baseline_checkpoint(cls, baseline_model_path: str, use_attention: bool = True, **kwargs):
        """
        Load weights from baseline UNetColorizer checkpoint.
        
        This allows seamless transfer learning from the baseline model.
        The semantic head is initialized randomly.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load baseline checkpoint
        checkpoint = torch.load(baseline_model_path, map_location=device)
        baseline_state = checkpoint.get("model_state_dict", checkpoint)
        
        # Create semantic model
        model = cls(use_attention=use_attention, **kwargs)
        
        # Load compatible weights
        semantic_state = model.state_dict()
        compatible_keys = []
        incompatible_keys = []
        
        for key in baseline_state:
            if key in semantic_state and baseline_state[key].shape == semantic_state[key].shape:
                semantic_state[key] = baseline_state[key]
                compatible_keys.append(key)
            else:
                incompatible_keys.append(key)
        
        model.load_state_dict(semantic_state)
        
        print(f"[INFO] Loaded {len(compatible_keys)} compatible weights from baseline")
        if incompatible_keys:
            print(f"[INFO] {len(incompatible_keys)} incompatible weights (expected for new semantic head)")
        
        return model

    def get_baseline_state_dict(self) -> dict:
        """
        Extract only the AB-prediction weights (without semantic head).
        Useful for saving a "backward-compatible" baseline checkpoint.
        """
        baseline_state = {}
        for name, param in self.named_parameters():
            if "semantic" not in name:
                baseline_state[name] = param.data
        return baseline_state


class UNetColorizer(nn.Module):
    """
    Backward-compatible baseline UNet (no semantic understanding).
    Can be replaced by UNetColorizerSemantic with return_semantic=False
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2, base_channels: int = 64):
        super().__init__()

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c5)

        self.up1 = Up(c5, c4, c4, use_attention=False)
        self.up2 = Up(c4, c3, c3, use_attention=False)
        self.up3 = Up(c3, c2, c2, use_attention=False)
        self.up4 = Up(c2, c1, c1, use_attention=False)

        self.outc = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.final_act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.final_act(self.outc(x))
