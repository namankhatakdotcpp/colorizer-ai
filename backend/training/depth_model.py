import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FeatureFusionBlock(nn.Module):
    """
    Combines high-resolution, low-semantic features from the encoder skip connection 
    with the low-resolution, high-semantic features from the decoder pathway.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 projection for the skip connection to match channel dimensions
        self.proj = nn.Conv2d(in_ch // 2, out_ch, kernel_size=1, bias=False)

    def forward(self, dec_feat, enc_feat):
        # 1. Bilinearly upscale the compressed decoder features
        dec_up = F.interpolate(dec_feat, scale_factor=2, mode="bilinear", align_corners=False)
        
        # 2. Project encoder features
        enc_proj = self.proj(enc_feat)
        
        # 3. Concatenate and refine
        fused = torch.cat([dec_up, enc_proj], dim=1) # (out_ch + out_ch -> in_ch)
        return self.out_conv(fused)

class MiDaSResNet50(nn.Module):
    """
    MiDaS-style CNN Depth Estimator based on ResNet-50.
    Outputs relative inverse depth aligned to 384x384 spatial dimensions natively.
    """
    def __init__(self, freeze_early_stages=True):
        super().__init__()
        
        # ----------------------------------------------------
        # 1. ENCODER (Pre-trained ResNet-50)
        # ----------------------------------------------------
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # We dissect the ResNet into 4 scales to tap skip connections
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # -> 1/4 res (channels: 256)
        self.layer2 = resnet.layer2                                # -> 1/8 res (channels: 512)
        self.layer3 = resnet.layer3                                # -> 1/16 res (channels: 1024)
        self.layer4 = resnet.layer4                                # -> 1/32 res (channels: 2048)
        
        # Option to freeze early encoder stages during initial fine-tuning
        if freeze_early_stages:
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False

        # ----------------------------------------------------
        # 2. MULTI-SCALE FEATURE FUSION DECODER
        # ----------------------------------------------------
        # Initial projection of the deep bottleneck (2048 -> 512)
        self.bottleneck_proj = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        
        # FeatureFusionBlock expects concatenated (dec + enc) channel definitions
        # 512 (Dec_Up) + 512 (Enc_Proj) = 1024 input -> 256 output
        self.fusion4 = FeatureFusionBlock(in_ch=1024, out_ch=256) 
        
        # 256 (Dec_Up) + 256 (Enc_Proj) = 512 input -> 128 output
        self.fusion3 = FeatureFusionBlock(in_ch=512, out_ch=128)
        
        # 128 (Dec_Up) + 128 (Enc_Proj) = 256 input -> 64 output
        self.fusion2 = FeatureFusionBlock(in_ch=256, out_ch=64)
        
        # Final explicit upscale to full resolution (1/4 -> 1/2 -> 1/1)
        self.final_upscale = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), # -> 1/2
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), # -> 1/1
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            # ReLU mapping ensures non-negative depth predictions
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Input constraint: x must be shape (B, 3, 384, 384)
        """
        # --- Encoder Path ---
        l1 = self.layer1(x)  # (B, 256, 96, 96)
        l2 = self.layer2(l1) # (B, 512, 48, 48)
        l3 = self.layer3(l2) # (B, 1024, 24, 24)
        l4 = self.layer4(l3) # (B, 2048, 12, 12)
        
        # Bottleneck reduction
        dec4 = self.bottleneck_proj(l4) # (B, 512, 12, 12)

        # --- Decoder Path (Multi-Scale Fusion) ---
        dec3 = self.fusion4(dec_feat=dec4, enc_feat=l3) # Upscales to (B, 256, 24, 24)
        dec2 = self.fusion3(dec_feat=dec3, enc_feat=l2) # Upscales to (B, 128, 48, 48)
        dec1 = self.fusion2(dec_feat=dec2, enc_feat=l1) # Upscales to (B, 64, 96, 96)

        # Final full-res mapping
        out_depth = self.final_upscale(dec1) # Upscales to (B, 1, 384, 384)
        
        return out_depth
