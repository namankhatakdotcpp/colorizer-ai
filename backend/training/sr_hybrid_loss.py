import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    """
    Computes Perceptual divergence using pre-trained VGG19.
    ESRGAN heavily utilizes features before activation ('relu5_4') for more continuous gradients,
    resulting in crisper, artifact-free textures.
    """
    def __init__(self, layer_idx=34):
        super(VGGPerceptualLoss, self).__init__()
        # VGG19 is standard for Super-Resolution (ESRGAN design)
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # We slice up to the requested layer. 'relu5_4' is layer 34.
        self.slice = vgg[: (layer_idx + 1)]
        
        for param in self.parameters():
            param.requires_grad = False
            
        # Standard ImageNet normalization parameters needed for VGG
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred_hr, target_hr):
        x = (pred_hr - self.mean) / self.std
        y = (target_hr - self.mean) / self.std
        
        return F.l1_loss(self.slice(x), self.slice(y))

class DiscriminatorLoss(nn.Module):
    """
    Relativistic Standard GAN (RaGAN) Loss Module.
    Predicts probability that real images are relatively more realistic than fake images.
    """
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.bce_logit_loss = nn.BCEWithLogitsLoss()

    def forward(self, real_logits, fake_logits, mode="generator"):
        """
        Args:
            real_logits: Raw outputs from Discriminator evaluating True HR images
            fake_logits: Raw outputs from Discriminator evaluating Generator output
            mode: 'generator' or 'discriminator' update tracking
        """
        # Relativistic distances
        d_real_fake = real_logits - torch.mean(fake_logits)
        d_fake_real = fake_logits - torch.mean(real_logits)
        
        # Real labels are 1s, Fake labels are 0s
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)

        if mode == "generator":
            # Generator tries to make fake look MORE real than true real
            loss_g = (
                self.bce_logit_loss(d_real_fake, fake_labels) + 
                self.bce_logit_loss(d_fake_real, real_labels)
            ) / 2.0
            return loss_g
            
        elif mode == "discriminator":
            # Discriminator standard objective
            loss_d = (
                self.bce_logit_loss(d_real_fake, real_labels) + 
                self.bce_logit_loss(d_fake_real, fake_labels)
            ) / 2.0
            return loss_d

class ESRGANLoss(nn.Module):
    """
    Combines:
    1. L1 (Pixel reconstruction) base loss
    2. VGG19 deep perceptual feature divergence
    3. Relativistic Adversarial Generator Loss (Optional)
    """
    def __init__(self, l1_weight=1e-2, vgg_weight=1.0, gan_weight=5e-3, use_gan=True):
        super().__init__()
        self.l1_weight = l1_weight
        self.vgg_weight = vgg_weight
        self.gan_weight = gan_weight
        self.use_gan = use_gan
        
        self.l1_criterion = nn.L1Loss()
        self.vgg_criterion = VGGPerceptualLoss(layer_idx=34) # relu5_4
        
        if self.use_gan:
            self.gan_criterion = DiscriminatorLoss()

    def forward(self, pred_hr, target_hr, real_logits=None, fake_logits=None):
        """
        Args:
            pred_hr (Tensor): Generator Output SR image.
            target_hr (Tensor): Ground truth natural HR image.
            real_logits (Tensor, optional): Descriminator D(HR).
            fake_logits (Tensor, optional): Descriminator D(SR).
            
        Returns:
            loss (Tensor): Total combined focal SR loss scalar for the Generator.
        """
        # 1. Pixel loss (L1 maintains sharper boundaries than L2)
        loss_pixel = self.l1_criterion(pred_hr, target_hr)
        
        # 2. Deep feature match for hallucinative textual structural accuracy
        loss_vgg = self.vgg_criterion(pred_hr, target_hr)
        
        total_loss = (self.l1_weight * loss_pixel) + (self.vgg_weight * loss_vgg)
        loss_gan = torch.tensor(0.0, device=pred_hr.device)
        
        # 3. Add Adversarial gradient if provided and active
        if self.use_gan and fake_logits is not None and real_logits is not None:
            loss_gan = self.gan_criterion(real_logits, fake_logits, mode="generator")
            total_loss += (self.gan_weight * loss_gan)
            
        return total_loss, loss_pixel, loss_vgg, loss_gan

# --- Example Usage ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 4 dummy batches of 3-channel 256x256 image patches
    dummy_pred_sr = torch.rand(4, 3, 256, 256, requires_grad=True).to(device)
    dummy_target_hr = torch.rand(4, 3, 256, 256).to(device)
    
    # Discriminator scalar logic maps for Relativistic GAN evaluating 128x128 feature patches 
    dummy_d_real = torch.rand(4, 1, 128, 128, requires_grad=True).to(device)
    dummy_d_fake = torch.rand(4, 1, 128, 128, requires_grad=True).to(device)
    
    criterion = ESRGANLoss(use_gan=True).to(device)
    
    # Run forward pass for Generator Update
    total_g_loss, l1, vgg, gan = criterion(
        pred_hr=dummy_pred_sr, 
        target_hr=dummy_target_hr,
        real_logits=dummy_d_real,
        fake_logits=dummy_d_fake
    )
    
    print(f"Total Combined SR Generator Loss: {total_g_loss.item():.4f}")
    print(f" -> L1 Pixel Component: {l1.item():.6f}")
    print(f" -> VGG Perceptual Component: {vgg.item():.4f}")
    print(f" -> Relativistic GAN Component: {gan.item():.4f}")
    
    total_g_loss.backward()
    print("Backward pass gradients computed successfully on Generator output!")
