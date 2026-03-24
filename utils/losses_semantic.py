"""
Semantic-augmented losses for semantic colorization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSupervisionLoss(nn.Module):
    """
    Cross-entropy loss for scene type classification.
    Auxiliary supervision to guide the encoder to learn semantic features.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, semantic_logits: torch.Tensor, scene_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            semantic_logits: [B, num_classes] from semantic head
            scene_labels: [B] with class indices (0-4)

        Returns:
            Cross-entropy loss
        """
        return self.criterion(semantic_logits, scene_labels)


class ColorResidualLoss(nn.Module):
    """
    Encourages predictions to respect scene-type color priors.
    
    E.g., if scene is "sky", encourage blue colors;
         if scene is "vegetation", encourage green colors.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

        # Define color priors for each scene (A, B channel biases)
        # In LAB space: A > 0 = red/magenta, A < 0 = green
        #               B > 0 = yellow,         B < 0 = blue
        self.color_priors = torch.nn.Parameter(
            torch.tensor([
                [-30.0, -60.0],  # water: greenish-blue
                [-20.0, -50.0],  # sky: bluish
                [-60.0, -10.0],  # vegetation: greenish
                [20.0, 5.0],     # skin: reddish-yellow
                [0.0, 0.0],      # other: neutral
            ]),
            requires_grad=False,
        )

    def forward(
        self, pred_ab: torch.Tensor, semantic_logits: torch.Tensor, ab_target: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            pred_ab: [B, 2, H, W] predicted AB in [-1, 1]
            semantic_logits: [B, num_classes] logits
            ab_target: [B, 2, H, W] optional target (not used in this version)

        Returns:
            Regularization loss encouraging color priors
        """
        B = pred_ab.shape[0]

        # Get soft class assignments
        class_probs = F.softmax(semantic_logits, dim=1)  # [B, num_classes]

        # Compute weighted color prior
        color_target = torch.einsum("bc,c...->b...", class_probs, self.color_priors)  # [B, 2, 1, 1]
        color_target = color_target.expand_as(pred_ab)  # [B, 2, H, W]

        # Normalize color priors to [-1, 1] range (assuming AB in [-128, 128])
        color_target = torch.clamp(color_target / 128.0, -1.0, 1.0)

        # L1 distance to color prior
        loss = F.l1_loss(pred_ab, color_target, reduction="mean")

        return loss * 0.1  # Light regularization


class SemanticConsistencyLoss(nn.Module):
    """
    Encourages spatially consistent semantic regions.
    Penalizes large color variations within predicted high-confidence regions.
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred_ab: torch.Tensor, semantic_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_ab: [B, 2, H, W]
            semantic_logits: [B, num_classes]

        Returns:
            Consistency regularization loss
        """
        B, C, H, W = pred_ab.shape

        # Get max confidence class per image
        class_confidence = torch.max(F.softmax(semantic_logits, dim=1), dim=1)[0]  # [B]

        # Compute spatial smoothness (variance of AB within high-confidence regions)
        ab_variance = torch.var(pred_ab.view(B, C, -1), dim=2)  # [B, 2]
        ab_std = torch.sqrt(ab_variance + 1e-6)

        # Penalize high variance in high-confidence regions
        loss = torch.mean((class_confidence.view(B, 1) * ab_std))

        return loss * 0.05  # Light regularization


class CombinedSemanticLoss(nn.Module):
    """
    Combined loss for semantic-aware colorization:
    - AB reconstruction (L1)
    - Perceptual loss (VGG features)
    - Semantic classification (CE)
    - Color prior regularization
    - Consistency regularization
    """

    def __init__(
        self,
        base_loss: nn.Module,
        perceptual_loss: nn.Module | None = None,
        semantic_weight: float = 0.5,
        color_prior_weight: float = 0.1,
        consistency_weight: float = 0.05,
    ):
        """
        Args:
            base_loss: Base colorization loss (e.g., L1 + histogram)
            perceptual_loss: Optional VGG perceptual loss
            semantic_weight: Weight for semantic classification loss
            color_prior_weight: Weight for color prior regularization
            consistency_weight: Weight for consistency loss
        """
        super().__init__()
        self.base_loss = base_loss
        self.perceptual_loss = perceptual_loss
        self.semantic_loss = SemanticSupervisionLoss(num_classes=5)
        self.color_prior_loss = ColorResidualLoss(num_classes=5)
        self.consistency_loss = SemanticConsistencyLoss(num_classes=5)

        self.semantic_weight = semantic_weight
        self.color_prior_weight = color_prior_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        pred_ab: torch.Tensor,
        target_ab: torch.Tensor,
        l_channel: torch.Tensor,
        semantic_logits: torch.Tensor,
        scene_labels: torch.Tensor | None = None,
        epoch: int = 0,
        total_epochs: int = 100,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred_ab: [B, 2, H, W] predicted AB
            target_ab: [B, 2, H, W] target AB
            l_channel: [B, 1, H, W] L channel for perceptual loss
            semantic_logits: [B, 5] semantic classification logits
            scene_labels: [B] scene class labels (for supervision)
            epoch: Current epoch (for scheduling loss weights)
            total_epochs: Total training epochs

        Returns:
            (loss, loss_dict) where loss_dict contains all component losses
        """
        loss_dict = {}

        # Base AB reconstruction loss
        base = self.base_loss(pred_ab, target_ab, epoch=epoch, total_epochs=total_epochs)
        loss_dict["base"] = float(base.item())

        # Perceptual loss
        if self.perceptual_loss is not None:
            perc = self.perceptual_loss(pred_ab, target_ab, l_channel)
            loss_dict["perceptual"] = float(perc.item())
        else:
            perc = 0.0

        # Semantic classification loss (if labels provided)
        if scene_labels is not None:
            sem = self.semantic_loss(semantic_logits, scene_labels)
            loss_dict["semantic"] = float(sem.item())
        else:
            sem = 0.0

        # Color prior regularization
        color_prior = self.color_prior_loss(pred_ab, semantic_logits, target_ab)
        loss_dict["color_prior"] = float(color_prior.item())

        # Consistency regularization
        consistency = self.consistency_loss(pred_ab, semantic_logits)
        loss_dict["consistency"] = float(consistency.item())

        # Combined loss
        total_loss = (
            base
            + perc
            + self.semantic_weight * sem
            + self.color_prior_weight * color_prior
            + self.consistency_weight * consistency
        )

        loss_dict["total"] = float(total_loss.item())
        return total_loss, loss_dict
