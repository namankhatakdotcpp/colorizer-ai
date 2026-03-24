"""
Inference utilities for baseline and semantic colorizers.

Provides:
- Load baseline or semantic model
- Inference with optional semantic output
- Mode switching (baseline/semantic)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Literal, Optional, Tuple

from models.unet_colorizer import UNetColorizer
from models.unet_colorizer_semantic import UNetColorizerSemantic


class ColorizerFactory:
    """Factory for loading colorizer models (baseline or semantic)."""

    @staticmethod
    def load_model(
        checkpoint_path: str,
        device: torch.device,
        mode: Literal["baseline", "semantic", "auto"] = "auto",
        use_attention: bool = True,
    ) -> nn.Module:
        """
        Load colorizer checkpoint.

        Args:
            checkpoint_path: Path to checkpoint (.pth file)
            device: Device to load on
            mode: "baseline" (standard UNet), "semantic" (with scene classification), or "auto" (detect)
            use_attention: For semantic mode, whether to use attention gates

        Returns:
            Loaded model (baseline UNet or semantic UNet)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Auto-detect mode from checkpoint
        if mode == "auto":
            is_semantic = checkpoint.get("semantic", False)
            mode = "semantic" if is_semantic else "baseline"

        if mode == "semantic":
            print("[INFO] Loading semantic colorizer (with attention gates & scene classification)")
            model = UNetColorizerSemantic(use_attention=use_attention).to(device)
        else:
            print("[INFO] Loading baseline colorizer")
            model = UNetColorizer().to(device)

        # Load weights
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        return model

    @staticmethod
    def inference_baseline(
        model: nn.Module, l_channel: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Baseline inference: L channel -> AB predictions.

        Args:
            model: Baseline UNet model
            l_channel: [1, 1, H, W] L channel (normalized to [0, 1])
            device: Inference device

        Returns:
            AB predictions [1, 2, H, W] in [-1, 1]
        """
        with torch.no_grad():
            l_channel = l_channel.to(device)
            ab_pred = model(l_channel)
        return ab_pred

    @staticmethod
    def inference_semantic(
        model: nn.Module,
        l_channel: torch.Tensor,
        device: torch.device,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Semantic inference: L channel -> AB predictions + scene classification.

        Args:
            model: Semantic UNet model
            l_channel: [1, 1, H, W] L channel (normalized to [0, 1])
            device: Inference device
            return_logits: Whether to return raw logits or class probabilities

        Returns:
            (ab_pred, scene_info) where:
            - ab_pred: [1, 2, H, W] AB predictions in [-1, 1]
            - scene_info: dict with scene classification results
        """
        with torch.no_grad():
            l_channel = l_channel.to(device)
            ab_pred, semantic_logits = model(l_channel, return_semantic=True)

        # Convert logits to class info
        scene_classes = model.SCENE_CLASSES
        probs = torch.softmax(semantic_logits, dim=1)[0]  # [num_classes]
        class_idx = torch.argmax(probs, dim=0).item()
        confidence = probs[class_idx].item()

        scene_info = {
            "class_idx": class_idx,
            "class_name": scene_classes[class_idx],
            "confidence": confidence,
            "all_probs": {scene_classes[i]: float(probs[i].item()) for i in range(len(scene_classes))},
            "logits": semantic_logits.cpu() if return_logits else None,
        }

        return ab_pred, scene_info


class SemanticColorizerInference:
    """High-level inference interface for semantic colorization."""

    def __init__(self, checkpoint_path: str, device: Optional[torch.device] = None):
        """
        Initialize semantic colorizer.

        Args:
            checkpoint_path: Path to semantic checkpoint
            device: Device (auto-detects GPU if available)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path

        # Try to load as semantic, fall back to baseline
        self.model = ColorizerFactory.load_model(checkpoint_path, self.device, mode="auto")
        self.is_semantic = isinstance(self.model, UNetColorizerSemantic)

    def colorize(self, l_channel: torch.Tensor) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Colorize L channel.

        Args:
            l_channel: [1, 1, H, W] or [H, W] L channel in [0, 1]

        Returns:
            (ab_pred, scene_info)
            - ab_pred: [1, 2, H, W] AB predictions in [-1, 1]
            - scene_info: dict with scene classification (None if baseline model)
        """
        # Ensure 4D tensor
        if l_channel.ndim == 2:
            l_channel = l_channel.unsqueeze(0).unsqueeze(0)
        elif l_channel.ndim == 3:
            l_channel = l_channel.unsqueeze(0)

        if self.is_semantic:
            ab_pred, scene_info = ColorizerFactory.inference_semantic(self.model, l_channel, self.device)
            return ab_pred, scene_info
        else:
            ab_pred = ColorizerFactory.inference_baseline(self.model, l_channel, self.device)
            return ab_pred, None

    def get_scene_info(self, l_channel: torch.Tensor) -> dict:
        """Get scene classification without full colorization (faster)."""
        if not self.is_semantic:
            return {"error": "Model is baseline, not semantic"}

        with torch.no_grad():
            l_channel = l_channel.unsqueeze(0).unsqueeze(0).to(self.device)
            _, semantic_logits = self.model(l_channel, return_semantic=True)

        scene_classes = self.model.SCENE_CLASSES
        probs = torch.softmax(semantic_logits, dim=1)[0]

        return {
            "classes": scene_classes,
            "probabilities": {scene_classes[i]: float(probs[i].item()) for i in range(len(scene_classes))},
        }


# Convenience functions
def load_semantic_model(checkpoint_path: str = "checkpoints/stage1_colorizer_semantic_latest.pth"):
    """Quick load semantic model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return SemanticColorizerInference(checkpoint_path, device)


def load_baseline_model(checkpoint_path: str = "checkpoints/stage1_colorizer_latest.pth"):
    """Quick load baseline model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorizerFactory.load_model(checkpoint_path, device, mode="baseline")
    return model
