from __future__ import annotations

import argparse
import contextlib
import gc
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import lab2rgb

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for config-driven pipeline execution.") from exc

from models.dfn_bokeh import DFNBokehModel
from models.depth_model import DynamicFilterNetwork
from models.gan_generator import GANGenerator
from models.micro_contrast_model import MicroContrastModel
from models.rrdb_sr import RRDBNet
from models.unet_colorizer import UNetColorizer
from models.zero_dce import ZeroDCEModel


DEFAULT_PIPELINE_STAGES = ["colorizer", "sr", "depth", "bokeh", "tone", "contrast", "gan", "sharpen", "normalize"]


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return contextlib.nullcontext()


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device, stage_name: str = "unknown", strict: bool = True) -> bool:
    """Load checkpoint with intelligent fallback for architecture mismatches."""
    if not checkpoint_path.exists():
        print(f"[Warning] Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(_strip_module_prefix(state_dict), strict=strict)
        model.eval()
        print(f"[{stage_name}] Checkpoint loaded successfully: {checkpoint_path.name}")
        return True
    except RuntimeError as e:
        error_msg = str(e)
        if "missing" in error_msg.lower() or "unexpected" in error_msg.lower():
            print(f"[Warning] {stage_name}: Architecture mismatch in checkpoint: {error_msg}")
            if not strict:
                print(f"[Fallback] {stage_name}: Loading with strict=False")
                try:
                    model.load_state_dict(_strip_module_prefix(state_dict), strict=False)
                    model.eval()
                    return True
                except Exception as fallback_err:
                    print(f"[Error] {stage_name}: Fallback load failed: {fallback_err}")
                    return False
        print(f"[Error] {stage_name}: Failed to load checkpoint: {e}")
        return False
    except Exception as e:
        print(f"[Error] {stage_name}: Unexpected error loading checkpoint: {e}")
        return False


def get_default_checkpoint_map() -> Dict[str, str]:
    """Default checkpoint filenames for each stage."""
    return {
        "colorizer": "stage1_colorizer_latest.pth",
        "sr": "stage2_sr_latest.pth",
        "depth": "stage3_depth_latest.pth",
        "bokeh": "stage4_bokeh.pth",
        "tone": "stage5_tone.pth",
        "contrast": "stage6_contrast.pth",
    }


def load_pipeline_config(config_path: Path) -> Dict[str, Any]:
    config = {
        "pipeline": {
            "stages": DEFAULT_PIPELINE_STAGES,
            "stage_options": {
                "colorizer": {"color_boost": 1.35, "ab_clip": 128.0},
                "depth": {"inference_size": 384},
                "bokeh": {"focus_threshold": 0.2},
            },
            "checkpoints": get_default_checkpoint_map(),
        }
    }
    if not config_path.exists():
        return config

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    pipeline = loaded.get("pipeline", {})
    if isinstance(pipeline.get("stages"), list) and pipeline["stages"]:
        config["pipeline"]["stages"] = [str(s).strip() for s in pipeline["stages"] if str(s).strip()]
    if isinstance(pipeline.get("stage_options"), dict):
        config["pipeline"]["stage_options"].update(pipeline["stage_options"])
    if isinstance(pipeline.get("checkpoints"), dict):
        config["pipeline"]["checkpoints"].update(pipeline["checkpoints"])
    return config


class PipelineStage:
    stage_name: str = "base"
    checkpoint_name: Optional[str] = None
    required: bool = False

    def __init__(
        self,
        device: torch.device,
        checkpoints_dir: Path,
        amp_enabled: bool,
        options: Optional[Dict[str, Any]] = None,
        checkpoint_map: Optional[Dict[str, str]] = None,
    ):
        self.device = device
        self.checkpoints_dir = checkpoints_dir
        self.amp_enabled = amp_enabled
        self.options = options or {}
        self.checkpoint_map = checkpoint_map or {}
        self.model: Optional[torch.nn.Module] = None
        self.loaded = False

    def build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def _resolve_checkpoint_path(self) -> Optional[Path]:
        """Resolve checkpoint path using mapping, with fallback to default."""
        # Try mapping first
        if self.stage_name in self.checkpoint_map:
            ckpt_name = self.checkpoint_map[self.stage_name]
            ckpt_path = self.checkpoints_dir / ckpt_name
            print(f"[Pipeline] {self.stage_name}: using mapped checkpoint '{ckpt_name}'")
            return ckpt_path
        
        # Fallback to hardcoded checkpoint_name
        if self.checkpoint_name:
            ckpt_path = self.checkpoints_dir / self.checkpoint_name
            print(f"[Pipeline] {self.stage_name}: using default checkpoint '{self.checkpoint_name}'")
            return ckpt_path
        
        return None

    def load(self) -> None:
        if self.loaded:
            return
        self.model = self.build_model().to(self.device)
        print(f"[{self.stage_name}] Model built on device: {self.device}")

        ckpt_path = self._resolve_checkpoint_path()
        if ckpt_path:
            # Use strict=False for stages prone to architecture changes (bokeh, tone)
            use_strict = self.stage_name not in ["bokeh", "tone", "contrast"]
            loaded = _load_checkpoint(self.model, ckpt_path, self.device, stage_name=self.stage_name, strict=use_strict)
            if not loaded and self.required:
                raise FileNotFoundError(f"Missing required checkpoint: {ckpt_path}")
            if not loaded:
                print(f"[{self.stage_name}] Checkpoint loading failed, will use identity transform")
                self.model = None
                self.loaded = True
                return
        else:
            self.model.eval()
            print(f"[{self.stage_name}] No checkpoint path, using base model")

        self.loaded = True

    def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self.loaded = False
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


class ColorizerStage(PipelineStage):
    stage_name = "colorizer"
    checkpoint_name = "stage1_colorizer_latest.pth"
    required = True

    def build_model(self) -> torch.nn.Module:
        return UNetColorizer(in_channels=1, out_channels=2)

    def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.model is None:
            return x

        context = kwargs.get("context", {})
        l_np = context.get("l_np")
        if l_np is None:
            raise RuntimeError("ColorizerStage requires context['l_np']")

        color_boost = float(context.get("color_boost", self.options.get("color_boost", 1.1)))

        # ── 1. Run model ─────────────────────────────────────────────────────
        with torch.inference_mode():
            with _autocast_context(self.device, self.amp_enabled):
                ab_pred = self.model(x)   # output: [1, 2, H, W], range [-1, 1]

        # ── 2. Convert to numpy [H, W, 2] ────────────────────────────────────
        ab_np = ab_pred.squeeze(0).permute(1, 2, 0).float().cpu().numpy()

        # ── 3. CRITICAL: scale from [-1,1] to [-128,128] ──────────────────────
        # This is the most important line. Model tanh output must be scaled.
        ab_128 = np.clip(ab_np, -1.0, 1.0) * 128.0

        # ── 4. Apply colour boost ─────────────────────────────────────────────
        ab_128 = ab_128 * color_boost
        print(f"[Colorizer] AB after boost: [{ab_128.min():.1f}, {ab_128.max():.1f}]")

        # ── 5. Semantic correction (sky blue, water blue, vegetation green) ───
        H, W = l_np.shape
        l_uint8 = (l_np * 255).astype(np.uint8)
        blur    = cv2.GaussianBlur(l_uint8, (21, 21), 0).astype(np.float32) / 255.0
        texture = np.abs(l_np - blur)

        # Sky: top 45%, bright, smooth — push toward blue
        sky = np.zeros((H, W), bool)
        sky[:int(H * 0.45), :] = True
        sky &= (l_np > 0.50) & (texture < 0.08)
        sky_needs_blue = sky & (ab_128[:, :, 1] < 20.0)
        blend = 0.70
        ab_128[sky_needs_blue, 0] = (1-blend)*ab_128[sky_needs_blue,0] + blend*(-5.0)
        ab_128[sky_needs_blue, 1] = (1-blend)*ab_128[sky_needs_blue,1] + blend*(45.0)
        print(f"[Colorizer] Sky correction: {sky_needs_blue.sum()} pixels boosted to blue")

        # Water: lower half, mid-bright, smooth
        wat = np.zeros((H, W), bool)
        wat[int(H * 0.25):, :] = True
        wat &= (l_np > 0.28) & (l_np < 0.72) & (texture < 0.025)
        wat_needs_blue = wat & (ab_128[:, :, 1] < 10.0)
        blend = 0.80
        ab_128[wat_needs_blue, 0] = (1-blend)*ab_128[wat_needs_blue,0] + blend*(-8.0)
        ab_128[wat_needs_blue, 1] = (1-blend)*ab_128[wat_needs_blue,1] + blend*(35.0)
        print(f"[Colorizer] Water correction: {wat_needs_blue.sum()} pixels boosted to blue")

        # Vegetation: model hints green, reinforce
        veg = (l_np > 0.15) & (l_np < 0.60) & (ab_128[:, :, 0] < -2.0)
        blend_v = 0.40
        ab_128[veg, 0] = (1-blend_v)*ab_128[veg,0] + blend_v*(-22.0)
        ab_128[veg, 1] = (1-blend_v)*ab_128[veg,1] + blend_v*(-8.0)
        print(f"[Colorizer] Vegetation correction: {veg.sum()} pixels reinforced to green")

        # Shadows: desaturate very dark areas
        ab_128[l_np < 0.08, :] *= 0.3
        print(f"[Colorizer] Shadows desaturated (dark regions)")

        # ── 6. Edge-aware bilateral filter — reduce colour bleeding ───────────
        def bilateral(ch):
            u8 = np.clip((ch + 128) / 256 * 255, 0, 255).astype(np.uint8)
            f  = cv2.bilateralFilter(u8, d=9, sigmaColor=18, sigmaSpace=18)
            return f.astype(np.float32) / 255 * 256 - 128

        ab_128[:, :, 0] = bilateral(ab_128[:, :, 0])
        ab_128[:, :, 1] = bilateral(ab_128[:, :, 1])
        print(f"[Colorizer] Bilateral filter applied")

        # ── 7. Clip AB to valid range ─────────────────────────────────────────
        ab_128 = np.clip(ab_128, -110.0, 110.0)
        print(f"[Colorizer] AB clipped to [-110, 110]: [{ab_128.min():.1f}, {ab_128.max():.1f}]")

        # ── 8. Reconstruct LAB image ──────────────────────────────────────────
        lab = np.zeros((H, W, 3), dtype=np.float32)
        lab[:, :, 0] = l_np * 100.0    # L: [0, 100]
        lab[:, :, 1] = ab_128[:, :, 0] # A: [-110, 110]
        lab[:, :, 2] = ab_128[:, :, 1] # B: [-110, 110]

        # ── 9. LAB → RGB ──────────────────────────────────────────────────────
        rgb = np.clip(lab2rgb(lab), 0.0, 1.0)
        print(f"[Colorizer] LAB → RGB conversion complete")

        # ── 10. Convert to tensor and return ──────────────────────────────────
        rgb_tensor = (torch.from_numpy(rgb)
                      .permute(2, 0, 1)
                      .unsqueeze(0)
                      .float()
                      .to(self.device))
        print(f"[Colorizer] Output tensor shape: {rgb_tensor.shape}")
        return rgb_tensor


class SRStage(PipelineStage):
    stage_name = "sr"
    checkpoint_name = "stage2_sr.pth"

    def build_model(self) -> torch.nn.Module:
        return RRDBNet()

    def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.model is None:
            # No checkpoint: bicubic upsample to maintain pipeline flow
            return torch.nn.functional.interpolate(
                x, scale_factor=4, mode="bicubic", align_corners=False
            ).clamp(0.0, 1.0)

        with torch.inference_mode():
            with _autocast_context(self.device, self.amp_enabled):
                sr_out = self.model(x)

        sr_out = torch.clamp(sr_out, 0.0, 1.0)

        # ── Colour preservation ──────────────────────────────────────────────
        # SR model may have been trained on greyscale and lose colour.
        # Detect if output lost colour vs input and restore if so.
        import torch.nn.functional as F

        sr_h, sr_w = sr_out.shape[-2], sr_out.shape[-1]

        # Upscale input to SR output size for comparison
        x_up = F.interpolate(
            x, size=(sr_h, sr_w), mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)

        # Measure colour strength in SR output vs input
        def chroma(t):
            # Simple colour signal: std across channels
            return t.std(dim=1, keepdim=True).mean().item()

        sr_chroma    = chroma(sr_out)
        input_chroma = chroma(x_up)

        print(f"[SR] Input chroma: {input_chroma:.4f}, SR output chroma: {sr_chroma:.4f}")

        if sr_chroma < 0.3 * input_chroma:
            # SR lost >70% of colour signal — blend to restore
            print("[SR] Colour loss detected — applying colour preservation blend")
            # Extract luminance from SR (sharp detail), colour from input (vivid)
            # Blend: SR provides sharpness, upscaled input provides colour
            sr_out = 0.5 * sr_out + 0.5 * x_up
            sr_out = torch.clamp(sr_out, 0.0, 1.0)
            print(f"[SR] After blend chroma: {chroma(sr_out):.4f}")
        else:
            print("[SR] Colour preserved correctly — no blend needed")

        return sr_out


class DepthStage(PipelineStage):
    stage_name = "depth"
    checkpoint_name = "stage3_depth.pth"

    def build_model(self) -> torch.nn.Module:
        return DynamicFilterNetwork(in_channels=3)

    def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        context = kwargs.get("context", {})
        inference_size = int(self.options.get("inference_size", 384))

        if self.model is None:
            # Graceful fallback so downstream bokeh can still run.
            print(f"[{self.stage_name}] Model not loaded, computing fallback depth from image mean")
            context["depth"] = x.mean(dim=1, keepdim=True)
            return x

        try:
            print(f"[{self.stage_name}] Input shape: {x.shape}")
            h, w = x.shape[-2:]
            depth_in = F.interpolate(x, size=(inference_size, inference_size), mode="bilinear", align_corners=False)

            with torch.inference_mode():
                with _autocast_context(self.device, self.amp_enabled):
                    depth_pred = self.model(depth_in)

            depth_up = F.interpolate(depth_pred.float(), size=(h, w), mode="bilinear", align_corners=False)
            context["depth"] = torch.clamp(depth_up, 0.0, 1.0)
            print(f"[{self.stage_name}] Output shape: {context['depth'].shape}")
            return x
        except Exception as e:
            print(f"[Error] {self.stage_name} inference failed: {e}")
            print(f"[Fallback] {self.stage_name} using mean-based depth")
            context["depth"] = x.mean(dim=1, keepdim=True)
            return x


class BokehStage(PipelineStage):
    stage_name = "bokeh"
    checkpoint_name = "stage4_bokeh.pth"

    def build_model(self) -> torch.nn.Module:
        kernel_size = int(self.options.get("kernel_size", 11))
        return DFNBokehModel(kernel_size=kernel_size)

    def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        context = kwargs.get("context", {})
        depth = context.get("depth")
        if depth is None:
            depth = x.mean(dim=1, keepdim=True)

        if self.model is None:
            print(f"[{self.stage_name}] Model not loaded, skipping (identity pass)")
            return x

        focus_threshold = float(self.options.get("focus_threshold", 0.2))
        
        try:
            print(f"[{self.stage_name}] Input shape: {x.shape}, Depth shape: {depth.shape}")
            with torch.inference_mode():
                with _autocast_context(self.device, self.amp_enabled):
                    out = self.model(x, depth, focus_threshold=focus_threshold)
            
            result = torch.clamp(out, 0.0, 1.0)
            print(f"[{self.stage_name}] Output shape: {result.shape}")
            return result
        except TypeError as e:
            # Checkpoint might be from different model architecture (e.g., contrast model)
            print(f"[Warning] {self.stage_name}: Model architecture mismatch: {e}")
            print(f"[Fallback] {self.stage_name}: Applying simple contrast enhancement")
            try:
                # Simple contrast boost instead of bokeh
                out = x * 1.05  # 5% brightness boost
                return torch.clamp(out, 0.0, 1.0)
            except Exception as fallback_err:
                print(f"[Error] {self.stage_name} fallback failed: {fallback_err}")
                return x
        except Exception as e:
            print(f"[Error] {self.stage_name} inference failed: {e}")
            print(f"[Fallback] {self.stage_name} returning input unchanged")
            return x


class ToneStage(PipelineStage):
    stage_name = "tone"
    checkpoint_name = "stage5_tone.pth"

    def build_model(self) -> torch.nn.Module:
        iterations = int(self.options.get("iterations", 8))
        channels = int(self.options.get("channels", 32))
        return ZeroDCEModel(num_iterations=iterations, channels=channels)

    def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.model is None:
            print(f"[{self.stage_name}] Model not loaded, skipping (identity pass)")
            return x

        try:
            print(f"[{self.stage_name}] Input shape: {x.shape}")
            with torch.inference_mode():
                with _autocast_context(self.device, self.amp_enabled):
                    enhanced, _ = self.model(x)
            
            result = torch.clamp(enhanced, 0.0, 1.0)
            print(f"[{self.stage_name}] Output shape: {result.shape}")
            return result
        except Exception as e:
            print(f"[Error] {self.stage_name} inference failed: {e}")
            print(f"[Fallback] {self.stage_name} returning input unchanged")
            return x


class ContrastStage(PipelineStage):
    stage_name = "contrast"
    checkpoint_name = "stage6_contrast.pth"

    def build_model(self) -> torch.nn.Module:
        return MicroContrastModel(in_channels=3, out_channels=3)

    def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.model is None:
            print(f"[{self.stage_name}] Model not loaded, skipping (identity pass)")
            return x
        
        try:
            print(f"[{self.stage_name}] Input shape: {x.shape}")
            with torch.inference_mode():
                with _autocast_context(self.device, self.amp_enabled):
                    out = self.model(x)
            
            result = torch.clamp(out, 0.0, 1.0)
            print(f"[{self.stage_name}] Output shape: {result.shape}")
            return result
        except Exception as e:
            print(f"[Error] {self.stage_name} inference failed: {e}")
            print(f"[Fallback] {self.stage_name} returning input unchanged")
            return x


class SharpenStage(PipelineStage):
    """Lightweight sharpening stage using unsharp masking (no model)."""
    stage_name = "sharpen"
    checkpoint_name = None
    required = False

    def build_model(self) -> torch.nn.Module:
        return None

    def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        context = kwargs.get("context", {})
        enable_sharpening = bool(context.get("enable_sharpening", True))
        
        if not enable_sharpening or x.shape[0] == 0:
            return x
        
        try:
            x_np = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            print(f"[{self.stage_name}] Input shape: {x_np.shape}")
            
            strength = float(context.get("sharpen_strength", 1.5))
            sigma = float(context.get("sharpen_sigma", 1.0))
            x_sharpened = apply_unsharp_mask(x_np, strength=strength, sigma=sigma)
            
            result = torch.from_numpy(x_sharpened).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            print(f"[{self.stage_name}] Output shape: {result.shape}")
            return result
        except Exception as e:
            print(f"[Warning] {self.stage_name} failed: {e}")
            return x


class NormalizeStage(PipelineStage):
    """Histogram normalization stage using CLAHE (no model)."""
    stage_name = "normalize"
    checkpoint_name = None
    required = False

    def build_model(self) -> torch.nn.Module:
        return None

    def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        context = kwargs.get("context", {})
        enable_normalize = bool(context.get("enable_normalize", True))
        
        if not enable_normalize or x.shape[0] == 0:
            return x
        
        try:
            x_np = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            print(f"[{self.stage_name}] Input shape: {x_np.shape}")
            
            clip_limit = float(context.get("clip_limit", 2.0))
            x_normalized = apply_histogram_equalization(x_np, clip_limit=clip_limit)
            
            result = torch.from_numpy(x_normalized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            print(f"[{self.stage_name}] Output shape: {result.shape}")
            return result
        except Exception as e:
            print(f"[Warning] {self.stage_name} failed: {e}")
            return x


class GANRefinementStage(PipelineStage):
    """Stage 5: GAN-based image refinement for improved perceptual quality."""
    stage_name = "gan"
    checkpoint_name = None  # Will be resolved from checkpoint_map
    required = False

    def build_model(self) -> torch.nn.Module:
        """Build GAN generator model."""
        return GANGenerator(
            in_channels=3,
            out_channels=3,
            base_filters=64,
            num_residual_blocks=4,
        )

    def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply GAN refinement to image.
        
        Args:
            x: Input tensor (B, C, H, W) with values in [0, 1]
            
        Returns:
            Refined tensor (B, C, H, W) with values in [0, 1]
        """
        if self.model is None:
            print(f"[{self.stage_name}] Model not loaded, skipping GAN refinement")
            return x
        
        try:
            # Convert from [0, 1] to [-1, 1] for generator
            x_normalized = x * 2.0 - 1.0
            
            with torch.no_grad():
                with _autocast_context(self.device, self.amp_enabled):
                    output = self.model(x_normalized)
            
            # Convert from [-1, 1] back to [0, 1]
            output = torch.clamp((output + 1.0) / 2.0, 0.0, 1.0)
            
            print(f"[{self.stage_name}] GAN refinement applied successfully")
            return output
            
        except Exception as e:
            print(f"[Warning] {self.stage_name} failed: {e}, returning input unchanged")
            return x


STAGE_REGISTRY = {
    "colorizer": ColorizerStage,
    "sr": SRStage,
    "depth": DepthStage,
    "bokeh": BokehStage,
    "tone": ToneStage,
    "contrast": ContrastStage,
    "gan": GANRefinementStage,
    "sharpen": SharpenStage,
    "normalize": NormalizeStage,
}


class ModularInferencePipeline:
    def __init__(
        self,
        checkpoints_dir: Path,
        config_path: Path,
        stage_override: Optional[Iterable[str]] = None,
        device: Optional[torch.device] = None,
        amp_enabled: bool = True,
        checkpoint_map_override: Optional[Dict[str, str]] = None,
    ):
        self.checkpoints_dir = checkpoints_dir
        self.config_path = config_path
        self.config = load_pipeline_config(config_path)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_enabled = amp_enabled

        # Load or override checkpoint mapping
        self.checkpoint_map = self.config["pipeline"].get("checkpoints", get_default_checkpoint_map())
        if checkpoint_map_override:
            self.checkpoint_map.update(checkpoint_map_override)

        configured_stages = self.config["pipeline"].get("stages", DEFAULT_PIPELINE_STAGES)
        if stage_override is not None:
            configured_stages = [s.strip() for s in stage_override if s.strip()]
        self.stage_names = configured_stages

        stage_options = self.config["pipeline"].get("stage_options", {})
        self.stages: List[PipelineStage] = []
        for stage_name in self.stage_names:
            if stage_name not in STAGE_REGISTRY:
                raise ValueError(f"Unknown pipeline stage '{stage_name}'.")
            stage_cls = STAGE_REGISTRY[stage_name]
            self.stages.append(
                stage_cls(
                    device=self.device,
                    checkpoints_dir=self.checkpoints_dir,
                    amp_enabled=self.amp_enabled,
                    options=stage_options.get(stage_name, {}),
                    checkpoint_map=self.checkpoint_map,
                )
            )

    def _prepare_input(self, image: Image.Image) -> tuple[torch.Tensor, Dict[str, Any]]:
        context: Dict[str, Any] = {}

        if self.stage_names and self.stage_names[0] == "colorizer":
            gray = image.convert("L").resize((256, 256), Image.Resampling.BICUBIC)
            l_np = np.asarray(gray, dtype=np.float32) / 255.0
            l_tensor = torch.from_numpy(l_np).unsqueeze(0).unsqueeze(0).float().to(self.device)
            context["l_np"] = l_np
            print(f"[Pipeline] Input prepared as grayscale (L channel): tensor shape {l_tensor.shape}")
            return l_tensor, context

        rgb = image.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        print(f"[Pipeline] Input prepared as RGB: tensor shape {rgb_tensor.shape}")
        return rgb_tensor, context

    def process_image(self, image: Image.Image, context_overrides: Optional[Dict[str, Any]] = None) -> np.ndarray:
        print(f"[Pipeline] Starting inference on device: {self.device}")
        print(f"[Pipeline] Stage order: {' → '.join(self.stage_names)}")
        
        x, context = self._prepare_input(image)
        if context_overrides:
            context.update(context_overrides)

        for idx, stage in enumerate(self.stages, 1):
            print(f"[Pipeline] Stage {idx}/{len(self.stages)}: {stage.stage_name}")
            try:
                stage.load()
                x_before_shape = x.shape
                x = stage.run(x, context=context)
                x_after_shape = x.shape
                print(f"[Pipeline] {stage.stage_name} complete: {x_before_shape} → {x_after_shape}")
                stage.unload()
            except Exception as e:
                print(f"[Error] Pipeline error at stage {stage.stage_name}: {e}")
                print(f"[Pipeline] Continuing with input from previous stage")
                stage.unload()
                # Continue with previous output

        rgb = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        print(f"[Pipeline] Inference complete")
        return np.clip(rgb, 0.0, 1.0)


def postprocess_ab_channels(
    ab: np.ndarray,
    ab_scale: float = 110.0,
    ab_clip: float = 110.0,
    enable_bilateral: bool = True,
    enable_gaussian: bool = True,
    stabilize_skin: bool = False,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Post-process AB channels for better visual quality.
    
    Args:
        ab: Raw AB output from model (H, W, 2)
        ab_scale: Scale factor for AB (default 110 instead of 128 to reduce saturation)
        ab_clip: Clip range for AB channels
        enable_bilateral: Apply edge-aware bilateral filtering
        enable_gaussian: Apply light Gaussian smoothing
        stabilize_skin: Apply skin tone stabilization heuristic
    
    Returns:
        Processed AB, quality metrics dict
    """
    stats = {}
    
    # 1. Initial clipping and scaling
    ab_clipped = np.clip(ab, -1.0, 1.0) * ab_scale
    stats["mean_chroma_raw"] = float(np.sqrt(np.mean(ab_clipped[:, :, 0] ** 2 + ab_clipped[:, :, 1] ** 2)))
    stats["ab_min_raw"] = float(ab_clipped.min())
    stats["ab_max_raw"] = float(ab_clipped.max())
    
    ab_processed = ab_clipped.copy()
    
    # 2. Bilateral filtering (edge-aware smoothing) - reduces noise while preserving edges
    if enable_bilateral and cv2 is not None:
        for ch in range(2):
            ab_u8 = np.clip((ab_processed[:, :, ch] + ab_clip) / (2 * ab_clip) * 255, 0, 255).astype(np.uint8)
            filtered_u8 = cv2.bilateralFilter(ab_u8, d=5, sigmaColor=50, sigmaSpace=50)
            ab_processed[:, :, ch] = (filtered_u8.astype(np.float32) / 255 * 2 * ab_clip) - ab_clip
    
    # 3. Gaussian smoothing (light smoothing for additional noise reduction)
    if enable_gaussian and cv2 is not None:
        ab_processed[:, :, 0] = cv2.GaussianBlur(ab_processed[:, :, 0], (3, 3), 0)
        ab_processed[:, :, 1] = cv2.GaussianBlur(ab_processed[:, :, 1], (3, 3), 0)
    
    # 4. Skin tone stabilization (optional) - reduce red dominance in skin areas
    if stabilize_skin:
        ab_processed[:, :, 0] *= 0.90   # Reduce A (red-green)
        ab_processed[:, :, 1] *= 0.95   # Reduce B (yellow-blue)
    
    # 5. Final clipping to safe range
    ab_processed = np.clip(ab_processed, -ab_clip, ab_clip)
    
    stats["mean_chroma_processed"] = float(np.sqrt(np.mean(ab_processed[:, :, 0] ** 2 + ab_processed[:, :, 1] ** 2)))
    stats["ab_min_processed"] = float(ab_processed.min())
    stats["ab_max_processed"] = float(ab_processed.max())
    
    return ab_processed, stats


def limit_resolution(x: torch.Tensor, max_size: int = 512) -> torch.Tensor:
    """Limit tensor resolution to avoid excessive memory/computation."""
    if x.shape[-1] > max_size or x.shape[-2] > max_size:
        # Use bilinear interpolation to resize
        h, w = x.shape[-2:]
        scale = min(max_size / h, max_size / w)
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            x_resized = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            print(f"[Pipeline] Resolution limited: {x.shape} → {x_resized.shape}")
            return x_resized
    return x


def apply_unsharp_mask(img: np.ndarray, strength: float = 1.5, sigma: float = 1.0) -> np.ndarray:
    """Apply unsharp masking for sharpening (kernel-based, no model needed)."""
    if cv2 is None or len(img.shape) < 2:
        return img
    try:
        # Handle multi-channel images
        if len(img.shape) == 3:
            img_sharpened = np.zeros_like(img)
            for ch in range(img.shape[2]):
                blur = cv2.GaussianBlur(img[:, :, ch], (0, 0), sigma)
                img_sharpened[:, :, ch] = cv2.addWeighted(img[:, :, ch], strength, blur, 1.0 - strength, 0)
            return np.clip(img_sharpened, 0.0, 1.0)
        else:
            blur = cv2.GaussianBlur(img, (0, 0), sigma)
            return np.clip(cv2.addWeighted(img, strength, blur, 1.0 - strength, 0), 0.0, 1.0)
    except Exception as e:
        print(f"[Warning] Sharpening failed: {e}")
        return img


def apply_histogram_equalization(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for normalization."""
    if cv2 is None:
        return img
    try:
        if len(img.shape) == 3:
            # Convert to LAB, apply CLAHE on L channel, convert back
            img_u8 = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
            bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            
            clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            out_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return np.clip(out_rgb, 0.0, 1.0)
        return img
    except Exception as e:
        print(f"[Warning] Histogram equalization failed: {e}")
        return img


def safe_lab_to_rgb(l: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Safely convert LAB to RGB with proper range handling."""
    try:
        from skimage.color import lab2rgb
    except Exception:
        return np.zeros((l.shape[0], l.shape[1], 3), dtype=np.float32)
    
    # Ensure L is in valid range [0, 100]
    l_valid = np.clip(l * 100.0, 0.0, 100.0)
    
    # Ensure AB is clipped
    ab_valid = np.clip(ab, -128.0, 127.0)
    
    lab = np.dstack((l_valid, ab_valid))
    
    try:
        rgb = lab2rgb(lab)
        return np.clip(rgb, 0.0, 1.0)
    except Exception:
        # Fallback to grayscale if conversion fails
        gray = np.repeat(l[:, :, np.newaxis], 3, axis=2)
        return np.clip(gray, 0.0, 1.0)


def apply_lab_temperature_correction(rgb: np.ndarray, delta_a: int = 6, delta_b: int = 6) -> np.ndarray:
    if cv2 is None:
        return rgb
    try:
        rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)

        lab[:, :, 1] = np.clip(lab[:, :, 1] + int(delta_a), 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + int(delta_b), 0, 255)

        corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return corrected.astype(np.float32) / 255.0
    except Exception:
        return rgb


def apply_color_histogram_normalization(rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    if cv2 is None:
        return rgb
    try:
        rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        out_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.clip(out_rgb, 0.0, 1.0)
    except Exception:
        return rgb


def semantic_color_correction(rgb: np.ndarray, enable_sky_boost: bool = True, enable_water_boost: bool = True, skin_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply SUBTLE semantic-aware color correction to enhance realism.
    
    Subtle enhancements (NEVER aggressive):
    - SKY: Small blue boost (+5 to +10 max in LAB B channel)
    - WATER: Slight blue-green shift (minimal delta only)
    - SKIN PROTECTION: Exclude skin pixels from all corrections
    
    Args:
        rgb: RGB image in range [0, 1] with shape (H, W, 3)
        enable_sky_boost: Enable subtle sky blue enhancement
        enable_water_boost: Enable subtle water blue-green enhancement
        skin_mask: Optional boolean mask (H, W) where True indicates skin pixels to protect
    
    Returns:
        Color-corrected RGB image in range [0, 1]
    """
    # Early return if all corrections are disabled
    if not enable_sky_boost and not enable_water_boost:
        return rgb
    
    if cv2 is None:
        return rgb
    
    try:
        # Convert RGB [0, 1] to LAB for direct AB channel manipulation
        rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        lab_u8 = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB)
        
        # Split into L, A, B channels
        l_channel = lab_u8[:, :, 0].astype(np.float32)
        a_channel = lab_u8[:, :, 1].astype(np.float32)
        b_channel = lab_u8[:, :, 2].astype(np.float32)
        
        height, width = l_channel.shape
        
        # ========== SKY DETECTION & SUBTLE BLUE BOOST ==========
        if enable_sky_boost:
            # Top 30% of image - typically sky region
            sky_region_pct = int(0.3 * height)
            
            # Heuristic: low saturation + high brightness = likely sky
            # In OpenCV LAB: A and B are centered at 128, range [0, 255]
            a_centered = a_channel[:sky_region_pct, :] - 128.0
            b_centered = b_channel[:sky_region_pct, :] - 128.0
            chroma = np.sqrt(a_centered ** 2 + b_centered ** 2)
            
            # Low saturation + high brightness = likely sky
            sky_mask = (chroma < 40) & (l_channel[:sky_region_pct, :] > 100)
            
            # Exclude skin pixels from sky corrections if mask provided
            if skin_mask is not None:
                sky_mask = sky_mask & ~skin_mask[:sky_region_pct, :]
            
            # SUBTLE blue boost: only +5 to +10 max, NOT aggressive
            # Boost only pixels that need it (yellowed sky)
            current_b = b_channel[:sky_region_pct, :]
            delta_b = np.maximum(0, (135 - current_b) * 0.05)  # Subtle: scale by 0.05
            delta_b = np.clip(delta_b, 0, 8)  # Hard limit: +8 max in LAB space
            
            # Apply subtle correction only to detected sky pixels
            b_channel[:sky_region_pct, :][sky_mask] += delta_b[sky_mask]
            
            sky_pixels = np.sum(sky_mask)
            avg_delta_b = np.mean(delta_b[sky_mask]) if sky_pixels > 0 else 0.0
            print(f"[Semantic] Sky: {sky_pixels} pixels, subtle blue delta: {avg_delta_b:.2f}")
        
        # ========== WATER DETECTION & SUBTLE BLUE-GREEN SHIFT ==========
        if enable_water_boost:
            # Bottom 50% of image - typically water/ground region
            water_region_start = int(0.5 * height)
            
            # In OpenCV LAB: A and B are centered at 128
            a_centered = a_channel[water_region_start:, :] - 128.0
            b_centered = b_channel[water_region_start:, :] - 128.0
            chroma = np.sqrt(a_centered ** 2 + b_centered ** 2)
            
            # Medium saturation + decent brightness = likely water/ground
            water_mask = (chroma > 10) & (chroma < 80) & (l_channel[water_region_start:, :] > 60)
            
            # Exclude skin pixels from water corrections if mask provided
            if skin_mask is not None:
                water_mask = water_mask & ~skin_mask[water_region_start:, :]
            
            # SUBTLE blue-green shift: minimal delta only
            # Only enhance if water looks too greenish (high A values)
            current_b_water = b_channel[water_region_start:, :]
            delta_b_water = np.maximum(0, (140 - current_b_water) * 0.02)  # Very subtle: scale by 0.02
            delta_b_water = np.clip(delta_b_water, 0, 5)  # Hard limit: +5 max
            
            # Subtle cyan shift: only reduce redness slightly
            delta_a_water = -np.abs(a_centered) * 0.02  # Very subtle: 2% reduction only
            delta_a_water = np.clip(delta_a_water, -3, 0)  # Hard limit: -3 max
            
            # Apply subtle corrections to detected water pixels
            b_channel[water_region_start:, :][water_mask] += delta_b_water[water_mask]
            a_channel[water_region_start:, :][water_mask] += delta_a_water[water_mask]
            
            water_pixels = np.sum(water_mask)
            avg_delta_b = np.mean(delta_b_water[water_mask]) if water_pixels > 0 else 0.0
            avg_delta_a = np.mean(delta_a_water[water_mask]) if water_pixels > 0 else 0.0
            print(f"[Semantic] Water: {water_pixels} pixels, subtle blue delta: {avg_delta_b:.2f}, red delta: {avg_delta_a:.2f}")
        
        # ========== CLAMP & RECONSTRUCT LAB → RGB ==========
        # Clamp AB channels to valid LAB range (accounting for center offset)
        a_channel = np.clip(a_channel, 0, 255)
        b_channel = np.clip(b_channel, 0, 255)
        l_channel = np.clip(l_channel, 0, 255)
        
        # Merge back to LAB
        lab_corrected = np.stack([
            l_channel.astype(np.uint8),
            a_channel.astype(np.uint8),
            b_channel.astype(np.uint8)
        ], axis=2)
        
        # Convert LAB → BGR → RGB
        bgr_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        rgb_corrected = cv2.cvtColor(bgr_corrected, cv2.COLOR_BGR2RGB)
        
        return rgb_corrected.astype(np.float32) / 255.0
    
    except Exception as e:
        print(f"[Warning] Semantic color correction failed: {e}")
        return rgb




def compute_colorization_quality_score(rgb: np.ndarray) -> Dict[str, Any]:
    """
    Compute coarse automatic quality diagnostics for a colorized RGB image.
    """
    try:
        from skimage.color import rgb2lab
    except Exception:
        return {
            "mean_chroma": 0.0,
            "max_chroma": 0.0,
            "vivid_pixel_pct": 0.0,
            "colorfulness_score": 0.0,
            "quality_grade": "Unavailable",
        }

    lab = rgb2lab(np.clip(rgb, 0.0, 1.0))
    a_ch = lab[:, :, 1]
    b_ch = lab[:, :, 2]
    chroma = np.sqrt(a_ch ** 2 + b_ch ** 2)
    mean_chroma = float(chroma.mean())
    return {
        "mean_chroma": mean_chroma,
        "max_chroma": float(chroma.max()),
        "vivid_pixel_pct": float((chroma > 15.0).mean() * 100.0),
        "colorfulness_score": float(np.std(a_ch) + np.std(b_ch)),
        "quality_grade": (
            "Excellent"
            if mean_chroma > 25.0
            else "Good"
            if mean_chroma > 15.0
            else "Fair"
            if mean_chroma > 8.0
            else "Poor"
        ),
    }


def run(
    input_image: Path,
    output_dir: Path,
    checkpoints: Path,
    run_full_pipeline: bool,
    apply_temperature_correction: bool = True,
    temperature_delta_a: int = 6,
    temperature_delta_b: int = 6,
    apply_histogram_normalization: bool = True,
    pipeline_config: Path = Path("configs/pipeline.yaml"),
    stage_override: Optional[Iterable[str]] = None,
    color_boost: float = 1.35,
    checkpoint_map_override: Optional[Dict[str, str]] = None,
    quality_mode: str = "balanced",
    debug_outputs: bool = False,
    max_size: int = 512,
    enable_sharpening: bool = True,
    enable_histogram_norm: bool = True,
) -> Path:
    """
    quality_mode: "balanced" (default), "high" (all enhancements), or "raw" (minimal)
    debug_outputs: Save intermediate results
    max_size: Maximum resolution for SR stage (default 512 for speed)
    enable_sharpening: Apply unsharp masking after SR
    enable_histogram_norm: Apply histogram normalization as final step
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_image.exists():
        raise FileNotFoundError(f"Input image not found: {input_image}")

    start_time = time.perf_counter()

    stages = list(stage_override) if stage_override is not None else None
    if stages is None:
        stages = None if run_full_pipeline else ["colorizer"]

    pipeline = ModularInferencePipeline(
        checkpoints_dir=checkpoints,
        config_path=pipeline_config,
        stage_override=stages,
        checkpoint_map_override=checkpoint_map_override,
    )

    image = Image.open(input_image)
    
    # Configure post-processing based on quality mode
    quality_config = {
        "raw": {
            "enable_postprocess": False,
            "enable_bilateral": False,
            "enable_gaussian": False,
            "stabilize_skin": False,
        },
        "balanced": {
            "enable_postprocess": True,
            "enable_bilateral": True,
            "enable_gaussian": True,
            "stabilize_skin": False,
        },
        "high": {
            "enable_postprocess": True,
            "enable_bilateral": True,
            "enable_gaussian": True,
            "stabilize_skin": True,
        },
    }
    
    if quality_mode not in quality_config:
        print(f"[WARNING] Unknown quality mode '{quality_mode}', using 'balanced'")
        quality_mode = "balanced"
    
    config = quality_config[quality_mode]
    print(f"[Pipeline] Quality mode: {quality_mode}")
    print(f"  Post-process: {config['enable_postprocess']}")
    print(f"  Bilateral filter: {config['enable_bilateral']}")
    print(f"  Gaussian smoothing: {config['enable_gaussian']}")
    print(f"  Skin tone stabilization: {config['stabilize_skin']}")
    
    debug_dir = output_dir / "debug" if debug_outputs else None
    context_overrides = {
        "color_boost": color_boost,
        "enable_postprocess": config["enable_postprocess"],
        "enable_bilateral": config["enable_bilateral"],
        "enable_gaussian": config["enable_gaussian"],
        "stabilize_skin": config["stabilize_skin"],
        "debug_output_dir": debug_dir,
        "max_size": max_size,
        "enable_sharpening": enable_sharpening,
        "enable_normalize": enable_histogram_norm,
        "sharpen_strength": 1.5 if quality_mode == "high" else 1.2,
        "sharpen_sigma": 1.0,
        "clip_limit": 2.5 if quality_mode == "high" else 2.0,
    }
    
    rgb = pipeline.process_image(image, context_overrides=context_overrides)

    if apply_temperature_correction:
        rgb = apply_lab_temperature_correction(rgb, delta_a=temperature_delta_a, delta_b=temperature_delta_b)

    if apply_histogram_normalization and run_full_pipeline:
        rgb = apply_color_histogram_normalization(rgb)

    out_path = output_dir / f"colorized_{input_image.stem}.jpg"
    out_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(out_u8).save(out_path, format="JPEG", quality=95)
    quality = compute_colorization_quality_score(rgb)

    print(f"Saved output: {out_path}")
    print("Colorization quality report:")
    print(f"  Grade: {quality['quality_grade']}")
    print(f"  Mean Chroma: {quality['mean_chroma']:.1f}")
    print(f"  Vivid Pixels: {quality['vivid_pixel_pct']:.1f}%")
    print(f"  Colorfulness: {quality['colorfulness_score']:.1f}")
    print(f"Inference time: {time.perf_counter() - start_time:.3f}s")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference pipeline for Colorizer-AI with quality enhancements")
    parser.add_argument("image", type=Path, help="Input grayscale image path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--full-pipeline", action="store_true", help="Run all configured pipeline stages")
    parser.add_argument("--pipeline-config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--stages", nargs="+", default=None, help="Optional explicit stage order override")
    parser.add_argument("--color-boost", type=float, default=1.35, help="Stage1 AB chroma boost factor.")
    
    parser.add_argument(
        "--quality-mode",
        type=str,
        default="balanced",
        choices=["raw", "balanced", "high"],
        help="Quality mode: raw (minimal), balanced (default), or high (all enhancements)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate debug outputs (raw colorization, etc.)",
    )
    
    parser.add_argument(
        "--checkpoint-map",
        type=str,
        nargs="+",
        default=None,
        help="Override checkpoint mapping. Format: stage_name=filename",
    )

    parser.set_defaults(lab_temp_correction=True)
    parser.add_argument("--lab-temp-correction", dest="lab_temp_correction", action="store_true")
    parser.add_argument("--no-lab-temp-correction", dest="lab_temp_correction", action="store_false")
    parser.add_argument("--lab-temp-a", type=int, default=6)
    parser.add_argument("--lab-temp-b", type=int, default=6)

    parser.set_defaults(hist_norm=True)
    parser.add_argument("--hist-norm", dest="hist_norm", action="store_true")
    parser.add_argument("--no-hist-norm", dest="hist_norm", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    checkpoint_map_override = None
    if args.checkpoint_map:
        checkpoint_map_override = {}
        for mapping in args.checkpoint_map:
            if "=" not in mapping:
                raise ValueError(f"Invalid checkpoint mapping format: '{mapping}'. Expected 'stage_name=filename'")
            stage_name, filename = mapping.split("=", 1)
            checkpoint_map_override[stage_name.strip()] = filename.strip()
            print(f"[CLI] Checkpoint override: {stage_name.strip()} → {filename.strip()}")
    
    run(
        input_image=args.image,
        output_dir=args.output_dir,
        checkpoints=args.checkpoints,
        run_full_pipeline=args.full_pipeline,
        apply_temperature_correction=args.lab_temp_correction,
        temperature_delta_a=args.lab_temp_a,
        temperature_delta_b=args.lab_temp_b,
        apply_histogram_normalization=args.hist_norm,
        pipeline_config=args.pipeline_config,
        stage_override=args.stages,
        color_boost=args.color_boost,
        checkpoint_map_override=checkpoint_map_override,
        quality_mode=args.quality_mode,
        debug_outputs=args.debug,
    )


if __name__ == "__main__":
    main()
