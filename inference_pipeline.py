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
from models.micro_contrast_model import MicroContrastModel
from models.rrdb_sr import RRDBNet
from models.unet_colorizer import UNetColorizer
from models.zero_dce import ZeroDCEModel


DEFAULT_PIPELINE_STAGES = ["colorizer", "sr", "depth", "bokeh", "tone", "contrast"]


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return contextlib.nullcontext()


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> bool:
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(_strip_module_prefix(state_dict), strict=True)
    model.eval()
    return True


def load_pipeline_config(config_path: Path) -> Dict[str, Any]:
    config = {
        "pipeline": {
            "stages": DEFAULT_PIPELINE_STAGES,
            "stage_options": {
                "colorizer": {"color_boost": 1.35, "ab_clip": 128.0},
                "depth": {"inference_size": 384},
                "bokeh": {"focus_threshold": 0.2},
            },
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
    ):
        self.device = device
        self.checkpoints_dir = checkpoints_dir
        self.amp_enabled = amp_enabled
        self.options = options or {}
        self.model: Optional[torch.nn.Module] = None
        self.loaded = False

    def build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def load(self) -> None:
        if self.loaded:
            return
        self.model = self.build_model().to(self.device)

        if self.checkpoint_name:
            ckpt_path = self.checkpoints_dir / self.checkpoint_name
            loaded = _load_checkpoint(self.model, ckpt_path, self.device)
            if not loaded and self.required:
                raise FileNotFoundError(f"Missing required checkpoint: {ckpt_path}")
            if not loaded:
                # Optional stage with missing checkpoint: stage will behave like identity.
                self.model = None
                self.loaded = True
                return
        else:
            self.model.eval()

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
    checkpoint_name = "stage1_colorizer.pth"
    required = True

    def build_model(self) -> torch.nn.Module:
        return UNetColorizer(in_channels=1, out_channels=2)

    def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.model is None:
            return x

        context = kwargs.get("context", {})
        l_np = context.get("l_np")
        if l_np is None:
            raise RuntimeError("ColorizerStage requires context['l_np'] for LAB reconstruction.")

        color_boost = float(context.get("color_boost", self.options.get("color_boost", 1.0)))
        ab_clip = float(self.options.get("ab_clip", 128.0))

        with torch.inference_mode():
            with _autocast_context(self.device, self.amp_enabled):
                ab_pred = self.model(x).squeeze(0).permute(1, 2, 0).float().cpu().numpy()

        lab = np.zeros((l_np.shape[0], l_np.shape[1], 3), dtype=np.float32)
        lab[:, :, 0] = l_np * 100.0

        ab = np.clip(ab_pred, -1.0, 1.0) * 128.0
        mean_chroma = float(np.sqrt(np.mean(ab[:, :, 0] ** 2 + ab[:, :, 1] ** 2)))
        adaptive_boost = color_boost
        if mean_chroma < 15.0:
            adaptive_boost = color_boost * 1.8
        elif mean_chroma < 25.0:
            adaptive_boost = color_boost * 1.3
        ab = ab * adaptive_boost
        lab[:, :, 1:] = np.clip(ab, -ab_clip, ab_clip)
        rgb = np.clip(lab2rgb(lab), 0.0, 1.0)

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return rgb_tensor


class SRStage(PipelineStage):
    stage_name = "sr"
    checkpoint_name = "stage2_sr.pth"

    def build_model(self) -> torch.nn.Module:
        return RRDBNet()

    def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.model is None:
            return x
        with torch.inference_mode():
            with _autocast_context(self.device, self.amp_enabled):
                out = self.model(x)
        return torch.clamp(out, 0.0, 1.0)


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
            context["depth"] = x.mean(dim=1, keepdim=True)
            return x

        h, w = x.shape[-2:]
        depth_in = F.interpolate(x, size=(inference_size, inference_size), mode="bilinear", align_corners=False)

        with torch.inference_mode():
            with _autocast_context(self.device, self.amp_enabled):
                depth_pred = self.model(depth_in)

        depth_up = F.interpolate(depth_pred.float(), size=(h, w), mode="bilinear", align_corners=False)
        context["depth"] = torch.clamp(depth_up, 0.0, 1.0)
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
            return x

        focus_threshold = float(self.options.get("focus_threshold", 0.2))
        with torch.inference_mode():
            with _autocast_context(self.device, self.amp_enabled):
                out = self.model(x, depth, focus_threshold=focus_threshold)
        return torch.clamp(out, 0.0, 1.0)


class ToneStage(PipelineStage):
    stage_name = "tone"
    checkpoint_name = "stage5_tone.pth"

    def build_model(self) -> torch.nn.Module:
        iterations = int(self.options.get("iterations", 8))
        channels = int(self.options.get("channels", 32))
        return ZeroDCEModel(num_iterations=iterations, channels=channels)

    def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.model is None:
            return x

        with torch.inference_mode():
            with _autocast_context(self.device, self.amp_enabled):
                enhanced, _ = self.model(x)
        return torch.clamp(enhanced, 0.0, 1.0)


class ContrastStage(PipelineStage):
    stage_name = "contrast"
    checkpoint_name = "stage6_contrast.pth"

    def build_model(self) -> torch.nn.Module:
        return MicroContrastModel(in_channels=3, out_channels=3)

    def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.model is None:
            return x
        with torch.inference_mode():
            with _autocast_context(self.device, self.amp_enabled):
                out = self.model(x)
        return torch.clamp(out, 0.0, 1.0)


STAGE_REGISTRY = {
    "colorizer": ColorizerStage,
    "sr": SRStage,
    "depth": DepthStage,
    "bokeh": BokehStage,
    "tone": ToneStage,
    "contrast": ContrastStage,
}


class ModularInferencePipeline:
    def __init__(
        self,
        checkpoints_dir: Path,
        config_path: Path,
        stage_override: Optional[Iterable[str]] = None,
        device: Optional[torch.device] = None,
        amp_enabled: bool = True,
    ):
        self.checkpoints_dir = checkpoints_dir
        self.config_path = config_path
        self.config = load_pipeline_config(config_path)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_enabled = amp_enabled

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
                )
            )

    def _prepare_input(self, image: Image.Image) -> tuple[torch.Tensor, Dict[str, Any]]:
        context: Dict[str, Any] = {}

        if self.stage_names and self.stage_names[0] == "colorizer":
            gray = image.convert("L").resize((256, 256), Image.Resampling.BICUBIC)
            l_np = np.asarray(gray, dtype=np.float32) / 255.0
            l_tensor = torch.from_numpy(l_np).unsqueeze(0).unsqueeze(0).float().to(self.device)
            context["l_np"] = l_np
            return l_tensor, context

        rgb = image.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return rgb_tensor, context

    def process_image(self, image: Image.Image, context_overrides: Optional[Dict[str, Any]] = None) -> np.ndarray:
        x, context = self._prepare_input(image)
        if context_overrides:
            context.update(context_overrides)

        for stage in self.stages:
            stage.load()
            x = stage.run(x, context=context)
            stage.unload()

        rgb = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        return np.clip(rgb, 0.0, 1.0)


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
) -> Path:
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
    )

    image = Image.open(input_image)
    rgb = pipeline.process_image(image, context_overrides={"color_boost": color_boost})

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
    parser = argparse.ArgumentParser(description="Inference pipeline for Colorizer-AI")
    parser.add_argument("image", type=Path, help="Input grayscale image path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--full-pipeline", action="store_true", help="Run all configured pipeline stages")
    parser.add_argument("--pipeline-config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--stages", nargs="+", default=None, help="Optional explicit stage order override")
    parser.add_argument("--color-boost", type=float, default=1.35, help="Stage1 AB chroma boost factor.")

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
    )


if __name__ == "__main__":
    main()
