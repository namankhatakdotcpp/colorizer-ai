import argparse
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from skimage.color import lab2rgb
try:
    import cv2
except Exception:
    cv2 = None

from models.unet_colorizer import UNetColorizer
from models.rrdb_sr import RRDBNet
from models.depth_model import DynamicFilterNetwork
from models.micro_contrast_model import MicroContrastModel


def _ensure_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if torch.isnan(tensor).any():
        raise RuntimeError(f"NaN detected in inference pipeline ({name})")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"NaN detected in pipeline ({name})")


def _ensure_valid_rgb(name: str, rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(f"Invalid RGB shape for {name}: {rgb.shape}")
    if np.isnan(rgb).any():
        raise RuntimeError(f"NaN detected in inference pipeline ({name})")
    if not np.isfinite(rgb).all():
        raise RuntimeError(f"NaN detected in pipeline ({name})")


def _assert_uint8_range(name: str, img_u8: np.ndarray) -> None:
    assert img_u8.min() >= 0, f"{name} min below 0"
    assert img_u8.max() <= 255, f"{name} max above 255"


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> bool:
    if not path.exists():
        print(f"[WARN] Missing checkpoint: {path}")
        return False

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Loaded checkpoint: {path}")
    return True


def stage1_colorize(gray_img: Image.Image, model: UNetColorizer, device: torch.device) -> np.ndarray:
    # Resize to training resolution for consistent behavior.
    gray = gray_img.resize((256, 256), Image.Resampling.BICUBIC)
    l_np = np.asarray(gray, dtype=np.float32) / 255.0
    if l_np.ndim != 2:
        raise RuntimeError(f"Invalid grayscale input shape: {l_np.shape}")
    if not np.isfinite(l_np).all():
        raise RuntimeError("NaN detected in pipeline (input_luminance)")

    l_tensor = torch.from_numpy(l_np).unsqueeze(0).unsqueeze(0).to(device)
    _ensure_finite_tensor("stage1_input", l_tensor)
    if float(l_tensor.min().item()) < -1e-6 or float(l_tensor.max().item()) > 1.0 + 1e-6:
        raise RuntimeError("Stage1 input tensor out of expected [0,1] range.")

    with torch.inference_mode():
        ab_pred = model(l_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
    if not np.isfinite(ab_pred).all():
        raise RuntimeError("NaN detected in pipeline (stage1_ab)")

    # Undo normalization: L in [0,100], AB in [-128,128].
    lab = np.zeros((l_np.shape[0], l_np.shape[1], 3), dtype=np.float32)
    lab[:, :, 0] = l_np * 100.0
    lab[:, :, 1:] = np.clip(ab_pred, -1.0, 1.0) * 128.0

    rgb = np.clip(lab2rgb(lab), 0.0, 1.0)
    _ensure_valid_rgb("stage1_output", rgb)
    return rgb


def apply_lab_temperature_correction(rgb: np.ndarray, delta_a: int = 6, delta_b: int = 6) -> np.ndarray:
    if cv2 is None:
        print("[WARN] OpenCV unavailable; skipping LAB temperature correction.")
        return rgb
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        print(f"[WARN] Unexpected image shape {rgb.shape}; skipping LAB temperature correction.")
        return rgb

    try:
        rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        _assert_uint8_range("temperature_rgb_u8", rgb_u8)
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)

        lab[:, :, 1] = np.clip(lab[:, :, 1] + int(delta_a), 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + int(delta_b), 0, 255)

        corrected_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
        _assert_uint8_range("temperature_corrected_rgb_u8", corrected_rgb)
        return corrected_rgb.astype(np.float32) / 255.0
    except Exception as exc:
        print(f"[WARN] LAB temperature correction failed, skipping: {exc}")
        return rgb


def apply_color_histogram_normalization(rgb: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    if cv2 is None:
        print("[WARN] OpenCV unavailable; skipping color histogram normalization.")
        return rgb
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        print(f"[WARN] Unexpected image shape {rgb.shape}; skipping color histogram normalization.")
        return rgb

    try:
        rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        _assert_uint8_range("histnorm_rgb_u8", rgb_u8)
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        out_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.clip(out_rgb, 0.0, 1.0)
    except Exception as exc:
        print(f"[WARN] Color histogram normalization failed, skipping: {exc}")
        return rgb


def maybe_run_full_pipeline(rgb: np.ndarray, device: torch.device, checkpoints: Path) -> np.ndarray:
    _ensure_valid_rgb("pipeline_input", rgb)
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    _ensure_finite_tensor("pipeline_input_tensor", rgb_tensor)

    sr = RRDBNet().to(device)
    depth = DynamicFilterNetwork(3).to(device)
    contrast = MicroContrastModel(3, 3).to(device)

    sr_loaded = load_checkpoint(sr, checkpoints / "stage2_sr_best.pth", device)
    depth_loaded = load_checkpoint(depth, checkpoints / "stage3_depth_best.pth", device)
    contrast_loaded = load_checkpoint(contrast, checkpoints / "stage4_contrast_best.pth", device)

    with torch.inference_mode():
        print("Stage2 start")
        out = sr(rgb_tensor) if sr_loaded else rgb_tensor
        _ensure_finite_tensor("stage2_output", out)
        print("Stage2 finished")

        print("Stage3 start")
        if depth_loaded:
            depth_out = depth(out)
            _ensure_finite_tensor("stage3_output", depth_out)
        print("Stage3 finished")

        print("Stage4 start")
        out = contrast(out) if contrast_loaded else out
        _ensure_finite_tensor("stage4_output", out)
        print("Stage4 finished")

    out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    out = np.clip(out, 0.0, 1.0)
    _ensure_valid_rgb("pipeline_output", out)
    return out


def run(
    input_image: Path,
    output_dir: Path,
    checkpoints: Path,
    run_full_pipeline: bool,
    apply_temperature_correction: bool = True,
    temperature_delta_a: int = 6,
    temperature_delta_b: int = 6,
    apply_histogram_normalization: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_image.exists():
        raise FileNotFoundError(f"Input image not found: {input_image}")

    start_time = time.perf_counter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    colorizer = UNetColorizer(in_channels=1, out_channels=2).to(device)
    stage1_best = checkpoints / "stage1_colorizer_best.pth"
    stage1_latest = checkpoints / "stage1_colorizer_latest.pth"
    if not load_checkpoint(colorizer, stage1_best, device):
        if not load_checkpoint(colorizer, stage1_latest, device):
            raise FileNotFoundError("Missing stage1 checkpoint (expected stage1_colorizer_best.pth or stage1_colorizer_latest.pth)")

    print("Stage1 start")
    gray = Image.open(input_image).convert("L")
    rgb = stage1_colorize(gray, colorizer, device)
    print("Stage1 finished")

    if apply_temperature_correction:
        rgb = apply_lab_temperature_correction(
            rgb,
            delta_a=temperature_delta_a,
            delta_b=temperature_delta_b,
        )
        _ensure_valid_rgb("temperature_corrected", rgb)

    if run_full_pipeline:
        rgb = maybe_run_full_pipeline(rgb, device, checkpoints)

    if run_full_pipeline and apply_histogram_normalization:
        rgb = apply_color_histogram_normalization(rgb)
        _ensure_valid_rgb("histogram_normalized", rgb)

    stem = input_image.stem
    out_path = output_dir / f"colorized_{stem}.jpg"
    out_u8 = (rgb * 255.0).astype(np.uint8)
    _assert_uint8_range("final_output_u8", out_u8)
    Image.fromarray(out_u8).save(out_path, format="JPEG", quality=95)
    print(f"Saved output: {out_path}")
    print(f"Inference time: {time.perf_counter() - start_time:.3f}s")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference pipeline for Colorizer-AI")
    parser.add_argument("image", type=Path, help="Input grayscale image path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--full-pipeline", action="store_true", help="Run SR/depth/contrast stages if checkpoints are present")
    parser.set_defaults(lab_temp_correction=True)
    parser.add_argument("--lab-temp-correction", dest="lab_temp_correction", action="store_true", help="Enable LAB temperature correction after Stage1.")
    parser.add_argument("--no-lab-temp-correction", dest="lab_temp_correction", action="store_false", help="Disable LAB temperature correction.")
    parser.add_argument("--lab-temp-a", type=int, default=6, help="LAB channel A additive correction.")
    parser.add_argument("--lab-temp-b", type=int, default=6, help="LAB channel B additive correction.")
    parser.set_defaults(hist_norm=True)
    parser.add_argument("--hist-norm", dest="hist_norm", action="store_true", help="Enable post-pipeline LAB histogram normalization.")
    parser.add_argument("--no-hist-norm", dest="hist_norm", action="store_false", help="Disable post-pipeline LAB histogram normalization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.image,
        args.output_dir,
        args.checkpoints,
        args.full_pipeline,
        apply_temperature_correction=args.lab_temp_correction,
        temperature_delta_a=args.lab_temp_a,
        temperature_delta_b=args.lab_temp_b,
        apply_histogram_normalization=args.hist_norm,
    )


if __name__ == "__main__":
    main()
