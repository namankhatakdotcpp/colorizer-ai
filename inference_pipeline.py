import argparse
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

    l_tensor = torch.from_numpy(l_np).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        ab_pred = model(l_tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Undo normalization: L in [0,100], AB in [-128,128].
    lab = np.zeros((l_np.shape[0], l_np.shape[1], 3), dtype=np.float32)
    lab[:, :, 0] = l_np * 100.0
    lab[:, :, 1:] = np.clip(ab_pred, -1.0, 1.0) * 128.0

    rgb = np.clip(lab2rgb(lab), 0.0, 1.0)
    return rgb


def apply_lab_temperature_correction(rgb: np.ndarray, delta_a: int = 5, delta_b: int = 5) -> np.ndarray:
    if cv2 is None:
        print("[WARN] OpenCV unavailable; skipping LAB temperature correction.")
        return rgb
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        print(f"[WARN] Unexpected image shape {rgb.shape}; skipping LAB temperature correction.")
        return rgb

    try:
        rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)

        lab[:, :, 1] = np.clip(lab[:, :, 1] + int(delta_a), 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + int(delta_b), 0, 255)

        corrected_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
        return corrected_rgb.astype(np.float32) / 255.0
    except Exception as exc:
        print(f"[WARN] LAB temperature correction failed, skipping: {exc}")
        return rgb


def maybe_run_full_pipeline(rgb: np.ndarray, device: torch.device, checkpoints: Path) -> np.ndarray:
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

    sr = RRDBNet().to(device)
    depth = DynamicFilterNetwork(3).to(device)
    contrast = MicroContrastModel(3, 3).to(device)

    sr_loaded = load_checkpoint(sr, checkpoints / "stage2_sr_best.pth", device)
    depth_loaded = load_checkpoint(depth, checkpoints / "stage3_depth_best.pth", device)
    contrast_loaded = load_checkpoint(contrast, checkpoints / "stage4_contrast_best.pth", device)

    with torch.no_grad():
        out = sr(rgb_tensor) if sr_loaded else rgb_tensor
        if depth_loaded:
            _ = depth(out)
        out = contrast(out) if contrast_loaded else out

    out = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    out = np.clip(out, 0.0, 1.0)
    return out


def run(
    input_image: Path,
    output_dir: Path,
    checkpoints: Path,
    run_full_pipeline: bool,
    apply_temperature_correction: bool = False,
    temperature_delta_a: int = 5,
    temperature_delta_b: int = 5,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}")

    colorizer = UNetColorizer(in_channels=1, out_channels=2).to(device)
    stage1_best = checkpoints / "stage1_colorizer_best.pth"
    stage1_latest = checkpoints / "stage1_colorizer_latest.pth"
    if not load_checkpoint(colorizer, stage1_best, device):
        if not load_checkpoint(colorizer, stage1_latest, device):
            raise FileNotFoundError("Missing stage1 checkpoint (expected stage1_colorizer_best.pth or stage1_colorizer_latest.pth)")

    gray = Image.open(input_image).convert("L")
    rgb = stage1_colorize(gray, colorizer, device)
    if apply_temperature_correction:
        rgb = apply_lab_temperature_correction(
            rgb,
            delta_a=temperature_delta_a,
            delta_b=temperature_delta_b,
        )

    if run_full_pipeline:
        rgb = maybe_run_full_pipeline(rgb, device, checkpoints)

    stem = input_image.stem
    out_path = output_dir / f"colorized_{stem}.jpg"
    Image.fromarray((rgb * 255.0).astype(np.uint8)).save(out_path, format="JPEG", quality=95)
    print(f"Saved output: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference pipeline for Colorizer-AI")
    parser.add_argument("image", type=Path, help="Input grayscale image path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--full-pipeline", action="store_true", help="Run SR/depth/contrast stages if checkpoints are present")
    parser.add_argument("--lab-temp-correction", action="store_true", help="Enable LAB temperature correction after Stage1.")
    parser.add_argument("--lab-temp-a", type=int, default=5, help="LAB channel A additive correction.")
    parser.add_argument("--lab-temp-b", type=int, default=5, help="LAB channel B additive correction.")
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
    )


if __name__ == "__main__":
    main()
