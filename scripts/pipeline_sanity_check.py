import argparse
from pathlib import Path
from typing import Dict, Iterable

import torch
from PIL import Image
import numpy as np

from models.depth_model import DynamicFilterNetwork
from models.micro_contrast_model import MicroContrastModel
from models.rrdb_sr import RRDBNet
from models.unet_colorizer import UNetColorizer


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _resolve_checkpoint(checkpoints: Path, preferred_name: str, fallback_name: str) -> Path:
    preferred = checkpoints / preferred_name
    fallback = checkpoints / fallback_name
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing checkpoint: expected {preferred} or {fallback}")


def _load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(_strip_module_prefix(state_dict), strict=True)
    model.eval()
    print(f"Loaded: {path}")


def _make_stage1_input(image_path: Path | None, device: torch.device) -> torch.Tensor:
    if image_path is None:
        return torch.zeros(1, 1, 256, 256, device=device)

    gray = Image.open(image_path).convert("L").resize((256, 256), Image.Resampling.BICUBIC)
    gray_np = np.asarray(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(gray_np).unsqueeze(0).unsqueeze(0).to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline sanity check for Stage1-Stage4 model loading")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--image", type=Path, default=None, help="Optional grayscale image for Stage1 inference.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is unavailable.")
    device = torch.device(args.device if args.device == "cpu" else "cuda:0")

    stage1 = UNetColorizer(in_channels=1, out_channels=2).to(device)
    stage2 = RRDBNet().to(device)
    stage3 = DynamicFilterNetwork(in_channels=3).to(device)
    stage4 = MicroContrastModel(in_channels=3, out_channels=3).to(device)

    stage_ckpts = {
        "stage1": _resolve_checkpoint(args.checkpoints, "stage1_colorizer_best.pth", "stage1_colorizer_latest.pth"),
        "stage2": _resolve_checkpoint(args.checkpoints, "stage2_sr_best.pth", "stage2_sr_latest.pth"),
        "stage3": _resolve_checkpoint(args.checkpoints, "stage3_depth_best.pth", "stage3_depth_latest.pth"),
        "stage4": _resolve_checkpoint(args.checkpoints, "stage4_contrast_best.pth", "stage4_contrast_latest.pth"),
    }

    _load_checkpoint(stage1, stage_ckpts["stage1"], device)
    _load_checkpoint(stage2, stage_ckpts["stage2"], device)
    _load_checkpoint(stage3, stage_ckpts["stage3"], device)
    _load_checkpoint(stage4, stage_ckpts["stage4"], device)

    stage1_input = _make_stage1_input(args.image, device)
    with torch.no_grad():
        stage1_ab = stage1(stage1_input)
        if stage1_ab.shape[1] != 2:
            raise RuntimeError(f"Stage1 output channel mismatch: expected 2, got {stage1_ab.shape}")

        stage1_rgb_proxy = torch.cat([stage1_input, (stage1_ab + 1.0) * 0.5], dim=1).clamp(0.0, 1.0)
        stage2_out = stage2(stage1_rgb_proxy)
        _ = stage3(stage2_out)
        stage4_out = stage4(stage2_out)

    print(f"Stage1 inference shape: {tuple(stage1_ab.shape)}")
    print(f"Stage2 output shape: {tuple(stage2_out.shape)}")
    print(f"Stage4 output shape: {tuple(stage4_out.shape)}")
    print("Pipeline sanity check passed: all stage models loaded and executed without crashes.")


if __name__ == "__main__":
    main()
