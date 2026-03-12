import argparse
from pathlib import Path
from typing import Dict

import torch

from models.depth_model import DynamicFilterNetwork
from models.micro_contrast_model import MicroContrastModel
from models.rrdb_sr import RRDBNet
from models.unet_colorizer import UNetColorizer


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _resolve_checkpoint(checkpoints_dir: Path, base_name: str) -> Path:
    best = checkpoints_dir / f"{base_name}_best.pth"
    latest = checkpoints_dir / f"{base_name}_latest.pth"
    if best.exists():
        return best
    if latest.exists():
        return latest
    raise FileNotFoundError(f"Missing checkpoint for {base_name}: expected {best} or {latest}")


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = _strip_module_prefix(state_dict)
    if not state_dict:
        raise RuntimeError(f"Checkpoint has empty state_dict: {ckpt_path}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return ckpt


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Health check for Stage1-Stage4 training/inference checkpoints")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda but CUDA is not available.")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    print(f"Running pipeline health check on device: {device}")

    stage1 = UNetColorizer(in_channels=1, out_channels=2).to(device)
    stage2 = RRDBNet().to(device)
    stage3 = DynamicFilterNetwork(in_channels=3).to(device)
    stage4 = MicroContrastModel(in_channels=3, out_channels=3).to(device)

    stage_models: Dict[str, torch.nn.Module] = {
        "stage1_colorizer": stage1,
        "stage2_sr": stage2,
        "stage3_depth": stage3,
        "stage4_contrast": stage4,
    }

    failures = []
    for stage_name, model in stage_models.items():
        try:
            ckpt_path = _resolve_checkpoint(args.checkpoints, stage_name)
            _ = _load_checkpoint(model, ckpt_path, device)
            params = _param_count(model)
            print(f"{stage_name}: OK | params={params:,} | checkpoint={ckpt_path}")
        except Exception as exc:
            failures.append((stage_name, str(exc)))
            print(f"{stage_name}: ERROR | {exc}")

    if failures:
        raise RuntimeError(f"Checkpoint loading failed for {len(failures)} stage(s).")

    with torch.no_grad():
        stage1_in = torch.zeros(1, 1, 256, 256, device=device)
        stage1_out = stage1(stage1_in)
        if stage1_out.shape != (1, 2, 256, 256):
            raise RuntimeError(f"stage1_colorizer output shape mismatch: {tuple(stage1_out.shape)}")

        stage1_rgb_proxy = torch.cat([stage1_in, (stage1_out + 1.0) * 0.5], dim=1).clamp(0.0, 1.0)
        stage2_out = stage2(stage1_rgb_proxy)
        if stage2_out.ndim != 4 or stage2_out.shape[1] != 3:
            raise RuntimeError(f"stage2_sr output shape mismatch: {tuple(stage2_out.shape)}")

        stage3_out = stage3(stage2_out)
        if stage3_out.ndim != 4 or stage3_out.shape[1] != 1:
            raise RuntimeError(f"stage3_depth output shape mismatch: {tuple(stage3_out.shape)}")

        stage4_out = stage4(stage2_out)
        if stage4_out.shape != stage2_out.shape:
            raise RuntimeError(
                f"stage4_contrast output shape mismatch: {tuple(stage4_out.shape)} vs {tuple(stage2_out.shape)}"
            )

    print(f"Dummy forward shapes | stage1={tuple(stage1_out.shape)} stage2={tuple(stage2_out.shape)} "
          f"stage3={tuple(stage3_out.shape)} stage4={tuple(stage4_out.shape)}")
    print("Pipeline health check passed.")


if __name__ == "__main__":
    main()
