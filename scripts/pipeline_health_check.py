import argparse
from pathlib import Path
from typing import Dict

import torch

from models.dfn_bokeh import DFNBokehModel
from models.depth_model import DynamicFilterNetwork
from models.micro_contrast_model import MicroContrastModel
from models.rrdb_sr import RRDBNet
from models.unet_colorizer import UNetColorizer
from models.zero_dce import ZeroDCEModel


CHECKPOINT_MAP = {
    "stage1_colorizer": "stage1_colorizer.pth",
    "stage2_sr": "stage2_sr.pth",
    "stage3_depth": "stage3_depth.pth",
    "stage4_bokeh": "stage4_bokeh.pth",
    "stage5_tone": "stage5_tone.pth",
    "stage6_contrast": "stage6_contrast.pth",
}


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return ckpt


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Health check for Stage1-Stage6 training/inference checkpoints")
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
    stage4 = DFNBokehModel().to(device)
    stage5 = ZeroDCEModel().to(device)
    stage6 = MicroContrastModel(in_channels=3, out_channels=3).to(device)

    stage_models: Dict[str, torch.nn.Module] = {
        "stage1_colorizer": stage1,
        "stage2_sr": stage2,
        "stage3_depth": stage3,
        "stage4_bokeh": stage4,
        "stage5_tone": stage5,
        "stage6_contrast": stage6,
    }

    failures = []
    for stage_name, model in stage_models.items():
        try:
            ckpt_path = args.checkpoints / CHECKPOINT_MAP[stage_name]
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
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
        stage1_ab = stage1(stage1_in)
        stage1_rgb = torch.cat([stage1_in, (stage1_ab + 1.0) * 0.5], dim=1).clamp(0.0, 1.0)

        stage2_out = stage2(stage1_rgb)
        stage3_depth = stage3(stage2_out)
        stage4_out = stage4(stage2_out, stage3_depth)
        stage5_out, _ = stage5(stage4_out)
        stage6_out = stage6(stage5_out)

        if stage3_depth.shape[1] != 1:
            raise RuntimeError(f"Depth channel mismatch: {tuple(stage3_depth.shape)}")
        if stage6_out.shape != stage2_out.shape:
            raise RuntimeError(f"Final shape mismatch: {tuple(stage6_out.shape)} vs {tuple(stage2_out.shape)}")

    print(
        f"Dummy forward shapes | ab={tuple(stage1_ab.shape)} sr={tuple(stage2_out.shape)} "
        f"depth={tuple(stage3_depth.shape)} bokeh={tuple(stage4_out.shape)} "
        f"tone={tuple(stage5_out.shape)} contrast={tuple(stage6_out.shape)}"
    )
    print("Pipeline health check passed.")


if __name__ == "__main__":
    main()
