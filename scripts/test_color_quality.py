import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from inference_pipeline import run


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(input_dir: Path, max_images: int) -> List[Path]:
    images = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images[:max_images]


def _color_statistics(image_path: Path) -> Dict[str, float]:
    arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    arr_01 = arr / 255.0

    # Channel histograms (normalized).
    hist_r, _ = np.histogram(arr[:, :, 0], bins=16, range=(0, 255), density=True)
    hist_g, _ = np.histogram(arr[:, :, 1], bins=16, range=(0, 255), density=True)
    hist_b, _ = np.histogram(arr[:, :, 2], bins=16, range=(0, 255), density=True)

    # Color variance proxy from channel spread.
    color_var = float(np.var(arr_01[:, :, 0]) + np.var(arr_01[:, :, 1]) + np.var(arr_01[:, :, 2]))

    return {
        "mean_r": float(arr_01[:, :, 0].mean()),
        "mean_g": float(arr_01[:, :, 1].mean()),
        "mean_b": float(arr_01[:, :, 2].mean()),
        "var_color": color_var,
        "hist_r_mean": float(hist_r.mean()),
        "hist_g_mean": float(hist_g.mean()),
        "hist_b_mean": float(hist_b.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Color quality test for colorization pipeline")
    parser.add_argument("--input-dir", type=Path, default=Path("assets"), help="Directory with sample images.")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test_results"))
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--full-pipeline", action="store_true")
    parser.set_defaults(lab_temp_correction=True)
    parser.add_argument("--lab-temp-correction", dest="lab_temp_correction", action="store_true")
    parser.add_argument("--no-lab-temp-correction", dest="lab_temp_correction", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Verify checkpoints exist up front.
    required = [args.checkpoints / "stage1_colorizer.pth"]
    if args.full_pipeline:
        required.extend(
            [
                args.checkpoints / "stage2_sr.pth",
                args.checkpoints / "stage3_depth.pth",
                args.checkpoints / "stage4_bokeh.pth",
                args.checkpoints / "stage5_tone.pth",
                args.checkpoints / "stage6_contrast.pth",
            ]
        )
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing required checkpoints: {missing}")

    images = _collect_images(args.input_dir, args.max_images)
    if not images:
        raise RuntimeError(f"No images found in {args.input_dir}")

    times: List[float] = []
    stats_all: List[Dict[str, float]] = []

    for image_path in images:
        start = time.perf_counter()
        out_path = run(
            input_image=image_path,
            output_dir=args.output_dir,
            checkpoints=args.checkpoints,
            run_full_pipeline=args.full_pipeline,
            apply_temperature_correction=args.lab_temp_correction,
        )
        elapsed = time.perf_counter() - start
        stats = _color_statistics(out_path)

        times.append(elapsed)
        stats_all.append(stats)
        print(
            f"{image_path.name} -> {out_path.name} | "
            f"time={elapsed:.3f}s | color_var={stats['var_color']:.6f}"
        )

    avg_time = float(np.mean(times))
    avg_var = float(np.mean([s["var_color"] for s in stats_all]))
    avg_hist_r = float(np.mean([s["hist_r_mean"] for s in stats_all]))
    avg_hist_g = float(np.mean([s["hist_g_mean"] for s in stats_all]))
    avg_hist_b = float(np.mean([s["hist_b_mean"] for s in stats_all]))

    print("\n=== Color Quality Summary ===")
    print(f"Images processed: {len(stats_all)}")
    print(f"Average inference time: {avg_time:.3f}s")
    print(f"Color variance: {avg_var:.6f}")
    print(f"Color histogram stats (mean density) -> R: {avg_hist_r:.6f}, G: {avg_hist_g:.6f}, B: {avg_hist_b:.6f}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
