import argparse
import time
from pathlib import Path
from typing import List

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


def _color_stats(img_path: Path) -> dict:
    arr = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
    mean_rgb = arr.mean(axis=(0, 1))
    std_rgb = arr.std(axis=(0, 1))
    saturation = (arr.max(axis=2) - arr.min(axis=2)).mean()
    return {
        "mean_r": float(mean_rgb[0]),
        "mean_g": float(mean_rgb[1]),
        "mean_b": float(mean_rgb[2]),
        "std_r": float(std_rgb[0]),
        "std_g": float(std_rgb[1]),
        "std_b": float(std_rgb[2]),
        "saturation_mean": float(saturation),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline quality test for Colorizer-AI")
    parser.add_argument("--input-dir", type=Path, default=Path("assets"), help="Directory containing sample images.")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test_results"))
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--full-pipeline", action="store_true", help="Run Stage2/3/4 in addition to Stage1.")
    parser.set_defaults(lab_temp_correction=True)
    parser.add_argument("--lab-temp-correction", dest="lab_temp_correction", action="store_true")
    parser.add_argument("--no-lab-temp-correction", dest="lab_temp_correction", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(args.input_dir, args.max_images)
    if not images:
        raise RuntimeError(f"No input images found in {args.input_dir}")

    all_times: List[float] = []
    all_stats: List[dict] = []

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
        stats = _color_stats(out_path)
        all_times.append(elapsed)
        all_stats.append(stats)
        print(
            f"{image_path.name} -> {out_path.name} | time={elapsed:.3f}s "
            f"| sat={stats['saturation_mean']:.4f}"
        )

    avg_time = float(np.mean(all_times)) if all_times else 0.0
    mean_sat = float(np.mean([s["saturation_mean"] for s in all_stats])) if all_stats else 0.0
    mean_rgb = np.mean([[s["mean_r"], s["mean_g"], s["mean_b"]] for s in all_stats], axis=0) if all_stats else np.zeros(3)
    std_rgb = np.mean([[s["std_r"], s["std_g"], s["std_b"]] for s in all_stats], axis=0) if all_stats else np.zeros(3)

    print("\n=== Quality Summary ===")
    print(f"Images processed: {len(images)}")
    print(f"Average inference time: {avg_time:.3f}s")
    print(f"Average saturation mean: {mean_sat:.4f}")
    print(f"Average mean RGB: ({mean_rgb[0]:.4f}, {mean_rgb[1]:.4f}, {mean_rgb[2]:.4f})")
    print(f"Average std RGB: ({std_rgb[0]:.4f}, {std_rgb[1]:.4f}, {std_rgb[2]:.4f})")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
