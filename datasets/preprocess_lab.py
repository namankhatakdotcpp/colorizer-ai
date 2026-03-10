import argparse
import hashlib
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **kwargs):
        return iterable


def _discover_images(input_dir: Path) -> List[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images: List[Path] = []
    for pattern in patterns:
        images.extend(input_dir.rglob(pattern))
    return sorted(images)


def _stable_name(path: Path, root: Path) -> str:
    # Deterministic filename safe for nested folders and duplicate basenames.
    rel = str(path.relative_to(root)).replace(os.sep, "_")
    digest = hashlib.md5(rel.encode("utf-8")).hexdigest()[:10]
    stem = Path(rel).stem
    return f"{stem}_{digest}"


def preprocess_images_to_lab(input_dir: Path, output_dir: Path, img_size: int = 256) -> int:
    l_dir = output_dir / "L"
    ab_dir = output_dir / "AB"
    l_dir.mkdir(parents=True, exist_ok=True)
    ab_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _discover_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found under {input_dir}. Please place RGB images before preprocessing."
        )

    for img_path in tqdm(image_paths, desc="RGB -> LAB", unit="img"):
        rgb = Image.open(img_path).convert("RGB")
        rgb = rgb.resize((img_size, img_size), Image.Resampling.BICUBIC)

        # rgb2lab is deterministic for fixed input.
        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
        lab = rgb2lab(rgb_np)

        l_channel = lab[:, :, 0].astype(np.float32)
        ab_channel = lab[:, :, 1:].astype(np.float32)

        base = _stable_name(img_path, input_dir)
        np.save(l_dir / f"{base}.npy", l_channel)
        np.save(ab_dir / f"{base}.npy", ab_channel)

    return len(image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic RGB to LAB preprocessing")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with source RGB images")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_lab"), help="Output directory")
    parser.add_argument("--img-size", type=int, default=256, help="Resize dimension (square)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = preprocess_images_to_lab(args.input_dir, args.output_dir, args.img_size)
    print(f"Processed {count} images")
    print(f"Output: {args.output_dir / 'L'}")
    print(f"Output: {args.output_dir / 'AB'}")


if __name__ == "__main__":
    main()
