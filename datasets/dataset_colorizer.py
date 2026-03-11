from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class ColorizationDataset(Dataset):
    """
    Loads preprocessed LAB tensors from:
      <root_dir>/L/*.npy
      <root_dir>/AB/*.npy
    """

    def __init__(self, root_dir: str = "datasets/flickr2k", augment: bool = True, min_samples: int = 101):
        self.root_dir = Path(root_dir)
        self.l_dir = self.root_dir / "L"
        self.ab_dir = self.root_dir / "AB"
        self.augment = augment

        if not self.l_dir.is_dir() or not self.ab_dir.is_dir():
            raise FileNotFoundError(
                "Colorization dataset not found. Expected directories:\n"
                f"  - {self.l_dir}\n"
                f"  - {self.ab_dir}\n"
                "Run preprocessing first, e.g. `python datasets/preprocess_lab.py --output-dir datasets/flickr2k`."
            )

        l_files = {p.stem: p for p in self.l_dir.glob("*.npy")}
        ab_files = {p.stem: p for p in self.ab_dir.glob("*.npy")}

        common_keys = sorted(set(l_files).intersection(ab_files))
        if not common_keys:
            raise RuntimeError(
                f"No matched .npy pairs found between {self.l_dir} and {self.ab_dir}."
            )

        missing_l = sorted(set(ab_files).difference(l_files))
        missing_ab = sorted(set(l_files).difference(ab_files))
        if missing_l or missing_ab:
            print(
                "Warning: unmatched pairs detected in dataset. "
                f"missing L: {len(missing_l)}, missing AB: {len(missing_ab)}. Using matched subset only."
            )

        self.samples: List[Tuple[Path, Path]] = [(l_files[k], ab_files[k]) for k in common_keys]
        if len(self.samples) <= 100:
            raise RuntimeError(
                f"Colorization dataset too small ({len(self.samples)} samples). Something is wrong with dataset loading."
            )
        if len(self.samples) < min_samples:
            raise RuntimeError(
                f"Colorization dataset has {len(self.samples)} samples, expected at least {min_samples}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        l_path, ab_path = self.samples[idx]
        l_arr = np.load(l_path)
        ab_arr = np.load(ab_path)

        if l_arr.ndim != 2:
            raise ValueError(f"Expected L channel shape (H, W), got {l_arr.shape} at {l_path}")
        if ab_arr.ndim != 3 or ab_arr.shape[2] != 2:
            raise ValueError(f"Expected AB shape (H, W, 2), got {ab_arr.shape} at {ab_path}")

        l = torch.from_numpy(l_arr).unsqueeze(0).float() / 100.0
        ab = torch.from_numpy(ab_arr).permute(2, 0, 1).float() / 128.0

        if self.augment and torch.rand(1).item() > 0.5:
            l = torch.flip(l, dims=[2])
            ab = torch.flip(ab, dims=[2])

        return l, ab


class ImageColorizationDataset(Dataset):
    """
    Loads RGB images and converts to LAB on-the-fly.
    Expected layout:
      <root_dir>/**/*.{png,jpg,jpeg,bmp,webp}
    """

    def __init__(self, root_dir: str, image_size: int = 256, augment: bool = True, min_samples: int = 101):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augment = augment

        if not self.root_dir.exists() or not self.root_dir.is_dir():
            raise FileNotFoundError(f"Colorization image dataset not found: {self.root_dir}")

        self.images: List[Path] = sorted(
            p for p in self.root_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(self.images) == 0:
            raise RuntimeError(f"Colorization image dataset is empty: {self.root_dir}")
        if len(self.images) < min_samples:
            raise RuntimeError(
                f"Colorization image dataset has {len(self.images)} samples, expected at least {min_samples}."
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        rgb = Image.open(img_path).convert("RGB")
        rgb = rgb.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)

        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
        lab = rgb2lab(rgb_np).astype(np.float32)

        l = torch.from_numpy(lab[:, :, 0]).unsqueeze(0) / 100.0
        ab = torch.from_numpy(lab[:, :, 1:]).permute(2, 0, 1) / 128.0

        if self.augment and torch.rand(1).item() > 0.5:
            l = torch.flip(l, dims=[2])
            ab = torch.flip(ab, dims=[2])

        return l, ab


class CombinedDataset(Dataset):
    """Concatenates multiple datasets while preserving deterministic indexing."""

    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = list(datasets)
        if not self.datasets:
            raise RuntimeError("CombinedDataset requires at least one dataset.")

        self.cumulative_sizes: List[int] = []
        running = 0
        for ds in self.datasets:
            ds_len = len(ds)
            if ds_len <= 0:
                raise RuntimeError("CombinedDataset received an empty dataset.")
            running += ds_len
            self.cumulative_sizes.append(running)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"CombinedDataset index out of range: {idx}")

        ds_idx = bisect_right(self.cumulative_sizes, idx)
        prev_cum = 0 if ds_idx == 0 else self.cumulative_sizes[ds_idx - 1]
        sample_idx = idx - prev_cum
        return self.datasets[ds_idx][sample_idx]


def build_combined_colorization_dataset(
    data_roots: Sequence[str],
    augment: bool = True,
    image_size: int = 256,
    min_total_samples: int = 50001,
) -> Tuple[CombinedDataset, Dict[str, int]]:
    """
    Build a Stage1 dataset from multiple roots.
    Uses preprocessed LAB pairs when <root>/L and <root>/AB exist, otherwise RGB image fallback.
    """

    if not data_roots:
        raise RuntimeError("No dataset roots provided for Stage1 training.")

    datasets: List[Dataset] = []
    stats: Dict[str, int] = {}

    for root in data_roots:
        root_path = Path(root)
        if not root_path.exists() or not root_path.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {root_path}")

        if (root_path / "L").is_dir() and (root_path / "AB").is_dir():
            ds = ColorizationDataset(root_dir=str(root_path), augment=augment, min_samples=1)
        else:
            ds = ImageColorizationDataset(
                root_dir=str(root_path),
                image_size=image_size,
                augment=augment,
                min_samples=1,
            )

        datasets.append(ds)
        stats[str(root_path)] = len(ds)

    combined = CombinedDataset(datasets)
    total = len(combined)

    if total < 1000:
        raise RuntimeError(f"Combined Stage1 dataset too small ({total} samples). Minimum required: 1000.")
    if total < min_total_samples:
        raise RuntimeError(
            f"Combined Stage1 dataset has {total} samples. Expected at least {min_total_samples} samples."
        )

    return combined, stats
