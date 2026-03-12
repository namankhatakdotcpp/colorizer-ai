import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, Dataset
import torchvision.transforms as T

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _resolution_bucket(width: int, height: int) -> str:
    short_side = min(width, height)
    if short_side <= 256:
        return "0-256"
    if short_side <= 512:
        return "257-512"
    if short_side <= 1024:
        return "513-1024"
    return "1025+"


class ColorizationDataset(Dataset):
    """
    Loads preprocessed LAB tensors from:
      <root_dir>/L/*.npy
      <root_dir>/AB/*.npy
    """

    def __init__(
        self,
        root_dir: str = "datasets/flickr2k",
        augment: bool = True,
        min_samples: int = 101,
        image_size: int = 256,
    ):
        self.root_dir = Path(root_dir)
        self.l_dir = self.root_dir / "L"
        self.ab_dir = self.root_dir / "AB"
        self.augment = augment
        self.image_size = image_size
        self.stats: Dict[str, Any] = {}

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

        # Lightweight dataset quality statistics for startup diagnostics.
        color_vars: List[float] = []
        res_counter: Counter[str] = Counter()
        sum_w = 0.0
        sum_h = 0.0
        counted = 0
        sample_count = min(256, len(self.samples))
        for l_path, ab_path in self.samples[:sample_count]:
            try:
                l_arr = np.load(l_path)
                ab_arr = np.load(ab_path)
            except Exception:
                continue

            if l_arr.ndim == 2:
                h, w = l_arr.shape
                res_counter[_resolution_bucket(w, h)] += 1
                sum_w += float(w)
                sum_h += float(h)
                counted += 1
            if ab_arr.ndim == 3 and ab_arr.shape[2] == 2:
                color_vars.append(float(np.var(ab_arr)))

        self.stats = {
            "size": len(self.samples),
            "color_variance": float(np.mean(color_vars)) if color_vars else 0.0,
            "average_resolution": (
                float(sum_w / counted) if counted > 0 else 0.0,
                float(sum_h / counted) if counted > 0 else 0.0,
            ),
            "resolution_distribution": dict(res_counter),
            "rejected_corrupt": 0,
            "rejected_grayscale": 0,
        }

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

        if self.augment:
            # Random horizontal flip.
            if torch.rand(1).item() > 0.5:
                l = torch.flip(l, dims=[2])
                ab = torch.flip(ab, dims=[2])

            # Random crop + resize back to target resolution (safe for fixed-shape pipelines).
            h, w = l.shape[-2], l.shape[-1]
            min_hw = min(h, w)
            crop_hw = int(min_hw * random.uniform(0.85, 1.0))
            crop_hw = max(32, min(crop_hw, min_hw))
            if crop_hw < min_hw:
                top = random.randint(0, h - crop_hw)
                left = random.randint(0, w - crop_hw)
                l = l[:, top:top + crop_hw, left:left + crop_hw]
                ab = ab[:, top:top + crop_hw, left:left + crop_hw]

                l = F.interpolate(
                    l.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                ab = F.interpolate(
                    ab.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            # Random Gaussian noise on L channel (small, stable).
            noise = torch.randn_like(l) * 0.01
            l = torch.clamp(l + noise, 0.0, 1.0)

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
        self.stats: Dict[str, Any] = {}
        self.aug_transform = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomResizedCrop(
                    size=(self.image_size, self.image_size),
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1),
                ),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            ]
        )

        if not self.root_dir.exists() or not self.root_dir.is_dir():
            raise FileNotFoundError(f"Colorization image dataset not found: {self.root_dir}")

        image_candidates: List[Path] = sorted(
            p for p in self.root_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        valid_images: List[Path] = []
        color_vars: List[float] = []
        res_counter: Counter[str] = Counter()
        sum_w = 0.0
        sum_h = 0.0
        counted = 0
        rejected_corrupt = 0
        rejected_grayscale = 0
        max_stats_samples = 512

        for path in image_candidates:
            try:
                # Fast corruption check.
                with Image.open(path) as probe:
                    probe.verify()

                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                    w, h = rgb.size
                    res_counter[_resolution_bucket(w, h)] += 1
                    sum_w += float(w)
                    sum_h += float(h)
                    counted += 1

                    # Reject near-grayscale images using per-channel differences.
                    thumb = rgb.resize((64, 64), Image.Resampling.BILINEAR)
                    arr = np.asarray(thumb, dtype=np.float32) / 255.0
                    rg_diff = np.mean(np.abs(arr[:, :, 0] - arr[:, :, 1]))
                    gb_diff = np.mean(np.abs(arr[:, :, 1] - arr[:, :, 2]))
                    if rg_diff < 0.01 and gb_diff < 0.01:
                        rejected_grayscale += 1
                        continue

                    if len(color_vars) < max_stats_samples:
                        color_vars.append(float(np.var(arr[:, :, 1:3])))

                valid_images.append(path)
            except Exception:
                rejected_corrupt += 1
                continue

        self.images = valid_images

        if len(self.images) == 0:
            raise RuntimeError(f"Colorization image dataset is empty: {self.root_dir}")
        if len(self.images) < min_samples:
            raise RuntimeError(
                f"Colorization image dataset has {len(self.images)} samples, expected at least {min_samples}."
            )

        self.stats = {
            "size": len(self.images),
            "color_variance": float(np.mean(color_vars)) if color_vars else 0.0,
            "average_resolution": (
                float(sum_w / counted) if counted > 0 else 0.0,
                float(sum_h / counted) if counted > 0 else 0.0,
            ),
            "resolution_distribution": dict(res_counter),
            "rejected_corrupt": rejected_corrupt,
            "rejected_grayscale": rejected_grayscale,
        }

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        rgb = Image.open(img_path).convert("RGB")
        rgb = rgb.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)

        if self.augment:
            rgb = self.aug_transform(rgb)

        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
        lab = rgb2lab(rgb_np).astype(np.float32)

        l = torch.from_numpy(lab[:, :, 0]).unsqueeze(0) / 100.0
        ab = torch.from_numpy(lab[:, :, 1:]).permute(2, 0, 1) / 128.0

        if self.augment:
            # Random Gaussian noise on luminance for robustness.
            noise = torch.randn_like(l) * 0.01
            l = torch.clamp(l + noise, 0.0, 1.0)

        return l, ab


def build_combined_colorization_dataset(
    data_roots: Sequence[str],
    augment: bool = True,
    image_size: int = 256,
    warn_min_total_samples: int = 50000,
) -> Tuple[ConcatDataset, Dict[str, Dict[str, Any]]]:
    """
    Build a Stage1 dataset from multiple roots.
    Uses preprocessed LAB pairs when <root>/L and <root>/AB exist, otherwise RGB image fallback.
    """

    if not data_roots:
        raise RuntimeError("No dataset roots provided for Stage1 training.")

    datasets: List[Dataset] = []
    stats: Dict[str, Dict[str, Any]] = {}

    for root in data_roots:
        root_path = Path(root)
        if not root_path.exists() or not root_path.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {root_path}")

        if (root_path / "L").is_dir() and (root_path / "AB").is_dir():
            ds = ColorizationDataset(
                root_dir=str(root_path),
                augment=augment,
                min_samples=1,
                image_size=image_size,
            )
        else:
            ds = ImageColorizationDataset(
                root_dir=str(root_path),
                image_size=image_size,
                augment=augment,
                min_samples=1,
            )

        datasets.append(ds)
        ds_stats = getattr(ds, "stats", None)
        if not ds_stats:
            ds_stats = {
                "size": len(ds),
                "color_variance": 0.0,
                "average_resolution": (0.0, 0.0),
                "resolution_distribution": {},
                "rejected_corrupt": 0,
                "rejected_grayscale": 0,
            }
        stats[str(root_path)] = ds_stats

    combined = ConcatDataset(datasets)
    total = len(combined)

    if total < 1000:
        raise RuntimeError(f"Combined Stage1 dataset too small ({total} samples). Minimum required: 1000.")
    if total < warn_min_total_samples:
        print(
            f"[WARN] Combined Stage1 dataset has {total} samples. "
            f"Recommended minimum for color diversity is {warn_min_total_samples}."
        )

    return combined, stats
