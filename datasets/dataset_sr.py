from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class SRDataset(Dataset):
    """
    Super-resolution dataset.

    Supported layouts:
    1) root/HR + root/LR paired by stem
    2) root with HR images only (LR generated on-the-fly)
    """

    def __init__(self, root_dir: str, crop_size: int = 256, scale_factor: int = 4, min_samples: int = 101):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.lr_size = crop_size // scale_factor

        if not self.root_dir.exists() or not self.root_dir.is_dir():
            raise FileNotFoundError(f"SR dataset directory not found: {self.root_dir}")

        hr_dir = self.root_dir / "HR"
        lr_dir = self.root_dir / "LR"

        self.paired = False
        self.samples: List[Tuple[Path, Optional[Path]]] = []

        if hr_dir.is_dir() and lr_dir.is_dir():
            hr_map = self._collect_by_stem(hr_dir)
            lr_map = self._collect_by_stem(lr_dir)
            common = sorted(set(hr_map).intersection(lr_map))
            self.samples = [(hr_map[k], lr_map[k]) for k in common]
            self.paired = True
        else:
            hr_images = self._collect_images(self.root_dir)
            self.samples = [(p, None) for p in hr_images]

        if len(self.samples) == 0:
            raise RuntimeError(f"SR dataset is empty: {self.root_dir}")
        if len(self.samples) <= 100:
            raise RuntimeError(
                f"SR dataset too small ({len(self.samples)} samples). Something is wrong with dataset loading."
            )
        if len(self.samples) < min_samples:
            raise RuntimeError(
                f"SR dataset has {len(self.samples)} samples, expected at least {min_samples}."
            )

    @staticmethod
    def _collect_images(root: Path) -> List[Path]:
        images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        return sorted(images)

    @classmethod
    def _collect_by_stem(cls, root: Path) -> Dict[str, Path]:
        mapping: Dict[str, Path] = {}
        for p in cls._collect_images(root):
            mapping[p.stem] = p
        return mapping

    def __len__(self) -> int:
        return len(self.samples)

    def _random_crop(self, img: Image.Image, size: int) -> Image.Image:
        w, h = img.size
        if w < size or h < size:
            img = img.resize((max(w, size), max(h, size)), Image.Resampling.BICUBIC)
            w, h = img.size

        left = random.randint(0, w - size)
        top = random.randint(0, h - size)
        return img.crop((left, top, left + size, top + size))

    def __getitem__(self, idx: int):
        hr_path, lr_path = self.samples[idx]

        hr_img = Image.open(hr_path).convert("RGB")
        hr_img = self._random_crop(hr_img, self.crop_size)

        if self.paired and lr_path is not None:
            lr_img = Image.open(lr_path).convert("RGB")
            lr_img = lr_img.resize((self.lr_size, self.lr_size), Image.Resampling.BICUBIC)
        else:
            lr_img = hr_img.resize((self.lr_size, self.lr_size), Image.Resampling.BICUBIC)

        hr_tensor = TF.to_tensor(hr_img)
        lr_tensor = TF.to_tensor(lr_img)

        return lr_tensor, hr_tensor
