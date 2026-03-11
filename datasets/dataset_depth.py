from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class DepthDataset(Dataset):
    """
    Depth dataset loader.

    Supported layouts:
    1) root/images + root/depth paired by stem
    2) root/train2017 + root/depth paired by stem
    3) root with images only (pseudo-depth target generated from luminance)
    """

    def __init__(self, root_dir: str, image_size: int = 256, min_samples: int = 101):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        if not self.root_dir.exists() or not self.root_dir.is_dir():
            raise FileNotFoundError(f"Depth dataset directory not found: {self.root_dir}")

        image_root = self._resolve_image_root(self.root_dir)
        depth_root = self._resolve_depth_root(self.root_dir)

        image_map = self._collect_by_stem(image_root)
        if not image_map:
            raise RuntimeError(f"No images found for depth training under: {image_root}")

        self.samples: List[Tuple[Path, Optional[Path]]] = []
        self.has_depth_labels = False

        if depth_root is not None:
            depth_map = self._collect_by_stem(depth_root)
            common = sorted(set(image_map).intersection(depth_map))
            if common:
                self.samples = [(image_map[k], depth_map[k]) for k in common]
                self.has_depth_labels = True

        if not self.samples:
            self.samples = [(image_map[k], None) for k in sorted(image_map)]

        if len(self.samples) == 0:
            raise RuntimeError(f"Depth dataset is empty: {self.root_dir}")
        if len(self.samples) <= 100:
            raise RuntimeError(
                f"Depth dataset too small ({len(self.samples)} samples). Something is wrong with dataset loading."
            )
        if len(self.samples) < min_samples:
            raise RuntimeError(
                f"Depth dataset has {len(self.samples)} samples, expected at least {min_samples}."
            )

    @staticmethod
    def _resolve_image_root(root: Path) -> Path:
        candidates = [root / "images", root / "train2017", root]
        for candidate in candidates:
            if candidate.is_dir():
                files = [p for p in candidate.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
                if files:
                    return candidate
        return root

    @staticmethod
    def _resolve_depth_root(root: Path) -> Optional[Path]:
        for name in ("depth", "Depth", "depth_maps"):
            candidate = root / name
            if candidate.is_dir():
                return candidate
        return None

    @staticmethod
    def _collect_by_stem(root: Path) -> Dict[str, Path]:
        mapping: Dict[str, Path] = {}
        for p in root.rglob("*"):
            if p.is_file() and (p.suffix.lower() in IMAGE_EXTENSIONS or p.suffix.lower() == ".npy"):
                mapping[p.stem] = p
        return mapping

    def __len__(self) -> int:
        return len(self.samples)

    def _load_depth_map(self, path: Path) -> torch.Tensor:
        if path.suffix.lower() == ".npy":
            arr = np.load(path).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[..., 0]
            depth = torch.from_numpy(arr).unsqueeze(0)
            if depth.max() > 0:
                depth = depth / depth.max()
            return depth

        depth_img = Image.open(path).convert("L").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return TF.to_tensor(depth_img)

    def __getitem__(self, idx: int):
        image_path, depth_path = self.samples[idx]

        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image_tensor = TF.to_tensor(image)

        if depth_path is not None:
            depth_tensor = self._load_depth_map(depth_path)
        else:
            # Pseudo-depth fallback derived from luminance; keeps pipeline trainable when GT depth is unavailable.
            depth_tensor = image_tensor.mean(dim=0, keepdim=True)

        return image_tensor, depth_tensor
