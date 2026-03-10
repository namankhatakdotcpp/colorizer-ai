from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ColorizationDataset(Dataset):
    """
    Loads preprocessed LAB tensors from:
      <root_dir>/L/*.npy
      <root_dir>/AB/*.npy
    """

    def __init__(self, root_dir: str = "datasets/flickr2k", augment: bool = True):
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
