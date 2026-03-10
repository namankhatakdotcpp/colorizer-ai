import os
import torch
import numpy as np
from torch.utils.data import Dataset


class ColorizationDataset(Dataset):

    def __init__(self, root_dir):
        self.l_dir = os.path.join(root_dir, "L")
        self.ab_dir = os.path.join(root_dir, "AB")

        self.l_files = sorted(os.listdir(self.l_dir))
        self.ab_files = sorted(os.listdir(self.ab_dir))

    def __len__(self):
        return len(self.l_files)

    def __getitem__(self, idx):

        L = np.load(os.path.join(self.l_dir, self.l_files[idx]))
        AB = np.load(os.path.join(self.ab_dir, self.ab_files[idx]))

        L = torch.from_numpy(L).unsqueeze(0).float() / 100.0
        AB = torch.from_numpy(AB).permute(2, 0, 1).float() / 128.0

        # Data augmentation
        if torch.rand(1) > 0.5:
            L = torch.flip(L, [2])
            AB = torch.flip(AB, [2])

        return L, AB