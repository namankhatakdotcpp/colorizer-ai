import torch
from torch.utils.data import Dataset

class DepthDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        # Placeholder logic
        self.length = 100

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Image, Depth map
        return torch.rand(3, 256, 256), torch.rand(1, 256, 256)
