import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from .dataset import ColorizationDataset

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_dataloaders(config_path: str) -> tuple[DataLoader, DataLoader]:
    """
    Factory function to initialize optimized Multiprocessing DataLoaders.
    Parses configuration to bind datasets to parallel workers.
    """
    config = load_config(config_path)
    
    data_cfg = config.get("dataset", {})
    loader_cfg = config.get("dataloader", {})
    
    root_dir = data_cfg.get("root_dir", "/data/imagenet")
    image_size = data_cfg.get("image_size", 256)
    
    batch_size = loader_cfg.get("batch_size", 32)
    num_workers = loader_cfg.get("num_workers", 4)
    pin_memory = loader_cfg.get("pin_memory", True)
    prefetch_factor = loader_cfg.get("prefetch_factor", 2)
    
    # 1. Instantiate the Datasets
    train_dataset = ColorizationDataset(
        root_dir=root_dir,
        split="train",
        image_size=image_size
    )
    
    val_dataset = ColorizationDataset(
        root_dir=root_dir,
        split="val",
        image_size=image_size
    )
    
    # 2. Bind into highly parallelized DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Shuffle strictly only on Train
        num_workers=num_workers,
        pin_memory=pin_memory, # Accelerates CPU RAM to GPU VRAM pushes
        prefetch_factor=prefetch_factor,
        drop_last=True         # Retains uniform tensor shapes, critical for BatchNorm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # Deterministic validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        drop_last=False
    )
    
    return train_loader, val_loader
