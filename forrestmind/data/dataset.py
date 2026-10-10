"""
Unified dataset interface for ForestMind.

Supports both on-the-fly synthetic generation and loading pre-generated
.npz files from disk. Provides a standard PyTorch Dataset interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from forestmind.data.synthetic import SyntheticForestGenerator
from forestmind.data.transforms import Compose, get_eval_transforms, get_train_transforms


class ForestMindDataset(Dataset):
    """PyTorch Dataset for ForestMind training and evaluation.
    
    Modes:
      1. On-the-fly synthetic generation (source="synthetic")
      2. Load from pre-generated .npz files (source="disk")
      
    Args:
        source: "synthetic" for on-the-fly generation, or path to directory of .npz files.
        num_samples: Number of samples (required for synthetic mode).
        transform: Optional transform pipeline.
        config: Data configuration dict.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        source: str = "synthetic",
        num_samples: int = 1000,
        transform: Optional[Callable] = None,
        config: Optional[dict] = None,
        seed: int = 42,
    ):
        self.source = source
        self.transform = transform
        self.config = config or {}

        if source == "synthetic":
            self.generator = SyntheticForestGenerator(
                patch_size=self.config.get("patch_size", 128),
                num_bands=self.config.get("num_bands", 12),
                sar_channels=self.config.get("sar_channels", 2),
                time_steps=self.config.get("time_steps", 12),
                seed=seed,
            )
            # Pre-generate all samples for consistent epoch-to-epoch behavior
            self.samples = self.generator.generate_dataset(num_samples)

        elif Path(source).is_dir():
            # Load from disk
            self.npz_files = sorted(Path(source).glob("*.npz"))
            if not self.npz_files:
                raise FileNotFoundError(f"No .npz files found in {source}")
            self.samples = None  # lazy loading
        else:
            raise ValueError(f"Unknown data source: {source}. Use 'synthetic' or a directory path.")

    def __len__(self) -> int:
        if self.samples is not None:
            return len(self.samples)
        return len(self.npz_files)

    def __getitem__(self, idx: int) -> dict:
        if self.samples is not None:
            # Return a copy to avoid in-place transform issues
            sample = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.samples[idx].items()}
        else:
            # Load from disk
            data = np.load(self.npz_files[idx])
            sample = {
                "optical": data["optical"],
                "sar": data["sar"],
                "timestamps": data["timestamps"],
                "health_state": int(data["health_state"]),
                "health_score": float(data["health_score"]),
                "drivers": data["drivers"],
                "driver_strengths": data["driver_strengths"],
                "trajectory": data["trajectory"],
                "health_history": data["health_history"],
            }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def create_dataloaders(
    config: dict,
    batch_size: int = 4,
    num_workers: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val/test dataloaders from config.
    
    Args:
        config: Data configuration dictionary.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_cfg = config.get("data", config)

    train_transform = get_train_transforms()
    eval_transform = get_eval_transforms()

    source = data_cfg.get("source", "synthetic")

    train_dataset = ForestMindDataset(
        source=source,
        num_samples=data_cfg.get("num_train_samples", 2000),
        transform=train_transform,
        config=data_cfg,
        seed=config.get("seed", 42),
    )

    val_dataset = ForestMindDataset(
        source=source,
        num_samples=data_cfg.get("num_val_samples", 400),
        transform=eval_transform,
        config=data_cfg,
        seed=config.get("seed", 42) + 10000,  # different seed for val
    )

    test_dataset = ForestMindDataset(
        source=source,
        num_samples=data_cfg.get("num_test_samples", 200),
        transform=eval_transform,
        config=data_cfg,
        seed=config.get("seed", 42) + 20000,  # different seed for test
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
