"""
Unified dataset interface for ForestMind.

Supports multiple data sources:
  1. On-the-fly synthetic generation (source="synthetic")
  2. Pre-generated .npz files from disk (source=directory path)
  3. Real satellite data via adapters (source="sentinel2", "landsat", "multi")

Provides a standard PyTorch Dataset interface with consistent output
format regardless of the underlying data source.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from forestmind.data.synthetic import SyntheticForestGenerator
from forestmind.data.transforms import Compose, get_eval_transforms, get_train_transforms

logger = logging.getLogger(__name__)

# Sentinel source identifiers that trigger real-data adapters
REAL_DATA_SOURCES = {"sentinel2", "sentinel1", "landsat", "multi"}


class ForestMindDataset(Dataset):
    """PyTorch Dataset for ForestMind training and evaluation.

    Modes:
      1. On-the-fly synthetic generation (source="synthetic")
      2. Load from pre-generated .npz files (source=path/to/dir)
      3. Real satellite data (source="sentinel2" | "landsat" | "multi")

    Args:
        source: Data source identifier or path to directory of .npz files.
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

        elif source in REAL_DATA_SOURCES:
            # Real satellite data mode — load from adapter cache
            self._init_real_data(source, config)

        elif Path(source).is_dir():
            # Load from disk
            self.npz_files = sorted(Path(source).glob("*.npz"))
            if not self.npz_files:
                raise FileNotFoundError(f"No .npz files found in {source}")
            self.samples = None  # lazy loading
        else:
            raise ValueError(f"Unknown data source: {source}. Use 'synthetic', a directory path, or {REAL_DATA_SOURCES}.")

    def _init_real_data(self, source: str, config: dict | None) -> None:
        """Initialize real satellite data from adapter cache directories.

        Scans the cache directory for preprocessed .npz files that were
        previously downloaded via the satellite adapters or download_data.py.

        Args:
            source: Adapter name ("sentinel2", "landsat", "multi").
            config: Configuration dict with cache_dir, etc.
        """
        config = config or {}
        cache_dir = Path(config.get("cache_dir", "data/real"))

        if source == "multi":
            # Multi-source: combine all available sensors
            sensor_dirs = ["sentinel2", "sentinel1", "landsat"]
        elif source == "sentinel2":
            sensor_dirs = ["sentinel2"]
        elif source == "sentinel1":
            sensor_dirs = ["sentinel1"]
        elif source == "landsat":
            sensor_dirs = ["landsat"]
        else:
            sensor_dirs = [source]

        self.npz_files = []
        for sensor in sensor_dirs:
            sensor_path = cache_dir / sensor
            if sensor_path.is_dir():
                self.npz_files.extend(sorted(sensor_path.glob("*.npz")))

        if not self.npz_files:
            logger.warning(
                f"No cached real data found for source='{source}' in {cache_dir}. "
                f"Run `python scripts/download_data.py` first, or falling back to empty dataset."
            )

        self.samples = None
        logger.info(f"[dataset] Real data source '{source}': {len(self.npz_files)} cached patches")

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
            sample = self._load_npz(self.npz_files[idx])

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _load_npz(self, path: Path) -> dict:
        """Load a sample from a .npz file.

        Handles both synthetic-format .npz files (with all labels) and
        real-data .npz files (which may lack labels and need synthetic fill).
        """
        data = np.load(path, allow_pickle=True)

        # Check if this is a real-data adapter cache file (has 'data' key)
        if "data" in data and "optical" not in data:
            return self._load_real_data_npz(data, path)

        # Standard synthetic / pre-generated format
        return {
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

    def _load_real_data_npz(self, data: np.lib.npyio.NpzFile, path: Path) -> dict:
        """Load a real-data adapter cache file and format for the model.

        Real-data .npz files from adapters contain raw sensor patches
        but lack labels (health state, drivers, etc.). We provide
        placeholder labels so the data can flow through the pipeline
        for inference (not training).

        For training on real data, labels should be provided via external
        annotation files (GFC, GLAD, MapBiomas).
        """
        patch = data["data"]  # (C, H, W) for single timestep
        sensor = str(data.get("sensor", "unknown"))

        # Determine if this is optical or SAR
        config = self.config
        num_bands = config.get("num_bands", 12)
        sar_channels = config.get("sar_channels", 2)
        time_steps = config.get("time_steps", 12)
        patch_size = patch.shape[-1] if patch.ndim >= 2 else config.get("patch_size", 128)

        if sensor in ("sentinel1",):
            # SAR-only: create placeholder optical
            optical = np.zeros((time_steps, num_bands, patch_size, patch_size), dtype=np.float32)
            sar = np.broadcast_to(
                patch[np.newaxis],  # (1, 2, H, W)
                (time_steps, sar_channels, patch_size, patch_size),
            ).copy()
        else:
            # Optical data: may need harmonization for Landsat
            if sensor in ("landsat", "landsat8", "landsat9") and patch.shape[0] == 7:
                from forestmind.data.adapters.harmonize import BandHarmonizer
                harmonizer = BandHarmonizer()
                patch = harmonizer.harmonize_landsat(patch)

            # Ensure we have 12 bands
            if patch.shape[0] < num_bands:
                # Pad with zeros
                pad = np.zeros((num_bands - patch.shape[0], patch_size, patch_size), dtype=np.float32)
                patch = np.concatenate([patch, pad], axis=0)
            elif patch.shape[0] > num_bands:
                patch = patch[:num_bands]

            optical = np.broadcast_to(
                patch[np.newaxis],
                (time_steps, num_bands, patch_size, patch_size),
            ).copy()
            sar = np.zeros((time_steps, sar_channels, patch_size, patch_size), dtype=np.float32)

        # Generate timestamps (monthly spacing as placeholder)
        timestamps = np.arange(time_steps, dtype=np.float32) * 30.0

        # Placeholder labels for inference mode
        health_history = np.linspace(0.5, 0.5, time_steps, dtype=np.float32)

        return {
            "optical": optical,
            "sar": sar,
            "timestamps": timestamps,
            "health_state": 0,  # placeholder — Healthy
            "health_score": 50.0,  # placeholder
            "drivers": np.zeros(8, dtype=np.float32),
            "driver_strengths": np.zeros(8, dtype=np.float32),
            "trajectory": np.full(5, 0.5, dtype=np.float32),
            "health_history": health_history,
        }


class RealDataTimeSeriesDataset(Dataset):
    """Dataset that loads real multi-temporal satellite data for a study region.

    Unlike ForestMindDataset (which loads individual cached patches),
    this dataset constructs proper temporal stacks by combining multiple
    observations from the same geographic location across time.

    Args:
        optical_dir: Path to cached optical data directory.
        sar_dir: Optional path to cached SAR data directory.
        time_steps: Number of time steps per sample.
        patch_size: Spatial patch size.
        transform: Optional transform pipeline.
        harmonize: Whether to harmonize Landsat to S2-equivalent.
        label_file: Optional path to label annotations (GeoJSON / CSV).
    """

    def __init__(
        self,
        optical_dir: str | Path,
        sar_dir: str | Path | None = None,
        time_steps: int = 12,
        patch_size: int = 128,
        transform: Optional[Callable] = None,
        harmonize: bool = True,
        label_file: Optional[str | Path] = None,
    ):
        self.optical_dir = Path(optical_dir)
        self.sar_dir = Path(sar_dir) if sar_dir else None
        self.time_steps = time_steps
        self.patch_size = patch_size
        self.transform = transform
        self.harmonize = harmonize

        # Discover optical files
        self.optical_files = sorted(self.optical_dir.glob("*.npz"))
        if not self.optical_files:
            raise FileNotFoundError(f"No .npz files found in {optical_dir}")

        # Group by location (bbox hash in filename)
        self.location_groups = self._group_by_location(self.optical_files)

        # SAR files (if available)
        self.sar_files = {}
        if self.sar_dir and self.sar_dir.is_dir():
            sar_npzs = sorted(self.sar_dir.glob("*.npz"))
            self.sar_files = self._group_by_location(sar_npzs)

        # Load labels if provided
        self.labels = {}
        if label_file:
            self.labels = self._load_labels(label_file)

        logger.info(
            f"[dataset] RealDataTimeSeries: {len(self.location_groups)} locations, "
            f"{len(self.optical_files)} optical, {sum(len(v) for v in self.sar_files.values())} SAR"
        )

    def _group_by_location(self, files: list[Path]) -> dict[str, list[Path]]:
        """Group cache files by bbox hash (location identifier)."""
        groups = {}
        for f in files:
            # Filename format: {scene_id}_{bbox_hash}.npz
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2:
                bbox_hash = parts[1]
            else:
                bbox_hash = "unknown"

            if bbox_hash not in groups:
                groups[bbox_hash] = []
            groups[bbox_hash].append(f)

        # Sort each group by filename (which includes datetime)
        for key in groups:
            groups[key].sort()

        return groups

    def _load_labels(self, label_file: str | Path) -> dict:
        """Load label annotations from file. Placeholder for future integration."""
        logger.info(f"Label loading from {label_file} — not yet implemented")
        return {}

    def __len__(self) -> int:
        return len(self.location_groups)

    def __getitem__(self, idx: int) -> dict:
        location_key = list(self.location_groups.keys())[idx]
        optical_paths = self.location_groups[location_key]

        # Load optical patches
        optical_patches = []
        timestamps = []
        for path in optical_paths[:self.time_steps]:
            npz = np.load(path, allow_pickle=True)
            patch = npz["data"].astype(np.float32)

            # Harmonize if needed
            sensor = str(npz.get("sensor", "sentinel2"))
            if self.harmonize and sensor in ("landsat", "landsat8", "landsat9"):
                from forestmind.data.adapters.harmonize import BandHarmonizer
                harmonizer = BandHarmonizer()
                patch = harmonizer.harmonize_landsat(patch)

            # Ensure 12 bands
            if patch.shape[0] < 12:
                pad = np.zeros((12 - patch.shape[0], *patch.shape[1:]), dtype=np.float32)
                patch = np.concatenate([patch, pad], axis=0)

            optical_patches.append(patch[:12])
            timestamps.append(float(npz["timestamp"]))

        # Pad if fewer than time_steps
        while len(optical_patches) < self.time_steps:
            optical_patches.append(optical_patches[-1].copy())
            timestamps.append(timestamps[-1] + 30.0 * 86400)

        # Stack: (T, 12, H, W)
        optical = np.stack(optical_patches[:self.time_steps], axis=0)

        # Convert timestamps to day offsets from first observation
        ts_array = np.array(timestamps[:self.time_steps], dtype=np.float64)
        ts_days = ((ts_array - ts_array[0]) / 86400.0).astype(np.float32)

        # Load SAR if available
        sar_patches = []
        if location_key in self.sar_files:
            for path in self.sar_files[location_key][:self.time_steps]:
                npz = np.load(path, allow_pickle=True)
                sar_patches.append(npz["data"].astype(np.float32))

        # Pad / create placeholder SAR
        if sar_patches:
            while len(sar_patches) < self.time_steps:
                sar_patches.append(sar_patches[-1].copy())
            sar = np.stack(sar_patches[:self.time_steps], axis=0)
        else:
            sar = np.zeros((self.time_steps, 2, self.patch_size, self.patch_size), dtype=np.float32)

        # Placeholder labels (for inference)
        health_history = np.linspace(0.5, 0.5, self.time_steps, dtype=np.float32)

        sample = {
            "optical": optical,
            "sar": sar,
            "timestamps": ts_days,
            "health_state": 0,
            "health_score": 50.0,
            "drivers": np.zeros(8, dtype=np.float32),
            "driver_strengths": np.zeros(8, dtype=np.float32),
            "trajectory": np.full(5, 0.5, dtype=np.float32),
            "health_history": health_history,
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
    source_p = Path(source)
    if source_p.is_dir() and (source_p / "train").is_dir():
        train_source = str(source_p / "train")
        val_source = str(source_p / "val") if (source_p / "val").is_dir() else str(source_p)
        test_source = str(source_p / "test") if (source_p / "test").is_dir() else str(source_p)
    else:
        train_source = source
        val_source = source
        test_source = source

    train_dataset = ForestMindDataset(
        source=train_source,
        num_samples=data_cfg.get("num_train_samples", 2000),
        transform=train_transform,
        config=data_cfg,
        seed=config.get("seed", 42),
    )

    val_dataset = ForestMindDataset(
        source=val_source,
        num_samples=data_cfg.get("num_val_samples", 400),
        transform=eval_transform,
        config=data_cfg,
        seed=config.get("seed", 42) + 10000,  # different seed for val
    )

    test_dataset = ForestMindDataset(
        source=test_source,
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
