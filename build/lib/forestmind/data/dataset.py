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
        self.seed = seed
        self.num_samples = num_samples

        if source == "synthetic":
            if num_samples < 1:
                raise ValueError("num_samples must be positive")
            # Per-index seeds preserve repeated observations without retaining
            # tens of GB of image arrays for the default dataset in RAM.
            self.samples = None

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
        if self.source == "synthetic":
            return self.num_samples
        if self.samples is not None:
            return len(self.samples)
        return len(self.npz_files)

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        if self.source == "synthetic":
            generator = SyntheticForestGenerator(
                patch_size=self.config.get("patch_size", 128),
                num_bands=self.config.get("num_bands", 12),
                sar_channels=self.config.get("sar_channels", 2),
                time_steps=self.config.get("time_steps", 12),
                seed=np.random.SeedSequence([self.seed, idx]),
            )
            sample = generator.generate_sample(horizons_days=self.config.get("horizon_days"))
        elif self.samples is not None:
            # Return a copy to avoid in-place transform issues
            sample = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.samples[idx].items()}
        else:
            # Load from disk
            sample = self._load_npz(self.npz_files[idx])

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _load_npz(self, path: Path) -> dict:
        """Load numeric data only. Unlabeled acquisitions remain unlabeled."""
        with np.load(path, allow_pickle=False) as data:
            if "data" in data and "optical" not in data:
                patch, sensor, timestamp, quality = _read_observation(data)
                return _assemble_observations([(patch, sensor, timestamp, quality)],
                                              self.config.get("time_steps", 12))
            sample = {key: data[key].copy() for key in data.files}
        for key, cast in (("health_state", int), ("health_score", float)):
            if key in sample:
                sample[key] = cast(sample[key])
        return sample


def _read_observation(data):
    patch = np.asarray(data["data"], dtype=np.float32)
    sensor = str(data["sensor"])
    timestamp = float(data["timestamp"])
    if patch.ndim != 3 or not np.isfinite(timestamp):
        raise ValueError("Cached observations need (C, H, W) data and a finite UTC timestamp")
    quality = np.asarray(data.get("quality_mask", np.ones(patch.shape[-2:], dtype=bool)), dtype=bool)
    if quality.shape != patch.shape[-2:]:
        raise ValueError("Cached quality mask must match the observation grid")
    if sensor in ("landsat", "landsat8", "landsat9"):
        from forestmind.data.adapters.harmonize import BandHarmonizer
        patch = BandHarmonizer(interpolate_red_edge=False).harmonize_landsat(patch)
    expected = 2 if sensor == "sentinel1" else 12
    if patch.shape[0] != expected:
        raise ValueError(f"{sensor} requires {expected} bands, got {patch.shape[0]}")
    quality = quality & np.isfinite(patch).all(axis=0)
    patch = np.where(quality[None], patch, 0).astype(np.float32)
    return patch, sensor, timestamp, quality


def _assemble_observations(observations, time_steps, sar_observations=None,
                           sar_max_age_days=12.0):
    """Pad with masked zeros and match SAR only from the same date or past."""
    if time_steps < 1 or not observations:
        raise ValueError("At least one observation and positive time_steps are required")
    observations = sorted(observations, key=lambda item: item[2])[-time_steps:]
    height, width = observations[0][0].shape[-2:]
    optical = np.zeros((time_steps, 12, height, width), dtype=np.float32)
    sar = np.zeros((time_steps, 2, height, width), dtype=np.float32)
    optical_mask = np.zeros(time_steps, dtype=bool)
    sar_mask = np.zeros(time_steps, dtype=bool)
    quality = np.zeros((time_steps, height, width), dtype=bool)
    timestamps = np.full(time_steps, observations[-1][2], dtype=np.float64)
    for index, (patch, sensor, timestamp, valid) in enumerate(observations):
        if patch.shape[-2:] != (height, width):
            raise ValueError("All observations must share a spatial grid")
        timestamps[index] = timestamp
        if sensor == "sentinel1":
            sar[index] = patch
            sar_mask[index] = valid.any()
        else:
            optical[index] = patch
            optical_mask[index] = valid.any()
            quality[index] = valid
            # Indexing SAR by list position incorrectly pairs unrelated dates.
            candidates = [item for item in (sar_observations or [])
                          if 0 <= timestamp - item[2] <= sar_max_age_days * 86400]
            if candidates:
                match = max(candidates, key=lambda item: item[2])
                if match[0].shape != (2, height, width):
                    raise ValueError("SAR and optical grids must match before temporal alignment")
                sar[index] = match[0]
                sar_mask[index] = match[3].any()
    return {
        "optical": optical, "sar": sar,
        "timestamps": ((timestamps - timestamps[0]) / 86400).astype(np.float32),
        "observation_mask": optical_mask | sar_mask,
        "optical_mask": optical_mask, "sar_mask": sar_mask,
        "quality_mask": quality, "label_valid": False,
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
        """Group by actual footprint, then order by acquisition time."""
        groups = {}
        timestamps = {}
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                bbox_keys = ["bbox_west", "bbox_south", "bbox_east", "bbox_north"]
                if all(key in data for key in bbox_keys):
                    location = ",".join(f"{float(data[key]):.6f}" for key in bbox_keys)
                else:
                    parts = path.stem.rsplit("_", 1)
                    if len(parts) != 2:
                        raise ValueError(f"Cache {path} has no geographic location metadata")
                    location = parts[-1]
                timestamps[path] = float(data["timestamp"])
            groups.setdefault(location, []).append(path)
        for paths in groups.values():
            paths.sort(key=lambda path: timestamps[path])
        return groups

    def _load_labels(self, label_file: str | Path) -> dict:
        raise NotImplementedError(
            "External annotation ingestion is not implemented. Supply labeled sequence NPZ "
            "files in explicit train/val/test directories for supervised learning."
        )

    def __len__(self) -> int:
        return len(self.location_groups)

    def __getitem__(self, idx: int) -> dict:
        location = list(self.location_groups)[idx]
        def read(paths):
            observations = []
            for path in paths:
                with np.load(path, allow_pickle=False) as data:
                    observations.append(_read_observation(data))
            return observations
        sample = _assemble_observations(
            read(self.location_groups[location]), self.time_steps,
            read(self.sar_files.get(location, [])),
        )
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
    data_cfg = dict(config.get("data", config))
    horizons = config.get("heads", {}).get("trajectory", {}).get("horizons")
    if horizons is not None:
        data_cfg["horizon_days"] = list(horizons)

    normalization = data_cfg.get("normalization")
    foundation_cfg = config.get("foundation", {})
    foundation_name = foundation_cfg.get("name", "custom_vit")
    pretrained = foundation_name in ("prithvi", "terramind") and not foundation_cfg.get(
        foundation_name, {}).get("use_proxy", False)
    if pretrained:
        required_stats = {"optical_mean", "optical_std", "sar_mean", "sar_std"}
        if not isinstance(normalization, dict) or not required_stats.issubset(normalization):
            raise ValueError(
                "Pretrained encoders require explicit data.normalization optical_mean/std and "
                "sar_mean/std matching the checkpoint AND input units. Synthetic default "
                "statistics are not compatible with pretrained models."
            )
    train_transform = get_train_transforms(normalization=normalization)
    eval_transform = get_eval_transforms(normalization=normalization)

    source = data_cfg.get("source", "synthetic")
    source_p = Path(source)
    if source == "synthetic":
        train_source = val_source = test_source = source
    elif source in REAL_DATA_SOURCES:
        raise ValueError(
            "Raw satellite caches have no supervised targets. Prepare labeled sequence NPZ "
            "files in geographically independent train/val/test directories first."
        )
    else:
        split_paths = [source_p / split for split in ("train", "val", "test")]
        if not all(path.is_dir() for path in split_paths):
            raise ValueError(
                "Disk datasets require explicit train/val/test directories; reusing the same "
                "files across splits would leak evaluation data. Split by geography and time."
            )
        train_source, val_source, test_source = map(str, split_paths)
        resolved = [path.resolve() for path in split_paths]
        if len(set(resolved)) != 3:
            raise ValueError("Dataset split directories must refer to distinct locations")
        # When locations are provided, enforce geographic holdouts as well as
        # separate directories. Metadata-free datasets need an external split audit.
        owners = {}
        for split, path in zip(("train", "val", "test"), split_paths):
            for file in path.glob("*.npz"):
                with np.load(file, allow_pickle=False) as record:
                    targets = {"health_state", "health_score", "drivers", "driver_strengths", "trajectory"}
                    if not targets.issubset(record.files):
                        raise ValueError(f"{file} lacks supervised targets; raw caches cannot train a model")
                    if "label_valid" in record and not bool(record["label_valid"]):
                        raise ValueError(f"{file} is explicitly unlabeled")
                    location = None
                    if "location_id" in record:
                        location = str(record["location_id"])
                    elif all(key in record for key in ("bbox_west", "bbox_south", "bbox_east", "bbox_north")):
                        location = tuple(float(record[key]) for key in
                                         ("bbox_west", "bbox_south", "bbox_east", "bbox_north"))
                    if location is not None:
                        if location in owners and owners[location] != split:
                            raise ValueError(f"Geographic leakage: location {location} appears in multiple splits")
                        owners[location] = split

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
        drop_last=False,
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
