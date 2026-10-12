"""
Abstract base class for satellite data adapters.

All real-data adapters (Sentinel-2, Sentinel-1, Landsat) implement this
interface, enabling a unified data pipeline regardless of the source sensor.

The adapter lifecycle:
  1. search()       — query a STAC catalog for scenes matching a region + time range
  2. download()     — fetch selected scenes to a local cache
  3. load_patch()   — load a single preprocessed patch from cache
  4. get_time_series() — load a temporal stack of patches for one location
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from forestmind.utils.geo import BoundingBox

logger = logging.getLogger(__name__)
PREPROCESSING_VERSION = "forestmind-grid-v2"


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SceneMetadata:
    """Metadata for a single satellite scene / granule."""
    scene_id: str
    sensor: str                   # "sentinel2", "sentinel1", "landsat8", "landsat9"
    datetime: datetime
    bbox: BoundingBox
    cloud_cover: Optional[float] = None   # 0–100, None for SAR
    stac_item: Optional[dict] = None      # raw STAC item dict
    local_path: Optional[Path] = None     # path to cached data
    extra: dict = field(default_factory=dict)

    @property
    def date_str(self) -> str:
        return self.datetime.strftime("%Y-%m-%d")


@dataclass
class PatchData:
    """A single preprocessed satellite observation patch."""
    data: np.ndarray              # (C, H, W) — channels × height × width
    timestamp: float              # day offset from reference date
    datetime: datetime
    sensor: str
    band_names: list[str]
    bbox: BoundingBox
    cloud_cover: Optional[float] = None
    quality_mask: Optional[np.ndarray] = None  # (H, W) boolean, True = valid


@dataclass
class TimeSeriesData:
    """A temporal stack of patches for one geographic location."""
    optical: np.ndarray           # (T, C_opt, H, W)
    sar: Optional[np.ndarray]     # (T_sar, C_sar, H, W) or None
    timestamps: np.ndarray        # (T,) day offsets
    sar_timestamps: Optional[np.ndarray] = None
    bbox: BoundingBox = None
    sensor_sources: list[str] = field(default_factory=list)
    quality_masks: Optional[np.ndarray] = None  # (T, H, W)


# ─── Abstract Base ────────────────────────────────────────────────────────────

class SatelliteAdapterBase(ABC):
    """Abstract interface for satellite data adapters.

    Each adapter wraps a specific satellite sensor and data source (typically
    accessed via STAC API from Planetary Computer). The adapter handles:
      - Scene discovery (STAC search)
      - Data download and caching
      - Preprocessing (cloud masking, band extraction, resampling)
      - Loading preprocessed patches

    Args:
        cache_dir: Local directory for caching downloaded data.
        patch_size: Target spatial size in pixels (e.g. 128 for 128×128).
        resolution: Target spatial resolution in meters (e.g. 10 for 10m).
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/real",
        patch_size: int = 128,
        resolution: float = 10.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.patch_size = patch_size
        self.resolution = resolution
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ─── Abstract Methods ─────────────────────────────────────────────────

    @property
    @abstractmethod
    def sensor_name(self) -> str:
        """Human-readable sensor name (e.g. 'sentinel2')."""
        ...

    @property
    @abstractmethod
    def band_names(self) -> list[str]:
        """List of output band names after preprocessing."""
        ...

    @property
    @abstractmethod
    def num_bands(self) -> int:
        """Number of output bands."""
        ...

    @abstractmethod
    def search(
        self,
        bbox: BoundingBox,
        start_date: str | datetime,
        end_date: str | datetime,
        max_cloud_cover: float = 30.0,
        max_results: int = 100,
    ) -> list[SceneMetadata]:
        """Search for available scenes in a region and time range.

        Args:
            bbox: Geographic bounding box (WGS84).
            start_date: Start of time range (ISO format string or datetime).
            end_date: End of time range.
            max_cloud_cover: Maximum cloud cover percentage (ignored for SAR).
            max_results: Maximum number of results to return.

        Returns:
            List of SceneMetadata, sorted by datetime ascending.
        """
        ...

    @abstractmethod
    def download(
        self,
        scenes: list[SceneMetadata],
        bbox: BoundingBox,
    ) -> list[Path]:
        """Download and preprocess scenes to local cache.

        Args:
            scenes: List of scenes to download (from search()).
            bbox: Bounding box to crop to.

        Returns:
            List of paths to cached preprocessed files.
        """
        ...

    @abstractmethod
    def load_patch(self, path: Path) -> PatchData:
        """Load a single preprocessed patch from the cache.

        Args:
            path: Path to the cached .npz file.

        Returns:
            PatchData with the loaded observation.
        """
        ...

    # ─── Concrete Methods ─────────────────────────────────────────────────

    def get_time_series(
        self,
        bbox: BoundingBox,
        start_date: str | datetime,
        end_date: str | datetime,
        max_cloud_cover: float = 30.0,
        max_scenes: int = 48,
    ) -> list[PatchData]:
        """Full pipeline: search → download → load a temporal stack.

        Args:
            bbox: Geographic bounding box.
            start_date: Start date.
            end_date: End date.
            max_cloud_cover: Maximum cloud cover.
            max_scenes: Maximum number of scenes in the time series.

        Returns:
            List of PatchData ordered by timestamp.
        """
        logger.info(
            f"[{self.sensor_name}] Fetching time series: "
            f"{start_date} → {end_date}, cloud ≤ {max_cloud_cover}%"
        )

        # Search
        scenes = self.search(bbox, start_date, end_date, max_cloud_cover, max_scenes)
        if not scenes:
            logger.warning(f"[{self.sensor_name}] No scenes found for query.")
            return []

        logger.info(f"[{self.sensor_name}] Found {len(scenes)} scenes.")

        # Download (skips already-cached)
        paths = self.download(scenes, bbox)

        # Load
        patches = []
        for path in paths:
            try:
                patch = self.load_patch(path)
                patches.append(patch)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        # Sort by timestamp
        patches.sort(key=lambda p: p.datetime)

        logger.info(
            f"[{self.sensor_name}] Loaded {len(patches)} patches "
            f"({patches[0].datetime.date().isoformat() if patches else '?'} → "
            f"{patches[-1].datetime.date().isoformat() if patches else '?'})"
        )
        return patches

    def cache_path_for_scene(self, scene: SceneMetadata, bbox: BoundingBox) -> Path:
        """Generate a deterministic cache path for a scene + bbox pair."""
        bbox_hash = hashlib.md5(
            f"{bbox.west:.6f}_{bbox.south:.6f}_{bbox.east:.6f}_{bbox.north:.6f}"
            f"_{self.patch_size}_{PREPROCESSING_VERSION}".encode()
        ).hexdigest()[:8]
        filename = f"{scene.scene_id}_{bbox_hash}.npz"
        sensor_dir = self.cache_dir / self.sensor_name
        sensor_dir.mkdir(parents=True, exist_ok=True)
        return sensor_dir / filename

    def is_cached(self, scene: SceneMetadata, bbox: BoundingBox) -> bool:
        """Check if a scene is already in the local cache."""
        return self.cache_path_for_scene(scene, bbox).exists()

    def save_patch_to_cache(
        self,
        path: Path,
        data: np.ndarray,
        scene: SceneMetadata,
        bbox: BoundingBox,
        quality_mask: np.ndarray | None = None,
        preprocessing_version: str | None = None,
        data_units: str = "unknown",
    ) -> None:
        """Save preprocessed data to the cache."""
        save_dict = {
            "data": data,
            "timestamp": scene.datetime.timestamp(),
            "datetime_iso": scene.datetime.isoformat(),
            "sensor": scene.sensor,
            "band_names": self.band_names,
            "bbox_west": bbox.west,
            "bbox_south": bbox.south,
            "bbox_east": bbox.east,
            "bbox_north": bbox.north,
            "cloud_cover": scene.cloud_cover if scene.cloud_cover is not None else -1.0,
            "scene_id": scene.scene_id,
            "preprocessing_version": preprocessing_version or "unverified",
            "provider": "microsoft-planetary-computer" if preprocessing_version else "unknown",
            "data_units": data_units,
            "collection": (scene.stac_item or {}).get("collection", "unrecorded"),
        }
        if preprocessing_version == PREPROCESSING_VERSION:
            dx = (bbox.east - bbox.west) / data.shape[-1]
            dy = (bbox.north - bbox.south) / data.shape[-2]
            save_dict.update({
                "grid_crs": "EPSG:4326",
                "grid_transform": np.asarray([dx, 0., bbox.west, 0., -dy, bbox.north]),
                "pixel_size_degrees": np.asarray([dx, dy]),
                "requested_source_resolution_m": self.resolution,
            })
        if quality_mask is not None:
            save_dict["quality_mask"] = quality_mask

        np.savez_compressed(path, **save_dict)

    def load_patch_from_cache(self, path: Path) -> PatchData:
        """Load a PatchData from a cached .npz file."""
        data = np.load(path, allow_pickle=False)
        cloud_cover = float(data["cloud_cover"])
        if cloud_cover < 0:
            cloud_cover = None

        quality_mask = data["quality_mask"] if "quality_mask" in data else None

        return PatchData(
            data=data["data"],
            timestamp=0.0,  # Will be recomputed relative to reference date
            datetime=datetime.fromisoformat(str(data["datetime_iso"])),
            sensor=str(data["sensor"]),
            band_names=list(data["band_names"]),
            bbox=BoundingBox(
                west=float(data["bbox_west"]),
                south=float(data["bbox_south"]),
                east=float(data["bbox_east"]),
                north=float(data["bbox_north"]),
            ),
            cloud_cover=cloud_cover,
            quality_mask=quality_mask,
        )
