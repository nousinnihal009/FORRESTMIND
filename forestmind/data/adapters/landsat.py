"""
Landsat-8/9 Collection 2 Level-2 data adapter for ForestMind.

Downloads and preprocesses Landsat surface reflectance data from
the Microsoft Planetary Computer STAC API.

Landsat provides:
  - 7 surface reflectance bands (30m native for most, 100m thermal)
  - ~16-day revisit per satellite (8 days combined Landsat-8 + 9)
  - Continuous archive since 1972 (Landsat-1), L8 since 2013, L9 since 2021
  - QA_PIXEL for cloud/shadow masking

Output format: (7, H, W) float32 surface reflectance in [0, 1]
Bands are harmonized to Sentinel-2 equivalent via the harmonize module.

Requires:
  pip install pystac-client planetary-computer rasterio
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from forestmind.data.adapters.base import (
    BoundingBox,
    PatchData,
    SatelliteAdapterBase,
    SceneMetadata,
    PREPROCESSING_VERSION,
)
from forestmind.data.adapters.raster import read_aligned_band
from forestmind.data.adapters.network import AssetSigner, is_retryable_error, open_catalog, open_raster, public_error

logger = logging.getLogger(__name__)

# ─── Landsat Band Configuration ──────────────────────────────────────────────

# 7 surface reflectance bands from Landsat-8/9 OLI
LANDSAT_BANDS = [
    "B1",   # Coastal Aerosol, 435nm, 30m
    "B2",   # Blue, 482nm, 30m
    "B3",   # Green, 562nm, 30m
    "B4",   # Red, 655nm, 30m
    "B5",   # NIR, 865nm, 30m
    "B6",   # SWIR 1, 1609nm, 30m
    "B7",   # SWIR 2, 2201nm, 30m
]

# STAC asset keys for Planetary Computer Landsat C2 L2
LANDSAT_ASSET_KEYS = {
    "B1": "coastal",
    "B2": "blue",
    "B3": "green",
    "B4": "red",
    "B5": "nir08",
    "B6": "swir16",
    "B7": "swir22",
    "QA": "qa_pixel",
}

# QA_PIXEL bit flags for cloud/shadow masking (Landsat Collection 2)
# Bit 1: Dilated Cloud, Bit 3: Cloud, Bit 4: Cloud Shadow
QA_CLOUD_BITS = sum(1 << bit for bit in (0, 1, 2, 3, 4, 5))

# Landsat C2 L2 scale and offset
# Surface reflectance = DN * 0.0000275 - 0.2
LANDSAT_SCALE = 0.0000275
LANDSAT_OFFSET = -0.2

# Planetary Computer STAC
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
LANDSAT_COLLECTIONS = ["landsat-c2-l2"]


class LandsatAdapter(SatelliteAdapterBase):
    """Adapter for Landsat-8/9 Collection 2 Level-2 data.

    Downloads surface reflectance data, applies QA_PIXEL cloud masking,
    and converts to [0, 1] reflectance. Outputs 7-band patches that can
    be harmonized to 12-band Sentinel-2 equivalent via BandHarmonizer.

    Args:
        cache_dir: Local directory for caching downloaded data.
        patch_size: Target spatial size in pixels.
        resolution: Nominal source resolution; final grid follows bbox / patch_size.
        platforms: List of Landsat platforms to include ("landsat-8", "landsat-9").
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/real",
        patch_size: int = 128,
        resolution: float = 30.0,
        platforms: Optional[list[str]] = None,
        timeout_seconds: float = 30.0,
    ):
        super().__init__(cache_dir=cache_dir, patch_size=patch_size, resolution=resolution,
                         timeout_seconds=timeout_seconds)
        self.platforms = platforms or ["landsat-8", "landsat-9"]

    @property
    def sensor_name(self) -> str:
        return "landsat"

    @property
    def band_names(self) -> list[str]:
        return LANDSAT_BANDS

    @property
    def num_bands(self) -> int:
        return len(LANDSAT_BANDS)

    def search(
        self,
        bbox: BoundingBox,
        start_date: str | datetime,
        end_date: str | datetime,
        max_cloud_cover: float = 30.0,
        max_results: int = 100,
    ) -> list[SceneMetadata]:
        """Search Planetary Computer for Landsat scenes.

        Args:
            bbox: Geographic bounding box (WGS84).
            start_date: Start of time range.
            end_date: End of time range.
            max_cloud_cover: Maximum cloud cover percentage (0–100).
            max_results: Maximum number of results.

        Returns:
            List of SceneMetadata sorted by datetime ascending.
        """
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%d")

        catalog = open_catalog(PC_STAC_URL, self.timeout_seconds)

        # Build query
        query = {
            "eo:cloud_cover": {"lt": max_cloud_cover},
        }
        if self.platforms:
            query["platform"] = {"in": self.platforms}

        search = catalog.search(
            collections=LANDSAT_COLLECTIONS,
            bbox=[bbox.west, bbox.south, bbox.east, bbox.north],
            datetime=f"{start_date}/{end_date}",
            query=query,
            max_items=max_results,
            limit=min(max_results, 100),
            sortby=[{"field": "datetime", "direction": "asc"}],
        )

        scenes = []
        for item in search.items():
            # Verify required assets exist
            required_assets = list(LANDSAT_ASSET_KEYS.values())
            if not all(a in item.assets for a in required_assets):
                continue

            scene_bbox = BoundingBox(
                west=item.bbox[0],
                south=item.bbox[1],
                east=item.bbox[2],
                north=item.bbox[3],
            )

            platform = item.properties.get("platform", "unknown")
            sensor_id = "landsat9" if "9" in platform else "landsat8"

            scenes.append(SceneMetadata(
                scene_id=item.id,
                sensor=sensor_id,
                datetime=item.datetime,
                bbox=scene_bbox,
                cloud_cover=item.properties.get("eo:cloud_cover"),
                stac_item=item.to_dict(),
                extra={
                    "platform": platform,
                    "path_row": f"{item.properties.get('landsat:wrs_path', '?')}_"
                                f"{item.properties.get('landsat:wrs_row', '?')}",
                },
            ))

        logger.info(f"[landsat] Found {len(scenes)} scenes (cloud ≤ {max_cloud_cover}%)")
        return scenes

    def download(
        self,
        scenes: list[SceneMetadata],
        bbox: BoundingBox,
    ) -> list[Path]:
        """Download and preprocess Landsat scenes.

        For each scene:
          1. Read each surface reflectance band (windowed to bbox)
          2. Apply scale/offset to get physical reflectance
          3. Apply QA_PIXEL cloud mask
          4. Resize to target patch_size
          5. Cache as .npz

        Args:
            scenes: Scenes to download.
            bbox: Bounding box to crop to.

        Returns:
            List of cache file paths.
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError(
                "Landsat adapter requires 'rasterio'. "
                "Install with: pip install rasterio"
            )

        try:
            import pystac
        except ImportError:
            raise ImportError(
                "Landsat adapter requires 'planetary-computer'. "
                "Install with: pip install planetary-computer"
            )

        paths = []
        self.download_failures = []
        signer = AssetSigner(self.timeout_seconds)

        for scene in scenes:
            cache_path = self.cache_path_for_scene(scene, bbox)

            if self.is_cached(scene, bbox):
                logger.debug(f"[landsat] Cached: {scene.scene_id}")
                paths.append(cache_path)
                continue

            logger.info(f"[landsat] Downloading: {scene.scene_id} ({scene.date_str})")

            try:
                item = pystac.Item.from_dict(scene.stac_item)

                band_arrays = []
                quality_mask = np.ones((self.patch_size, self.patch_size), dtype=bool)
                for band_name in LANDSAT_BANDS:
                    asset_key = LANDSAT_ASSET_KEYS[band_name]
                    href = signer.sign(item.assets[asset_key].href)

                    with open_raster(href, self.timeout_seconds) as src:
                        band_data, valid = read_aligned_band(src, bbox, self.patch_size)
                        quality_mask &= valid

                        # Apply Collection 2 scale and offset
                        band_data = band_data * LANDSAT_SCALE + LANDSAT_OFFSET
                        band_data = np.clip(band_data, 0.0, 1.0)

                        band_arrays.append(band_data)

                # Stack bands: (7, H, W)
                patch = np.stack(band_arrays, axis=0)

                # Read QA_PIXEL for cloud masking
                qa_key = LANDSAT_ASSET_KEYS["QA"]
                if qa_key in item.assets:
                    qa_href = signer.sign(item.assets[qa_key].href)
                    with open_raster(qa_href, self.timeout_seconds) as src:
                        qa, qa_valid = read_aligned_band(src, bbox, self.patch_size, categorical=True)
                        qa = qa.astype(np.uint16)
                        # True = valid pixel (no cloud/shadow bits set)
                        quality_mask &= qa_valid & ((qa & QA_CLOUD_BITS) == 0)

                else:
                    raise ValueError("Landsat scene lacks QA_PIXEL quality data")

                if not quality_mask.any():
                    raise ValueError("Scene has no valid pixels in the requested region")
                patch[:, ~quality_mask] = 0

                # Save to cache
                self.save_patch_to_cache(cache_path, patch, scene, bbox, quality_mask,
                                         preprocessing_version=PREPROCESSING_VERSION,
                                         data_units="surface_reflectance_0_1",
                                         processing_metadata={"scale": LANDSAT_SCALE, "offset": LANDSAT_OFFSET,
                                                              "clip_range": [0.0, 1.0], "invalid_qa_bits": list(range(6)),
                                                              "resampling": "bilinear; nearest for QA_PIXEL"})
                paths.append(cache_path)

            except Exception as e:
                reason = public_error(e)
                self.download_failures.append({"scene_id": scene.scene_id, "reason": reason,
                                               "retryable": is_retryable_error(e),
                                               "error_type": type(e).__name__})
                logger.error(f"[landsat] Failed to download {scene.scene_id}: {reason}")
                continue

        return paths

    def load_patch(self, path: Path) -> PatchData:
        """Load a preprocessed Landsat patch from cache."""
        return self.load_patch_from_cache(path)
