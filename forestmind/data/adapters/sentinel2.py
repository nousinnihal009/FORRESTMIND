"""
Sentinel-2 Level-2A data adapter for ForestMind.

Downloads and preprocesses Sentinel-2 L2A surface reflectance data
from the Microsoft Planetary Computer STAC API.

Sentinel-2 provides:
  - 13 spectral bands (we use 12, excluding B10/cirrus)
  - 10m, 20m, 60m native resolutions (aligned to the requested AOI grid)
  - ~5-day revisit time (at equator, more frequent at higher latitudes)
  - Scene Classification Layer (SCL) for cloud masking

Output format: (12, H, W) float32 surface reflectance in [0, 1]

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
from forestmind.data.adapters.network import AssetSigner, open_catalog, open_raster, public_error

logger = logging.getLogger(__name__)

# ─── Sentinel-2 Band Configuration ───────────────────────────────────────────

# 12 bands we extract (B10/cirrus excluded)
S2_BANDS = [
    "B01",  # Coastal aerosol, 443nm, 60m
    "B02",  # Blue, 490nm, 10m
    "B03",  # Green, 560nm, 10m
    "B04",  # Red, 665nm, 10m
    "B05",  # Veg Red Edge 1, 705nm, 20m
    "B06",  # Veg Red Edge 2, 740nm, 20m
    "B07",  # Veg Red Edge 3, 783nm, 20m
    "B08",  # NIR, 842nm, 10m
    "B8A",  # Narrow NIR, 865nm, 20m
    "B09",  # Water Vapour, 945nm, 60m
    "B11",  # SWIR 1, 1610nm, 20m
    "B12",  # SWIR 2, 2190nm, 20m
]

# STAC asset keys for Planetary Computer Sentinel-2 L2A collection
S2_ASSET_KEYS = {
    "B01": "B01", "B02": "B02", "B03": "B03", "B04": "B04",
    "B05": "B05", "B06": "B06", "B07": "B07", "B08": "B08",
    "B8A": "B8A", "B09": "B09", "B11": "B11", "B12": "B12",
    "SCL": "SCL",  # Scene Classification Layer for cloud masking
}

# SCL values to mask as invalid (clouds, shadows, etc.)
SCL_INVALID = {0, 1, 3, 8, 9, 10, 11}
# 0=No data, 1=Saturated, 3=Cloud shadow, 8=Med cloud, 9=High cloud,
# 10=Thin cirrus, 11=Snow/ice

# Sentinel-2 reflectance scale factor (L2A stores as uint16 × 10000)
S2_SCALE_FACTOR = 10000.0

# Planetary Computer STAC catalog URL
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"


class Sentinel2Adapter(SatelliteAdapterBase):
    """Adapter for Sentinel-2 Level-2A data from Planetary Computer.

    Downloads, cloud-masks, and preprocesses Sentinel-2 L2A surface
    reflectance data into ForestMind-compatible 12-band patches.

    Args:
        cache_dir: Local directory for caching downloaded data.
        patch_size: Target spatial size in pixels.
        resolution: Nominal source resolution; final grid follows bbox / patch_size.
        scale_factor: Reflectance scale factor for L2A products.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/real",
        patch_size: int = 128,
        resolution: float = 10.0,
        scale_factor: float = S2_SCALE_FACTOR,
        timeout_seconds: float = 30.0,
    ):
        super().__init__(cache_dir=cache_dir, patch_size=patch_size, resolution=resolution,
                         timeout_seconds=timeout_seconds)
        self.scale_factor = scale_factor

    @property
    def sensor_name(self) -> str:
        return "sentinel2"

    @property
    def band_names(self) -> list[str]:
        return S2_BANDS

    @property
    def num_bands(self) -> int:
        return len(S2_BANDS)

    def search(
        self,
        bbox: BoundingBox,
        start_date: str | datetime,
        end_date: str | datetime,
        max_cloud_cover: float = 30.0,
        max_results: int = 100,
    ) -> list[SceneMetadata]:
        """Search Planetary Computer for Sentinel-2 L2A scenes.

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

        search = catalog.search(
            collections=[S2_COLLECTION],
            bbox=[bbox.west, bbox.south, bbox.east, bbox.north],
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=max_results,
            limit=min(max_results, 100),
            sortby=[{"field": "datetime", "direction": "asc"}],
        )

        scenes = []
        for item in search.items():
            scene_bbox = BoundingBox(
                west=item.bbox[0],
                south=item.bbox[1],
                east=item.bbox[2],
                north=item.bbox[3],
            )
            scenes.append(SceneMetadata(
                scene_id=item.id,
                sensor="sentinel2",
                datetime=item.datetime,
                bbox=scene_bbox,
                cloud_cover=item.properties.get("eo:cloud_cover"),
                stac_item=item.to_dict(),
            ))

        logger.info(f"[sentinel2] Found {len(scenes)} scenes (cloud ≤ {max_cloud_cover}%)")
        return scenes

    def download(
        self,
        scenes: list[SceneMetadata],
        bbox: BoundingBox,
    ) -> list[Path]:
        """Download and preprocess Sentinel-2 scenes.

        For each scene:
          1. Read each band as a windowed read (cropped to bbox)
          2. Reproject every band to one geographic AOI grid
          3. Apply SCL cloud mask
          4. Scale to [0, 1] surface reflectance
          5. Resize to target patch_size
          6. Cache as .npz

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
                "Sentinel-2 adapter requires 'rasterio'. "
                "Install with: pip install rasterio"
            )

        try:
            import pystac
        except ImportError:
            raise ImportError(
                "Sentinel-2 adapter requires 'planetary-computer'. "
                "Install with: pip install planetary-computer"
            )

        paths = []
        self.download_failures = []
        signer = AssetSigner(self.timeout_seconds)

        for scene in scenes:
            cache_path = self.cache_path_for_scene(scene, bbox)

            if self.is_cached(scene, bbox):
                logger.debug(f"[sentinel2] Cached: {scene.scene_id}")
                paths.append(cache_path)
                continue

            logger.info(f"[sentinel2] Downloading: {scene.scene_id} ({scene.date_str})")

            try:
                # Reconstruct STAC item for asset signing
                item = pystac.Item.from_dict(scene.stac_item)

                # Read and stack bands
                band_arrays = []
                scaling = {}
                quality_mask = np.ones((self.patch_size, self.patch_size), dtype=bool)
                for band_name in S2_BANDS:
                    asset_key = S2_ASSET_KEYS[band_name]
                    href = signer.sign(item.assets[asset_key].href)

                    with open_raster(href, self.timeout_seconds) as src:
                        band_data, valid = read_aligned_band(src, bbox, self.patch_size)
                        quality_mask &= valid

                        # Scale to [0, 1]
                        raster_meta = item.assets[asset_key].extra_fields.get("raster:bands", [{}])[0]
                        scale = raster_meta.get("scale", 1.0 / self.scale_factor)
                        offset = raster_meta.get("offset", 0.0)
                        scaling[band_name] = {"scale": scale, "offset": offset}
                        band_data = band_data * scale + offset
                        band_data = np.clip(band_data, 0.0, 1.0)

                        band_arrays.append(band_data)

                # Stack bands: (12, H, W)
                patch = np.stack(band_arrays, axis=0)

                # Read SCL for cloud masking
                if "SCL" in item.assets:
                    scl_href = signer.sign(item.assets["SCL"].href)
                    with open_raster(scl_href, self.timeout_seconds) as src:
                        scl, scl_valid = read_aligned_band(src, bbox, self.patch_size, categorical=True)
                        # True = valid pixel
                        quality_mask &= scl_valid & ~np.isin(scl, list(SCL_INVALID))

                else:
                    raise ValueError("Sentinel-2 scene lacks SCL quality data")

                if not quality_mask.any():
                    raise ValueError("Scene has no valid pixels in the requested region")
                patch[:, ~quality_mask] = 0

                # Save to cache
                self.save_patch_to_cache(cache_path, patch, scene, bbox, quality_mask,
                                         preprocessing_version=PREPROCESSING_VERSION,
                                         data_units="surface_reflectance_0_1",
                                         processing_metadata={
                                             "band_scaling": scaling,
                                             "processing_baseline": item.properties.get("s2:processing_baseline"),
                                             "clip_range": [0.0, 1.0],
                                             "invalid_scl_classes": sorted(SCL_INVALID),
                                             "resampling": "bilinear; nearest for SCL",
                                         })
                paths.append(cache_path)

            except Exception as e:
                reason = public_error(e)
                self.download_failures.append({"scene_id": scene.scene_id, "reason": reason})
                logger.error(f"[sentinel2] Failed to download {scene.scene_id}: {reason}")
                continue

        return paths

    def load_patch(self, path: Path) -> PatchData:
        """Load a preprocessed Sentinel-2 patch from cache."""
        return self.load_patch_from_cache(path)


def compute_ndvi(patch: np.ndarray) -> np.ndarray:
    """Compute NDVI from a Sentinel-2 12-band patch.

    NDVI = (B08 - B04) / (B08 + B04)

    Args:
        patch: (12, H, W) or (T, 12, H, W) surface reflectance.

    Returns:
        NDVI array of shape (H, W) or (T, H, W), values in [-1, 1].
    """
    if patch.ndim == 3:
        nir = patch[7]   # B08
        red = patch[3]   # B04
    elif patch.ndim == 4:
        nir = patch[:, 7]
        red = patch[:, 3]
    else:
        raise ValueError(f"Expected 3D or 4D input, got {patch.ndim}D")

    denominator = nir + red
    ndvi = np.where(denominator > 0, (nir - red) / (denominator + 1e-8), 0.0)
    return ndvi.astype(np.float32)


def compute_evi(patch: np.ndarray) -> np.ndarray:
    """Compute Enhanced Vegetation Index from Sentinel-2 data.

    EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)

    Args:
        patch: (12, H, W) surface reflectance.

    Returns:
        EVI array of shape (H, W).
    """
    nir = patch[7]    # B08
    red = patch[3]    # B04
    blue = patch[1]   # B02

    denominator = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = 2.5 * (nir - red) / (denominator + 1e-8)
    return np.clip(evi, -1.0, 1.0).astype(np.float32)
