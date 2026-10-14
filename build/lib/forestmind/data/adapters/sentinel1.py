"""
Sentinel-1 GRD IW SAR data adapter for ForestMind.

Downloads and preprocesses Sentinel-1 Ground Range Detected (GRD)
Interferometric Wide (IW) swath data from Microsoft Planetary Computer.

Sentinel-1 provides:
  - VV and VH polarization backscatter
  - ~12-day revisit time per satellite (6 days with both S1A + S1B)
  - All-weather, day/night imaging (active microwave)
  - Sensitive to forest structure, biomass, and moisture

Output format: (2, H, W) float32 normalized backscatter in [0, 1]

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

# ─── Sentinel-1 Configuration ────────────────────────────────────────────────

S1_BANDS = ["VV", "VH"]

# Planetary Computer STAC
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S1_COLLECTION = "sentinel-1-grd"

# Backscatter normalization range (dB)
# Typical forest backscatter: VV ≈ -6 to -12 dB, VH ≈ -12 to -20 dB
S1_DB_MIN = -25.0
S1_DB_MAX = 0.0


class Sentinel1Adapter(SatelliteAdapterBase):
    """Adapter for Sentinel-1 GRD IW data from Planetary Computer.

    Downloads VV and VH polarization backscatter data, converts from
    linear power to dB scale, normalizes to [0, 1], and caches locally.

    Args:
        cache_dir: Local directory for caching downloaded data.
        patch_size: Target spatial size in pixels.
        resolution: Nominal source resolution; final grid follows bbox / patch_size.
        orbit_direction: Filter by orbit direction ("ascending", "descending", or None).
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/real",
        patch_size: int = 128,
        resolution: float = 10.0,
        orbit_direction: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ):
        super().__init__(cache_dir=cache_dir, patch_size=patch_size, resolution=resolution,
                         timeout_seconds=timeout_seconds)
        self.orbit_direction = orbit_direction

    @property
    def sensor_name(self) -> str:
        return "sentinel1"

    @property
    def band_names(self) -> list[str]:
        return S1_BANDS

    @property
    def num_bands(self) -> int:
        return len(S1_BANDS)

    def search(
        self,
        bbox: BoundingBox,
        start_date: str | datetime,
        end_date: str | datetime,
        max_cloud_cover: float = 100.0,  # ignored for SAR
        max_results: int = 100,
    ) -> list[SceneMetadata]:
        """Search Planetary Computer for Sentinel-1 GRD scenes.

        Args:
            bbox: Geographic bounding box (WGS84).
            start_date: Start of time range.
            end_date: End of time range.
            max_cloud_cover: Ignored for SAR sensor.
            max_results: Maximum number of results.

        Returns:
            List of SceneMetadata sorted by datetime ascending.
        """
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%d")
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%d")

        catalog = open_catalog(PC_STAC_URL, self.timeout_seconds)

        # Build query filters
        query = {
            "sar:instrument_mode": {"eq": "IW"},
        }
        if self.orbit_direction:
            query["sat:orbit_state"] = {"eq": self.orbit_direction}

        search = catalog.search(
            collections=[S1_COLLECTION],
            bbox=[bbox.west, bbox.south, bbox.east, bbox.north],
            datetime=f"{start_date}/{end_date}",
            query=query,
            max_items=max_results,
            limit=min(max_results, 100),
            sortby=[{"field": "datetime", "direction": "asc"}],
        )

        scenes = []
        for item in search.items():
            # Verify VV and VH assets exist
            if "vv" not in item.assets or "vh" not in item.assets:
                continue

            scene_bbox = BoundingBox(
                west=item.bbox[0],
                south=item.bbox[1],
                east=item.bbox[2],
                north=item.bbox[3],
            )
            scenes.append(SceneMetadata(
                scene_id=item.id,
                sensor="sentinel1",
                datetime=item.datetime,
                bbox=scene_bbox,
                cloud_cover=None,  # SAR is cloud-independent
                stac_item=item.to_dict(),
                extra={
                    "orbit_direction": item.properties.get("sat:orbit_state"),
                    "relative_orbit": item.properties.get("sat:relative_orbit"),
                },
            ))

        logger.info(f"[sentinel1] Found {len(scenes)} scenes")
        return scenes

    def download(
        self,
        scenes: list[SceneMetadata],
        bbox: BoundingBox,
    ) -> list[Path]:
        """Download and preprocess Sentinel-1 scenes.

        For each scene:
          1. Read VV and VH bands as windowed reads (cropped to bbox)
          2. Convert from linear power to dB: 10 * log10(power)
          3. Normalize dB to [0, 1] using min/max scaling
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
                "Sentinel-1 adapter requires 'rasterio'. "
                "Install with: pip install rasterio"
            )

        try:
            import pystac
        except ImportError:
            raise ImportError(
                "Sentinel-1 adapter requires 'planetary-computer'. "
                "Install with: pip install planetary-computer"
            )

        paths = []
        self.download_failures = []
        signer = AssetSigner(self.timeout_seconds)

        for scene in scenes:
            cache_path = self.cache_path_for_scene(scene, bbox)

            if self.is_cached(scene, bbox):
                logger.debug(f"[sentinel1] Cached: {scene.scene_id}")
                paths.append(cache_path)
                continue

            logger.info(f"[sentinel1] Downloading: {scene.scene_id} ({scene.date_str})")

            try:
                item = pystac.Item.from_dict(scene.stac_item)

                band_arrays = []
                quality_mask = np.ones((self.patch_size, self.patch_size), dtype=bool)
                for pol in ["vv", "vh"]:
                    href = signer.sign(item.assets[pol].href)

                    with open_raster(href, self.timeout_seconds) as src:
                        band_data, valid = read_aligned_band(src, bbox, self.patch_size)
                        quality_mask &= valid

                        quality_mask &= band_data > 0
                        # Convert linear power to dB
                        band_data = np.where(
                            band_data > 0,
                            10.0 * np.log10(band_data + 1e-10),
                            S1_DB_MIN,
                        )

                        # Normalize to [0, 1]
                        band_data = (band_data - S1_DB_MIN) / (S1_DB_MAX - S1_DB_MIN)
                        band_data = np.clip(band_data, 0.0, 1.0)

                        band_arrays.append(band_data)

                # Stack: (2, H, W)
                patch = np.stack(band_arrays, axis=0)

                if not quality_mask.any():
                    raise ValueError("Scene has no valid pixels in the requested region")
                patch[:, ~quality_mask] = 0

                # Save to cache
                self.save_patch_to_cache(cache_path, patch, scene, bbox, quality_mask,
                                         preprocessing_version=PREPROCESSING_VERSION,
                                         data_units="backscatter_db_scaled_minus25_0",
                                         processing_metadata={"db_range": [S1_DB_MIN, S1_DB_MAX],
                                                              "resampling": "bilinear"})
                paths.append(cache_path)

            except Exception as e:
                reason = public_error(e)
                self.download_failures.append({"scene_id": scene.scene_id, "reason": reason})
                logger.error(f"[sentinel1] Failed to download {scene.scene_id}: {reason}")
                continue

        return paths

    def load_patch(self, path: Path) -> PatchData:
        """Load a preprocessed Sentinel-1 patch from cache."""
        return self.load_patch_from_cache(path)


def compute_vv_vh_ratio(patch: np.ndarray) -> np.ndarray:
    """Compute VV/VH ratio (cross-pol ratio) from SAR data.

    Higher ratios indicate volume scattering (forest canopy).
    Lower ratios indicate surface scattering (bare ground, water).

    Args:
        patch: (2, H, W) normalized SAR data [VV, VH].

    Returns:
        Ratio array of shape (H, W).
    """
    vv = patch[0]
    vh = patch[1]
    ratio = np.where(vh > 0.01, vv / (vh + 1e-8), 0.0)
    return ratio.astype(np.float32)


def compute_rfdi(patch: np.ndarray) -> np.ndarray:
    """Compute Radar Forest Degradation Index (RFDI).

    RFDI = (VV - VH) / (VV + VH)

    Values near 0 indicate dense forest, values near 1 indicate degradation.

    Args:
        patch: (2, H, W) normalized SAR data [VV, VH].

    Returns:
        RFDI array of shape (H, W), values in [-1, 1].
    """
    vv = patch[0]
    vh = patch[1]
    denominator = vv + vh
    rfdi = np.where(denominator > 0.01, (vv - vh) / (denominator + 1e-8), 0.0)
    return rfdi.astype(np.float32)
