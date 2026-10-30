"""
Sentinel-2 Level-2A data adapter for ForestMind.

Downloads and preprocesses Sentinel-2 L2A surface reflectance data
from Microsoft Planetary Computer or Element 84 Earth Search.

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
import re
import hashlib
import json
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
from forestmind.data.adapters.network import (
    AssetSigner, check_acquisition, is_retryable_error, open_catalog, open_raster, public_error,
)

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

# Earth Search C1 asset names verified against its EO band metadata. B8A is
# narrow NIR (nir08); B09 is the reflectance band nir09, not the WVP product.
EARTH_SEARCH_ASSET_KEYS = {
    "B01": "coastal", "B02": "blue", "B03": "green", "B04": "red",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3", "B08": "nir",
    "B8A": "nir08", "B09": "nir09", "B11": "swir16", "B12": "swir22",
    "SCL": "scl",
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
EARTH_SEARCH_STAC_URL = "https://earth-search.aws.element84.com/v1"
EARTH_SEARCH_COLLECTION = "sentinel-2-c1-l2a"
# Keep only one scene's successful aligned reads across caller-budgeted retries.
# Large research grids may exceed this bound and simply reread uncached bands.
MAX_RETRY_BAND_BYTES = 16 * 1024 * 1024


def _band_failure_reason(error: Exception, band: str) -> str:
    """Retain the failed band and bounded causes without asset access details."""
    details, seen = [], set()
    current = error
    while current is not None and id(current) not in seen and len(details) < 4:
        seen.add(id(current))
        message = public_error(current)
        # The band identifies the asset; its URL (including embedded credentials)
        # is unnecessary. GDAL can also truncate a signed URL before a block error.
        message = re.sub(r"https?://[^\s'\"]+", "[raster asset]", message, flags=re.IGNORECASE)
        block = re.search(r"band \d+: IReadBlock failed\b", message)
        if block:
            message = "[raster asset] " + message[block.start():]
        message = " ".join(message.split())
        if len(message) > 220:
            message = message[:217] + "..."
        details.append(f"{type(current).__name__}: {message}")
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return (f"Band {band}: " + "; caused by ".join(details))[:1000]


def reflectance_scaling(asset_metadata: dict, properties: dict,
                        scale_factor: float = S2_SCALE_FACTOR,
                        require_explicit: bool = False) -> dict:
    """Resolve DN conversion without mixing old and new processing baselines.

    Explicit raster scale/offset take precedence. Planetary Computer currently
    omits them, so its native L2A assets need the baseline >=04.00 DN correction.
    Use the processing baseline, not acquisition date: old acquisitions can be
    reprocessed. Reference: Microsoft's PlanetaryComputerExamples notebook
    datasets/sentinel-2-l2a/baseline-change.ipynb and ESA's L2A BOA_ADD_OFFSET.
    """
    bands = asset_metadata.get("raster:bands") or [{}]
    metadata = bands[0]
    if "scale" in metadata and "offset" in metadata:
        scale, offset = float(metadata["scale"]), float(metadata["offset"])
        source = "stac-raster-scale-offset"
    else:
        if require_explicit:
            raise ValueError("Earth Search reflectance requires explicit raster:bands scale and offset")
        try:
            baseline = float(properties["s2:processing_baseline"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Sentinel-2 needs raster scale/offset or a processing baseline") from error
        if not np.isfinite(baseline) or baseline < 0:
            raise ValueError("Sentinel-2 processing baseline must be finite and nonnegative")
        scale = float(metadata.get("scale", 1.0 / scale_factor))
        offset = float(metadata.get("offset", -1000.0 * scale if baseline >= 4.0 else 0.0))
        source = "planetary-computer-processing-baseline"
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
        raise ValueError("Sentinel-2 reflectance scale/offset must be finite, with positive scale")
    return {"scale": scale, "offset": offset, "source": source}


class Sentinel2Adapter(SatelliteAdapterBase):
    """Adapter for Sentinel-2 Level-2A data from an explicitly selected provider.

    Downloads, cloud-masks, and preprocesses Sentinel-2 L2A surface
    reflectance data into ForestMind-compatible 12-band patches.

    Args:
        cache_dir: Local directory for caching downloaded data.
        patch_size: Target spatial size in pixels.
        resolution: Nominal source resolution; final grid follows bbox / patch_size.
        scale_factor: Reflectance scale factor for L2A products.
        provider: planetary-computer (default) or earth-search; no automatic fallback.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/real",
        patch_size: int = 128,
        resolution: float = 10.0,
        scale_factor: float = S2_SCALE_FACTOR,
        timeout_seconds: float = 30.0,
        provider: str = "planetary-computer",
    ):
        if provider not in {"planetary-computer", "earth-search"}:
            raise ValueError("provider must be planetary-computer or earth-search")
        super().__init__(cache_dir=cache_dir, patch_size=patch_size, resolution=resolution,
                         timeout_seconds=timeout_seconds)
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError("scale_factor must be positive and finite")
        self.scale_factor = scale_factor
        self.provider = provider
        self.catalog_url = EARTH_SEARCH_STAC_URL if provider == "earth-search" else PC_STAC_URL
        self.collection = EARTH_SEARCH_COLLECTION if provider == "earth-search" else S2_COLLECTION
        self.provider_name = "element84-earth-search" if provider == "earth-search" else "microsoft-planetary-computer"
        self.asset_keys = EARTH_SEARCH_ASSET_KEYS if provider == "earth-search" else S2_ASSET_KEYS
        self._retry_band_identity = None
        self._retry_bands = {}
        self._retry_band_bytes = 0

    def _clear_retry_bands(self):
        self._retry_band_identity = None
        self._retry_bands.clear()
        self._retry_band_bytes = 0

    def _prepare_retry_bands(self, scene, bbox):
        identity = (scene.scene_id, scene.datetime.isoformat(), self.provider, self.collection,
                    bbox.west, bbox.south, bbox.east, bbox.north, self.patch_size,
                    self.scale_factor, PREPROCESSING_VERSION,
                    hashlib.sha256(json.dumps(scene.stac_item, sort_keys=True, allow_nan=False,
                                               separators=(",", ":")).encode()).hexdigest())
        if identity != self._retry_band_identity:
            self._clear_retry_bands()
            self._retry_band_identity = identity

    def _read_band(self, href, band_name, bbox, signer, *, categorical=False):
        check_acquisition(href)
        key = (band_name, href, categorical)
        if key in self._retry_bands:
            return self._retry_bands[key]
        with open_raster(signer.sign(href), self.timeout_seconds) as src:
            data, valid = read_aligned_band(src, bbox, self.patch_size, categorical=categorical)
        # A failed close or cancellation never leaves a completed-band entry.
        check_acquisition()
        size = data.nbytes + valid.nbytes
        if self._retry_band_bytes + size <= MAX_RETRY_BAND_BYTES:
            self._retry_bands[key] = (data.copy(), valid.copy())
            self._retry_band_bytes += size
        return data, valid

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
        """Search the selected catalog for Sentinel-2 L2A scenes.

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

        catalog = open_catalog(self.catalog_url, self.timeout_seconds)

        search_options = dict(
            collections=[self.collection],
            bbox=[bbox.west, bbox.south, bbox.east, bbox.north],
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=max_results,
            limit=min(max_results, 100),
        )
        # Earth Search C1 rejects a datetime sort. Sort returned candidates
        # locally; max_results bounds the provider's returned subset.
        if self.provider == "planetary-computer":
            search_options["sortby"] = [{"field": "datetime", "direction": "asc"}]
        search = catalog.search(**search_options)

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
                extra={"provider": self.provider_name, "catalog_url": self.catalog_url},
            ))

        logger.info(f"[sentinel2] Found {len(scenes)} scenes (cloud ≤ {max_cloud_cover}%)")
        return sorted(scenes, key=lambda scene: scene.datetime)

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
                    "Sentinel-2 adapter requires 'pystac'. "
                    "Install with: pip install pystac"
            )

        paths = []
        self.download_failures = []
        signer = AssetSigner(self.timeout_seconds)

        for scene in scenes:
            check_acquisition()
            # A scene from another provider must never be silently processed
            # using this provider's keys or scaling assumptions.
            scene_provider = scene.extra.get("provider", self.provider_name)
            scene_collection = (scene.stac_item or {}).get("collection")
            if scene_provider != self.provider_name or (
                scene_collection is not None and scene_collection != self.collection
            ):
                self.download_failures.append({"scene_id": scene.scene_id,
                                               "reason": "Scene does not match selected Sentinel-2 provider/collection",
                                               "retryable": False, "error_type": "ValueError"})
                self._clear_retry_bands()
                continue
            scene.extra["provider"] = self.provider_name
            cache_path = self.cache_path_for_scene(scene, bbox)

            if self.is_cached(scene, bbox):
                self._clear_retry_bands()
                logger.debug(f"[sentinel2] Cached: {scene.scene_id}")
                paths.append(cache_path)
                continue

            logger.info(f"[sentinel2] Downloading: {scene.scene_id} ({scene.date_str})")

            active_band = None
            try:
                # Reconstruct STAC item for asset signing
                item = pystac.Item.from_dict(scene.stac_item)
                self._prepare_retry_bands(scene, bbox)

                # Validate the complete Earth Search contract before any COG
                # read. Do not apply Planetary Computer baseline heuristics.
                if self.provider == "earth-search":
                    if scene_collection != self.collection:
                        raise ValueError("Earth Search scene must declare its C1 collection")
                    scl_asset = item.assets.get(self.asset_keys["SCL"])
                    if scl_asset is None:
                        raise ValueError("Sentinel-2 scene lacks SCL quality data")
                    if "data" not in (scl_asset.roles or []) or not scl_asset.href.startswith("https://"):
                        raise ValueError("Earth Search SCL requires a public HTTPS data asset")
                    for band_name in S2_BANDS:
                        asset = item.assets[self.asset_keys[band_name]]
                        eo_bands = asset.extra_fields.get("eo:bands", [])
                        if len(eo_bands) != 1 or eo_bands[0].get("name") != band_name:
                            raise ValueError(f"Earth Search asset metadata does not identify {band_name}")
                        if "data" not in (asset.roles or []) or not asset.href.startswith("https://"):
                            raise ValueError(f"Earth Search {band_name} requires a public HTTPS data asset")
                        reflectance_scaling(asset.extra_fields, item.properties, require_explicit=True)

                # Read and stack bands
                band_arrays = []
                scaling = {}
                quality_mask = np.ones((self.patch_size, self.patch_size), dtype=bool)
                for band_name in S2_BANDS:
                    active_band = band_name
                    asset_key = self.asset_keys[band_name]
                    band_data, valid = self._read_band(item.assets[asset_key].href, band_name, bbox, signer)
                    quality_mask &= valid

                    # Scale to [0, 1] after reuse of raw aligned values.
                    conversion = reflectance_scaling(item.assets[asset_key].extra_fields,
                                                     item.properties, self.scale_factor,
                                                     require_explicit=self.provider == "earth-search")
                    scaling[band_name] = conversion
                    band_data = band_data * conversion["scale"] + conversion["offset"]
                    band_data = np.clip(band_data, 0.0, 1.0)

                    band_arrays.append(band_data)
                    active_band = None

                # Stack bands: (12, H, W)
                patch = np.stack(band_arrays, axis=0)

                # Read SCL for cloud masking
                scl_key = self.asset_keys["SCL"]
                if scl_key in item.assets:
                    active_band = "SCL"
                    scl, scl_valid = self._read_band(item.assets[scl_key].href, "SCL", bbox, signer,
                                                     categorical=True)
                    # True = valid pixel
                    quality_mask &= scl_valid & ~np.isin(scl, list(SCL_INVALID))
                    active_band = None

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
                                             "catalog_url": self.catalog_url,
                                             "asset_keys": self.asset_keys,
                                         })
                paths.append(cache_path)
                self._clear_retry_bands()

            except Exception as e:
                reason = _band_failure_reason(e, active_band) if active_band else public_error(e)
                retryable = is_retryable_error(e)
                self.download_failures.append({"scene_id": scene.scene_id, "reason": reason,
                                               "retryable": retryable, "error_type": type(e).__name__})
                if not retryable:
                    self._clear_retry_bands()
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
