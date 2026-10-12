"""
Real satellite data adapters for ForestMind.

Provides unified access to multi-sensor satellite imagery:
  - Sentinel-2 L2A (12-band optical, 10m)
  - Sentinel-1 GRD IW (VV/VH SAR, 10m)
  - Landsat-8/9 C2 L2 (7-band optical, 30m)

All adapters share the SatelliteAdapterBase interface for consistent
search → download → load pipelines via the Planetary Computer STAC API.
"""

from forestmind.data.adapters.base import (
    PatchData,
    SatelliteAdapterBase,
    SceneMetadata,
    TimeSeriesData,
)
from forestmind.data.adapters.harmonize import BandHarmonizer, landsat_to_sentinel2
from forestmind.data.adapters.sentinel2 import Sentinel2Adapter
from forestmind.data.adapters.sentinel1 import Sentinel1Adapter
from forestmind.data.adapters.landsat import LandsatAdapter

__all__ = [
    "SatelliteAdapterBase",
    "SceneMetadata",
    "PatchData",
    "TimeSeriesData",
    "Sentinel2Adapter",
    "Sentinel1Adapter",
    "LandsatAdapter",
    "BandHarmonizer",
    "landsat_to_sentinel2",
]
