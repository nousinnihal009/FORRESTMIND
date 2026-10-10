"""
Geospatial utilities for ForestMind.

Coordinate systems, bounding box handling, and CRS utilities.
These will be expanded when real satellite data adapters are integrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BoundingBox:
    """Geographic bounding box in WGS84 (EPSG:4326)."""
    west: float   # min longitude
    south: float  # min latitude
    east: float   # max longitude
    north: float  # max latitude

    @property
    def center(self) -> tuple[float, float]:
        """Return (lat, lon) center."""
        return (
            (self.north + self.south) / 2,
            (self.east + self.west) / 2,
        )

    @property
    def width_deg(self) -> float:
        return self.east - self.west

    @property
    def height_deg(self) -> float:
        return self.north - self.south

    @property
    def area_deg2(self) -> float:
        return self.width_deg * self.height_deg

    def approx_area_km2(self) -> float:
        """Approximate area in km² using the center latitude."""
        lat_center = self.center[0]
        lat_km = 111.32  # km per degree latitude
        lon_km = 111.32 * np.cos(np.radians(lat_center))
        return self.height_deg * lat_km * self.width_deg * lon_km

    def to_dict(self) -> dict:
        return {"west": self.west, "south": self.south, "east": self.east, "north": self.north}

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat <= self.north and self.west <= lon <= self.east


@dataclass
class GeoLocation:
    """A geographic point with metadata."""
    latitude: float
    longitude: float
    name: Optional[str] = None
    country: Optional[str] = None
    biome: Optional[str] = None

    def to_bbox(self, radius_km: float = 10.0) -> BoundingBox:
        """Create a bounding box centered on this point.
        
        Args:
            radius_km: Half-width of the box in km.
        """
        lat_deg = radius_km / 111.32
        lon_deg = radius_km / (111.32 * np.cos(np.radians(self.latitude)))
        return BoundingBox(
            west=self.longitude - lon_deg,
            south=self.latitude - lat_deg,
            east=self.longitude + lon_deg,
            north=self.latitude + lat_deg,
        )


def pixel_to_geo(
    pixel_x: int,
    pixel_y: int,
    bbox: BoundingBox,
    img_width: int,
    img_height: int,
) -> tuple[float, float]:
    """Convert pixel coordinates to geographic coordinates.
    
    Returns (latitude, longitude).
    """
    lon = bbox.west + (pixel_x / img_width) * bbox.width_deg
    lat = bbox.north - (pixel_y / img_height) * bbox.height_deg  # y-axis inverted
    return lat, lon


def geo_to_pixel(
    lat: float,
    lon: float,
    bbox: BoundingBox,
    img_width: int,
    img_height: int,
) -> tuple[int, int]:
    """Convert geographic coordinates to pixel coordinates.
    
    Returns (pixel_x, pixel_y).
    """
    pixel_x = int((lon - bbox.west) / bbox.width_deg * img_width)
    pixel_y = int((bbox.north - lat) / bbox.height_deg * img_height)
    return pixel_x, pixel_y


# ─── Notable Forest Regions ──────────────────────────────────────────────────

AMAZON_BASIN = BoundingBox(west=-73.0, south=-15.0, east=-45.0, north=5.0)
CONGO_BASIN = BoundingBox(west=15.0, south=-5.0, east=30.0, north=5.0)
BORNEO = BoundingBox(west=108.0, south=-4.5, east=119.5, north=7.5)
BOREAL_CANADA = BoundingBox(west=-140.0, south=50.0, east=-60.0, north=70.0)

STUDY_REGIONS = {
    "amazon": AMAZON_BASIN,
    "congo": CONGO_BASIN,
    "borneo": BORNEO,
    "boreal_canada": BOREAL_CANADA,
}
