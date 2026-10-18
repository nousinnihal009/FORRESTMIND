"""Bounded ESA WorldCover context, sampled from official public categorical COGs.

This is a dated tree-cover reference for topology, not a current ecological
forest boundary, land-loss detector, or extra input to the optical model.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import math

import numpy as np

from forestmind.api.schemas import Geometry
from forestmind.data.adapters.base import BoundingBox
from forestmind.data.adapters.network import open_raster, public_error, restricted_acquisition
from forestmind.data.adapters.raster import read_aligned_band
from forestmind.monitoring.spatial import geometry_bbox, polygon_mask

SOURCE = "https://esa-worldcover.org/en/data-access"
HOST = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
VERSIONS = {2020: "v100", 2021: "v200"}
PUBLICATION_DATES = {2020: "2021-10-20", 2021: "2022-10-28"}
CLASSES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
           50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
           80: "Permanent water bodies", 90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss and lichen"}
TREE_CLASSES = (10, 95)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def selected_epoch(cutoff: str) -> int:
    parsed = date.fromisoformat(cutoff)
    if parsed.isoformat() != cutoff:
        raise ValueError("Land-cover cutoff must use YYYY-MM-DD")
    if parsed >= date(2021, 12, 31):
        return 2021
    if parsed >= date(2020, 12, 31):
        return 2020
    raise ValueError("WorldCover has complete annual reference maps for 2020 and 2021 only; no epoch ends at or before this cutoff")


def tile_urls(bbox: list[float], year: int) -> list[str]:
    """At most four 3-degree tiles for the service's compact AOI bound."""
    if year not in VERSIONS:
        raise ValueError("WorldCover supports the 2020/v100 and 2021/v200 epochs only")
    w, s, e, n = bbox
    west = math.floor(w / 3) * 3
    south = math.floor(s / 3) * 3
    east = math.floor(math.nextafter(e, -math.inf) / 3) * 3
    north = math.floor(math.nextafter(n, -math.inf) / 3) * 3
    count = ((east - west) // 3 + 1) * ((north - south) // 3 + 1)
    if not 1 <= count <= 4:
        raise ValueError("Land-cover request exceeds the four-tile bound")
    urls = []
    version = VERSIONS[year]
    for latitude in range(south, north + 1, 3):
        for longitude in range(west, east + 1, 3):
            tile = f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}{'E' if longitude >= 0 else 'W'}{abs(longitude):03d}"
            urls.append(f"{HOST}/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif")
    return urls


def _read_tile(url: str, bbox: list[float], patch_size: int, timeout_seconds: float):
    with open_raster(url, timeout_seconds=timeout_seconds) as source:
        if source.count != 1 or source.crs is None:
            raise ValueError("WorldCover source must be a georeferenced single-band class raster")
        values, valid = read_aligned_band(source, BoundingBox(*bbox), patch_size, categorical=True)
        metadata = {"source_url": url, "source_crs": source.crs.to_string(),
                    "source_shape": [source.height, source.width],
                    "source_transform": list(source.transform)[:6],
                    "source_pixel_size": list(source.res), "source_nodata": source.nodata,
                    "source_dtype": source.dtypes[0], "full_source_sha256": None,
                    "checksum_scope": "The saved aligned class/mask sample is hashed; the remote full-tile file was not downloaded"}
        return values, valid, metadata


def acquire_landcover(geometry: dict, *, cutoff: str, patch_size: int = 128,
                      timeout_seconds: float = 30) -> dict:
    """Read only bounded AOI windows; retain unknown/nodata rather than nonforest.

    Rasterio/GDAL reads COG byte ranges. Nothing downloads the complete tile or
    accepts arbitrary HTTP URLs. Source failures leave an explicit incomplete
    mask; absence of every valid source pixel raises a useful provider error.
    """
    validated = Geometry.model_validate(geometry).model_dump()
    bbox = geometry_bbox(validated)
    w, s, e, n = bbox
    area = 6371.0088 ** 2 * math.radians(e - w) * (math.sin(math.radians(n)) - math.sin(math.radians(s)))
    if abs(s) > 85 or abs(n) > 85 or e - w > 1 or n - s > 1 or not .0001 <= area <= 25:
        raise ValueError("Land-cover collection requires a compact AOI of at most 25 km² between 85°S and 85°N")
    if type(patch_size) is not int or not 16 <= patch_size <= 128:
        raise ValueError("Land-cover sample grid must have 16–128 pixels per side")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
        raise ValueError("Land-cover request timeout must be finite and at most 60 seconds")
    year = selected_epoch(cutoff)
    urls = tile_urls(bbox, year)
    shape = (patch_size, patch_size)
    classes = np.zeros(shape, dtype=np.uint8)
    valid = np.zeros(shape, dtype=bool)
    inside = polygon_mask(validated, shape)
    sources, failures = [], []
    with restricted_acquisition():
        for url in urls:
            try:
                values, mask, metadata = _read_tile(url, bbox, patch_size, timeout_seconds)
                if values.shape != shape or mask.shape != shape:
                    raise ValueError("WorldCover aligned raster shape differs from the selected grid")
                mask = mask.astype(bool) & np.isfinite(values) & (values == np.rint(values)) & np.isin(values, list(CLASSES)) & inside
                conflict = valid & mask & (classes != values)
                if conflict.any():
                    raise ValueError("WorldCover tile overlap has conflicting classes on the same grid")
                classes[mask] = values[mask].astype(np.uint8)
                valid |= mask
                metadata["accepted_sample_pixels"] = int(mask.sum())
                metadata["sample_sha256"] = _digest({"classes": np.where(mask, values, 0).astype(np.uint8).tolist(),
                                                      "valid_mask": mask.tolist()})
                sources.append(metadata)
            except (OSError, ValueError) as error:
                failures.append({"source_url": url, "reason": public_error(error)})
    if not valid.any():
        detail = failures[0]["reason"] if failures else "No valid classified samples intersect this AOI"
        raise ValueError(f"WorldCover {year} has no usable land-cover samples for this area: {detail}")
    forest = np.isin(classes, TREE_CLASSES) & valid
    result = {"available": True, "provider": "ESA WorldCover / VITO public AWS archive",
        "source": f"ESA WorldCover {year} {VERSIONS[year]} (official public categorical COGs)",
        "source_url": SOURCE, "dataset": "ESA WorldCover", "version": VERSIONS[year],
        "year": year, "observed_at": f"{year}-12-31", "epoch_start": f"{year}-01-01",
        "epoch_end": f"{year}-12-31", "published_at": PUBLICATION_DATES[year],
        "observation_date_interpretation": "Annual reference period end, not a scene acquisition timestamp",
        "temporal_role": "Retrospective reference; publication follows its observation year. Not an operational historical predictor.",
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "requested_cutoff": cutoff,
        "geometry": validated, "geometry_sha256": _digest(validated), "bbox_wgs84": bbox,
        "data_kind": "classified_forest_cover", "classification": "WorldCover class 10 (tree cover) or 95 (mangroves); a woody-cover proxy, not a universal ecological forest definition",
        "forest_classes": list(TREE_CLASSES), "forest": forest.tolist(), "valid_mask": valid.tolist(),
        "class_raster": classes.tolist(), "class_legend": {str(key): value for key, value in CLASSES.items()},
        "valid_fraction": float(valid.sum() / inside.sum()) if inside.any() else 0,
        "tree_sample_fraction": float(forest.sum() / valid.sum()),
        "native_resolution_m": 10, "grid_shape": list(shape), "grid_crs": "EPSG:4326",
        "grid_pixel_size_degrees": [(e - w) / patch_size, (n - s) / patch_size],
        "resampling": "nearest-neighbour categorical sample; no interpolation of class IDs",
        "grid_orientation": "north-up; row zero is the northern edge",
        "sources": sources, "failures": failures, "incomplete": bool(failures) or not np.all(valid[inside]),
        "role": "context_only", "model_input": False, "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": f"© ESA WorldCover project {year} / Contains modified Copernicus Sentinel data ({year}) processed by ESA WorldCover consortium",
        "doi": "https://doi.org/10.5281/zenodo." + ("7254221" if year == 2021 else "5571936"),
        "available_epochs": [2020, 2021], "update_frequency": "Fixed published annual epochs; this integration does not imply a current annual feed",
        "classification_uncertainty": "Provider classification errors apply; no per-pixel class probability or independent AOI field validation is available from this map read",
        "change_detection": {"available": False, "reason": "2020/v100 and 2021/v200 use different algorithms. Their differences mix classification changes and real changes; no land-loss estimate is produced."},
        "limitations": [
            "This is historical classified tree/mangrove cover, not a verified forest boundary or present-day forest inventory.",
            "Native resolution is approximately 10 m. The output grid samples the AOI extent; resampling adds no detail and changes apparent patch topology.",
            "No-data and failed tile reads remain unknown. The AWS map does not include the separately distributed InputQuality layer or per-pixel class confidence.",
            "Reference year, product publication and retrieval dates differ. This context is excluded from model inputs and historical forecast/backtest predictors.",
            "Changing 2020/v100 to 2021/v200 changes both reference year and classification algorithm; do not interpret the difference as measured forest loss.",
        ]}
    result["sha256"] = _digest(result)
    return result
