"""Prepare strict monitoring imports from cached, dated optical acquisitions.

This module never downloads imagery, imputes a month, or manufactures auxiliary
observations. It reads the adapter's physical reflectance cache, uses named bands
and an explicit quality mask, and aggregates valid pixel samples within UTC months.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SENSORS = {
    "s2": "sentinel2", "sentinel-2": "sentinel2", "sentinel2": "sentinel2",
    "landsat": "landsat", "landsat8": "landsat8", "landsat9": "landsat9",
}
BANDS = {"sentinel2": ("B08", "B04", "B11", "B12"),
         "landsat": ("B5", "B4", "B6", "B7")}
REQUIRED_PREPROCESSING_VERSION = "forestmind-grid-v3"


@dataclass(frozen=True)
class PreparedImport:
    payload: dict[str, Any]
    provenance: dict[str, Any]


def _text_scalar(cache, key: str, default: str | None = None) -> str:
    if key not in cache:
        if default is not None:
            return default
        raise ValueError(f"Missing cache metadata: {key}")
    value = cache[key]
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"Cache metadata {key} must be a text scalar")
    scalar = value.item()
    return scalar.decode("utf-8") if isinstance(scalar, bytes) else str(scalar)


def validate_bbox(values) -> tuple[float, float, float, float]:
    if len(values) != 4 or any(isinstance(value, bool) for value in values):
        raise ValueError("Bounding box requires west,south,east,north")
    west, south, east, north = map(float, values)
    if not all(np.isfinite([west, south, east, north])):
        raise ValueError("Bounding box coordinates must be finite")
    if not (-180 <= west < east <= 180 and -90 < south < north < 90 and east - west < 180):
        raise ValueError("Expected a nonempty WGS84 bbox without pole or antimeridian crossing")
    return west, south, east, north


def _load_acquisition(path: Path, requested: str, target_bbox=None) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as cache:
            sensor = _text_scalar(cache, "sensor")
            if sensor not in ({"landsat", "landsat8", "landsat9"} if requested == "landsat" else {requested}):
                return None
            family = "landsat" if sensor.startswith("landsat") else sensor
            bbox = validate_bbox([float(cache[f"bbox_{edge}"]) for edge in ("west", "south", "east", "north")])
            if target_bbox is not None and not np.allclose(bbox, target_bbox, rtol=0, atol=1e-7):
                return None
            version = _text_scalar(cache, "preprocessing_version", "legacy-unversioned")
            if version != REQUIRED_PREPROCESSING_VERSION:
                raise ValueError("Legacy or unverified preprocessing: regenerate acquisitions with the current "
                                 f"download_data.py adapters (required version: {REQUIRED_PREPROCESSING_VERSION})")
            if _text_scalar(cache, "data_units") != "surface_reflectance_0_1":
                raise ValueError("Expected surface_reflectance_0_1 optical units; raw DN or normalized model inputs are not supported")
            if _text_scalar(cache, "grid_crs") != "EPSG:4326":
                raise ValueError("Cache must use the adapter's declared EPSG:4326 output grid")
            acquired = datetime.fromisoformat(_text_scalar(cache, "datetime_iso").replace("Z", "+00:00"))
            if acquired.tzinfo is None or acquired.utcoffset() is None:
                raise ValueError("Acquisition datetime_iso must include its timezone")
            acquired = acquired.astimezone(timezone.utc)
            if "timestamp" in cache:
                timestamp = float(cache["timestamp"])
                if not np.isfinite(timestamp) or abs(timestamp - acquired.timestamp()) > 1:
                    raise ValueError("Acquisition timestamp and datetime_iso disagree")
            data = np.asarray(cache["data"])
            if data.ndim != 3 or min(data.shape) < 1 or data.dtype.kind != "f":
                raise ValueError("Expected a nonempty floating-point reflectance array (bands,height,width)")
            west, south, east, north = bbox
            height, width = data.shape[1:]
            transform = np.asarray(cache["grid_transform"], dtype=float)
            expected = [(east - west) / width, 0, west, 0, -(north - south) / height, north]
            if transform.shape != (6,) or not np.isfinite(transform).all() or not np.allclose(transform, expected, rtol=0, atol=1e-10):
                raise ValueError("Cache grid transform does not match its bbox and spatial shape")
            names_array = cache["band_names"]
            if names_array.ndim != 1 or names_array.dtype.kind not in {"U", "S"}:
                raise ValueError("band_names must be a one-dimensional text array")
            names = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in names_array.tolist()]
            if len(names) != data.shape[0] or len(names) != len(set(names)):
                raise ValueError("band_names must uniquely identify every array channel")
            if any(name not in names for name in BANDS[family]):
                raise ValueError(f"{sensor} cache requires named bands {', '.join(BANDS[family])}")
            selected = np.asarray(data[[names.index(name) for name in BANDS[family]]], dtype=np.float64)
            if "quality_mask" not in cache:
                raise ValueError("An explicit cloud/quality mask is required; scene cloud percentage cannot replace it")
            mask = np.asarray(cache["quality_mask"])
            if mask.shape != data.shape[1:]:
                raise ValueError("Quality mask must match the reflectance spatial grid")
            if mask.dtype.kind not in {"b", "i", "u"} or not np.isin(mask, [0, 1]).all():
                raise ValueError("Quality mask must use boolean or 0/1 values (True means valid)")
            finite = np.isfinite(selected).all(axis=0)
            reflectance_valid = (selected >= 0).all(axis=0) & (selected <= 1).all(axis=0)
            nir, red, swir1, swir2 = selected
            denominators = np.stack([nir + red, nir + swir1, nir + swir2])
            valid = mask.astype(bool) & finite & reflectance_valid & (denominators > 1e-8).all(axis=0)
            numerators = np.stack([nir - red, nir - swir1, nir - swir2])
            indices = np.zeros_like(numerators)
            np.divide(numerators, denominators, out=indices, where=valid[None, ...])
            count = int(valid.sum())
            record = {
                "file": path, "sensor": sensor, "family": family, "bbox": bbox,
                "datetime": acquired, "valid_samples": count, "total_samples": valid.size,
                "index_sums": indices[:, valid].sum(axis=1), "grid_shape": list(data.shape[1:]),
                "source_band_names": names,
                "invalid_numeric_samples": int((~finite | ~reflectance_valid).sum()),
                "scene_id": _text_scalar(cache, "scene_id", path.stem),
                "provider": _text_scalar(cache, "provider"),
                "collection": _text_scalar(cache, "collection", "Not recorded in cache"),
                "preprocessing_version": version, "grid_crs": "EPSG:4326",
                "grid_transform": transform.tolist(), "data_units": "surface_reflectance_0_1",
            }
            # Optional provider provenance is preserved only when actually present.
            for key in ("crs", "grid_crs", "reflectance_units"):
                if key in cache:
                    record[key] = _text_scalar(cache, key)
            return record
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise ValueError(f"Cannot prepare {path.name}: {error}") from error


def prepare_cached_observations(
    cache_dir: str | Path,
    *,
    sensor: str,
    name: str,
    country: str = "Unspecified",
    bbox: tuple[float, float, float, float] | None = None,
    source_paths: list[str | Path] | None = None,
) -> PreparedImport:
    """Return a strict API payload and a detailed provenance sidecar manifest.

    One sensor family and one exact cached AOI are selected. Monthly values are
    valid-pixel-sample-weighted means, and coverage is the mean valid fraction
    across acquisitions, not a union of spatial coverage. No interpolation occurs.
    source_paths restricts aggregation to explicitly accepted files in this cache;
    other cached candidates are excluded rather than accidentally averaged in.
    """
    if sensor not in SENSORS:
        raise ValueError(f"Unsupported optical sensor {sensor!r}; choose sentinel2 or landsat")
    requested = SENSORS[sensor]
    family = "landsat" if requested.startswith("landsat") else requested
    root = Path(cache_dir)
    directory = root / family if (root / family).is_dir() else root
    if not directory.is_dir():
        raise ValueError(f"Cache directory does not exist: {directory}")
    if source_paths is None:
        paths = sorted(directory.rglob("*.npz"))
    else:
        if not source_paths:
            raise ValueError("source_paths must contain at least one accepted acquisition")
        selected_directory = directory.resolve()
        paths = []
        seen_paths = set()
        for candidate in source_paths:
            path = Path(candidate).resolve()
            if not path.is_relative_to(selected_directory) or not path.is_file() or path.suffix.lower() != ".npz":
                raise ValueError("Every selected acquisition must be an existing NPZ inside the selected cache directory")
            if path in seen_paths:
                raise ValueError("Duplicate selected acquisition path")
            seen_paths.add(path)
            paths.append(path)
        paths.sort()
        # Keep provenance relative paths consistent when selection is absolute.
        root = root.resolve()
    target_bbox = validate_bbox(bbox) if bbox is not None else None
    records = []
    for path in paths:
        record = _load_acquisition(path, requested, target_bbox)
        if record is None:
            if source_paths is not None:
                raise ValueError(f"Selected acquisition {path.name} does not match the requested sensor and AOI")
            continue
        records.append(record)
    if not records:
        raise ValueError("No matching optical acquisitions found in the selected cache and region")
    groups = {tuple(round(value, 8) for value in row["bbox"]) for row in records}
    if len(groups) != 1:
        raise ValueError("Cache contains multiple AOIs; select one with --region=west,south,east,north")
    if len({tuple(row["grid_shape"]) for row in records}) != 1:
        raise ValueError("Selected acquisitions have different grid shapes; regenerate one consistent AOI cache")
    records.sort(key=lambda row: (row["datetime"], row["scene_id"], str(row["file"])))
    seen = set()
    for row in records:
        key = row["sensor"], row["scene_id"]
        if key in seen:
            raise ValueError(f"Duplicate cached scene {row['scene_id']!r}; remove duplicate inputs before aggregation")
        seen.add(key)
    month_groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        month_groups[row["datetime"].strftime("%Y-%m")].append(row)
    observations = []
    months = []
    for month, acquisitions in sorted(month_groups.items()):
        valid = sum(row["valid_samples"] for row in acquisitions)
        total = sum(row["total_samples"] for row in acquisitions)
        sums = np.sum([row["index_sums"] for row in acquisitions], axis=0)
        last_date = acquisitions[-1]["datetime"].date().isoformat()
        observation = {"date": last_date, "valid_fraction": round(valid / total, 6)}
        observation.update({key: round(float(sums[index] / valid), 6) if valid else None
                            for index, key in enumerate(("ndvi", "ndmi", "nbr"))})
        observations.append(observation)
        months.append({"month": month, "label_date": last_date,
                       "acquisition_count": len(acquisitions), "valid_samples": valid,
                       "total_samples": total})
    providers = sorted({row["provider"] for row in records})
    collections = sorted({row["collection"] for row in records})
    versions = sorted({row["preprocessing_version"] for row in records})
    west, south, east, north = records[0]["bbox"]
    method = "Monthly mean of valid pixel index samples from cached optical acquisitions"
    notes = (
        f"{method}. UTC month labels use the last actual acquisition date; no missing month is filled. "
        "valid_fraction is valid pixel samples / all pixel samples across that month's acquisitions, "
        "not spatial union coverage. NDVI=(NIR-red)/(NIR+red); NDMI uses SWIR1; NBR uses SWIR2. "
        "Explicit quality mask, finite [0,1] reflectance and positive denominators required. "
        "No climate, fire, SAR change or forest-cover values were supplied. "
        f"{len(records)} acquisitions; preprocessing: {', '.join(versions)}. "
        + "The companion provenance JSON records source files, acquisition timestamps and hashes. "
        "Cache metadata and provider claims are not independently verified."
    )
    source = f"Cached {family} surface reflectance | providers: {', '.join(providers)} | collections: {', '.join(collections)}"
    # Reuse the actual input schema so output cannot silently drift from the API.
    from forestmind.api.schemas import ImportRegion
    payload = ImportRegion.model_validate({
        "name": name, "country": country, "source": source,
        "notes": notes, "synthetic": False,
        "geometry": {"type": "Polygon", "coordinates": [[
            [west, south], [east, south], [east, north], [west, north], [west, south],
        ]]}, "observations": observations,
    }).model_dump(exclude_none=True)
    # Keep unavailable optical indices explicitly null; omit unobserved auxiliary channels.
    payload["observations"] = observations
    acquisition_manifest = []
    for row in records:
        digest = hashlib.sha256()
        with row["file"].open("rb") as handle:
            while block := handle.read(64 * 1024):
                digest.update(block)
        acquisition_manifest.append({
            **{key: value for key, value in row.items() if key not in {"file", "index_sums", "datetime", "family"}},
            "bbox": list(row["bbox"]),
            "cache_file": str(row["file"].relative_to(root)), "sha256": digest.hexdigest(),
            "datetime_utc": row["datetime"].isoformat(),
        })
    provenance = {
        "format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "sensor_selection": requested, "bbox_wgs84": [west, south, east, north],
        "method": method, "providers": providers, "collections": collections,
        "preprocessing_versions": versions, "legacy_cache": False,
        "independently_verified": False, "monthly_aggregation": months,
        "acquisitions": acquisition_manifest,
        "limitations": [
            "Scene-average sampling can vary with cloud cover and acquisition footprints.",
            "Bandpass, atmospheric correction and residual georegistration differences are not harmonized here.",
            "Scene/cloud metadata cannot establish forest-only land cover; these summarize the entire cached AOI.",
            "NDVI, NDMI and NBR alone cannot establish disturbance causes or tipping points.",
        ],
    }
    return PreparedImport(payload=payload, provenance=provenance)


def write_prepared_import(prepared: PreparedImport, output: str | Path) -> tuple[Path, Path]:
    """Write the API payload and companion manifest, refusing accidental overwrite."""
    output = Path(output)
    sidecar = output.with_suffix(".provenance.json")
    if output == sidecar:
        raise ValueError("Choose an output name that does not end in .provenance.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Output or provenance file already exists; choose a new --output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_text = json.dumps(prepared.payload, indent=2, ensure_ascii=False, allow_nan=False)
    provenance_text = json.dumps(prepared.provenance, indent=2, ensure_ascii=False, allow_nan=False)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload_text + "\n")
    with sidecar.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(provenance_text + "\n")
    return output, sidecar
