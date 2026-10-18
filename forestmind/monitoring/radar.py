"""Bounded, orbit-matched Sentinel-1 RTC measurements as contextual evidence.

Positive linear gamma-naught power is aligned before conversion to dB. No
min/max clipping, optical-input fusion, invented labels or disturbance cause.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np

from forestmind.api.schemas import Geometry
from forestmind.data.adapters.base import BoundingBox
from forestmind.data.adapters.network import AssetSigner, open_raster, restricted_acquisition, validate_destination
from forestmind.data.adapters.raster import read_aligned_band
from forestmind.data.history import _sanitize
from forestmind.monitoring.spatial import geometry_bbox, polygon_mask

CATALOG = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
SOURCE = "https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MIN_COMMON_VALID_FRACTION = .4


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Radar catalog redirected; refusing an unapproved destination")


def _search(bbox, start, end, search_limit, timeout_seconds):
    query = urlencode({"collections": COLLECTION, "bbox": ",".join(str(v) for v in bbox),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59.999999Z", "limit": search_limit,
        "sortby": "-datetime"})
    url = CATALOG + "/search?" + query
    validate_destination(url, catalog=True)
    request = Request(url, headers={"User-Agent": "ForestMind-local-research/0.6", "Accept": "application/geo+json"})
    with build_opener(_NoRedirect()).open(request, timeout=timeout_seconds) as stream:
        raw = stream.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Radar catalog response exceeds the 4 MiB limit")
    response = json.loads(raw)
    items = response.get("features")
    if not isinstance(items, list) or len(items) > search_limit:
        raise ValueError("Radar catalog returned invalid or unbounded candidates")
    return items, {"request_url": url, "response_sha256": hashlib.sha256(raw).hexdigest(),
                   "returned_candidates": len(items), "search_limit": search_limit,
                   "pagination": "Only this bounded first page was searched; no next links followed"}


def _candidate(item, start, end):
    if not isinstance(item, dict) or item.get("collection") != COLLECTION or not isinstance(item.get("id"), str):
        raise ValueError("Radar item collection/identity does not match Sentinel-1 RTC")
    props = item["properties"]
    acquired = datetime.fromisoformat(props["datetime"].replace("Z", "+00:00"))
    if acquired.tzinfo is None:
        raise ValueError("Radar acquisition timestamp requires a timezone")
    acquired = acquired.astimezone(timezone.utc)
    if not start <= acquired.date().isoformat() <= end:
        raise ValueError("Radar acquisition lies outside the selected cutoff window")
    orbit = props["sat:relative_orbit"]
    orbit_state = props["sat:orbit_state"]
    platform = props["platform"]
    if (type(orbit) is not int or not 1 <= orbit <= 175 or orbit_state not in {"ascending", "descending"}
            or not isinstance(platform, str) or not platform or props.get("sar:instrument_mode") != "IW"
            or sorted(props.get("sar:polarizations", [])) != ["VH", "VV"]):
        raise ValueError("Radar pairing requires explicit platform, relative orbit, pass, IW mode and VV/VH")
    for pol in ("vv", "vh"):
        asset = item["assets"][pol]
        validate_destination(asset["href"])
        if "gamma" not in asset.get("description", "").lower():
            raise ValueError("RTC assets must explicitly identify terrain-corrected gamma-naught power")
        band = asset.get("raster:bands", [{}])[0]
        scale, offset = band.get("scale", 1), band.get("offset", 0)
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in (scale, offset)) or scale <= 0:
            raise ValueError("Invalid radar scale/offset metadata")
    safe = _sanitize(item)
    return {"item": safe, "datetime": acquired.isoformat(),
            "group": (platform, orbit, orbit_state, "IW", "VV", "VH")}


def select_pair(items, *, start, end):
    groups, rejected = defaultdict(list), []
    seen = set()
    for item in items:
        try:
            candidate = _candidate(item, start, end)
            if item["id"] in seen:
                raise ValueError("Duplicate radar scene identity")
            seen.add(item["id"])
            groups[candidate["group"]].append(candidate)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            rejected.append({"scene_id": item.get("id") if isinstance(item, dict) else None, "reason": str(error)})
    pairs = []
    for rows in groups.values():
        rows.sort(key=lambda row: (row["datetime"], row["item"]["id"]), reverse=True)
        latest = rows[0]
        previous = next((row for row in rows[1:] if datetime.fromisoformat(latest["datetime"]) - datetime.fromisoformat(row["datetime"]) >= timedelta(days=1)), None)
        if previous:
            pairs.append([previous, latest])
    if not pairs:
        raise ValueError("No pair at least one day apart has matching satellite, relative orbit, pass and VV/VH in the bounded catalog subset; extend the period or retry another area")
    pair = max(pairs, key=lambda rows: (rows[1]["datetime"], rows[0]["datetime"]))
    return [row["item"] for row in pair], rejected


def _read_scene(item, bbox, patch_size, timeout_seconds):
    signer = AssetSigner(timeout_seconds)
    arrays, valid_masks, metadata = [], [], []
    for pol in ("vv", "vh"):
        asset = item["assets"][pol]
        # Tokens exist only in this local variable and the bounded reader.
        signed = signer.sign(asset["href"])
        with open_raster(signed, timeout_seconds) as source:
            values, valid = read_aligned_band(source, BoundingBox(*bbox), patch_size)
            band = asset.get("raster:bands", [{}])[0]
            values = values * band.get("scale", 1) + band.get("offset", 0)
            valid &= np.isfinite(values) & (values > 0)
            arrays.append(np.where(valid, values, 0))
            valid_masks.append(valid)
            metadata.append({"polarization": pol.upper(), "source_url": asset["href"],
                "source_crs": source.crs.to_string(), "source_transform": list(source.transform)[:6],
                "source_pixel_size": list(source.res), "source_shape": [source.height, source.width],
                "source_nodata": source.nodata, "asset_raster_metadata": band,
                "full_source_sha256": None})
    return np.stack(arrays), np.logical_and.reduce(valid_masks), metadata


def _map(values, mask):
    return [[round(float(value), 6) if keep else None for value, keep in zip(row, valid)]
            for row, valid in zip(values, mask)]


def acquire_radar(geometry: dict, *, start: str, end: str, patch_size: int = 64,
                  timeout_seconds: float = 30, search_limit: int = 30) -> dict:
    validated = Geometry.model_validate(geometry).model_dump()
    bbox = geometry_bbox(validated)
    w, s, e, n = bbox
    area = 6371.0088 ** 2 * math.radians(e - w) * (math.sin(math.radians(n)) - math.sin(math.radians(s)))
    if abs(s) > 85 or abs(n) > 85 or e - w > 1 or n - s > 1 or not .0001 <= area <= 25:
        raise ValueError("Radar collection requires a compact AOI of at most 25 km² between 85°S and 85°N")
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first.isoformat() != start or last.isoformat() != end or first > last or last > datetime.now(timezone.utc).date():
        raise ValueError("Radar requires ordered YYYY-MM-DD dates ending no later than today UTC")
    if type(patch_size) is not int or not 16 <= patch_size <= 128:
        raise ValueError("Radar sample grid must have 16–128 pixels per side")
    if type(search_limit) is not int or not 2 <= search_limit <= 30:
        raise ValueError("Radar search must be bounded to 2–30 candidates")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
        raise ValueError("Radar timeout must be finite and at most 60 seconds")
    effective_start = max(first, last - timedelta(days=179)).isoformat()
    inside = polygon_mask(validated, (patch_size, patch_size))
    with restricted_acquisition():
        items, search = _search(bbox, effective_start, end, search_limit, timeout_seconds)
        selected, rejected = select_pair(items, start=effective_start, end=end)
        records, powers, masks = [], [], []
        for item in selected:
            power, valid, metadata = _read_scene(item, bbox, patch_size, timeout_seconds)
            if power.shape != (2, patch_size, patch_size) or valid.shape != inside.shape:
                raise ValueError("Radar aligned grid differs from the confirmed AOI sample shape")
            valid &= np.isfinite(power).all(axis=0) & (power > 0).all(axis=0) & inside
            if not valid.any():
                raise ValueError("Radar acquisition has no positive valid calibrated samples inside the AOI")
            powers.append(power)
            masks.append(valid)
            properties = item["properties"]
            acquired = datetime.fromisoformat(properties["datetime"].replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
            records.append({"id": item["id"], "datetime": acquired,
                "platform": properties["platform"], "relative_orbit": properties["sat:relative_orbit"],
                "orbit_state": properties["sat:orbit_state"], "instrument_mode": "IW", "polarizations": ["VV", "VH"],
                "valid_fraction": float(valid.sum() / inside.sum()), "valid_mask": valid.tolist(),
                "vv_linear_gamma0": _map(power[0], valid), "vh_linear_gamma0": _map(power[1], valid),
                "native_resolution_range_m": properties.get("sar:resolution_range"),
                "native_resolution_azimuth_m": properties.get("sar:resolution_azimuth"),
                "provider_pixel_spacing_range_m": properties.get("sar:pixel_spacing_range"),
                "provider_pixel_spacing_azimuth_m": properties.get("sar:pixel_spacing_azimuth"),
                "source_metadata_sha256": _digest(item), "source_metadata": item, "assets": metadata})
    common = masks[0] & masks[1]
    common_fraction = float(common.sum() / inside.sum()) if inside.any() else 0.
    if common_fraction < MIN_COMMON_VALID_FRACTION:
        raise ValueError(f"Radar pair common valid coverage is {common_fraction:.1%}; at least 40% is required for this screening comparison")
    db_arrays = []
    for record, power, mask in zip(records, powers, masks):
        db = np.zeros_like(power, dtype=np.float64)
        db[:, mask] = 10 * np.log10(power[:, mask])
        record.update(vv_db=_map(db[0], mask), vh_db=_map(db[1], mask),
                      mean_vv_db=float(db[0][mask].mean()), mean_vh_db=float(db[1][mask].mean()))
        record["sample_sha256"] = _digest({"vv": record["vv_linear_gamma0"], "vh": record["vh_linear_gamma0"], "valid_mask": record["valid_mask"]})
        db_arrays.append(db)
    delta = db_arrays[1] - db_arrays[0]
    result = {"available": True, "provider": "Microsoft Planetary Computer", "dataset": "Sentinel-1 RTC",
        "collection": COLLECTION, "source_url": SOURCE, "start": start, "end": end,
        "search_start": effective_start, "observed_at": records[-1]["datetime"][:10],
        "geometry": validated, "geometry_sha256": _digest(validated), "bbox_wgs84": bbox,
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "search": search, "rejected_candidates": rejected,
        "selection": "Most recent pair at least one day apart within bounded candidates; same satellite, relative orbit, pass, IW and VV/VH",
        "scenes": records, "units": "dB gamma0", "source_units": "positive linear gamma-naught power",
        "conversion": "10 * log10(linear power); no min-max scaling or clipping",
        "grid_shape": [patch_size, patch_size], "grid_crs": "EPSG:4326",
        "grid_pixel_size_degrees": [(e - w) / patch_size, (n - s) / patch_size],
        "resampling": "bilinear interpolation of linear power onto exact AOI grid before logarithm",
        "comparison": {"start": records[0]["datetime"], "end": records[1]["datetime"],
            "common_valid_fraction": common_fraction, "common_valid_mask": common.tolist(),
            "vv_change_db": float(delta[0][common].mean()), "vh_change_db": float(delta[1][common].mean()),
            "vv_change_map_db": _map(delta[0], common), "vh_change_map_db": _map(delta[1], common),
            "meaning": "Mean of paired pixel dB differences on common valid samples; not a forest-loss or causal estimate"},
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Contains modified Copernicus Sentinel data; Sentinel-1 RTC provided by Microsoft Planetary Computer",
        "role": "context_only", "model_input": False,
        "limits": {"catalog_candidates": search_limit, "max_scene_reads": 2, "max_band_reads": 4,
                   "max_lookback_days": 180, "minimum_common_valid_fraction": MIN_COMMON_VALID_FRACTION},
        "limitations": [
            "Radar backscatter responds to moisture, structure, terrain, speckle and acquisition geometry. A change does not identify disturbance or its cause.",
            "Two dates are contextual evidence, not a seasonally calibrated radar history or a forest-only average.",
            "RTC source pixel spacing is distinct from instrument resolution. Native range/azimuth resolution and source raster spacing are retained when declared; resampling adds no detail.",
            "No independently verified layover/shadow mask or local uncertainty calibration is supplied; finite positive values and nodata masks alone cannot establish ecological reliability.",
            "This separately dated context is not inserted into optical model inputs or monthly observations. A paired trained multimodal model remains a separate requirement.",
        ]}
    result["sha256"] = _digest(result)
    return result
