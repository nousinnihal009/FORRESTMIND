"""As-of alignment of measured RTC radar and daily climate to optical acquisitions.

This module does not collect data, fill temporal gaps, infer labels, or fit
normalization. ERA5 alignment is retrospective: a date/latency rule is not proof
that this particular reanalysis vintage was available at a historical cutoff.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math

import numpy as np

from forestmind.monitoring.ingest import validate_bbox


ALIGNMENT_VERSION = "asof-multimodal-v1"
SAR_UNITS = "gamma0_db"
CLIMATE_FEATURE_NAMES = (
    "temperature_mean_c", "precipitation_sum_mm", "reference_et0_sum_mm",
    "soil_moisture_mean_m3_m3",
)
CLIMATE_FEATURE_UNITS = ("degC", "mm", "mm", "m3/m3")
CLIMATE_SOURCE_VARIABLES = (
    "temperature_2m_mean", "precipitation_sum", "et0_fao_evapotranspiration",
    "soil_moisture_0_to_7cm_mean",
)
CLIMATE_SOURCE_UNITS = ("°C", "mm", "mm", "m³/m³")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _utc(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be a timezone-aware ISO timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must have a timezone")
    return parsed.astimezone(timezone.utc)


def _day(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} requires YYYY-MM-DD")
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError(f"{name} requires YYYY-MM-DD")
    return parsed


def _number(value, name, minimum, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and lie in [{minimum}, {maximum}]")
    return value


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _nullable_grid(values, shape, name):
    raw = np.asarray(values, dtype=object)
    if raw.shape != shape:
        raise ValueError(f"{name} does not match the optical grid")
    if any(value is not None and type(value) not in (int, float) for value in raw.flat):
        raise ValueError(f"{name} must contain numbers or nulls")
    return np.asarray(values, dtype=np.float64)


def _align_radar(context, dates, bbox, shape, max_age_days, minimum_valid_fraction):
    count = len(dates)
    arrays = {
        "sar": np.zeros((count, 2, *shape), dtype=np.float32),
        "sar_mask": np.zeros(count, dtype=bool),
        "sar_quality_mask": np.zeros((count, *shape), dtype=bool),
        "sar_acquisition_dates": np.full(count, "", dtype="U32"),
        "sar_age_days": np.full(count, -1., dtype=np.float32),
        "sar_units": np.asarray(SAR_UNITS),
    }
    evidence = {"present": False, "units": SAR_UNITS, "max_age_days": max_age_days,
                "minimum_valid_fraction": minimum_valid_fraction, "matches": [],
                "source_sha256": _digest(context), "same_orbit": True,
                "historical_availability_verified": False,
                "selection": "Latest same-geometry RTC acquisition at or before each optical timestamp, within max_age_days; no interpolation"}
    if not isinstance(context, dict):
        raise ValueError("Radar context must be a JSON object")
    if context.get("available") is not True:
        evidence["reason"] = "Supplied radar context is unavailable; all radar inputs are masked"
        return arrays, evidence
    if context.get("synthetic", False) is not False:
        raise ValueError("Synthetic radar cannot be attached to observed satellite inputs")
    if (context.get("collection") != "sentinel-1-rtc" or context.get("units") != "dB gamma0"
            or context.get("source_units") != "positive linear gamma-naught power"):
        raise ValueError("Radar alignment requires calibrated Sentinel-1 RTC positive linear gamma-naught power")
    supplied_bbox = validate_bbox(context.get("bbox_wgs84", []))
    if not np.allclose(supplied_bbox, bbox, rtol=0, atol=1e-8):
        raise ValueError("Radar context AOI differs from the optical bounding box")
    if context.get("grid_crs") != "EPSG:4326" or context.get("grid_shape") != list(shape):
        raise ValueError("Radar context must have the exact optical grid and EPSG:4326 CRS; resampling is not implicit")
    expected_spacing = [(bbox[2] - bbox[0]) / shape[1], (bbox[3] - bbox[1]) / shape[0]]
    spacing = context.get("grid_pixel_size_degrees")
    if (not isinstance(spacing, list) or len(spacing) != 2 or
            not np.allclose(spacing, expected_spacing, rtol=0, atol=1e-10)):
        raise ValueError("Radar grid spacing does not match the optical pixel grid")
    if context.get("sha256") and context["sha256"] != _digest({k: v for k, v in context.items() if k != "sha256"}):
        raise ValueError("Radar context hash changed")
    scenes = context.get("scenes")
    if not isinstance(scenes, list) or not 1 <= len(scenes) <= 600:
        raise ValueError("Radar context requires 1–600 scenes")
    records, identities, timestamps, group = [], set(), set(), None
    for scene in scenes:
        if not isinstance(scene, dict) or not isinstance(scene.get("id"), str) or not scene["id"]:
            raise ValueError("Radar scene requires an explicit identity")
        acquired = _utc(scene.get("datetime"), "Radar acquisition")
        if scene["id"] in identities or acquired in timestamps:
            raise ValueError("Duplicate radar scene identity or timestamp")
        identities.add(scene["id"])
        timestamps.add(acquired)
        orbit = scene.get("relative_orbit")
        scene_group = (scene.get("platform"), orbit, scene.get("orbit_state"), scene.get("instrument_mode"))
        if (not isinstance(scene_group[0], str) or not scene_group[0] or type(orbit) is not int
                or not 1 <= orbit <= 175 or scene_group[2] not in {"ascending", "descending"}
                or scene_group[3] != "IW" or scene.get("polarizations") != ["VV", "VH"]):
            raise ValueError("Radar scene requires platform, relative orbit, pass, IW mode and VV/VH")
        if group is not None and scene_group != group:
            raise ValueError("Radar input must use one platform, relative orbit and pass")
        group = scene_group
        # Future measurements can be present in an archive but never enter a sequence.
        if acquired > dates[-1]:
            continue
        mask = np.asarray(scene.get("valid_mask"))
        if mask.shape != shape or mask.dtype.kind != "b":
            raise ValueError("Radar valid_mask must be a boolean array on the optical grid")
        power = np.stack([_nullable_grid(scene.get(key), shape, key)
                          for key in ("vv_linear_gamma0", "vh_linear_gamma0")])
        if np.any(mask & (~np.isfinite(power).all(axis=0) | (power <= 0).any(axis=0))):
            raise ValueError("Valid radar samples require finite positive gamma-naught power")
        sample = {"vv": scene["vv_linear_gamma0"], "vh": scene["vh_linear_gamma0"],
                  "valid_mask": scene["valid_mask"]}
        sample_hash = _digest(sample)
        if scene.get("sample_sha256") and scene["sample_sha256"] != sample_hash:
            raise ValueError("Radar sample hash changed")
        valid_fraction = float(mask.mean())
        db = np.zeros_like(power)
        db[:, mask] = 10 * np.log10(power[:, mask])
        records.append({"id": scene["id"], "acquired": acquired, "values": db.astype(np.float32),
                        "mask": mask, "valid_fraction": valid_fraction,
                        "sample_sha256": sample_hash,
                        "source_metadata_sha256": scene.get("source_metadata_sha256")})
    records.sort(key=lambda row: (row["acquired"], row["id"]))
    for index, observed in enumerate(dates):
        eligible = [row for row in records if 0 <= (observed - row["acquired"]).total_seconds() / 86400 <= max_age_days
                    and row["mask"].any() and row["valid_fraction"] >= minimum_valid_fraction]
        match = {"optical_acquired_at": observed.isoformat(), "scene_id": None,
                 "reason": "No sufficiently valid past radar scene within the configured age"}
        if eligible:
            selected = eligible[-1]
            age = (observed - selected["acquired"]).total_seconds() / 86400
            arrays["sar"][index] = selected["values"]
            arrays["sar_mask"][index] = True
            arrays["sar_quality_mask"][index] = selected["mask"]
            arrays["sar_acquisition_dates"][index] = selected["acquired"].isoformat()
            arrays["sar_age_days"][index] = age
            match.update(scene_id=selected["id"], acquired_at=selected["acquired"].isoformat(),
                         age_days=age, valid_fraction=selected["valid_fraction"],
                         sample_sha256=selected["sample_sha256"],
                         source_metadata_sha256=selected["source_metadata_sha256"], reason=None)
        evidence["matches"].append(match)
    evidence.update(present=bool(arrays["sar_mask"].any()), matched_steps=int(arrays["sar_mask"].sum()),
                    orbit_group=list(group) if group else None)
    return arrays, evidence


def _align_climate(context, dates, bbox, window_days, lag_days, minimum_coverage):
    count, features = len(dates), len(CLIMATE_FEATURE_NAMES)
    arrays = {"climate_context": np.zeros((count, features), dtype=np.float32),
              "climate_mask": np.zeros((count, features), dtype=bool),
              "climate_coverage": np.zeros((count, features), dtype=np.float32),
              "climate_feature_names": np.asarray(CLIMATE_FEATURE_NAMES),
              "climate_feature_units": np.asarray(CLIMATE_FEATURE_UNITS)}
    evidence = {"present": False, "source_sha256": _digest(context), "window_days": window_days,
                "lag_days": lag_days, "minimum_coverage": minimum_coverage,
                "feature_names": list(CLIMATE_FEATURE_NAMES), "feature_units": list(CLIMATE_FEATURE_UNITS),
                "windows": [], "historical_availability_verified": False,
                "vintage_caveat": "Retrospective ERA5 reanalysis may have been revised after an historical cutoff; a fixed date lag does not establish historical-vintage availability"}
    if not isinstance(context, dict):
        raise ValueError("Climate context must be a JSON object")
    if context.get("available") is not True:
        evidence["reason"] = "Supplied climate context is unavailable; all climate inputs are masked"
        return arrays, evidence
    if context.get("synthetic", False) is not False or context.get("dataset") != "ERA5":
        raise ValueError("Climate alignment requires nonsynthetic ERA5 daily context")
    requested = context.get("location_requested")
    centroid = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    if (not isinstance(requested, list) or len(requested) != 2 or
            not np.allclose(requested, centroid, rtol=0, atol=1e-8)):
        raise ValueError("Climate context requested location differs from the optical AOI centroid")
    response = context.get("raw_response")
    if not isinstance(response, dict):
        raise ValueError("Climate model inputs require saved daily raw_response, not monthly aggregates")
    if response.get("utc_offset_seconds") != 0:
        raise ValueError("Climate daily observations must use UTC/GMT")
    daily, units = response.get("daily", {}), response.get("daily_units", {})
    times = daily.get("time")
    if not isinstance(times, list) or not 1 <= len(times) <= 3660:
        raise ValueError("Climate context requires 1–3660 dated daily rows")
    parsed = [_day(value, "Climate observation date") for value in times]
    if len(set(parsed)) != len(parsed):
        raise ValueError("Duplicate climate dates")
    start, end = _day(context.get("start"), "Climate start"), _day(context.get("end"), "Climate end")
    if start > end or any(not start <= day <= end for day in parsed):
        raise ValueError("Climate dates lie outside their declared collection window")
    values = []
    for variable, expected in zip(CLIMATE_SOURCE_VARIABLES, CLIMATE_SOURCE_UNITS):
        if units.get(variable) != expected:
            raise ValueError(f"Climate unit mismatch for {variable}: expected {expected}")
        source = daily.get(variable)
        if not isinstance(source, list) or len(source) != len(parsed):
            raise ValueError("Climate dates and measurement counts disagree")
        if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value)) for value in source):
            raise ValueError("Climate measurements must be finite numbers or nulls")
        if variable in {"precipitation_sum", "et0_fao_evapotranspiration"} and any(value is not None and value < 0 for value in source):
            raise ValueError("Climate precipitation and reference ET0 cannot be negative")
        if variable == "soil_moisture_0_to_7cm_mean" and any(value is not None and not 0 <= value <= 1 for value in source):
            raise ValueError("Volumetric soil moisture must lie in [0, 1]")
        values.append(source)
    for index, observed in enumerate(dates):
        last = observed.date() - timedelta(days=lag_days)
        first = last - timedelta(days=window_days - 1)
        selected = [i for i, day in enumerate(parsed) if first <= day <= last]
        valid_days = []
        for feature, source in enumerate(values):
            measured = [source[i] for i in selected if source[i] is not None]
            coverage = len(measured) / window_days
            valid_days.append(len(measured))
            arrays["climate_coverage"][index, feature] = coverage
            if measured and coverage >= minimum_coverage:
                aggregate = sum(measured) if feature in {1, 2} else sum(measured) / len(measured)
                if not np.isfinite(np.float32(aggregate)):
                    raise ValueError("Climate aggregate cannot be represented as a finite float32 input")
                arrays["climate_context"][index, feature] = aggregate
                arrays["climate_mask"][index, feature] = True
        evidence["windows"].append({"optical_acquired_at": observed.isoformat(),
            "start": first.isoformat(), "end": last.isoformat(), "requested_days": window_days,
            "valid_days": valid_days, "mask": arrays["climate_mask"][index].tolist()})
    evidence.update(present=bool(arrays["climate_mask"].any()),
                    matched_steps=int(arrays["climate_mask"].any(axis=1).sum()),
                    raw_response_sha256=_digest(response),
                    provider_response_sha256=context.get("response_sha256"))
    return arrays, evidence


def align_context(*, acquisition_dates, bbox, grid_shape, radar_context=None, climate_context=None,
                  radar_max_age_days=12, radar_min_valid_fraction=.4,
                  climate_window_days=30, climate_lag_days=5, climate_min_coverage=1.0):
    """Return additional NPZ arrays and a reproducible, explicitly timed manifest.

    Missing values are stored as zero with false masks, never treated as zero
    observations. Radar and optical use independent pixel masks. Climate sums
    with deliberately relaxed coverage thresholds remain partial, not scaled up.
    """
    radar_max_age_days = _number(radar_max_age_days, "radar_max_age_days", 0, 180)
    radar_min_valid_fraction = _number(radar_min_valid_fraction, "radar_min_valid_fraction", 0, 1)
    climate_window_days = _integer(climate_window_days, "climate_window_days", 1, 366)
    # Full-day aggregates on the optical acquisition date would contain future hours.
    climate_lag_days = _integer(climate_lag_days, "climate_lag_days", 1, 365)
    climate_min_coverage = _number(climate_min_coverage, "climate_min_coverage", 0, 1)
    dates = [_utc(value, "Optical acquisition") for value in acquisition_dates]
    if not 1 <= len(dates) <= 600 or any(right <= left for left, right in zip(dates, dates[1:])):
        raise ValueError("Optical acquisition timestamps must be nonempty, unique and increasing")
    bbox = validate_bbox(bbox)
    shape = tuple(grid_shape)
    if len(shape) != 2 or any(type(value) is not int or not 1 <= value <= 4096 for value in shape):
        raise ValueError("Optical grid_shape must contain two positive bounded integers")
    arrays, manifest = {}, {"version": ALIGNMENT_VERSION, "available_modalities": ["optical"],
        "parameters": {"radar_max_age_days": radar_max_age_days,
                       "radar_min_valid_fraction": radar_min_valid_fraction,
                       "climate_window_days": climate_window_days, "climate_lag_days": climate_lag_days,
                       "climate_min_coverage": climate_min_coverage},
        "limitations": ["Alignment is not evidence of a trained multimodal model or ecological predictive accuracy.",
                        "Sensor observation dates are enforced; exact historical product publication/vintage is not verified.",
                        "Radar gamma0 RTC is not interchangeable with GRD/sigma0 checkpoint inputs.",
                        "Climate is a coarse reanalysis at the AOI centroid; forest-specific measurements and causal attribution are not inferred."]}
    if radar_context is not None:
        radar_arrays, radar_evidence = _align_radar(radar_context, dates, bbox, shape,
                                                   radar_max_age_days, radar_min_valid_fraction)
        arrays.update(radar_arrays)
        manifest["radar"] = radar_evidence
        if radar_evidence["present"]:
            manifest["available_modalities"].append("sar")
    if climate_context is not None:
        climate_arrays, climate_evidence = _align_climate(climate_context, dates, bbox,
                                                         climate_window_days, climate_lag_days,
                                                         climate_min_coverage)
        arrays.update(climate_arrays)
        manifest["climate"] = climate_evidence
        if climate_evidence["present"]:
            manifest["available_modalities"].append("climate")
    manifest["sha256"] = _digest(manifest)
    arrays["alignment_sha256"] = np.asarray(manifest["sha256"])
    return arrays, manifest
