"""Join real observation sequences to independent, explicitly partial labels.

The converter audits partitions before writing anything. A missing label always
has a false task mask; its numeric placeholder is never a negative observation.
Passing this audit establishes data contracts, not label authenticity or model
accuracy. All normalization statistics are fitted on observed training inputs.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np
import yaml

from forestmind.monitoring.ingest import validate_bbox
from forestmind.training.reference_experiment import footprint_separation_lower_bound_km


DRIVERS = ("agriculture", "cattle_pasture", "mining", "logging", "road_construction",
           "wildfire", "drought", "urban_expansion")
STATES = ("healthy", "stressed", "degraded", "severely_degraded", "cleared")
TARGET_UNITS = {"health_state": "canonical_five_state", "health_score": "score_0_100",
                "drivers": "binary_presence", "driver_strengths": "strength_0_1",
                "trajectory": "condition_0_1"}
SPLITS = ("train", "val", "calibration", "test")
CLIMATE_NAMES = ["temperature_mean_c", "precipitation_sum_mm", "reference_et0_sum_mm",
                 "soil_moisture_mean_m3_m3"]
CLIMATE_UNITS = ["degC", "mm", "mm", "m3/m3"]
MAX_FILE_BYTES = 512 * 1024 * 1024


def _hash_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _date(value):
    if not isinstance(value, str):
        raise ValueError("Dates must use YYYY-MM-DD")
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD")
    return parsed


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} requires nonempty text")
    return value


def _number(value, low, high, field):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{field} must be a finite number in [{low}, {high}]")
    return value


def _scalar(arrays, name):
    if name not in arrays or arrays[name].shape != ():
        raise ValueError(f"Sequence requires scalar {name}")
    return arrays[name].item()


def _mask(arrays, name, shape):
    value = arrays.get(name)
    if value is None or value.shape != shape or value.dtype.kind != "b":
        raise ValueError(f"Sequence {name} must be a boolean array with shape {shape}")
    return value


def _read_sequence(path, example, time_steps, grid_shape):
    """Validate an unpadded or previously padded, raw observation sequence."""
    if path.suffix.lower() != ".npz" or not path.is_file():
        raise ValueError("sequence must point to an existing .npz file")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 128 or sum(entry.file_size for entry in entries) > MAX_FILE_BYTES:
            raise ValueError("Sequence exceeds 128 arrays / 512 MiB uncompressed")
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError("Sequence contains duplicate array names")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if _scalar(arrays, "synthetic") is not False or _scalar(arrays, "normalization") != "raw":
        raise ValueError("Training references require explicit nonsynthetic raw observations")
    if any(value.dtype.kind not in "bifuUS" for value in arrays.values()):
        raise ValueError("Sequence arrays must be numeric, boolean or text; objects are forbidden")
    if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "fiu"):
        raise ValueError("Sequence must preserve missing values with masks, not nonfinite numbers")
    optical, sar = arrays.get("optical"), arrays.get("sar")
    if optical is None or optical.ndim != 4 or optical.shape[1:] != (12, *grid_shape):
        raise ValueError("Optical sequence must match 12 bands and the declared grid_shape")
    steps = len(optical)
    if steps < 1 or steps > time_steps or sar is None or sar.shape != (steps, 2, *grid_shape):
        raise ValueError("Sequence exceeds time_steps or SAR/grid dimensions do not match; no scenes are dropped")
    observed = _mask(arrays, "observation_mask", (steps,))
    optical_mask = _mask(arrays, "optical_mask", (steps,))
    sar_mask = _mask(arrays, "sar_mask", (steps,))
    quality = _mask(arrays, "quality_mask", (steps, *grid_shape))
    if not observed.any() or not np.array_equal(observed, optical_mask | sar_mask):
        raise ValueError("observation_mask must equal the union of actual modality masks")
    if np.any(np.diff(observed.astype(int)) > 0):
        raise ValueError("Only trailing temporal padding is supported")
    if (quality[~optical_mask]).any() or any(not quality[i].any() for i in np.flatnonzero(optical_mask)):
        raise ValueError("Optical quality pixels and modality availability disagree")
    valid_values = optical.transpose(0, 2, 3, 1)[quality]
    if valid_values.size and (valid_values.min() < 0 or valid_values.max() > 1):
        raise ValueError("Observed optical data must use surface reflectance [0,1]")
    if sar_mask.any():
        sar_quality = _mask(arrays, "sar_quality_mask", (steps, *grid_shape))
        if sar_quality[~sar_mask].any() or any(not sar_quality[i].any() for i in np.flatnonzero(sar_mask)):
            raise ValueError("SAR quality pixels and modality availability disagree")
        if _scalar(arrays, "sar_units") not in {"gamma0_db", "sigma0_db", "gamma0_linear", "sigma0_linear"}:
            raise ValueError("Observed SAR requires explicit physical sar_units")
    else:
        arrays["sar_quality_mask"] = np.zeros_like(quality)
    actual_dates = arrays.get("acquisition_dates")
    if actual_dates is None or actual_dates.shape != (steps,) or actual_dates.dtype.kind not in "US":
        raise ValueError("Actual acquisition_dates are required for feature cutoff auditing")
    parsed = []
    for item in actual_dates.astype(str):
        instant = datetime.fromisoformat(item.replace("Z", "+00:00"))
        if "T" in item and instant.utcoffset() is None:
            raise ValueError("Acquisition timestamps need a timezone")
        parsed.append(instant.replace(tzinfo=timezone.utc) if instant.utcoffset() is None
                      else instant.astimezone(timezone.utc))
    actual = [instant for instant, present in zip(parsed, observed) if present]
    if any(a >= b for a, b in zip(actual, actual[1:])):
        raise ValueError("Actual acquisition dates must strictly increase")
    timestamps = arrays.get("timestamps")
    expected = np.asarray([(instant - actual[0]).total_seconds() / 86400 for instant in parsed])
    if timestamps is None or timestamps.shape != (steps,) or not np.allclose(timestamps, expected, atol=1e-4, rtol=0):
        raise ValueError("timestamps must match acquisition_dates exactly")
    if any(instant != actual[-1] for instant, present in zip(parsed, observed) if not present):
        raise ValueError("Padding must repeat the last real date without advancing time")
    cutoff = _date(example["cutoff"])
    if actual[-1].date() > cutoff or _date(_scalar(arrays, "observed_through")) != actual[-1].date():
        raise ValueError("Future features or mismatched observed_through date")
    if sar_mask.any():
        radar_dates, radar_age = arrays.get("sar_acquisition_dates"), arrays.get("sar_age_days")
        if radar_dates is None or radar_dates.shape != (steps,) or radar_age is None or radar_age.shape != (steps,):
            raise ValueError("Observed SAR requires per-step acquisition dates and ages")
        for index in np.flatnonzero(sar_mask):
            value = str(radar_dates[index])
            instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if instant.utcoffset() is None:
                raise ValueError("SAR timestamps require an explicit timezone")
            age = (parsed[index] - instant.astimezone(timezone.utc)).total_seconds() / 86400
            if age < 0 or not np.isclose(age, radar_age[index], atol=1e-4, rtol=0):
                raise ValueError("SAR features must precede their optical timestep and retain the actual age")
    if arrays.get("bbox", np.asarray([])).shape != (4,) or not np.allclose(arrays["bbox"], example["bbox"], atol=1e-8, rtol=0):
        raise ValueError("Sequence bbox must match the labeled footprint exactly")
    if "climate_context" in arrays:
        context = arrays["climate_context"]
        climate_mask = _mask(arrays, "climate_mask", (steps, 4))
        if context.shape != (steps, 4) or climate_mask[~observed].any():
            raise ValueError("Climate context requires four explicit features at actual timesteps")
        if arrays.get("climate_feature_names", np.asarray([])).tolist() != CLIMATE_NAMES:
            raise ValueError("Climate feature names do not match the physical feature contract")
        if arrays.get("climate_feature_units", np.asarray([])).tolist() != CLIMATE_UNITS:
            raise ValueError("Climate feature units do not match the physical feature contract")
        coverage = arrays.get("climate_coverage")
        if coverage is None or coverage.shape != (steps, 4) or (coverage < 0).any() or (coverage > 1).any():
            raise ValueError("Climate coverage must contain a [0,1] fraction per feature and timestep")
    if "activity_context" in arrays:
        raise ValueError("Activity context needs a versioned physical feature contract before training")
    return arrays


def _label(entry, field, cutoff, *, horizon=None, max_age=0):
    if not isinstance(entry, dict) or set(entry) != {"value", "date", "source"}:
        raise ValueError(f"{field} requires value, date and independent source")
    when = _date(entry["date"])
    if horizon is not None:
        if (when - cutoff).days != horizon:
            raise ValueError("Trajectory target date must equal cutoff plus its exact horizon")
    elif not 0 <= (cutoff - when).days <= max_age:
        raise ValueError("Current label date is future or exceeds max_current_label_age_days")
    source = entry["source"]
    if not isinstance(source, dict) or source.get("independent") is not True:
        raise ValueError("Labels require independent=true; indicator-generated pseudo-labels are not references")
    for key in ("reference", "version", "uncertainty", "license"):
        _text(source.get(key), f"label source.{key}")
    available = _date(source.get("available_at"))
    if available < when:
        raise ValueError("Label source availability cannot predate its observed target")
    value = entry["value"]
    if field == "health_state":
        if value not in STATES:
            raise ValueError("health_state must name one of the five canonical states")
        value = STATES.index(value)
    else:
        value = _number(value, 0, 100 if field == "health_score" else 1, field)
        if field == "drivers" and value not in (0, 1):
            raise ValueError("Driver labels must be 0 or 1; omit unknown drivers")
    return value, available


def _labels(example, document):
    supplied = example.get("labels", {})
    if not isinstance(supplied, dict) or not supplied or set(supplied) - set(TARGET_UNITS):
        raise ValueError("Every example needs at least one supported observed target")
    horizons = document["horizons_days"]
    result = {"health_state": np.asarray(0, dtype=np.int64), "health_score": np.asarray(0., dtype=np.float32),
              "drivers": np.zeros(8, dtype=np.float32), "driver_strengths": np.zeros(8, dtype=np.float32),
              "trajectory": np.zeros(len(horizons), dtype=np.float32)}
    for key, value in list(result.items()):
        result[key + "_mask"] = np.zeros(value.shape, dtype=bool)
    latest = _date(example["cutoff"])
    for field, entries in supplied.items():
        if field not in document["target_spec"]:
            raise ValueError(f"Missing frozen target_spec definition for {field}")
        if field in {"health_state", "health_score"}:
            entries = {None: entries}
        elif not isinstance(entries, dict) or not entries:
            raise ValueError(f"{field} labels require a nonempty named mapping")
        for key, entry in entries.items():
            if field == "trajectory":
                if key not in {str(h) for h in horizons}:
                    raise ValueError("Trajectory key must be a configured horizon in days")
                index, horizon = horizons.index(int(key)), int(key)
            elif field in {"drivers", "driver_strengths"}:
                if key not in DRIVERS:
                    raise ValueError("Unknown canonical driver name")
                index, horizon = DRIVERS.index(key), None
            else:
                index, horizon = (), None
            value, available = _label(entry, field, _date(example["cutoff"]), horizon=horizon,
                                      max_age=document.get("max_current_label_age_days", 0))
            result[field][index] = value
            result[field + "_mask"][index] = True
            latest = max(latest, available)
    result.update(label_valid=np.asarray(True), trajectory_units=np.asarray("condition_0_1"))
    return result, latest


def audit_training_manifest(manifest_path: str | Path) -> tuple[dict, list[dict], dict]:
    """Read and audit a manifest and all sequences without writing training files."""
    path = Path(manifest_path).resolve()
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Training manifest exceeds 16 MiB")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1 or document.get("synthetic") is not False:
        raise ValueError("Require schema_version=1 and synthetic=false")
    for field in ("dataset_id", "version", "intended_use"):
        _text(document.get(field), field)
    time_steps, grid = document.get("time_steps"), document.get("grid_shape")
    if type(time_steps) is not int or not 1 <= time_steps <= 600:
        raise ValueError("time_steps must be an integer in [1,600]")
    if not isinstance(grid, list) or len(grid) != 2 or any(type(n) is not int or not 16 <= n <= 1024 for n in grid):
        raise ValueError("grid_shape must contain two integers from 16 to 1024")
    horizons = document.get("horizons_days")
    if (not isinstance(horizons, list) or not horizons or len(horizons) > 16
            or any(type(h) is not int or not 1 <= h <= 3653 for h in horizons)
            or horizons != sorted(set(horizons))):
        raise ValueError("horizons_days must contain sorted unique positive integers up to ten years")
    for name, default in (("max_current_label_age_days", 0), ("max_feature_age_days", 31)):
        value = document.get(name, default)
        if type(value) is not int or not 0 <= value <= 366:
            raise ValueError(f"{name} must be an integer in [0,366]")
    spec = document.get("target_spec")
    if not isinstance(spec, dict) or not spec or set(spec) - set(TARGET_UNITS):
        raise ValueError("target_spec must define the supported tasks and their scientific meaning")
    for key, definition in spec.items():
        if not isinstance(definition, dict) or definition.get("units") != TARGET_UNITS[key]:
            raise ValueError(f"{key} target units must be {TARGET_UNITS[key]}; canopy loss is not health")
        for name in ("definition", "reference_protocol"):
            _text(definition.get(name), f"target_spec.{key}.{name}")
    policy = document.get("split_policy", {})
    buffer = _number(policy.get("spatial_buffer_km"), 0.001, 20000, "spatial_buffer_km")
    embargo = policy.get("temporal_embargo_days")
    if type(embargo) is not int or embargo < 0 or policy.get("protected_test") is not True:
        raise ValueError("Declare nonnegative temporal_embargo_days and protected_test=true")
    examples = document.get("examples")
    if not isinstance(examples, list) or not 3 <= len(examples) <= 10000:
        raise ValueError("Provide 3–10000 labeled examples; minimum counts are software limits, not study adequacy")
    ids, regions, hashes, rows = set(), {}, {}, []
    loaded_bytes = 0
    counts = {split: 0 for split in SPLITS}
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("Every example must be an object")
        identity, split = example.get("id"), example.get("split")
        if (not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", identity)
                or identity.casefold() in ids or split not in SPLITS):
            raise ValueError("Examples need unique filename-safe IDs and train/val/calibration/test splits")
        if identity.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
            raise ValueError("Example ID is a reserved filename")
        ids.add(identity.casefold())
        region = _text(example.get("region_id"), "region_id")
        if region in regions and regions[region] != split:
            raise ValueError("Region identity crosses protected splits")
        regions[region] = split
        example["bbox"] = list(validate_bbox(example.get("bbox", [])))
        cutoff, available = _date(example.get("cutoff")), _date(example.get("features_available_at"))
        if available > cutoff:
            raise ValueError("Features are not available by the prediction cutoff")
        sequence_path = (path.parent / _text(example.get("sequence"), "sequence")).resolve()
        if not sequence_path.is_relative_to(path.parent):
            raise ValueError("Sequence paths must remain inside the manifest directory")
        sequence_hash = _hash_file(sequence_path)
        expected_hash = example.get("sequence_sha256")
        if expected_hash != sequence_hash:
            raise ValueError("sequence_sha256 must match the exact reviewed NPZ")
        if sequence_hash in hashes:
            raise ValueError("Duplicate sequence content across examples; avoid reweighting/reusing observations")
        hashes[sequence_hash] = identity
        arrays = _read_sequence(sequence_path, example, time_steps, tuple(grid))
        loaded_bytes += sum(value.nbytes for value in arrays.values())
        if loaded_bytes > 2 * 1024 * 1024 * 1024:
            raise ValueError("Local conversion exceeds 2 GiB of input arrays; partition the dataset")
        last = _date(_scalar(arrays, "observed_through"))
        if available < last or (cutoff - last).days > document.get("max_feature_age_days", 31):
            raise ValueError("Feature availability predates observations or the sequence is too stale")
        labels, latest = _labels(example, document)
        rows.append({"example": example, "path": sequence_path, "sha256": sequence_hash,
                     "arrays": arrays, "labels": labels, "latest_label_available": latest})
        counts[split] += 1
    if any(counts[split] < 1 for split in ("train", "val", "test")):
        raise ValueError("Separate nonempty train, val and test partitions are required")
    partitions = {split: [r for r in rows if r["example"]["split"] == split] for split in SPLITS if counts[split]}
    unique = {split: sorted({tuple(r["example"]["bbox"]) for r in values}) for split, values in partitions.items()}
    pairs = list(itertools.combinations(partitions, 2))
    if sum(len(unique[a]) * len(unique[b]) for a, b in pairs) > 1000000:
        raise ValueError("Spatial audit exceeds one million footprint comparisons")
    nearest = None
    for left, right in pairs:
        for a in unique[left]:
            for b in unique[right]:
                distance = footprint_separation_lower_bound_km(a, b)
                nearest = distance if nearest is None else min(nearest, distance)
                if distance < buffer:
                    raise ValueError(f"{left}/{right} footprints violate the spatial buffer")
        latest = max(r["latest_label_available"] for r in partitions[left])
        earliest = min(_date(r["example"]["cutoff"]) for r in partitions[right])
        if (earliest - latest).days <= embargo:
            raise ValueError(f"{left}/{right} labels cross the temporal embargo or protected cutoff")
    audit = {"passed": True, "counts": counts, "nearest_split_separation_lower_bound_km": nearest,
             "split_policy": policy, "label_counts": {split: {key: int(sum(r["labels"][key + "_mask"].sum() for r in values))
                                                    for key in TARGET_UNITS} for split, values in partitions.items()},
             "limitations": ["Independent source declarations and label accuracy still require review.",
                             "Small sample counts are not evidence of scientific sufficiency.",
                             "No model has been trained, calibrated or validated by this conversion."]}
    audit["label_counts_by_channel"] = {split: {
        "drivers": dict(zip(DRIVERS, np.sum([r["labels"]["drivers_mask"] for r in values], axis=0).astype(int).tolist())),
        "driver_strengths": dict(zip(DRIVERS, np.sum([r["labels"]["driver_strengths_mask"] for r in values], axis=0).astype(int).tolist())),
        "trajectory": dict(zip(map(str, horizons), np.sum([r["labels"]["trajectory_mask"] for r in values], axis=0).astype(int).tolist()))}
        for split, values in partitions.items()}
    trained = {key: np.any([r["labels"][key + "_mask"] for r in partitions["train"]], axis=0) for key in TARGET_UNITS}
    for row in rows:
        for key in TARGET_UNITS:
            if np.any(row["labels"][key + "_mask"] & ~trained[key]):
                raise ValueError(f"{key} has held-out labels without any matching training supervision")
    return document, rows, audit


def _normalization(rows):
    moments = {"optical": [np.zeros(12), np.zeros(12), np.zeros(12)],
               "sar": [np.zeros(2), np.zeros(2), np.zeros(2)]}
    climate = any("climate_context" in row["arrays"] for row in rows)
    if climate:
        moments["climate"] = [np.zeros(4), np.zeros(4), np.zeros(4)]
    sar_units = {str(row["arrays"]["sar_units"].item()) for row in rows if row["arrays"]["sar_mask"].any()}
    if len(sar_units) > 1:
        raise ValueError("SAR units must be identical across all examples")
    for row in rows:
        if row["example"]["split"] != "train":
            continue
        arrays = row["arrays"]
        for modality, quality in (("optical", "quality_mask"), ("sar", "sar_quality_mask")):
            valid = arrays[quality] & arrays[modality + "_mask"][:, None, None]
            values = arrays[modality].transpose(0, 2, 3, 1)[valid].astype(np.float64)
            count, total, square = moments[modality]
            count += len(values)
            total += values.sum(axis=0)
            square += np.square(values).sum(axis=0)
        if "climate_context" in arrays:
            values, valid = arrays["climate_context"].astype(np.float64), arrays["climate_mask"]
            count, total, square = moments["climate"]
            count += valid.sum(axis=0)
            total += np.where(valid, values, 0).sum(axis=0)
            square += np.where(valid, values * values, 0).sum(axis=0)
    normalization, evidence = {}, {}
    for name, (count, total, square) in moments.items():
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        variance = np.maximum(0., np.divide(square, count, out=np.zeros_like(square), where=count > 0) - mean * mean)
        std = np.where(count > 0, np.maximum(np.sqrt(variance), 1e-6), 1.)
        normalization[name + "_mean"], normalization[name + "_std"] = mean.tolist(), std.tolist()
        evidence[name] = {"observed_training_count": count.astype(int).tolist(),
                          "constant_observed_features": ((count > 0) & (variance < 1e-12)).tolist(),
                          "unobserved_training_features": (count == 0).tolist()}
        for row in rows:
            if name == "climate":
                observed = row["arrays"].get("climate_mask", np.zeros((1, 4), dtype=bool)).any(axis=0)
            else:
                observed = np.repeat(row["arrays"][name + "_mask"].any(), len(count))
            if np.any(observed & (count == 0)):
                raise ValueError(f"{name} is observed outside training without fitted training statistics")
    return normalization, evidence, next(iter(sar_units), "absent")


def _pad(arrays, document, climate):
    steps, total = len(arrays["optical"]), document["time_steps"]
    temporal = ("optical", "sar", "timestamps", "observation_mask", "optical_mask", "sar_mask",
                "quality_mask", "sar_quality_mask", "acquisition_dates", "sar_acquisition_dates",
                "sar_age_days", "climate_context", "climate_mask", "climate_coverage")
    # Canonical fields make missing-modality samples batch-compatible. The source
    # NPZ hash and original manifest retain all provenance not copied as tensors.
    metadata = ("synthetic", "normalization", "observed_through", "bbox")
    result = {key: value.copy() for key, value in arrays.items() if key in temporal + metadata}
    result.setdefault("sar_acquisition_dates", np.full(steps, "", dtype="U40"))
    result.setdefault("sar_age_days", np.full(steps, -1., dtype=np.float32))
    if climate and "climate_context" not in result:
        result.update(climate_context=np.zeros((steps, 4), dtype=np.float32),
                      climate_mask=np.zeros((steps, 4), dtype=bool),
                      climate_coverage=np.zeros((steps, 4), dtype=np.float32),
                      climate_feature_names=np.asarray(CLIMATE_NAMES), climate_feature_units=np.asarray(CLIMATE_UNITS))
    if climate:
        result["climate_feature_names"] = np.asarray(CLIMATE_NAMES)
        result["climate_feature_units"] = np.asarray(CLIMATE_UNITS)
    for key in temporal:
        if key not in result or steps == total:
            continue
        value = result[key]
        if value.shape[0] != steps:
            raise ValueError(f"Temporal array {key} does not match the sequence length")
        padded = np.zeros((total, *value.shape[1:]), dtype=value.dtype)
        padded[:steps] = value
        if key in {"timestamps", "acquisition_dates"}:
            padded[steps:] = value[-1]
        elif key == "sar_age_days":
            padded[steps:] = -1
        result[key] = padded
    # Zero only explicitly invalid pixels. Missingness remains represented by masks.
    result["optical"] = np.where(result["quality_mask"][:, None], result["optical"], 0)
    result["sar"] = np.where(result["sar_quality_mask"][:, None], result["sar"], 0)
    return result


def prepare_training_dataset(manifest_path: str | Path, output: str | Path) -> dict:
    """Audit and create a NEW dataset directory, config and reproducible report."""
    destination = Path(output).resolve()
    if destination.exists():
        raise ValueError("Output directory already exists; preserve datasets and choose a new version")
    document, rows, audit = audit_training_manifest(manifest_path)
    normalization, statistics, sar_units = _normalization(rows)
    climate = "climate_mean" in normalization
    manifest_hash = _hash_json(document)
    config = {
        "data": {"source": str(destination), "synthetic": False, "time_steps": document["time_steps"],
                 "patch_size": document["grid_shape"][0], "num_bands": 12, "sar_channels": 2,
                 "batch_size": min(4, audit["counts"]["train"]), "num_workers": 0,
                 "normalization": normalization, "target_spec": document["target_spec"],
                 "dataset_manifest_sha256": manifest_hash, "sar_units": sar_units},
        "foundation": {"name": "custom_vit", "embedding_dim": 64,
                       "custom_vit": {"patch_embed_size": 16, "depth": 2, "num_heads": 4, "mlp_ratio": 2., "drop_rate": 0.}},
        "fusion": {"method": "cross_attention", "num_heads": 4, "num_layers": 1, "dropout": 0.},
        "temporal": {"depth": 2, "num_heads": 4, "dim_feedforward": 128, "dropout": 0.,
                     "max_seq_len": document["time_steps"], "use_continuous_pe": True},
        "heads": {"health": {"num_classes": 5, "continuous_score": True},
                  "attribution": {"num_drivers": 8, "multi_label": True},
                  "trajectory": {"horizons": document["horizons_days"], "predict_uncertainty": True}},
        "training": {"epochs": 20, "learning_rate": 1e-4, "weight_decay": 1e-5,
                     "scheduler": "cosine", "warmup_epochs": 0, "gradient_clip": 1.,
                     "use_amp": False, "checkpoint_dir": str(destination / "checkpoints"), "save_every": 5},
        "device": "cpu", "seed": 42}
    if climate:
        config["data"].update(climate_feature_names=CLIMATE_NAMES, climate_feature_units=CLIMATE_UNITS)
        config["fusion"]["climate_dim"] = 4
    # Audit every source again immediately before writing to detect changed inputs.
    if any(_hash_file(row["path"]) != row["sha256"] for row in rows):
        raise ValueError("A reviewed source sequence changed during preparation")
    destination.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for row in rows:
        example = row["example"]
        directory = destination / example["split"]
        directory.mkdir(exist_ok=True)
        arrays = _pad(row["arrays"], document, climate)
        arrays.update(row["labels"], location_id=np.asarray(example["region_id"]),
                      example_id=np.asarray(example["id"]), cutoff=np.asarray(example["cutoff"]),
                      sar_units=np.asarray(sar_units), source_manifest_sha256=np.asarray(row["sha256"]),
                      reference_manifest_sha256=np.asarray(manifest_hash),
                      label_source_json=np.asarray(json.dumps(example["labels"], sort_keys=True, allow_nan=False)))
        path = directory / (example["id"] + ".npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        artifacts.append({"id": example["id"], "split": example["split"],
                          "path": path.relative_to(destination).as_posix(), "sha256": _hash_file(path),
                          "source_sha256": row["sha256"]})
    report = {"schema_version": 1, "kind": "forestmind_training_dataset", "dataset_id": document["dataset_id"],
              "version": document["version"], "synthetic": False, "manifest_sha256": manifest_hash,
              "target_spec": document["target_spec"], "audit": audit, "normalization": normalization,
              "normalization_evidence": statistics, "normalization_method": "Observed training values only; population std floor 1e-6; absent modalities use neutral 0/1 and false masks",
              "artifacts": artifacts, "implementation_sha256": _hash_file(__file__), "numpy_version": np.__version__,
              "scientifically_validated": False}
    for name, content in (("manifest.json", document), ("dataset-report.json", report)):
        with (destination / name).open("x", encoding="utf-8") as stream:
            json.dump(content, stream, indent=2, allow_nan=False)
            stream.write("\n")
    with (destination / "training-config.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return report
