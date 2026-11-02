"""Review independently supplied annual loss references without inventing labels.

This is an intake/audit bridge, not an ecological labeler. Source hashes detect
changes; reviewer declarations do not authenticate the underlying measurements.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import shutil

from forestmind.training.reference_experiment import audit_references, content_hash

TARGET = {
    "id": "annual-gross-tree-cover-loss-v1",
    "kind": "continuous", "units": "fraction_0_1", "horizon_days": 365,
    "name": "Next-year gross tree-cover-loss fraction",
    "definition": "Area of independently reviewed stand-replacement tree-cover loss during the next full calendar year, divided by independently documented tree-covered area at the prediction cutoff, within a fixed study patch.",
    "intended_use": "Compare annual loss forecasts on protected places and later years; prioritize subsequent investigation.",
    "spatial_unit_ha": 100., "spatial_area_tolerance_fraction": .05,
    "minimum_baseline_forest_ha": 10.,
    "significant_loss_fraction": .10, "minimum_detectable_loss_ha": 1.,
    "maximum_reference_uncertainty_fraction": .05,
    "reference_protocol": "Independent dated baseline forest mask and loss-area assessment for the identical footprint/window; full baseline-mask coverage; recorded uncertainty and human review. Annual products retain annual time precision.",
    "forest_definition": "Trees taller than 5 metres and at least 30 percent canopy cover, documented by an independent baseline reference; plantations and natural forests must be distinguished in study metadata.",
}
LIMITATIONS = [
    "Gross tree-cover loss is a measurable disturbance target, not general ecosystem health, permanent deforestation, a causal driver or an irreversible tipping point.",
    "The 100 ha patch, 10 percent/1 ha criterion and 5 percent uncertainty ceiling are preregistered engineering study choices, not universal ecological thresholds or sensor accuracy guarantees.",
    "A human review declaration and file hash do not independently authenticate source quality, test custody or independence from satellite inputs.",
    "Annual references cannot establish an exact event day or day-level warning lead time. This converter exports a continuous target only.",
]
MAX_JSON_BYTES = 16 * 1024 * 1024


def _number(value, low, high, field):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{field} must be a finite number in [{low}, {high}]")
    return float(value)


def _text(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError(f"{field} requires nonempty text (at most 2000 characters)")
    return value


def _date(value):
    if not isinstance(value, str):
        raise ValueError("Dates must use YYYY-MM-DD")
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD")
    return result


def _read_json(path):
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Reference JSON exceeds 16 MiB")
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _evidence(source, root, cache):
    for key in ("reference", "version", "license", "method", "uncertainty", "independence_basis"):
        _text(source.get(key), f"source.{key}")
    if source.get("independent") is not True:
        raise ValueError("Reference source requires independent=true and a reviewed independence_basis")
    relative = Path(_text(source.get("file"), "source.file"))
    resolved = (root / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("Evidence file must exist inside the reference manifest directory")
    expected = source.get("sha256")
    stat = resolved.stat()
    fingerprint = (stat.st_size, stat.st_mtime_ns)
    if resolved in cache:
        if cache[resolved][0] != fingerprint:
            raise ValueError("Reference evidence changed during the audit")
        actual = cache[resolved][1]
    else:
        if stat.st_size > 512 * 1024 * 1024 or sum(value[0][0] for value in cache.values()) + stat.st_size > 2 * 1024**3:
            raise ValueError("Reference evidence exceeds 512 MiB per file or 2 GiB per intake")
        actual = _file_hash(resolved)
        cache[resolved] = (fingerprint, actual)
    if not isinstance(expected, str) or len(expected) != 64 or actual != expected:
        raise ValueError("Reference evidence SHA256 is missing or changed")
    _date(source.get("available_at"))


def _area_ha(bbox):
    if (not isinstance(bbox, list) or len(bbox) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox)
            or not -180 <= bbox[0] < bbox[2] <= 180 or not -85 <= bbox[1] < bbox[3] <= 85
            or bbox[2] - bbox[0] >= 180):
        raise ValueError("Reference bbox must be a WGS84 rectangle within 85 degrees latitude")
    return (6371008.8 ** 2 * math.radians(bbox[2] - bbox[0])
            * (math.sin(math.radians(bbox[3])) - math.sin(math.radians(bbox[1]))) / 10000)


def validate_target(target):
    """Only the predeclared annual protocol is accepted; new targets need a new version."""
    expected = deepcopy(TARGET)
    # Leap-year horizons must be frozen in a separate study, never rounded to 365.
    if target.get("horizon_days") not in (365, 366) or type(target.get("horizon_days")) is not int:
        raise ValueError("Annual target horizon must be exactly 365 or 366 days")
    expected["horizon_days"] = target["horizon_days"]
    if target != expected:
        raise ValueError("Target differs from the versioned annual loss protocol; do not silently change its definition or thresholds")


def reference_template():
    """An empty intake document, deliberately unable to train a model."""
    return {"schema_version": 1, "kind": "forestmind_reference_review", "dataset_id": "my-annual-loss-study",
            "version": "v1", "synthetic": False, "target": deepcopy(TARGET),
            "split_policy": {"spatial_buffer_km": 10., "temporal_embargo_days": 30, "protected_test": True},
            "feature_names": ["ndvi_mean", "ndmi_mean", "nbr_mean"], "examples": []}


def annual_reference_baselines(history, *, cutoff, horizon_days):
    """Compute persistence and a least-squares annual trend on past references only.

    Each record is a loss fraction for a complete previous calendar year. For
    this annual target, seasonal naive equals the latest annual observation.
    Publication lag is respected. Three consecutive available years are needed
    for the trend; an absent history never becomes a zero-loss prediction.
    """
    current = _date(cutoff)
    if not isinstance(history, list) or not 3 <= len(history) <= 50:
        raise ValueError("Baseline history needs 3–50 consecutive annual reference records")
    if type(horizon_days) is not int or horizon_days not in (365, 366):
        raise ValueError("Baseline horizon must be 365 or 366 days")
    years, values, availability = [], [], []
    for row in history:
        end, available = _date(row.get("period_end")), _date(row.get("available_at"))
        if (end.month, end.day) != (12, 31) or not end <= available <= current:
            raise ValueError("Past annual labels and their publication must be available by cutoff")
        if years and end.year != years[-1] + 1:
            raise ValueError("Baseline history must be ordered and consecutive without duplicate years")
        years.append(end.year)
        values.append(_number(row.get("loss_fraction"), 0, 1, "historical loss_fraction"))
        availability.append(available)
    if current.year - years[-1] > 2:
        raise ValueError("Annual baseline history is more than two years stale")
    mean_year, mean_value = sum(years) / len(years), sum(values) / len(values)
    slope = sum((year - mean_year) * (value - mean_value) for year, value in zip(years, values)) / sum((year - mean_year)**2 for year in years)
    future_year = (current + timedelta(days=horizon_days)).year
    trend = max(0., min(1., mean_value + slope * (future_year - mean_year)))
    available = max(availability).isoformat()
    return {"persistence": values[-1], "seasonal": values[-1], "historical_trend": trend,
            "persistence_available_at": available, "seasonal_available_at": available,
            "historical_trend_available_at": available,
            "baseline_method": "annual-reference-persistence-and-ordinary-least-squares-v1",
            "baseline_history_sha256": content_hash(history)}


def review_references(manifest_path):
    path = Path(manifest_path).resolve()
    document = _read_json(path)
    if (document.get("schema_version") != 1 or document.get("kind") != "forestmind_reference_review"
            or document.get("synthetic") is not False):
        raise ValueError("Expected nonsynthetic forestmind_reference_review schema_version=1")
    for key in ("dataset_id", "version"):
        _text(document.get(key), key)
    validate_target(document.get("target", {}))
    target = document["target"]
    features = document.get("feature_names")
    if (not isinstance(features, list) or not 1 <= len(features) <= 256
            or any(not isinstance(f, str) or not f for f in features) or len(set(features)) != len(features)):
        raise ValueError("Declare 1–256 unique feature_names before intake")
    rows = document.get("examples")
    if not isinstance(rows, list) or len(rows) > 10000:
        raise ValueError("Reference intake accepts a list of at most 10000 examples")
    decisions, accepted, identities, evidence_cache = [], [], set(), {}
    for row in rows:
        row = deepcopy(row)
        identity = _text(row.get("id"), "example.id")
        if identity in identities:
            raise ValueError("Reference example IDs must be unique")
        identities.add(identity)
        _text(row.get("region_id"), "region_id")
        if row.get("split") not in {"train", "val", "calibration", "test"}:
            raise ValueError("Declare train/val/calibration/test before reference review")
        area = _area_ha(row.get("bbox"))
        if abs(area / target["spatial_unit_ha"] - 1) > target["spatial_area_tolerance_fraction"]:
            raise ValueError("Footprint must be a fixed 100 ha study patch within 5 percent area tolerance")
        cutoff, end = _date(row.get("cutoff")), _date(row.get("target_date"))
        if (cutoff.month, cutoff.day) != (12, 31) or end != date(cutoff.year + 1, 12, 31):
            raise ValueError("Annual references require a December 31 cutoff and the next December 31 target")
        if (end - cutoff).days != target["horizon_days"]:
            raise ValueError("Annual target window disagrees with the frozen horizon; separate leap-year studies")
        if _date(row.get("features_available_at")) > cutoff:
            raise ValueError("Features must be available by the prediction cutoff")
        if "baseline_history" in row:
            computed = annual_reference_baselines(row["baseline_history"], cutoff=row["cutoff"], horizon_days=target["horizon_days"])
            for historic in row["baseline_history"]:
                _evidence(historic.get("source", {}), path.parent, evidence_cache)
                if historic["source"]["available_at"] != historic.get("available_at"):
                    raise ValueError("Baseline reference availability disagrees with its source")
            for key in ("persistence", "seasonal", "historical_trend"):
                if key in row and (row[key] != computed[key] or row.get(key + "_available_at") != computed[key + "_available_at"]):
                    raise ValueError("Supplied baseline disagrees with the pinned historical references")
            row.update(computed)
        values = row.get("features", {})
        if not isinstance(values, dict) or set(values) != set(features):
            raise ValueError("Feature keys must exactly match feature_names; use explicit null for missing values")
        for key, value in values.items():
            if value is not None:
                _number(value, -1e12, 1e12, f"features.{key}")
        for key in ("persistence", "seasonal", "historical_trend"):
            if key in row:
                _number(row[key], 0, 1, key)
                if _date(row.get(key + "_available_at")) > cutoff:
                    raise ValueError("Baseline availability exceeds the prediction cutoff")
        if row.get("quality_group") not in {"high", "partial", "low"}:
            raise ValueError("quality_group must be high, partial or low")
        _text(row.get("forest_type"), "forest_type")
        review = row.get("review", {})
        if review.get("status") not in {"pending", "accepted", "rejected"}:
            raise ValueError("Reference review status must be pending, accepted or rejected")
        reason, label, interval, significant = None, None, None, None
        reference = row.get("reference")
        if reference is None:
            if review["status"] == "accepted":
                raise ValueError("Accepted reviews require actual reference evidence")
            reason = "No independent reference assessment supplied"
        else:
            source = reference.get("source", {})
            _evidence(source, path.parent, evidence_cache)
            baseline_source = reference.get("baseline_source", {})
            _evidence(baseline_source, path.parent, evidence_cache)
            if reference.get("baseline_observed_at") != cutoff.isoformat():
                raise ValueError("Baseline forest reference must explicitly describe tree cover at the prediction cutoff")
            if _date(baseline_source["available_at"]) < cutoff:
                raise ValueError("Baseline reference availability cannot predate its observation")
            if reference.get("bbox") != row["bbox"]:
                raise ValueError("Reference geometry must exactly match the prediction footprint")
            if (reference.get("period_start") != (cutoff + timedelta(days=1)).isoformat()
                    or reference.get("period_end") != end.isoformat()):
                raise ValueError("Reference period must cover the exact future calendar year")
            if _date(source["available_at"]) < end:
                raise ValueError("A complete annual reference cannot be available before the target period ends")
            forest = _number(reference.get("baseline_forest_area_ha"), 0, area, "baseline_forest_area_ha")
            coverage = _number(reference.get("coverage_fraction"), 0, 1, "coverage_fraction")
            loss = _number(reference.get("loss_area_ha"), 0, forest, "loss_area_ha")
            uncertainty = _number(reference.get("uncertainty_area_ha"), 0, forest, "uncertainty_area_ha")
            if review["status"] in {"accepted", "rejected"}:
                for key in ("reviewer", "basis"):
                    _text(review.get(key), "review." + key)
                if _date(review.get("reviewed_at")) < max(_date(source["available_at"]), _date(baseline_source["available_at"])):
                    raise ValueError("Review date must not predate the reference availability")
            if review["status"] != "accepted":
                reason = "Reference is pending review" if review["status"] == "pending" else "Reference rejected by reviewer"
            elif forest < target["minimum_baseline_forest_ha"]:
                reason = "Insufficient independently documented baseline tree cover"
            elif coverage != 1.:
                reason = "Incomplete reference coverage; missing area is not absence of loss"
            elif uncertainty / forest > target["maximum_reference_uncertainty_fraction"]:
                reason = "Reference uncertainty exceeds the preregistered tolerance"
            else:
                low, high = max(0., loss - uncertainty), min(forest, loss + uncertainty)
                interval = [low / forest, high / forest]
                threshold = max(target["minimum_detectable_loss_ha"], forest * target["significant_loss_fraction"])
                if low >= threshold:
                    significant = True
                elif high < threshold:
                    significant = False
                # Threshold ambiguity affects the event interpretation only.
                # Dropping these valid continuous labels would select easy
                # outcomes and bias the continuous training/test distribution.
                label = loss / forest
        decision = {"id": identity, "split": row["split"], "status": "eligible" if label is not None else "excluded",
                    "classification": "uncertain" if significant is None else "positive" if significant else "negative",
                    "target": label, "reference_interval": interval, "reason": reason,
                    "classification_reason": "Reference interval straddles the significant-loss criterion" if label is not None and significant is None else None,
                    "review": deepcopy(review), "reference_sha256": content_hash(reference) if reference else None}
        decisions.append(decision)
        if label is not None:
            if any(key not in row for key in ("persistence", "seasonal")):
                raise ValueError("Eligible labels need actual dated persistence and seasonal baseline predictions")
            example = {key: deepcopy(row[key]) for key in (
                "id", "region_id", "bbox", "split", "cutoff", "features_available_at", "target_date", "features",
                "persistence", "persistence_available_at", "seasonal", "seasonal_available_at", "quality_group", "forest_type")}
            for key in ("historical_trend", "historical_trend_available_at", "season", "baseline_method", "baseline_history_sha256"):
                if key in row:
                    example[key] = deepcopy(row[key])
            example.update(target=label, reference_interval=interval, review=deepcopy(review),
                           baseline_source=deepcopy(baseline_source), baseline_observed_at=reference["baseline_observed_at"],
                           label_source={**deepcopy(reference["source"]),
                                         "source_available_at": reference["source"]["available_at"],
                                         "available_at": max(source["available_at"], baseline_source["available_at"], review["reviewed_at"]),
                                         "review_sha256": content_hash(decision)})
            accepted.append(example)
    for source_path, (fingerprint, _) in evidence_cache.items():
        stat = source_path.stat()
        if (stat.st_size, stat.st_mtime_ns) != fingerprint:
            raise ValueError("Reference evidence changed during the audit")
    export = {"schema_version": 1, "dataset_id": document["dataset_id"], "version": document["version"],
              "synthetic": False, "target": deepcopy(target), "split_policy": deepcopy(document.get("split_policy", {})),
              "feature_names": features, "examples": accepted, "review_manifest_sha256": content_hash(document),
              "evidence_root": str(path.parent),
              "evidence_files": {str(source_path.relative_to(path.parent)): hashed for source_path, (_, hashed) in evidence_cache.items()}}
    export["split_policy"]["required_splits"] = ["train", "val", "calibration", "test"] if any(row.get("split") == "val" for row in rows) else ["train", "calibration", "test"]
    export["required_baselines"] = ["persistence", "seasonal", "historical_trend"] if any("historical_trend" in row or "baseline_history" in row for row in rows) else ["persistence", "seasonal"]
    try:
        partition_audit = audit_references(export)
        ready, blockers = True, []
    except (ValueError, KeyError, TypeError) as error:
        partition_audit, ready, blockers = None, False, [str(error)]
    return {"schema_version": 1, "kind": "forestmind_reference_intake_report", "target": target,
            "source_manifest_path": str(path), "evidence_root": str(path.parent),
            "manifest_sha256": content_hash(document), "implementation_sha256": _file_hash(Path(__file__)),
            "scientifically_validated": False, "ready_for_baseline_experiment": ready,
            "counts": {"submitted": len(rows), "eligible": len(accepted), "excluded": len(rows) - len(accepted)},
            "class_counts": dict(Counter(d["classification"] for d in decisions)), "decisions": decisions,
            "partition_audit": partition_audit, "blockers": blockers, "limitations": list(LIMITATIONS)}, export


def prepare_reviewed_references(manifest_path, output_directory):
    """Write a new auditable packet; insufficient references never produce a training manifest."""
    report, document = review_references(manifest_path)
    source = _read_json(Path(manifest_path))
    if content_hash(source) != report["manifest_sha256"]:
        raise ValueError("Reference manifest changed during preparation")
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=False)
    report["output_manifest"] = "references.json" if report["ready_for_baseline_experiment"] else None
    (root / "source-manifest.json").write_text(json.dumps(source, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    evidence_root = Path(document["evidence_root"])
    mapping = {name: "evidence/" + digest for name, digest in document["evidence_files"].items()}
    if mapping:
        (root / "evidence").mkdir()
    for name, digest in document["evidence_files"].items():
        destination = root / mapping[name]
        if not destination.exists():
            shutil.copyfile(evidence_root / name, destination)
        if _file_hash(destination) != digest:
            raise ValueError("Reference evidence changed while creating the packet")
    document["evidence_files"] = {mapping[name]: digest for name, digest in document["evidence_files"].items()}
    document["evidence_root"] = str(root.resolve())
    for example in document["examples"]:
        for field in ("label_source", "baseline_source"):
            example[field]["file"] = mapping[example[field]["file"]]
    report["evidence_mapping"] = mapping
    report["packet_evidence_root"] = str(root.resolve())
    if report["ready_for_baseline_experiment"]:
        (root / "references.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report["output_manifest_sha256"] = content_hash(document)
    (root / "intake-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
