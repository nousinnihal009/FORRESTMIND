"""Reproducible conventional baselines on independently supplied references.

This harness does not manufacture labels or infer that a declared independent
reference is scientifically valid. It enforces auditable temporal/spatial/data
contracts before comparing models on exactly the same protected examples.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_CONFIG = {"seed": 17, "ridge_alpha": 1., "forest_trees": 100, "forest_min_samples_leaf": 3,
                "boost_stages": 100, "boost_learning_rate": .05, "boost_max_depth": 2,
                "boost_min_samples_leaf": 3, "boost_subsample": 1., "boost_n_iter_no_change": None,
                "boost_continuous_loss": "squared_error", "boost_event_loss": "log_loss",
                "event_threshold": .5, "minimum_subgroup_size": 5, "interval_nominal_coverage": .9}


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def calendar(value):
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD")
    return result


def _distance_km(a, b):
    p1, p2 = math.radians(a[1]), math.radians(b[1])
    dp, dl = p2 - p1, math.radians(b[0] - a[0])
    h = math.sin(dp / 2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1., max(0., h))))


def footprint_separation_lower_bound_km(a, b):
    """Conservative spherical bounding-circle separation; never trusts centroids alone."""
    centres = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in (a, b)]
    radii = [max(_distance_km(c, p) for p in ((box[0], box[1]), (box[0], box[3]),
                                            (box[2], box[1]), (box[2], box[3]))) for box, c in zip((a, b), centres)]
    return max(0., _distance_km(*centres) - sum(radii))


def audit_references(document: dict) -> dict:
    if document.get("schema_version") != 1 or not document.get("dataset_id") or not document.get("version"):
        raise ValueError("Reference manifest requires schema_version=1, dataset_id and version")
    if document.get("synthetic") is not False:
        raise ValueError("Real reference experiments require explicit synthetic=false; fabricated labels cannot establish validation")
    target = document.get("target", {})
    if target.get("kind") not in {"continuous", "event"} or any(not target.get(k) for k in ("name", "units", "definition", "intended_use")):
        raise ValueError("Freeze a named continuous/event target, units, definition and intended_use")
    if type(target.get("horizon_days")) is not int or not 1 <= target["horizon_days"] <= 3653:
        raise ValueError("target.horizon_days must be a positive integer at most ten years")
    policy = document.get("split_policy", {})
    buffer = policy.get("spatial_buffer_km")
    embargo = policy.get("temporal_embargo_days")
    if (type(buffer) not in (int, float) or not math.isfinite(buffer) or buffer <= 0
            or type(embargo) is not int or embargo < 0 or policy.get("protected_test") is not True):
        raise ValueError("A positive spatial buffer, nonnegative temporal embargo and protected_test=true are required")
    features = document.get("feature_names", [])
    examples = document.get("examples", [])
    if (not isinstance(features, list) or not 1 <= len(features) <= 256
            or any(not isinstance(name, str) or not name for name in features) or len(set(features)) != len(features)):
        raise ValueError("Provide 1–256 unique feature names")
    if not 20 <= len(examples) <= 100000:
        raise ValueError("Reference dataset must contain 20–100000 explicitly labeled examples")
    ids, regions, footprints = set(), {}, {name: [] for name in ("train", "calibration", "test")}
    event_references = {}
    splits = {name: [] for name in footprints}
    for example in examples:
        identity = example.get("id")
        split = example.get("split")
        if not isinstance(identity, str) or not identity or identity in ids or split not in splits:
            raise ValueError("Every example requires a unique id and train/calibration/test split")
        ids.add(identity)
        region = example.get("region_id")
        if not isinstance(region, str) or not region:
            raise ValueError("Every example needs an explicit region_id")
        if region in regions and regions[region] != split:
            raise ValueError("Region identity crosses protected splits")
        regions[region] = split
        bbox = example.get("bbox", [])
        if (len(bbox) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox)
                or not -180 <= bbox[0] < bbox[2] <= 180 or not -90 < bbox[1] < bbox[3] < 90 or bbox[2] - bbox[0] >= 180):
            raise ValueError("Reference footprints must be valid WGS84 rectangles without antimeridian crossing")
        cutoff, observed = calendar(example["cutoff"]), calendar(example["features_available_at"])
        target_date = calendar(example["target_date"])
        if observed > cutoff or (target_date - cutoff).days != target["horizon_days"]:
            raise ValueError("Features must be available by cutoff; target_date must match the frozen horizon exactly")
        values = example.get("features", {})
        if set(values) != set(features) or any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in values.values()):
            raise ValueError("Every feature must have a finite numeric value or explicit null")
        label = example.get("label_source", {})
        if (label.get("independent") is not True or any(not label.get(k) for k in ("reference", "version"))
                or "uncertainty" not in label or label["uncertainty"] is None):
            raise ValueError("Each outcome needs an independent label reference, version and declared uncertainty")
        outcome = example.get("target")
        if type(outcome) not in (int, float) or not math.isfinite(outcome) or (target["kind"] == "event" and outcome not in (0, 1)):
            raise ValueError("Outcomes must be finite; event labels must be 0 or 1")
        for field in ("persistence", "seasonal"):
            value = example.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or (target["kind"] == "event" and not 0 <= value <= 1):
                raise ValueError("Each identical example requires finite persistence and seasonal baselines (event scores in [0,1])")
            if calendar(example[field + "_available_at"]) > cutoff:
                raise ValueError("Baseline feature availability exceeds the prediction cutoff")
        if target["kind"] == "event" and outcome == 1:
            if not isinstance(example.get("event_id"), str) or not example["event_id"] or not cutoff < calendar(example["event_date"]) <= target_date:
                raise ValueError("Positive event references require event_id and event_date strictly after cutoff within horizon")
            reference = (region, example["event_date"], split)
            if example["event_id"] in event_references and event_references[example["event_id"]] != reference:
                raise ValueError("An event_id must identify one region, event date and protected split consistently")
            event_references[example["event_id"]] = reference
        if example.get("quality_group") not in {"high", "partial", "low"}:
            raise ValueError("Declare quality_group high, partial or low for subgroup reporting")
        if "season" in example and (not isinstance(example["season"], str) or not example["season"]):
            raise ValueError("Optional season labels must be nonempty strings")
        footprints[split].append(bbox)
        splits[split].append(example)
    if len(splits["train"]) < 10 or len(splits["calibration"]) < 5 or len(splits["test"]) < 5:
        raise ValueError("Need at least 10 training, 5 calibration and 5 protected test examples")
    if target["kind"] == "event" and any(len({r["target"] for r in splits[s]}) < 2 for s in ("train", "calibration")):
        raise ValueError("Event training/calibration partitions must each contain positive and negative outcomes")
    # Audit unique footprints: repeated dates at one site do not multiply work.
    unique = {key: sorted(set(map(tuple, values))) for key, values in footprints.items()}
    if sum(len(unique[a]) * len(unique[b]) for a, b in (("train", "calibration"), ("train", "test"), ("calibration", "test"))) > 1000000:
        raise ValueError("Spatial audit exceeds one million footprint comparisons; partition the study or pre-audit spatial blocks")
    nearest = None
    for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        for a in unique[left]:
            for b in unique[right]:
                separation = footprint_separation_lower_bound_km(a, b)
                nearest = separation if nearest is None else min(nearest, separation)
                if separation < buffer:
                    raise ValueError(f"Protected {left}/{right} footprints violate the {buffer} km spatial buffer")
        latest_target = max(calendar(r["target_date"]) for r in splits[left])
        earliest_cutoff = min(calendar(r["cutoff"]) for r in splits[right])
        if (earliest_cutoff - latest_target).days <= embargo:
            raise ValueError(f"Protected {left}/{right} temporal split violates target-window purging or embargo")
    return {"passed": True, "counts": {key: len(values) for key, values in splits.items()},
            "spatial_distance_method": "Conservative spherical bounding-circle separation", "nearest_split_separation_lower_bound_km": nearest,
            "policy": policy, "limitations": ["Manifest declarations and independent label quality require human/source review.",
                                               "Availability dates cannot prove an upstream feature builder avoided future information.",
                                               "Protected-test custody and preregistration cannot be authenticated by a local hash."]}


def freeze_protocol(document: dict) -> dict:
    import sklearn
    audit = audit_references(document)
    protocol = {"schema_version": 1, "kind": "forestmind_frozen_reference_protocol", "dataset_sha256": content_hash(document),
                "target": document["target"], "split_policy": document["split_policy"], "feature_names": document["feature_names"],
                "models": dict(MODEL_CONFIG), "audit": audit, "created_at": datetime.now(timezone.utc).isoformat(),
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "runtime": {"numpy": np.__version__, "scikit_learn": sklearn.__version__}}
    protocol["protocol_sha256"] = content_hash(protocol)
    return protocol


def _continuous_metrics(truth, prediction):
    return {"mae": float(mean_absolute_error(truth, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(truth, prediction))), "bias": float(np.mean(prediction - truth))}


def _event_metrics(rows, truth, scores):
    predicted = scores >= MODEL_CONFIG["event_threshold"]
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
    calibration = []
    for low in np.linspace(0, .8, 5):
        mask = (scores >= low) & (scores < low + .2 if low < .8 else scores <= 1)
        calibration.append({"lower": float(low), "upper": float(low + .2), "count": int(mask.sum()),
                            "mean_prediction": float(scores[mask].mean()) if mask.any() else None,
                            "observed_frequency": float(truth[mask].mean()) if mask.any() else None})
    events = {}
    exposure = {}
    for row, alert in zip(rows, predicted):
        exposure.setdefault(row["region_id"], []).append((calendar(row["cutoff"]).toordinal(), calendar(row["target_date"]).toordinal()))
        if row["target"]:
            event = events.setdefault(row["event_id"], {"detected": False, "lead_days": []})
            event["detected"] |= bool(alert)
            if alert:
                event["lead_days"].append((calendar(row["event_date"]) - calendar(row["cutoff"])).days)
    days = 0
    for intervals in exposure.values():
        end = -1
        for a, b in sorted(intervals):
            days += max(0, b - max(a, end))
            end = max(end, b)
    lead = [max(event["lead_days"]) for event in events.values() if event["lead_days"]]
    false_count = int(((truth == 0) & predicted).sum())
    return {"brier": float(brier_score_loss(truth, scores)),
            "pr_auc_average_precision": float(average_precision_score(truth, scores)) if truth.any() else None,
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "threshold": MODEL_CONFIG["event_threshold"], "reliability": calibration,
            "false_alert_origins": false_count, "false_alert_origins_per_region_year": false_count / (days / 365.25) if days else None,
            "unique_reference_events": len(events), "missed_reference_events": sum(not v["detected"] for v in events.values()),
            "mean_first_warning_lead_days": float(np.mean(lead)) if lead else None,
            "exposure_region_days": days,
            "alert_definition": "Threshold crossing per prediction origin; false-alert episodes are not deduplicated. True event detections group explicit event_id."}


def run_reference_experiment(document: dict, protocol: dict) -> dict:
    """Fit train-only models; calibrate on calibration; inspect protected test once."""
    import sklearn
    recorded = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if protocol.get("protocol_sha256") != content_hash(recorded) or protocol.get("dataset_sha256") != content_hash(document):
        raise ValueError("Dataset or frozen protocol changed; do not retune targets/labels after protected-test inspection")
    if protocol.get("models") != MODEL_CONFIG:
        raise ValueError("Model configuration differs from the frozen implementation")
    if (protocol.get("implementation_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            or protocol.get("runtime") != {"numpy": np.__version__, "scikit_learn": sklearn.__version__}):
        raise ValueError("Experiment code or numerical runtime differs from the frozen protocol; restore the pinned implementation/runtime before running")
    audit = audit_references(document)
    splits = {s: [r for r in document["examples"] if r["split"] == s] for s in ("train", "calibration", "test")}
    features = document["feature_names"]
    arrays = {s: np.array([[np.nan if row["features"][f] is None else row["features"][f] for f in features] for row in rows], dtype=float)
              for s, rows in splits.items()}
    if np.isnan(arrays["train"]).all(axis=0).any():
        raise ValueError("A feature is entirely missing in training; exclude it before freezing the protocol")
    medians = np.nanmedian(arrays["train"], axis=0)
    # Missingness flags let the model distinguish imputation from measured zeros.
    arrays = {s: np.concatenate([np.where(np.isnan(x), medians, x), np.isnan(x).astype(float)], axis=1) for s, x in arrays.items()}
    truth = {s: np.array([r["target"] for r in rows]) for s, rows in splits.items()}
    event = document["target"]["kind"] == "event"
    models = {"regularized_features": make_pipeline(StandardScaler(), LogisticRegression(C=1., random_state=17, max_iter=1000) if event else Ridge(alpha=1.)),
              "random_forest_features": (RandomForestClassifier if event else RandomForestRegressor)(n_estimators=100, min_samples_leaf=3, random_state=17, n_jobs=1),
              "gradient_boosted_features": (GradientBoostingClassifier if event else GradientBoostingRegressor)(
                  loss=MODEL_CONFIG["boost_event_loss" if event else "boost_continuous_loss"],
                  n_estimators=MODEL_CONFIG["boost_stages"], learning_rate=MODEL_CONFIG["boost_learning_rate"],
                  max_depth=MODEL_CONFIG["boost_max_depth"], min_samples_leaf=MODEL_CONFIG["boost_min_samples_leaf"],
                  subsample=MODEL_CONFIG["boost_subsample"], random_state=MODEL_CONFIG["seed"],
                  n_iter_no_change=MODEL_CONFIG["boost_n_iter_no_change"])}
    predictions = {}
    for name in ("persistence", "seasonal"):
        predictions[name] = {s: np.array([r[name] for r in rows], dtype=float) for s, rows in splits.items() if s != "train"}
    for name, estimator in models.items():
        estimator.fit(arrays["train"], truth["train"])
        predictions[name] = {s: estimator.predict_proba(arrays[s])[:, 1] if event else estimator.predict(arrays[s]) for s in ("calibration", "test")}
    outputs = {}
    for name, values in predictions.items():
        calibrator = None
        if event:
            calibrator = LogisticRegression(C=1., random_state=17).fit(values["calibration"][:, None], truth["calibration"])
            scores = calibrator.predict_proba(values["test"][:, None])[:, 1]
            metrics = _event_metrics(splits["test"], truth["test"], scores)
            raw_metrics = _event_metrics(splits["test"], truth["test"], values["test"])
            calibration = {"method": "regularized sigmoid on calibration scores only", "coefficient": calibrator.coef_.tolist(), "intercept": calibrator.intercept_.tolist(), "count": len(splits["calibration"])}
            lower = upper = None
        else:
            scores = values["test"]
            metrics, raw_metrics = _continuous_metrics(truth["test"], scores), None
            residuals = truth["calibration"] - values["calibration"]
            offsets = np.quantile(residuals, [.05, .95])
            lower, upper = scores + offsets[0], scores + offsets[1]
            metrics["empirical_interval_coverage"] = float(((truth["test"] >= lower) & (truth["test"] <= upper)).mean())
            calibration = {"method": "calibration residual 5th/95th quantiles; not guaranteed coverage under spatial/temporal shift", "offsets": offsets.tolist(), "count": len(splits["calibration"])}
        groups = {}
        for key in ("region_id", "quality_group", "season"):
            labels = [row.get(key, "calendar_quarter_" + str((calendar(row["cutoff"]).month - 1) // 3 + 1)) for row in splits["test"]]
            groups[key] = []
            for label_value in sorted(set(labels)):
                mask = np.asarray([v == label_value for v in labels])
                count = int(mask.sum())
                subset = [row for row, selected in zip(splits["test"], mask) if selected]
                measured = (_event_metrics(subset, truth["test"][mask], scores[mask]) if event else _continuous_metrics(truth["test"][mask], scores[mask])) if count >= 5 else None
                groups[key].append({"group": label_value, "count": count, "metrics": measured, "reason": None if count >= 5 else "Fewer than five examples; subgroup estimate withheld"})
        outputs[name] = {"metrics": metrics, "raw_metrics": raw_metrics, "calibration": calibration, "subgroups": groups,
                         "examples": [{"id": row["id"], "region_id": row["region_id"], "cutoff": row["cutoff"], "target_date": row["target_date"],
                                       "target": float(target), "raw_prediction": float(raw), "prediction": float(prediction),
                                       "lower": float(lower[i]) if lower is not None else None, "upper": float(upper[i]) if upper is not None else None,
                                       "label_source": row["label_source"]}
                                      for i, (row, target, raw, prediction) in enumerate(zip(splits["test"], truth["test"], values["test"], scores))]}
    primary = "brier" if event else "mae"
    baseline = outputs["persistence"]["metrics"][primary]
    for output in outputs.values():
        output["improvement_over_persistence"] = baseline - output["metrics"][primary]
    return {"schema_version": 1, "kind": "forestmind_reference_baseline_experiment", "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_id": document["dataset_id"], "dataset_version": document["version"], "dataset_sha256": content_hash(document),
            "data_provenance": "user_declared_nonsynthetic_references_not_independently_authenticated",
            "protocol": protocol, "target": document["target"], "audit": audit, "models": outputs,
            "training_preprocessing": {"feature_names": features, "train_medians": medians.tolist(), "missingness_flags": True},
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
            "limitations": ["References are supplied and declared independent; this tool does not authenticate their quality or test-set custody.",
                            "Repeated origins and spatial/temporal dependence invalidate naive independent-sample significance claims; none are reported.",
                            "Training/calibration/test geography and periods differ; calibration may not transfer.",
                            "Negative improvements are retained. Do not change targets, thresholds or examples after reading protected-test results.",
                            "Pretrained, recurrent, multimodal and resilience-feature model comparisons require compatible actual predictions on these same reference IDs."]}


def write_new_json(value: dict, path: str | Path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")

