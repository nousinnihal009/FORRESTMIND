"""Retrospective point-reference experiment, separate from operational forecasts.

Published visual interpretations are real annotations but were compiled after
the historical origins. This benchmark never backdates their availability or
exports a deployable ForestMind model. The 100 ha annual-area protocol is intact.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.neighbors import BallTree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EARTH_KM = 6371.0088
FEATURES = ["latitude_sin", "latitude_cos", "longitude_sin", "longitude_cos",
            "past_forest_fraction", "loss_last_year", "past_loss_frequency",
            "past_gain_frequency", "forest_run_years"]
CONFIG = {
    "seed": 17, "block_degrees": 5, "boundary_margin_km": 5.1,
    "minimum_between_split_centres_km": 10.04, "lookback_years": 5,
    "cutoffs": {"train": list(range(2005, 2010)), "val": [2012],
                "calibration": [2015], "test": [2018, 2019]},
    "split_percent_edges": [60, 70, 85, 100],
    "logistic_c": 1., "forest_trees": 100, "forest_depth": 8,
    "forest_leaf": 10, "boost_stages": 100, "boost_depth": 2,
    "boost_leaf": 10, "boost_rate": .05, "threshold": .5,
    "bootstrap_replicates": 1000, "minimum_bootstrap_blocks": 5,
    "duplicate_coordinate_policy": "exclude_all_copies_before_label_inspection",
}
TARGET = {
    "id": "guo-retrospective-point-loss-next-calendar-year-v1",
    "definition": "An explicitly dated forest-loss event in the source interpretation during the next calendar year, at a sample interpreted as forest at the origin.",
    "kind": "binary_event", "resolution_m_declared_by_source": 30,
    "time_precision": "calendar_year", "area_fraction": False,
    "operational_forecast": False, "historical_availability_verified": False,
}
LIMITATIONS = [
    "A retrospective feasibility benchmark using annotations published in 2022; annotations were not available at historical forecast origins.",
    "Predictors are prior interpreted state/event histories and coordinates, not satellite imagery, climate, pretrained embeddings or the proposed multimodal architecture.",
    "The source's change-enriched sampling and exclusions do not represent population prevalence. Unweighted scores and calibration describe retained samples only.",
    "Ambiguous event years and states remain unknown. Requiring interpretable histories changes the evaluated population; exclusion counts are retained.",
    "Visual interpretation can use later imagery and other maps. This does not independently verify an operational early-warning system or eliminate annotation error.",
    "Point loss is not the primary 100 ha annual loss-area target, a five-state health label, a causal driver or a tipping point.",
    "Spatial buffers and cell bootstrap do not establish independence at every ecological scale. Intervals condition on fitted models and exclude training/calibration uncertainty.",
]


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _versions():
    return {"numpy": np.__version__, "scikit_learn": sklearn.__version__}


def _implementations():
    from forestmind.data import public_forest_reference
    paths = [Path(__file__), Path(public_forest_reference.__file__)]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def spatial_assignment(lon: float, lat: float):
    """Label-blind 5-degree cells with a gap on both sides of every boundary."""
    if (type(lon) not in (int, float) or type(lat) not in (int, float)
            or not math.isfinite(lon) or not math.isfinite(lat)
            or not -180 <= lon < 180 or not -90 < lat < 90):
        raise ValueError("Reference coordinates must be finite WGS84 points away from exact poles/180")
    width = CONFIG["block_degrees"]
    x, y = math.floor((lon + 180) / width), math.floor((lat + 90) / width)
    west, south = x * width - 180, y * width - 90
    delta_lon = min(lon - west, west + width - lon)
    delta_lat = min(lat - south, south + width - lat)
    # Distance to the meridian great-circle is conservative for the cell edge.
    meridian = EARTH_KM * math.asin(abs(math.cos(math.radians(lat)) * math.sin(math.radians(delta_lon))))
    parallel = EARTH_KM * math.radians(delta_lat)
    block = f"{x}:{y}"
    bucket = int(hashlib.sha256(f"forestmind-public-v1:{CONFIG['seed']}:{block}".encode()).hexdigest()[:16], 16) % 100
    split = next(s for s, edge in zip(CONFIG["cutoffs"], CONFIG["split_percent_edges"]) if bucket < edge)
    return block, split, min(meridian, parallel) >= CONFIG["boundary_margin_km"]


def past_features(sample: dict, cutoff: int):
    """Only dated values up to the origin are read; unknowns are never filled."""
    years = range(cutoff - CONFIG["lookback_years"] + 1, cutoff + 1)
    states = [sample["annual_state"].get(str(y)) for y in years]
    losses = [sample["annual_loss"].get(str(y)) for y in years]
    if any(type(v) is not bool for v in states) or any(type(v) is not int or v not in (0, 1) for v in losses):
        return None
    lat, lon = math.radians(sample["lat"]), math.radians(sample["lon"])
    gains = sum(not a and b for a, b in zip(states[:-1], states[1:]))
    run = 0
    for year in range(cutoff, 1999, -1):
        if sample["annual_state"].get(str(year)) is not True:
            break
        run += 1
    return [math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon),
            sum(states) / len(states), losses[-1], sum(losses) / len(losses),
            gains / (len(states) - 1), run]


def _spatial_audit(rows):
    points = {}
    for row in rows:
        points.setdefault(row["split"], {})[row["sample_id"]] = (row["lat"], row["lon"])
    nearest = None
    splits = list(CONFIG["cutoffs"])
    for i, left in enumerate(splits):
        a = np.radians(list(points.get(left, {}).values()))
        if not len(a):
            continue
        for right in splits[i + 1:]:
            b = np.radians(list(points.get(right, {}).values()))
            if not len(b):
                continue
            distance = float(BallTree(b, metric="haversine").query(a, k=1)[0].min()) * EARTH_KM
            nearest = distance if nearest is None else min(nearest, distance)
    if nearest is not None and nearest < CONFIG["minimum_between_split_centres_km"]:
        raise ValueError("Reference locations violate the protected spatial buffer")
    return nearest


def build_examples(packet: dict):
    """Fixed geography/time split; source strata and future summaries are excluded."""
    if not isinstance(packet.get("samples"), list) or not 1 <= len(packet["samples"]) <= 20000:
        raise ValueError("Expected a bounded public reference sample list")
    rows, exclusions, ids = [], Counter(), set()
    positions = Counter((sample["lon"], sample["lat"]) for sample in packet["samples"])
    for sample in sorted(packet["samples"], key=lambda r: r["id"]):
        identity = sample.get("id")
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("Reference sample identities must be unique nonempty strings")
        ids.add(identity)
        block, split, interior = spatial_assignment(sample["lon"], sample["lat"])
        position = (sample["lon"], sample["lat"])
        if positions[position] > 1:
            exclusions["samples_at_duplicated_coordinates"] += 1
            continue
        if not interior:
            exclusions["samples_in_boundary_buffer"] += 1
            continue
        for cutoff in CONFIG["cutoffs"][split]:
            if sample["annual_state"].get(str(cutoff)) is not True:
                exclusions["origins_not_known_forest"] += 1
                continue
            features = past_features(sample, cutoff)
            if features is None:
                exclusions["origins_with_unknown_history"] += 1
                continue
            target = sample["annual_loss"].get(str(cutoff + 1))
            if type(target) is not int or target not in (0, 1):
                exclusions["origins_with_unknown_outcome"] += 1
                continue
            rows.append({"id": f"{identity}@{cutoff}", "sample_id": identity,
                         "block": block, "split": split, "lon": sample["lon"], "lat": sample["lat"],
                         "cutoff_year": cutoff, "target_year": cutoff + 1,
                         "features": features, "target": target})
    counts = {s: sum(r["split"] == s for r in rows) for s in CONFIG["cutoffs"]}
    # This audit deliberately reports no held-out label distribution at freeze.
    audit = {"source_samples": len(ids), "examples": len(rows), "split_counts": counts,
             "excluded": dict(exclusions), "nearest_between_split_centres_km": _spatial_audit(rows),
             "sample_ids_by_split": {s: sorted({r["sample_id"] for r in rows if r["split"] == s}) for s in counts}}
    return rows, audit


def freeze_public_protocol(packet: dict):
    rows, audit = build_examples(packet)
    if any(audit["split_counts"][s] < (20 if s == "train" else 5) for s in CONFIG["cutoffs"]):
        raise ValueError("Fixed partitions have insufficient interpretable examples; no outcome-based resplitting is allowed")
    protocol = {"schema_version": 1, "kind": "forestmind_retrospective_public_reference_protocol",
                "created_at": datetime.now(timezone.utc).isoformat(), "target": TARGET,
                "config": CONFIG, "features": FEATURES, "source_sha256": _hash(packet),
                "examples_sha256": _hash(rows), "implementation_sha256": _implementations(),
                "runtime": _versions(), "audit": audit, "limitations": LIMITATIONS}
    protocol["protocol_sha256"] = _hash(protocol)
    return protocol


def binary_metrics(truth, scores):
    truth, scores = np.asarray(truth), np.asarray(scores, dtype=float)
    if (truth.ndim != 1 or scores.shape != truth.shape or not len(truth)
            or not np.isin(truth, [0, 1]).all() or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Binary metrics require matched binary labels and finite probabilities")
    labels = scores >= CONFIG["threshold"]
    precision, recall, f1, support = precision_recall_fscore_support(truth, labels, labels=[0, 1], zero_division=0)
    bins = []
    edges = np.linspace(0, 1, 6)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        chosen = (scores >= a) & ((scores < b) if i < 4 else (scores <= b))
        bins.append({"lower": float(a), "upper": float(b), "count": int(chosen.sum()),
                     "mean_prediction": float(scores[chosen].mean()) if chosen.any() else None,
                     "observed_frequency": float(truth[chosen].mean()) if chosen.any() else None})
    return {"count": len(truth), "positive_count": int(truth.sum()),
            "brier": float(brier_score_loss(truth, scores)),
            "auroc": float(roc_auc_score(truth, scores)) if len(np.unique(truth)) == 2 else None,
            "average_precision": float(average_precision_score(truth, scores)) if truth.any() else None,
            "threshold": CONFIG["threshold"], "confusion_matrix": confusion_matrix(truth, labels, labels=[0, 1]).tolist(),
            "confusion_matrix_order": "rows=true, columns=predicted; classes=[0,1]",
            "per_class": [{"class": c, "precision": float(precision[c]), "recall": float(recall[c]),
                           "f1": float(f1[c]), "support": int(support[c])} for c in [0, 1]],
            "reliability": bins, "expected_calibration_error": sum(
                r["count"] * abs(r["mean_prediction"] - r["observed_frequency"]) for r in bins if r["count"]) / len(truth),
            "lead_time_days": None, "lead_time_reason": "Annual labels have no exact event day"}


def block_improvement(rows, baseline, scores):
    blocks = sorted({r["block"] for r in rows})
    result = {"available": False, "metric": "brier_improvement", "confidence_level": .95,
              "method": "paired-geographic-cell-percentile-bootstrap", "blocks": len(blocks),
              "replicates": CONFIG["bootstrap_replicates"], "seed": CONFIG["seed"],
              "lower": None, "upper": None}
    if len(blocks) < CONFIG["minimum_bootstrap_blocks"]:
        return {**result, "reason": "Fewer than five test cells"}
    truth = np.asarray([r["target"] for r in rows])
    differences = (np.asarray(baseline) - truth)**2 - (np.asarray(scores) - truth)**2
    block_index = {b: i for i, b in enumerate(blocks)}
    indices = np.asarray([block_index[r["block"]] for r in rows])
    sums, counts = np.bincount(indices, weights=differences), np.bincount(indices)
    generator = np.random.default_rng(CONFIG["seed"])
    values = []
    for _ in range(CONFIG["bootstrap_replicates"]):
        selected = generator.integers(0, len(blocks), len(blocks))
        values.append(float(sums[selected].sum() / counts[selected].sum()))
    low, high = np.quantile(values, [.025, .975])
    return {**result, "available": True, "reason": None, "lower": float(low), "upper": float(high)}


def _logits(probabilities):
    scores = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(scores / (1 - scores)).reshape(-1, 1)


def run_public_benchmark(packet: dict, protocol: dict):
    recorded = {k: v for k, v in protocol.items() if k != "protocol_sha256"}
    if protocol.get("protocol_sha256") != _hash(recorded):
        raise ValueError("Frozen protocol integrity mismatch")
    if (protocol.get("source_sha256") != _hash(packet) or protocol.get("implementation_sha256") != _implementations()
            or protocol.get("config") != CONFIG or protocol.get("features") != FEATURES
            or protocol.get("target") != TARGET or protocol.get("runtime") != _versions()):
        raise ValueError("Source, implementation, configuration or runtime differs from frozen study; do not refreeze inspected tests")
    rows, audit = build_examples(packet)
    if protocol.get("examples_sha256") != _hash(rows) or protocol.get("audit") != audit:
        raise ValueError("Protected example identities or geography changed")
    partitions = {s: [r for r in rows if r["split"] == s] for s in CONFIG["cutoffs"]}
    arrays = {s: np.asarray([r["features"] for r in part]) for s, part in partitions.items()}
    targets = {s: np.asarray([r["target"] for r in part]) for s, part in partitions.items()}
    if any(len(np.unique(targets[s])) != 2 for s in ["train", "calibration"]):
        raise ValueError("Fixed training and calibration partitions need both classes; preserve the failed protocol")
    train_x, train_y = arrays["train"], targets["train"]
    learners = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=CONFIG["logistic_c"], max_iter=2000, random_state=CONFIG["seed"])),
        "random_forest": RandomForestClassifier(n_estimators=CONFIG["forest_trees"], max_depth=CONFIG["forest_depth"],
                                                min_samples_leaf=CONFIG["forest_leaf"], random_state=CONFIG["seed"], n_jobs=1),
        "gradient_boosted": GradientBoostingClassifier(n_estimators=CONFIG["boost_stages"], learning_rate=CONFIG["boost_rate"],
                                                       max_depth=CONFIG["boost_depth"], min_samples_leaf=CONFIG["boost_leaf"], random_state=CONFIG["seed"]),
    }
    scores = {"no_new_loss": {s: np.zeros(len(part)) for s, part in partitions.items()},
              "last_year_event": {s: arrays[s][:, FEATURES.index("loss_last_year")] for s in partitions},
              "past_event_rate": {s: arrays[s][:, FEATURES.index("past_loss_frequency")] for s in partitions},
              "training_event_rate": {s: np.full(len(part), float(train_y.mean())) for s, part in partitions.items()}}
    fitted = {}
    for name, learner in learners.items():
        learner.fit(train_x, train_y)
        raw = {s: learner.predict_proba(x)[:, 1] for s, x in arrays.items()}
        calibration = LogisticRegression(C=1., max_iter=2000, random_state=CONFIG["seed"])
        calibration.fit(_logits(raw["calibration"]), targets["calibration"])
        scores[name] = raw
        scores[name + "_calibrated"] = {s: calibration.predict_proba(_logits(v))[:, 1] for s, v in raw.items()}
        fitted[name] = {"class": type(learner).__name__, "train_count": len(train_y),
                        "calibration_count": len(targets["calibration"]),
                        "calibration_coefficient": float(calibration.coef_[0, 0]),
                        "calibration_intercept": float(calibration.intercept_[0])}
    models = {}
    for name, values in scores.items():
        models[name] = {"test": binary_metrics(targets["test"], values["test"]),
                        "validation": binary_metrics(targets["val"], values["val"]),
                        "improvement_vs_no_new_loss": block_improvement(partitions["test"], scores["no_new_loss"]["test"], values["test"])}
    predictions = [{**{k: v for k, v in row.items() if k != "features"},
                    "scores": {name: float(values["test"][i]) for name, values in scores.items()}}
                   for i, row in enumerate(partitions["test"])]
    result = {"schema_version": 1, "kind": "forestmind_retrospective_public_reference_benchmark",
              "created_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol["protocol_sha256"],
              "source_sha256": protocol["source_sha256"], "implementation_sha256": _implementations(),
              "runtime": _versions(), "target": TARGET, "audit": audit, "models": models,
              "fitting": fitted, "test_predictions": predictions,
              "split_positive_counts": {s: int(v.sum()) for s, v in targets.items()},
              "validation_used_for_model_selection": False, "scientifically_validated_operational_model": False,
              "deployable_checkpoint": None, "limitations": LIMITATIONS}
    result["result_sha256"] = _hash(result)
    return result


def write_new_record(value: dict, destination):
    path = Path(destination)
    raw = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(raw)

