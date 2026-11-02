"""Fabricated inputs exercise intake contracts only; these are not field labels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from forestmind.data.reference_review import annual_reference_baselines, reference_template, review_references, prepare_reviewed_references
from forestmind.training.reference_experiment import freeze_protocol, run_reference_experiment


def study(tmp_path, *, validation=False):
    evidence = tmp_path / "controlled-evidence.txt"
    evidence.write_text("CONTROLLED SOFTWARE FIXTURE. No real observations or ecological validation.")
    document = reference_template()
    specs = [("train", 2016, -60., 12), ("calibration", 2020, 10., 6), ("test", 2022, 80., 6)]
    if validation:
        specs.insert(1, ("val", 2018, -20., 5))
    for split, year, lon, count in specs:
        for i in range(count):
            bbox = [lon + i * .02, 1., lon + i * .02 + .009, 1.009]
            cutoff, end = f"{year}-12-31", f"{year + 1}-12-31"
            document["examples"].append({
                "id": f"{split}-{i}", "region_id": f"{split}-block-{i}", "bbox": bbox,
                "split": split, "cutoff": cutoff, "target_date": end, "features_available_at": cutoff,
                "features": {"ndvi_mean": .7, "ndmi_mean": .4, "nbr_mean": None if i % 3 else .5},
                "persistence": .02, "seasonal": .03, "historical_trend": .04,
                "persistence_available_at": cutoff, "seasonal_available_at": cutoff,
                "historical_trend_available_at": cutoff, "quality_group": "high", "forest_type": "fixture-only",
                "reference": {"bbox": bbox, "period_start": f"{year+1}-01-01", "period_end": end,
                    "baseline_forest_area_ha": 80., "loss_area_ha": 0. if i % 2 else 20.,
                    "uncertainty_area_ha": .5, "coverage_fraction": 1.,
                    "source": {"independent": True, "reference": "CONTROLLED TEST FIXTURE",
                        "version": "fixture-v1", "license": "Test fixture", "method": "Fabricated fixture areas",
                        "uncertainty": "Fixture bounds only", "independence_basis": "Software fixture only",
                        "available_at": f"{year+2}-01-31", "file": evidence.name,
                        "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}},
                "review": {"status": "accepted", "reviewer": "test-fixture", "basis": "Software test only",
                           "reviewed_at": f"{year+2}-02-01"}})
    path = tmp_path / "intake.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def save(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")


def test_empty_intake_has_no_labels_and_no_training_manifest(tmp_path):
    path = tmp_path / "empty.json"
    save(path, reference_template())
    report = prepare_reviewed_references(path, tmp_path / "packet")
    assert not report["ready_for_baseline_experiment"]
    assert not report["scientifically_validated"]
    assert report["counts"] == {"submitted": 0, "eligible": 0, "excluded": 0}
    assert report["output_manifest"] is None
    assert not (tmp_path / "packet/references.json").exists()
    with pytest.raises(FileExistsError):
        prepare_reviewed_references(path, tmp_path / "packet")


def test_reviewed_continuous_labels_feed_identical_baseline_partitions(tmp_path):
    path, original = study(tmp_path, validation=True)
    report, document = review_references(path)
    assert report["ready_for_baseline_experiment"]
    assert report["partition_audit"]["counts"] == {"train": 12, "val": 5, "calibration": 6, "test": 6}
    assert {row["target"] for row in document["examples"]} == {0., .25}
    assert all("event_date" not in row for row in document["examples"])
    result = run_reference_experiment(document, freeze_protocol(document))
    assert "historical_trend" in result["models"]
    assert all([r["id"] for r in model["examples"]] == [f"test-{i}" for i in range(6)] for model in result["models"].values())
    assert all(model["improvement_interval"]["available"] for model in result["models"].values())
    assert json.loads(path.read_text()) == original
    prepare_reviewed_references(path, tmp_path / "prepared")
    assert json.loads((tmp_path / "prepared/references.json").read_text()) == document


@pytest.mark.parametrize("change,reason", [
    (lambda row: row.update(reference=None, review={"status": "pending"}), "No independent reference"),
    (lambda row: row["review"].update(status="pending"), "pending review"),
    (lambda row: row["review"].update(status="rejected"), "rejected"),
    (lambda row: row["reference"].update(coverage_fraction=.99), "Incomplete reference coverage"),
    (lambda row: row["reference"].update(uncertainty_area_ha=5.), "uncertainty exceeds"),
    (lambda row: row["reference"].update(loss_area_ha=8., uncertainty_area_ha=.5), "straddles"),
])
def test_unknown_incomplete_or_disputed_reference_never_becomes_negative(tmp_path, change, reason):
    path, doc = study(tmp_path)
    change(doc["examples"][0])
    save(path, doc)
    report, exported = review_references(path)
    decision = report["decisions"][0]
    assert decision["status"] == "excluded" and decision["classification"] == "uncertain"
    assert decision["target"] is None and reason in decision["reason"]
    assert "train-0" not in [row["id"] for row in exported["examples"]]


@pytest.mark.parametrize("change,match", [
    (lambda d: d["target"].update(significant_loss_fraction=.2), "versioned annual"),
    (lambda d: d["examples"][0].update(features_available_at="2025-01-01"), "Features must"),
    (lambda d: d["examples"][0]["reference"].update(bbox=[0, 1, .01, 1.01]), "geometry"),
    (lambda d: d["examples"][0]["reference"].update(period_start="2017-02-01"), "exact future"),
    (lambda d: d["examples"][0]["reference"]["source"].update(available_at="2017-01-01"), "before the target"),
    (lambda d: d["examples"][0]["reference"]["source"].update(independent=False), "independent=true"),
    (lambda d: d["examples"][0]["reference"]["source"].update(file="../outside.txt"), "inside"),
    (lambda d: d["examples"][0]["reference"]["source"].update(sha256="0" * 64), "SHA256"),
    (lambda d: d["examples"][0]["review"].update(reviewed_at="2016-01-01"), "Review date"),
    (lambda d: d["examples"][0]["reference"].update(loss_area_ha=True), "finite number"),
    (lambda d: d["examples"][0]["reference"].update(loss_area_ha=float("nan")), "finite number"),
    (lambda d: d["examples"][0].update(bbox=[-60, 1, -59, 2]), "100 ha"),
    (lambda d: d["examples"][0].update(cutoff="2016-12-30"), "December 31"),
    (lambda d: d["examples"][0].update(target_date="2018-12-31"), "December 31"),
    (lambda d: d["examples"][0].update(reference=None), "Accepted reviews"),
    (lambda d: d["examples"][1].update(id=d["examples"][0]["id"]), "unique"),
])
def test_invalid_reference_contracts_are_rejected(tmp_path, change, match):
    path, doc = study(tmp_path)
    change(doc)
    save(path, doc)
    with pytest.raises(ValueError, match=match):
        review_references(path)


def test_changed_source_file_is_detected(tmp_path):
    path, _ = study(tmp_path)
    (tmp_path / "controlled-evidence.txt").write_text("changed")
    with pytest.raises(ValueError, match="SHA256"):
        review_references(path)


def test_annual_baseline_trend_uses_only_available_history():
    history = [{"period_end": f"{year}-12-31", "available_at": f"{year+1}-02-01", "loss_fraction": value}
               for year, value in [(2012, .01), (2013, .02), (2014, .03)]]
    result = annual_reference_baselines(history, cutoff="2015-12-31", horizon_days=366)
    assert result["persistence"] == result["seasonal"] == .03
    assert result["historical_trend"] == pytest.approx(.05)
    assert result["historical_trend_available_at"] == "2015-02-01"
    history[-1]["available_at"] = "2016-02-01"
    with pytest.raises(ValueError, match="available by cutoff"):
        annual_reference_baselines(history, cutoff="2015-12-31", horizon_days=366)


def test_baseline_history_is_integrated_and_cannot_override_dated_measurements(tmp_path):
    path, doc = study(tmp_path)
    row = doc["examples"][0]
    history = []
    for year in [2013, 2014, 2015]:
        source = deepcopy(row["reference"]["source"])
        source["available_at"] = f"{year+1}-02-01"
        history.append({"period_end": f"{year}-12-31", "available_at": source["available_at"],
                        "loss_fraction": .02, "source": source})
    row["baseline_history"] = history
    for key in ["persistence", "seasonal", "historical_trend"]:
        del row[key]
        del row[key + "_available_at"]
    save(path, doc)
    report, export = review_references(path)
    assert report["ready_for_baseline_experiment"]
    assert export["examples"][0]["historical_trend"] == .02
    row["historical_trend"] = .9
    save(path, doc)
    with pytest.raises(ValueError, match="disagrees"):
        review_references(path)


def test_leap_year_window_requires_separate_protocol(tmp_path):
    path, doc = study(tmp_path)
    row = doc["examples"][0]
    row["cutoff"], row["target_date"] = "2015-12-31", "2016-12-31"
    save(path, doc)
    with pytest.raises(ValueError, match="leap-year"):
        review_references(path)


def test_reference_publication_embargo_and_nearby_holdout_block_preparation(tmp_path):
    path, doc = study(tmp_path)
    doc["examples"][0]["reference"]["source"]["available_at"] = "2021-01-01"
    doc["examples"][0]["review"]["reviewed_at"] = "2021-01-02"
    save(path, doc)
    report, _ = review_references(path)
    assert not report["ready_for_baseline_experiment"]
    assert "temporal split" in report["blockers"][0]


def test_cli_empty_template_is_honest_and_does_not_overwrite(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/review_references.py"
    path = tmp_path / "review.json"
    args = [sys.executable, str(script), "init", "--output", str(path)]
    assert subprocess.run(args, capture_output=True).returncode == 0
    before = path.read_bytes()
    assert subprocess.run(args, capture_output=True).returncode == 2
    assert path.read_bytes() == before
    checked = subprocess.run([sys.executable, str(script), "check", "--manifest", str(path)], capture_output=True, text=True)
    assert checked.returncode == 2
    assert '"ready_for_baseline_experiment": false' in checked.stdout
