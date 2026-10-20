"""Controlled fixtures test audit/calibration software, not real predictive validity."""
from copy import deepcopy
from datetime import date, timedelta

import pytest

from forestmind.training.reference_experiment import audit_references, freeze_protocol, run_reference_experiment, write_new_json


def references(kind="continuous"):
    rows = []
    for split, year, lon, count in (("train", 2020, -55, 12), ("calibration", 2022, 10, 6), ("test", 2024, 100, 6)):
        for i in range(count):
            cutoff = date(year, 1, 1) + timedelta(days=i * 30)
            target = i % 2 if kind == "event" else .3 + i * .02
            row = {"id": f"{split}-{i}", "split": split, "region_id": split + "-controlled-fixture", "bbox": [lon, 1, lon + .01, 1.01],
                   "cutoff": cutoff.isoformat(), "features_available_at": cutoff.isoformat(),
                   "target_date": (cutoff + timedelta(days=30)).isoformat(), "target": target,
                   "features": {"observed_index": float(target), "missing_context": None if i % 3 == 0 else .2},
                   "persistence": .5 if kind == "event" else target - .1,
                   "seasonal": .6 if kind == "event" else target + .2,
                   "persistence_available_at": cutoff.isoformat(), "seasonal_available_at": (cutoff - timedelta(days=365)).isoformat(),
                   "quality_group": "high", "label_source": {"independent": True, "reference": "CONTROLLED TEST FIXTURE ONLY", "version": "test-v1", "uncertainty": "Not real observations"}}
            if kind == "event" and target:
                row.update(event_id=f"event-{split}-{i}", event_date=(cutoff + timedelta(days=15)).isoformat())
            rows.append(row)
    return {"schema_version": 1, "dataset_id": "software-contract-fixture", "version": "v1", "synthetic": False,
            "target": {"kind": kind, "name": "Controlled audit fixture", "definition": "Test contract only", "units": "event" if kind == "event" else "index",
                       "intended_use": "Unit testing, not scientific validation", "horizon_days": 30},
            "split_policy": {"spatial_buffer_km": 10., "temporal_embargo_days": 30, "protected_test": True},
            "feature_names": ["observed_index", "missing_context"], "examples": rows}


def test_frozen_experiment_identical_ids_train_only_preprocessing_and_negative_results():
    document = references()
    protocol = freeze_protocol(document)
    result = run_reference_experiment(document, protocol)
    assert result["audit"]["counts"] == {"train": 12, "calibration": 6, "test": 6}
    assert result["training_preprocessing"]["train_medians"][1] == .2
    ids = [[row["id"] for row in model["examples"]] for model in result["models"].values()]
    assert all(value == ids[0] for value in ids)
    assert result["models"]["seasonal"]["improvement_over_persistence"] < 0
    assert result["models"]["persistence"]["calibration"]["count"] == 6
    assert result["models"] == run_reference_experiment(document, protocol)["models"]


def test_event_metrics_calibration_false_alerts_and_lead_time():
    document = references("event")
    result = run_reference_experiment(document, freeze_protocol(document))
    model = result["models"]["regularized_features"]
    assert model["metrics"]["unique_reference_events"] == 3
    assert model["metrics"]["missed_reference_events"] == 0
    assert model["metrics"]["mean_first_warning_lead_days"] == 15.
    assert 0 <= model["metrics"]["brier"] <= 1
    assert sum(bin["count"] for bin in model["metrics"]["reliability"]) == 6
    assert model["raw_metrics"] is not None
    assert model["calibration"]["count"] == 6


@pytest.mark.parametrize("kind", ["continuous", "event"])
def test_gradient_boosting_keeps_protected_examples_calibration_and_deterministic_results(kind):
    document = references(kind)
    protocol = freeze_protocol(document)
    result = run_reference_experiment(document, protocol)
    boosted = result["models"]["gradient_boosted_features"]
    expected = [row for row in document["examples"] if row["split"] == "test"]
    assert len(result["models"]) == 5
    for model in result["models"].values():
        assert [row["id"] for row in model["examples"]] == [row["id"] for row in expected]
        assert [row["target"] for row in model["examples"]] == [row["target"] for row in expected]
    assert boosted["calibration"]["count"] == 6
    assert boosted == run_reference_experiment(document, protocol)["models"]["gradient_boosted_features"]
    if kind == "event":
        assert boosted["raw_metrics"] is not None
        assert 0 <= boosted["metrics"]["brier"] <= 1
        assert sum(bin["count"] for bin in boosted["metrics"]["reliability"]) == len(expected)
        assert all(0 <= row["raw_prediction"] <= 1 and 0 <= row["prediction"] <= 1
                   for row in boosted["examples"])
    else:
        assert boosted["raw_metrics"] is None
        assert boosted["metrics"]["mae"] >= 0
        offsets = boosted["calibration"]["offsets"]
        for row in boosted["examples"]:
            assert row["lower"] == pytest.approx(row["prediction"] + offsets[0])
            assert row["upper"] == pytest.approx(row["prediction"] + offsets[1])


@pytest.mark.parametrize("kind", ["continuous", "event"])
def test_gradient_boosting_never_fits_imputation_or_model_on_held_out_features(kind):
    document = references(kind)
    probe = next(row for row in document["examples"] if row["split"] == "test")
    probe["features"] = {name: None for name in document["feature_names"]}
    altered = deepcopy(document)
    # A missing held-out probe uses the training medians. Other held-out inputs
    # must not influence either its imputation or its raw fitted-model output.
    for row in altered["examples"]:
        if row["split"] != "train" and row["id"] != probe["id"]:
            row["features"] = {name: 10000. for name in altered["feature_names"]}
    before = run_reference_experiment(document, freeze_protocol(document))
    after = run_reference_experiment(altered, freeze_protocol(altered))
    assert before["training_preprocessing"] == after["training_preprocessing"]
    before_probe = before["models"]["gradient_boosted_features"]["examples"][0]
    after_probe = after["models"]["gradient_boosted_features"]["examples"][0]
    assert before_probe["id"] == after_probe["id"] == probe["id"]
    assert before_probe["raw_prediction"] == after_probe["raw_prediction"]


def test_gradient_boosting_configuration_cannot_change_after_protocol_freeze(monkeypatch):
    from forestmind.training import reference_experiment as experiment
    document = references()
    protocol = freeze_protocol(document)
    assert protocol["models"]["boost_stages"] == 100
    assert protocol["models"]["boost_n_iter_no_change"] is None
    monkeypatch.setitem(experiment.MODEL_CONFIG, "boost_learning_rate", .2)
    with pytest.raises(ValueError, match="Model configuration differs"):
        run_reference_experiment(document, protocol)


@pytest.mark.parametrize("change,match", [
    (lambda d: d["examples"][-1].update(features_available_at="2027-01-01"), "Features must"),
    (lambda d: d["examples"][-1].update(bbox=d["examples"][0]["bbox"]), "spatial buffer"),
    (lambda d: d["examples"][-1].update(region_id=d["examples"][0]["region_id"]), "identity crosses"),
    (lambda d: d["examples"][-1]["label_source"].update(independent=False), "independent label"),
    (lambda d: d.update(synthetic=True), "synthetic=false"),
    (lambda d: d["examples"][-1].update(persistence_available_at="2027-01-01"), "Baseline feature availability"),
    (lambda d: d["examples"][-1].update(id=d["examples"][0]["id"]), "unique id"),
])
def test_leakage_and_reference_guards(change, match):
    document = references()
    change(document)
    with pytest.raises(ValueError, match=match):
        audit_references(document)


def test_target_window_and_embargo_are_purged():
    document = references()
    example = document["examples"][12]
    example.update(cutoff="2020-10-01", features_available_at="2020-10-01", target_date="2020-10-31",
                   persistence_available_at="2020-10-01", seasonal_available_at="2019-10-01")
    with pytest.raises(ValueError, match="temporal split"):
        audit_references(document)


def test_event_identity_cannot_merge_unrelated_events_or_cross_splits():
    document = references("event")
    document["examples"][-1]["event_id"] = document["examples"][1]["event_id"]
    with pytest.raises(ValueError, match="event_id must identify one"):
        audit_references(document)


def test_consistent_event_identity_groups_repeated_warning_origins():
    from forestmind.training.reference_experiment import _event_metrics
    import numpy as np
    rows = [{"region_id": "same-site", "cutoff": "2024-01-01", "target_date": "2024-03-01", "target": 1,
             "event_id": "one-disturbance", "event_date": "2024-02-15"},
            {"region_id": "same-site", "cutoff": "2024-02-01", "target_date": "2024-04-01", "target": 1,
             "event_id": "one-disturbance", "event_date": "2024-02-15"}]
    metrics = _event_metrics(rows, np.array([1, 1]), np.array([.8, .9]))
    assert metrics["unique_reference_events"] == 1
    assert metrics["mean_first_warning_lead_days"] == 45
    assert metrics["exposure_region_days"] == 91  # overlapping horizons count once


def test_target_protocol_and_test_labels_cannot_change_after_freeze(tmp_path):
    document = references()
    protocol = freeze_protocol(document)
    changed = deepcopy(document)
    changed["examples"][-1]["target"] += .1
    with pytest.raises(ValueError, match="changed"):
        run_reference_experiment(changed, protocol)
    altered_protocol = deepcopy(protocol)
    altered_protocol["models"]["event_threshold"] = .2
    with pytest.raises(ValueError, match="changed"):
        run_reference_experiment(document, altered_protocol)
    output = tmp_path / "protected-result.json"
    write_new_json(protocol, output)
    with pytest.raises(FileExistsError):
        write_new_json(protocol, output)


def test_frozen_protocol_pins_numerical_runtime(monkeypatch):
    import numpy as np
    document = references()
    protocol = freeze_protocol(document)
    monkeypatch.setattr(np, "__version__", "different-runtime")
    with pytest.raises(ValueError, match="numerical runtime differs"):
        run_reference_experiment(document, protocol)


def test_insufficient_subgroups_report_counts_without_unstable_metrics():
    document = references()
    document["examples"][-1]["quality_group"] = "low"
    report = run_reference_experiment(document, freeze_protocol(document))
    groups = report["models"]["persistence"]["subgroups"]["quality_group"]
    small = next(row for row in groups if row["group"] == "low")
    assert small["count"] == 1 and small["metrics"] is None
