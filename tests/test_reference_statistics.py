"""Analytic statistics fixtures; no accuracy claim about a real forest model."""
from copy import deepcopy

import numpy as np
import pytest

from forestmind.training.reference_experiment import (
    _event_metrics, audit_references, freeze_protocol, run_reference_experiment, paired_region_bootstrap,
)
from test_reference_experiment import references


def test_event_metrics_have_known_confusion_auroc_and_ece():
    rows = [r for r in references("event")["examples"] if r["split"] == "test"][:4]
    truth = np.array([0, 1, 0, 1])
    scores = np.array([.1, .8, .6, .3])
    metrics = _event_metrics(rows, truth, scores)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["auroc"] == .75
    assert metrics["expected_calibration_error"] == pytest.approx(.4)
    assert [c["support"] for c in metrics["per_class"]] == [2, 2]
    assert metrics["brier"] == pytest.approx(.225)


def test_reliability_bins_do_not_drop_boundary_scores_and_undefined_auc_is_null():
    rows = [r for r in references("event")["examples"] if r["split"] == "test"]
    metrics = _event_metrics(rows, np.array([r["target"] for r in rows]), np.array([0, .2, .4, .6, .8, 1.]))
    assert sum(b["count"] for b in metrics["reliability"]) == 6
    negative_rows = [r for r in rows if r["target"] == 0]
    metrics = _event_metrics(negative_rows, np.zeros(3), np.zeros(3))
    assert metrics["auroc"] is None
    assert metrics["pr_auc_average_precision"] is None
    assert metrics["expected_calibration_error"] == 0


def test_region_bootstrap_withholds_single_region_and_keeps_paired_negative_results():
    rows = [{"region_id": "same", "target": 0.} for _ in range(100)]
    assert not paired_region_bootstrap(rows, np.zeros(100), np.ones(100), event=False)["available"]
    rows = [{"region_id": f"region-{i}", "target": 0.} for i in range(6) for _ in range(i + 1)]
    baseline, prediction = np.zeros(len(rows)), np.ones(len(rows))
    result = paired_region_bootstrap(rows, baseline, prediction, event=False)
    assert result["available"] and result["lower"] == result["upper"] == -1.
    assert result == paired_region_bootstrap(rows, baseline, prediction, event=False)
    same = paired_region_bootstrap(rows, prediction, prediction, event=True)
    assert same["lower"] == same["upper"] == 0


def test_historical_trend_requires_identical_examples_and_past_availability():
    doc = references()
    for row in doc["examples"]:
        row.update(historical_trend=.2, historical_trend_available_at=row["cutoff"])
    result = run_reference_experiment(doc, freeze_protocol(doc))
    assert len(result["models"]) == 6
    assert [r["id"] for r in result["models"]["historical_trend"]["examples"]] == [r["id"] for r in result["models"]["persistence"]["examples"]]
    invalid = deepcopy(doc)
    del invalid["examples"][0]["historical_trend"]
    with pytest.raises(ValueError, match="finite persistence"):
        audit_references(invalid)
    doc["examples"][0]["historical_trend_available_at"] = "2099-01-01"
    with pytest.raises(ValueError, match="availability"):
        audit_references(doc)
