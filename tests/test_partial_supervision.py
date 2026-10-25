"""Unknown ecological labels must not contribute to loss or reported accuracy."""
import copy
import json
from dataclasses import asdict

import pytest
import torch
import numpy as np

from forestmind.training.losses import ForestMindLoss
from forestmind.training.metrics import ForestMindMetrics
from test_inference import config, raw_sequence


def batch():
    outputs = {
        "health": {"logits": torch.zeros(2, 5, requires_grad=True),
                   "score": torch.tensor([60., 20.], requires_grad=True), "state": torch.tensor([1, 4])},
        "attribution": {"logits": torch.zeros(2, 8, requires_grad=True), "active": torch.zeros(2, 8),
                        "strength": torch.zeros(2, 8, requires_grad=True)},
        "trajectory": {"mean": torch.full((2, 2), .4, requires_grad=True),
                       "logvar": torch.zeros(2, 2, requires_grad=True)},
    }
    targets = {"health_state": torch.tensor([1, 0]), "health_score": torch.zeros(2),
               "drivers": torch.zeros(2, 8), "driver_strengths": torch.zeros(2, 8),
               "trajectory": torch.tensor([[0., 0.], [.6, 0.]]),
               "health_state_mask": torch.tensor([True, False]), "health_score_mask": torch.zeros(2, dtype=torch.bool),
               "drivers_mask": torch.zeros(2, 8, dtype=torch.bool), "driver_strengths_mask": torch.zeros(2, 8, dtype=torch.bool),
               "trajectory_mask": torch.tensor([[False, False], [True, False]])}
    return outputs, targets


def test_unknown_labels_have_no_loss_or_gradient_even_with_bad_placeholders():
    outputs, targets = batch()
    expected = ForestMindLoss()(outputs, targets)["total"]
    altered = copy.deepcopy(targets)
    for name in ("health_state", "health_score", "drivers", "driver_strengths", "trajectory"):
        altered[name][~altered[f"{name}_mask"]] = -999
    actual = ForestMindLoss()(outputs, altered)["total"]
    assert actual.item() == pytest.approx(expected.item())
    actual.backward()
    assert outputs["health"]["logits"].grad[1].abs().sum() == 0
    assert outputs["trajectory"]["mean"].grad[0].abs().sum() == 0
    assert outputs["trajectory"]["mean"].grad[1, 1] == 0
    assert outputs["attribution"]["logits"].grad.abs().sum() == 0


def test_metrics_count_only_observed_labels_and_serialize_missing_as_null():
    outputs, targets = batch()
    metrics = ForestMindMetrics(num_horizons=2)
    metrics.update(outputs, targets)
    report = asdict(metrics.compute())
    assert report["health_accuracy"] == 1
    assert report["health_mae"] is None
    assert report["attribution_f1"] is None
    assert report["trajectory_per_horizon_mae"] == [pytest.approx(.2), None]
    assert report["observed_counts"]["trajectory"] == [1, 0]
    json.dumps(report, allow_nan=False)
    assert "unavailable" in metrics.summary_str()


def test_no_known_target_is_rejected_and_masks_cannot_be_numeric():
    outputs, targets = batch()
    targets["health_state_mask"][:] = False
    with pytest.raises(ValueError, match="at least one"):
        ForestMindLoss()(outputs, targets)
    targets["health_state_mask"] = torch.ones(2)
    with pytest.raises(ValueError, match="boolean"):
        ForestMindLoss()(outputs, targets)


def test_inference_withholds_untrained_outputs(tmp_path, config):
    from forestmind.models.forestmind_model import ForestMindModel
    from forestmind.inference.predict import predict_checkpoint
    support = {"health_state": False, "health_score": False,
               "drivers": [False] * 5 + [True, False, False],
               "driver_strengths": [False] * 8, "trajectory": [True, False]}
    checkpoint, sequence = tmp_path / "partial.pt", tmp_path / "sequence.npz"
    torch.save({"config": config, "model_state_dict": ForestMindModel(config).state_dict(),
                "epoch": 0, "supervised_targets": support}, checkpoint)
    np.savez(sequence, **raw_sequence())
    report = predict_checkpoint(checkpoint, sequence, region_id="fixture", observed_through="2025-06-01", synthetic_input=True)
    prediction = report["predictions"]
    assert prediction["health"] == {"state": None, "score": None, "class_scores": []}
    assert prediction["drivers"][0]["score"] is None
    assert prediction["drivers"][5]["score"] is not None
    assert all(row["strength"] is None for row in prediction["drivers"])
    assert prediction["trajectory"][0]["score"] is not None
    assert prediction["trajectory"][1]["score"] is None
    assert prediction["trajectory"][1]["stddev"] is None
