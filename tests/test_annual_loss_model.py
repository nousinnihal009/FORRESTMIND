"""Annual-area contract plumbing on explicit software fixtures, not real training."""
from copy import deepcopy
from datetime import date
import json

import numpy as np
import pytest
import torch
import yaml

from forestmind.data.reference_review import TARGET
from forestmind.models.forestmind_model import ForestMindModel
from forestmind.models.heads.annual_loss import annual_loss_contract, annual_loss_target_date
from forestmind.training.losses import ForestMindLoss
from forestmind.training.metrics import ForestMindMetrics
from test_inference import config, raw_sequence, example_report
from test_partial_supervision import batch
from test_training_data import study, change_sequence, save, _label


def enable(config, horizon=365):
    result = deepcopy(config)
    result.setdefault("data", {}).setdefault("target_spec", {})["annual_loss"] = {**TARGET, "horizon_days": horizon}
    result.setdefault("heads", {})["annual_loss"] = {"enabled": True, "horizon_days": horizon,
        "units": "fraction_0_1", "protocol_id": TARGET["id"]}
    return result


def annual_study(tmp_path):
    path, doc = study(tmp_path)
    doc["target_spec"] = {"annual_loss": deepcopy(TARGET)}
    for index, row in enumerate(doc["examples"]):
        year = int(row["cutoff"][:4])
        row["cutoff"] = row["features_available_at"] = f"{year}-12-31"
        row["bbox"][2] = row["bbox"][0] + .009
        row["bbox"][3] = row["bbox"][1] + .009
        row["labels"] = {"annual_loss": _label(.1 + index * .1, f"{year + 1}-12-31")}
        def change(arrays):
            arrays["observed_through"] = np.asarray(row["cutoff"])
            arrays["bbox"] = np.asarray(row["bbox"])
            arrays["acquisition_dates"] = np.asarray([f"{year}-12-01", f"{year}-12-31"])
            arrays["timestamps"] = np.asarray([0., 30.])
        change_sequence(path, doc, index, change)
    save(path, doc)
    return path, doc


def test_disabled_head_preserves_legacy_state_keys_and_requires_exact_protocol(config):
    original = ForestMindModel(config)
    disabled = deepcopy(config)
    disabled["heads"]["annual_loss"] = {"enabled": False}
    assert set(original.state_dict()) == set(ForestMindModel(disabled).state_dict())
    active = enable(config)
    assert any(name.startswith("annual_loss_head.") for name in ForestMindModel(active).state_dict())
    active["data"]["target_spec"]["annual_loss"]["units"] = "condition_0_1"
    with pytest.raises(ValueError, match="complete annual"):
        annual_loss_contract(active)


@pytest.mark.parametrize("cutoff,horizon", [("2020-12-30", 365), ("2019-12-31", 365), ("2020-12-31", 366), ("2020-12-31", True)])
def test_annual_horizon_never_silently_changes(cutoff, horizon):
    with pytest.raises(ValueError):
        annual_loss_target_date(cutoff, horizon)


def test_leap_year_is_a_separate_exact_contract():
    assert annual_loss_target_date("2019-12-31", 366) == "2020-12-31"
    assert annual_loss_target_date("2020-12-31", 365) == "2021-12-31"


def test_annual_unknowns_have_no_gradient_and_metrics_keep_units():
    outputs, targets = batch()
    outputs["annual_loss"] = {"mean": torch.tensor([.2, .7], requires_grad=True)}
    targets["annual_loss"] = torch.tensor([.1, -999.])
    targets["annual_loss_mask"] = torch.tensor([True, False])
    losses = ForestMindLoss()(outputs, targets)
    assert losses["annual_loss_mse"].item() == pytest.approx(.01)
    losses["total"].backward()
    assert outputs["annual_loss"]["mean"].grad.tolist() == pytest.approx([.2, 0])
    metrics = ForestMindMetrics(num_horizons=2)
    metrics.update(outputs, targets)
    result = metrics.compute()
    assert result.annual_loss_mae == pytest.approx(.1)
    assert result.annual_loss_rmse == pytest.approx(.1)
    assert result.observed_counts["annual_loss"] == 1
    targets["annual_loss_mask"] = torch.ones(2)
    with pytest.raises(ValueError, match="boolean"):
        ForestMindLoss()(outputs, targets)


@pytest.mark.parametrize("disable", [False, True])
def test_annual_converter_train_checkpoint_evaluate_chain(tmp_path, disable):
    from forestmind.data.training_data import prepare_training_dataset
    from forestmind.data.dataset import create_dataloaders
    from forestmind.training.trainer import ForestMindTrainer
    from forestmind.training.evaluation import evaluate_checkpoint
    path, doc = annual_study(tmp_path)
    destination = tmp_path / "prepared"
    prepared = prepare_training_dataset(path, destination)
    config = yaml.safe_load((destination / "training-config.yaml").read_text())
    config["training"].update(epochs=1, save_every=1)
    config["training"]["loss_weights"] = {"annual_loss_mse": 0. if disable else 1.}
    assert prepared["supervised_targets"]["annual_loss"] is True
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train, val, _ = create_dataloaders(config, batch_size=1)
        one = next(iter(train))
        assert one["annual_loss"].shape == (1,) and one["annual_loss_mask"].dtype == torch.bool
        trainer = ForestMindTrainer(ForestMindModel(config), config, train, val)
        trained = trainer.train()
        assert np.isfinite(trained["best_val_score"])
        checkpoint = destination / "checkpoints/best_model.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["training_target_counts"]["annual_loss"] == 1
        assert saved["supervised_targets"]["annual_loss"] is (not disable)
        assert saved["supervised_targets"]["health_state"] is False
        report = evaluate_checkpoint(checkpoint, split="test")
        assert report["metrics"]["annual_loss_mae"] is not None
        assert report["metrics"]["observed_counts"]["annual_loss"] == 1
        assert report["metrics"]["health_accuracy"] is None
        assert report["loss_observed_counts"]["annual_loss_mse"] == 1
        assert report["label_schema"]["annual_loss"] == TARGET
        assert report["examples"][0]["targets"]["annual_loss_mask"] is True
        assert "annual_loss" in report["examples"][0]["predictions"]
        json.dumps(report, allow_nan=False)
        override = deepcopy(config)
        override["heads"]["annual_loss"]["horizon_days"] = 366
        override["data"]["target_spec"]["annual_loss"]["horizon_days"] = 366
        override_path = tmp_path / "override.yaml"
        override_path.write_text(yaml.safe_dump(override))
        with pytest.raises(ValueError, match="must match the checkpoint"):
            evaluate_checkpoint(checkpoint, config_path=override_path)
    finally:
        torch.set_num_threads(previous)


def test_annual_training_rejects_wrong_spatial_unit(tmp_path):
    from forestmind.data.training_data import audit_training_manifest
    path, doc = annual_study(tmp_path)
    doc["examples"][0]["bbox"][2] += .02
    save(path, doc)
    with pytest.raises(ValueError, match="100 ha"):
        audit_training_manifest(path)


@pytest.mark.parametrize("supported,cutoff", [(True, "2025-12-31"), (False, "2025-12-31"), (True, "2025-06-01")])
def test_inference_checks_actual_support_and_annual_origin(tmp_path, config, supported, cutoff):
    from forestmind.inference.predict import predict_checkpoint
    cfg = enable(config)
    support = {"health_state": False, "health_score": False, "drivers": [False] * 8,
               "driver_strengths": [False] * 8, "trajectory": [False] * 2, "annual_loss": supported}
    checkpoint = tmp_path / "annual.pt"
    torch.save({"config": cfg, "model_state_dict": ForestMindModel(cfg).state_dict(),
                "supervised_targets": support, "training_target_counts": {"annual_loss": 2}}, checkpoint)
    sample = raw_sequence()
    sample["bbox"] = np.asarray([12., 0., 12.009, .009])
    sample["observed_through"] = np.asarray(cutoff)
    previous = date.fromisoformat(cutoff).toordinal() - 31
    sample["acquisition_dates"] = np.asarray([date.fromordinal(previous).isoformat(), cutoff])
    source = tmp_path / "input.npz"
    np.savez(source, **sample)
    report = predict_checkpoint(checkpoint, source, region_id="test", observed_through=cutoff, synthetic_input=True)
    annual = report["predictions"]["annual_loss"]
    assert annual["units"] == "fraction_0_1" and annual["calibration"] == "uncalibrated"
    if supported and cutoff.endswith("12-31"):
        assert 0 <= annual["fraction"] <= 1 and annual["reason"] is None
        assert annual["target_date"] == "2026-12-31"
    else:
        assert annual["fraction"] is None and annual["reason"]
    sample["bbox"] = np.asarray([12., 0., 13., 1.])
    np.savez(source, **sample)
    with pytest.raises(ValueError, match="100 ha"):
        predict_checkpoint(checkpoint, source, region_id="test", observed_through=cutoff, synthetic_input=True)


def test_legacy_report_serialization_has_no_extra_prediction_head():
    from forestmind.inference.schemas import InferenceReport
    report = InferenceReport.model_validate(example_report()).model_dump(mode="json")
    assert set(report["predictions"]) == {"health", "drivers", "trajectory"}
