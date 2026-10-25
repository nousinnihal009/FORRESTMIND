"""Controlled fixtures test conversion contracts, never real ecological accuracy."""
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from forestmind.data.training_data import (
    CLIMATE_NAMES, CLIMATE_UNITS, audit_training_manifest, prepare_training_dataset,
)


def _label(value, when):
    return {"value": value, "date": when, "source": {
        "independent": True, "reference": "CONTROLLED SOFTWARE TEST FIXTURE ONLY",
        "version": "fixture-v1", "uncertainty": "Not real reference observations",
        "license": "Test fixture", "available_at": when}}


def study(tmp_path, *, climate=False, calibration=False):
    examples = []
    split_specs = [("train", 2020, -70, .2), ("val", 2022, -40, .8), ("test", 2026, 40, .95)]
    if calibration:
        split_specs.insert(2, ("calibration", 2024, 0, .9))
    for split, year, west, reflectance in split_specs:
        cutoff = f"{year}-02-01"
        bbox = [west, -5, west + .01, -4.99]
        arrays = {"optical": np.full((2, 12, 16, 16), reflectance, dtype=np.float32),
                  "sar": np.zeros((2, 2, 16, 16), dtype=np.float32),
                  "timestamps": np.asarray([0., 31.]), "observation_mask": np.ones(2, dtype=bool),
                  "optical_mask": np.ones(2, dtype=bool), "sar_mask": np.zeros(2, dtype=bool),
                  "quality_mask": np.ones((2, 16, 16), dtype=bool), "synthetic": np.asarray(False),
                  "normalization": np.asarray("raw"), "observed_through": np.asarray(cutoff),
                  "acquisition_dates": np.asarray([f"{year}-01-01T12:00:00+00:00", f"{cutoff}T12:00:00+00:00"]),
                  "bbox": np.asarray(bbox)}
        if climate:
            arrays.update(climate_context=np.asarray([[20., 10., 50., .2], [22., 20., 60., .4]], dtype=np.float32) + np.asarray([0, 0, 0, 0] if split == "train" else [20, 20, 20, 0]),
                          climate_mask=np.ones((2, 4), dtype=bool), climate_coverage=np.ones((2, 4), dtype=np.float32),
                          climate_feature_names=np.asarray(CLIMATE_NAMES), climate_feature_units=np.asarray(CLIMATE_UNITS))
        path = tmp_path / (split + ".npz")
        np.savez_compressed(path, **arrays)
        examples.append({"id": split + "-patch", "split": split, "region_id": split + "-location",
                         "bbox": bbox, "sequence": path.name, "sequence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "cutoff": cutoff, "features_available_at": cutoff,
                         "labels": {"health_state": _label("stressed", cutoff),
                                    "drivers": {"wildfire": _label(1, cutoff)},
                                    "trajectory": {"30": _label(.6, (date.fromisoformat(cutoff) + timedelta(days=30)).isoformat())}}})
    document = {"schema_version": 1, "dataset_id": "software-fixture", "version": "v1", "synthetic": False,
                "intended_use": "Unit testing only, not real labels", "time_steps": 4, "grid_shape": [16, 16],
                "horizons_days": [30, 365], "split_policy": {"spatial_buffer_km": 10, "temporal_embargo_days": 30, "protected_test": True},
                "target_spec": {"health_state": {"units": "canonical_five_state", "definition": "Fixture condition", "reference_protocol": "Unit-test labels"},
                                "drivers": {"units": "binary_presence", "definition": "Fixture wildfire presence", "reference_protocol": "Unit-test labels"},
                                "trajectory": {"units": "condition_0_1", "definition": "Fixture future health", "reference_protocol": "Unit-test labels"}},
                "examples": examples}
    path = tmp_path / "study.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def save(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")


def change_sequence(path, document, index, modify):
    example = document["examples"][index]
    source = path.parent / example["sequence"]
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    modify(arrays)
    np.savez_compressed(source, **arrays)
    example["sequence_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    save(path, document)


def test_converter_preserves_unknown_targets_and_pads_only_missing_steps(tmp_path):
    path, document = study(tmp_path, calibration=True)
    output = tmp_path / "prepared"
    report = prepare_training_dataset(path, output)
    assert report["audit"]["counts"] == {"train": 1, "val": 1, "calibration": 1, "test": 1}
    assert report["scientifically_validated"] is False
    assert report["audit"]["label_counts_by_channel"]["train"]["trajectory"] == {"30": 1, "365": 0}
    with np.load(output / "train/train-patch.npz", allow_pickle=False) as data:
        assert data["health_state"].item() == 1
        assert data["health_state_mask"].item() is True
        assert data["health_score_mask"].item() is False
        assert data["drivers_mask"].tolist() == [False] * 5 + [True, False, False]
        assert data["driver_strengths_mask"].any() is np.False_
        assert data["trajectory_mask"].tolist() == [True, False]
        assert data["trajectory_units"].item() == "condition_0_1"
        assert data["observation_mask"].tolist() == [True, True, False, False]
        assert data["timestamps"].tolist() == [0, 31, 31, 31]
        assert len(data["acquisition_dates"]) == 4
        assert data["acquisition_dates"][2] == data["acquisition_dates"][1]
        assert not data["sar_quality_mask"].any()
        assert not data["optical"][2:].any()
    config = yaml.safe_load((output / "training-config.yaml").read_text())
    assert config["data"]["synthetic"] is False
    assert config["heads"]["trajectory"]["horizons"] == [30, 365]
    assert Path(config["data"]["source"]) == output
    assert config["data"]["dataset_report_sha256"] == hashlib.sha256((output / "dataset-report.json").read_bytes()).hexdigest()
    assert report["supervised_targets"] == config["data"]["supervised_targets"]
    assert config["foundation"]["custom_vit"]["img_size"] == 16
    assert report["input_contract"]["horizon_days"] == [30, 365]


def test_statistics_use_training_observed_pixels_and_context_only(tmp_path):
    path, document = study(tmp_path, climate=True)

    def mask_inputs(arrays):
        arrays["optical"][0, :, 0, 0] = 1
        arrays["quality_mask"][0, 0, 0] = False
        arrays["climate_context"][0, 0] = 9999
        arrays["climate_mask"][0, 0] = False

    change_sequence(path, document, 0, mask_inputs)
    report = prepare_training_dataset(path, tmp_path / "prepared")
    assert report["normalization"]["optical_mean"] == pytest.approx([.2] * 12)
    assert report["normalization"]["sar_mean"] == [0., 0.]
    assert report["normalization"]["sar_std"] == [1., 1.]
    assert report["normalization"]["climate_mean"] == pytest.approx([22., 15., 55., .3])
    assert report["normalization_evidence"]["optical"]["observed_training_count"] == [511] * 12
    assert report["normalization_evidence"]["climate"]["observed_training_count"] == [1, 2, 2, 2]
    config = yaml.safe_load((tmp_path / "prepared/training-config.yaml").read_text())
    assert config["fusion"]["climate_dim"] == 4
    with np.load(tmp_path / "prepared/train/train-patch.npz") as prepared:
        assert prepared["climate_context"][0, 0] == 0


@pytest.mark.parametrize("change,match", [
    (lambda d: d.update(synthetic=True), "synthetic=false"),
    (lambda d: d["examples"][1].update(region_id=d["examples"][0]["region_id"]), "Region identity"),
    (lambda d: d["examples"][0]["labels"]["health_state"]["source"].update(independent=False), "independent=true"),
    (lambda d: d["examples"][0]["labels"]["health_state"]["source"].update(uncertainty=""), "uncertainty"),
    (lambda d: d["examples"][0]["labels"]["health_state"]["source"].update(available_at="2023-01-01"), "temporal embargo"),
    (lambda d: d["examples"][0].update(features_available_at="2020-02-02"), "not available"),
    (lambda d: d["examples"][0]["labels"]["trajectory"]["30"].update(date="2020-03-03"), "exact horizon"),
    (lambda d: d["target_spec"]["trajectory"].update(units="canopy_loss_fraction"), "canopy loss is not health"),
    (lambda d: d["examples"][0]["labels"]["drivers"].update(unknown_driver=_label(1, "2020-02-01")), "canonical driver"),
    (lambda d: d["examples"][0].update(sequence_sha256="0" * 64), "sequence_sha256"),
    (lambda d: d["examples"][0].update(id="../outside"), "filename-safe"),
    (lambda d: d["examples"][0].update(id="CON"), "reserved"),
    (lambda d: d["examples"][0].update(labels={}), "observed target"),
    (lambda d: d["examples"][1]["labels"]["drivers"].update(logging=_label(1, "2022-02-01")), "without any matching training"),
])
def test_references_reject_invalid_or_leaking_labels(tmp_path, change, match):
    path, document = study(tmp_path)
    change(document)
    save(path, document)
    with pytest.raises(ValueError, match=match):
        prepare_training_dataset(path, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_geographic_buffer_uses_footprints_not_only_ids(tmp_path):
    path, document = study(tmp_path)
    document["examples"][1]["bbox"] = [-69.99, -5, -69.98, -4.99]
    change_sequence(path, document, 1, lambda a: a.update(bbox=np.asarray(document["examples"][1]["bbox"])))
    with pytest.raises(ValueError, match="spatial buffer"):
        audit_training_manifest(path)


@pytest.mark.parametrize("modify,match", [
    (lambda a: a.update(synthetic=np.asarray(True)), "nonsynthetic raw"),
    (lambda a: a.update(normalization=np.asarray("normalized")), "nonsynthetic raw"),
    (lambda a: a.update(observation_mask=np.asarray([1, 1])), "boolean"),
    (lambda a: a.update(hidden=np.asarray([{}], dtype=object)), "Object arrays"),
    (lambda a: a.update(timestamps=np.asarray([0., 35.])), "timestamps must match"),
    (lambda a: a.update(observed_through=np.asarray("2020-01-31")), "mismatched observed_through"),
    (lambda a: a["quality_mask"].__setitem__(0, False), "availability disagree"),
    (lambda a: a["optical"].__setitem__((0, 0, 0, 0), np.nan), "nonfinite"),
])
def test_sequence_rejects_invalid_masks_provenance_or_pickle(tmp_path, modify, match):
    path, document = study(tmp_path)
    change_sequence(path, document, 0, modify)
    with pytest.raises(ValueError, match=match):
        audit_training_manifest(path)


def test_all_scenes_preserved_or_refused_and_no_overwrite(tmp_path):
    path, document = study(tmp_path)
    document["time_steps"] = 1
    save(path, document)
    with pytest.raises(ValueError, match="no scenes are dropped"):
        audit_training_manifest(path)
    document["time_steps"] = 4
    save(path, document)
    output = tmp_path / "prepared"
    prepare_training_dataset(path, output)
    before = (output / "dataset-report.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        prepare_training_dataset(path, output)
    assert (output / "dataset-report.json").read_bytes() == before


def test_future_radar_cannot_enter_training(tmp_path):
    path, document = study(tmp_path)

    def radar(arrays):
        arrays.update(sar_mask=np.ones(2, dtype=bool), sar_quality_mask=np.ones((2, 16, 16), dtype=bool),
                      sar_units=np.asarray("gamma0_db"), sar_age_days=np.asarray([-1., -1.]),
                      sar_acquisition_dates=np.asarray(["2020-01-02T12:00:00+00:00", "2020-02-02T12:00:00+00:00"]))

    change_sequence(path, document, 0, radar)
    with pytest.raises(ValueError, match="SAR features must precede"):
        audit_training_manifest(path)


def test_heldout_modality_without_training_observations_is_rejected(tmp_path):
    path, document = study(tmp_path, climate=True)
    change_sequence(path, document, 0, lambda a: a["climate_mask"].__setitem__((slice(None), 1), False))
    with pytest.raises(ValueError, match="without fitted training statistics"):
        prepare_training_dataset(path, tmp_path / "prepared")


def test_mixed_context_presence_emits_batch_compatible_fields(tmp_path):
    path, document = study(tmp_path, climate=True)
    change_sequence(path, document, 1, lambda a: [a.pop(k) for k in list(a) if k.startswith("climate_")])
    output = tmp_path / "prepared"
    prepare_training_dataset(path, output)
    with np.load(output / "train/train-patch.npz") as train, np.load(output / "val/val-patch.npz") as val:
        assert set(train.files) == set(val.files)
        assert not val["climate_mask"].any()
        assert not val["climate_context"].any()


def test_cli_check_has_no_output_side_effects(tmp_path):
    path, _ = study(tmp_path)
    before = {p.name for p in tmp_path.iterdir()}
    result = subprocess.run([sys.executable, "scripts/prepare_training_data.py", "--manifest", str(path), "--check"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["passed"] is True
    assert {p.name for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("climate,disable_attribution", [(False, False), (True, False), (True, True)])
def test_prepared_partial_targets_train_and_evaluate_end_to_end(tmp_path, climate, disable_attribution):
    """One-epoch plumbing smoke on controlled fixtures, not a real trained model."""
    import torch
    from forestmind.data.dataset import create_dataloaders
    from forestmind.models.forestmind_model import ForestMindModel
    from forestmind.training.trainer import ForestMindTrainer
    from forestmind.training.evaluation import evaluate_checkpoint

    path, _ = study(tmp_path, climate=climate, calibration=True)
    output = tmp_path / "prepared"
    prepare_training_dataset(path, output)
    config = yaml.safe_load((output / "training-config.yaml").read_text())
    config["training"].update(epochs=1, save_every=1)
    if disable_attribution:
        config["training"]["loss_weights"] = {"task_attribution": 0.0}
    assert config["data"]["supervised_targets"]["health_score"] is False
    assert config["data"]["supervised_targets"]["drivers"] == [False] * 5 + [True, False, False]
    assert config["data"]["supervised_targets"]["trajectory"] == [True, False]
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        train, val, test = create_dataloaders(config, batch_size=1)
        model = ForestMindModel(config)
        trainer = ForestMindTrainer(model, config, train, val)
        result = trainer.train()
        assert np.isfinite(result["best_val_score"])
        checkpoint = output / "checkpoints/best_model.pt"
        assert checkpoint.is_file()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["training_target_counts"] == {
            "health_state": 1, "health_score": 0,
            "drivers": [0, 0, 0, 0, 0, 1, 0, 0],
            "driver_strengths": [0] * 8, "trajectory": [1, 0],
        }
        expected_support = deepcopy(config["data"]["supervised_targets"])
        if disable_attribution:
            expected_support["drivers"] = [False] * 8
        assert saved["supervised_targets"] == expected_support
        report = evaluate_checkpoint(checkpoint, split="test")
        assert report["counts"]["samples"] == 1
        assert report["per_horizon"][0]["n"] == 1
        assert report["per_horizon"][1]["n"] == 0
        assert report["per_horizon"][1]["mae"] is None
        assert report["loss_observed_counts"]["health_mse"] == 0
        assert report["loss_observed_counts"]["attribution_bce"] == 1
        assert report["examples"][0]["targets"]["trajectory_mask"] == [True, False]
        assert report["dataset"]["split"] == "test"
        assert not any("calibration" in row["path"] for row in report["dataset"]["source_manifest"])
        json.dumps(report, allow_nan=False)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("mutation,match", [
    ("sample", "sample changed"),
    ("manifest", "manifest changed"),
    ("report", "configured hash"),
    ("additional", "unaudited additional"),
    ("normalization", "normalization differs"),
    ("supervision", "supervised_targets differs"),
    ("horizons", "horizon_days differs"),
    ("units", "sar_units differs"),
    ("scale", "trajectory_target_scale differs"),
    ("climate", "climate_feature_names differs"),
])
def test_prepared_dataset_loader_rejects_mutated_evidence_or_contract(tmp_path, mutation, match):
    from forestmind.data.dataset import create_dataloaders

    path, _ = study(tmp_path, climate=True, calibration=True)
    output = tmp_path / "prepared"
    prepare_training_dataset(path, output)
    config = yaml.safe_load((output / "training-config.yaml").read_text())
    if mutation == "sample":
        # Calibration files are also audited even though the trainer never reads them.
        target = output / "calibration/calibration-patch.npz"
        with target.open("ab") as stream:
            stream.write(b"altered")
    elif mutation == "manifest":
        target = output / "manifest.json"
        document = json.loads(target.read_text())
        document["examples"][0]["labels"]["health_state"]["value"] = "healthy"
        target.write_text(json.dumps(document))
    elif mutation == "report":
        with (output / "dataset-report.json").open("a") as stream:
            stream.write(" ")
    elif mutation == "additional":
        (output / "test/extra.npz").write_bytes((output / "test/test-patch.npz").read_bytes())
    elif mutation == "normalization":
        config["data"]["normalization"]["optical_mean"][0] += .01
    elif mutation == "supervision":
        config["data"]["supervised_targets"]["health_score"] = True
    elif mutation == "horizons":
        config["heads"]["trajectory"]["horizons"] = [60, 365]
    elif mutation == "units":
        config["data"]["sar_units"] = "gamma0_db"
    elif mutation == "scale":
        config["data"]["trajectory_target_scale"] = "percent"
    elif mutation == "climate":
        config["data"]["climate_feature_names"] = list(reversed(CLIMATE_NAMES))
    with pytest.raises(ValueError, match=match):
        create_dataloaders(config)


def test_grid_that_would_drop_encoder_border_pixels_is_rejected(tmp_path):
    path, document = study(tmp_path)
    document["grid_shape"] = [17, 16]
    save(path, document)
    with pytest.raises(ValueError, match="border pixels cannot be dropped"):
        audit_training_manifest(path)


@pytest.mark.parametrize("shape,turns", [((16, 32), 2), ((16, 16), 1)])
def test_spatial_augmentation_preserves_rectangular_batch_shape_and_masks(monkeypatch, shape, turns):
    from forestmind.data.transforms import SpatialFlipRotate

    pixels = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    sample = {"optical": np.broadcast_to(pixels, (2, 12, *shape)).copy(),
              "sar": np.broadcast_to(pixels + 100, (2, 2, *shape)).copy(),
              "quality_mask": np.broadcast_to(pixels % 3 == 0, (2, *shape)).copy(),
              "sar_quality_mask": np.broadcast_to(pixels % 7 == 0, (2, *shape)).copy()}
    original = deepcopy(sample)
    monkeypatch.setattr(np.random, "random", lambda: 0.)
    monkeypatch.setattr(np.random, "randint", lambda low, high: 1)
    result = SpatialFlipRotate()(sample)
    for key in original:
        assert result[key].shape == original[key].shape
        np.testing.assert_array_equal(result[key], np.rot90(original[key], turns, axes=(-2, -1)))
