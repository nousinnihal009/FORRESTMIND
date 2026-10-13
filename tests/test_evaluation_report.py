"""Checkpoint report reproducibility, provenance, split and finite-value contracts."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from forestmind.data.dataset import create_dataloaders
from forestmind.models.forestmind_model import ForestMindModel
from forestmind.training.evaluation import (
    evaluate_checkpoint, sha256_file, tensor_mapping_hash, write_evaluation_report,
)


@pytest.fixture(scope="module")
def checkpoint_fixture(tmp_path_factory):
    directory = tmp_path_factory.mktemp("evaluation-checkpoint")
    config = {
        "device": "cpu", "seed": 17,
        "data": {"source": "synthetic", "patch_size": 16, "time_steps": 3,
                 "num_train_samples": 1, "num_val_samples": 2, "num_test_samples": 3,
                 "batch_size": 2, "num_workers": 0},
        "foundation": {"name": "custom_vit", "embedding_dim": 16,
                       "custom_vit": {"patch_embed_size": 16, "depth": 1, "num_heads": 2,
                                      "mlp_ratio": 2, "drop_rate": 0}},
        "fusion": {"method": "gated", "num_heads": 2, "num_layers": 1, "dropout": 0},
        "temporal": {"depth": 1, "num_heads": 2, "dim_feedforward": 32,
                     "dropout": 0, "max_seq_len": 12},
        "heads": {"health": {"num_classes": 5, "continuous_score": True},
                  "attribution": {"num_drivers": 8},
                  "trajectory": {"horizons": [45, 120], "predict_uncertainty": True}},
    }
    torch.manual_seed(9)
    model = ForestMindModel(config)
    path = directory / "tiny.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": 0,
                "training_status": "random_initialization_smoke_test"}, path)
    return path, config


@pytest.fixture(scope="module")
def test_report(checkpoint_fixture):
    return evaluate_checkpoint(checkpoint_fixture[0], split="test")


def test_synthetic_horizon_propagation_preserves_config(checkpoint_fixture):
    _, config = checkpoint_fixture
    original = deepcopy(config)
    loaders = create_dataloaders(config, batch_size=2)
    for loader in loaders:
        assert loader.dataset[0]["trajectory"].shape == (2,)
        assert loader.dataset.config["horizon_days"] == [45, 120]
    assert config == original


def test_report_contains_exact_checkpoint_inputs_and_calibration_evidence(checkpoint_fixture, test_report):
    path, config = checkpoint_fixture
    assert test_report["kind"] == "forestmind_checkpoint_evaluation"
    assert test_report["checkpoint"]["sha256"] == sha256_file(path)
    assert test_report["checkpoint"]["training_status"] == "random_initialization_smoke_test"
    assert test_report["counts"] == {"samples": 3, "batches": 2}
    assert test_report["dataset"]["split"] == "test"
    assert test_report["dataset"]["seed"] == config["seed"] + 20000
    assert test_report["dataset"]["synthetic"] is True
    assert "do not establish real-world" in test_report["synthetic_warning"]
    assert [item["horizon_days"] for item in test_report["per_horizon"]] == [45, 120]
    assert test_report["calibration"]["status"] == "uncalibrated"
    assert test_report["calibration"]["fitted"] is False
    assert test_report["calibration"]["example_role"] == "held_out_evaluation_only"
    json.dumps(test_report, allow_nan=False)
    for index, example in enumerate(test_report["examples"]):
        assert example["sample_id"] == f"test:{index}"
        assert len(example["input_sha256"]) == 64
        assert len(example["target_sha256"]) == 64
        assert example["source_ref"]["seed_sequence"] == [20017, index]
        predictions = example["predictions"]
        assert len(predictions["health"]["logits"]) == 5
        assert sum(predictions["health"]["probs"]) == pytest.approx(1)
        assert len(predictions["attribution"]["probs"]) == 8
        assert len(predictions["trajectory"]["mean"]) == 2
        assert len(predictions["trajectory"]["uncertainty"]) == 2


def test_val_and_test_keep_distinct_labels_seeds_and_hashes(checkpoint_fixture, test_report):
    validation = evaluate_checkpoint(checkpoint_fixture[0], split="val")
    assert validation["dataset"]["split"] == "val"
    assert validation["counts"]["samples"] == 2
    assert validation["dataset"]["seed"] == 10017
    assert validation["calibration"]["example_role"] == "validation_only"
    assert all(row["split"] == "val" for row in validation["examples"])
    assert validation["dataset"]["input_manifest_sha256"] != test_report["dataset"]["input_manifest_sha256"]
    assert validation["dataset"]["target_manifest_sha256"] != test_report["dataset"]["target_manifest_sha256"]


def test_input_and_target_hashes_repeat_and_metrics_ignore_batch_partition(checkpoint_fixture, test_report, tmp_path):
    checkpoint, config = checkpoint_fixture
    altered = deepcopy(config)
    altered["data"]["batch_size"] = 1
    override = tmp_path / "batch-one.yaml"
    override.write_text(yaml.safe_dump(altered), encoding="utf-8")
    second = evaluate_checkpoint(checkpoint, config_path=override)
    assert second["counts"] == {"samples": 3, "batches": 3}
    for field in ("input_manifest_sha256", "target_manifest_sha256"):
        assert second["dataset"][field] == test_report["dataset"][field]
    assert second["config"]["file_sha256"] == sha256_file(override)
    assert second["config"]["sha256"] != test_report["config"]["sha256"]
    assert second["losses"]["total"] == pytest.approx(test_report["losses"]["total"], rel=1e-5)
    assert second["metrics"]["trajectory_mae"] == pytest.approx(test_report["metrics"]["trajectory_mae"], abs=1e-6)


def test_per_horizon_errors_are_recomputable_from_example_predictions(test_report):
    predicted = np.array([row["predictions"]["trajectory"]["mean"] for row in test_report["examples"]])
    actual = np.array([row["targets"]["trajectory"] for row in test_report["examples"]])
    errors = predicted - actual
    for index, result in enumerate(test_report["per_horizon"]):
        assert result["n"] == len(actual)
        assert result["mae"] == pytest.approx(np.abs(errors[:, index]).mean())
        assert result["rmse"] == pytest.approx(np.sqrt(np.mean(errors[:, index] ** 2)))


def test_checkpoint_loading_is_explicitly_weights_only(checkpoint_fixture, monkeypatch):
    original = torch.load
    calls = []
    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "load", checked_load)
    evaluate_checkpoint(checkpoint_fixture[0], split="val")
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_tensor_hash_depends_on_values_shape_dtype_and_field_name():
    base = torch.tensor([1.0, 2.0])
    original = tensor_mapping_hash({"optical": base})
    assert original == tensor_mapping_hash({"optical": base.clone()})
    assert original != tensor_mapping_hash({"sar": base})
    assert original != tensor_mapping_hash({"optical": base.reshape(1, 2)})
    assert original != tensor_mapping_hash({"optical": base.double()})
    assert original != tensor_mapping_hash({"optical": base + 1})


def test_nonfinite_predictions_fail_without_a_plausible_report(checkpoint_fixture, monkeypatch):
    original = ForestMindModel.forward
    def invalid_forward(self, *args, **kwargs):
        values = original(self, *args, **kwargs)
        values["health"]["score"][0] = float("nan")
        return values
    monkeypatch.setattr(ForestMindModel, "forward", invalid_forward)
    with pytest.raises(ValueError, match="Non-finite.*outputs.health.score"):
        evaluate_checkpoint(checkpoint_fixture[0], split="val")


def test_report_writer_rejects_nonfinite_json_and_checkpoint_overwrite(test_report, tmp_path):
    invalid = deepcopy(test_report)
    invalid["metrics"]["health_mae"] = float("nan")
    output = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        write_evaluation_report(invalid, output)
    assert not output.exists()
    with pytest.raises(ValueError, match="must not overwrite"):
        write_evaluation_report(test_report, test_report["checkpoint"]["path"])


def test_no_inference_of_batch_dependent_percent_target_scale(checkpoint_fixture, tmp_path):
    checkpoint, config = checkpoint_fixture
    config = deepcopy(config)
    config["data"]["trajectory_target_scale"] = "percent"
    override = tmp_path / "bad-scale.yaml"
    override.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic trajectories always"):
        evaluate_checkpoint(checkpoint, config_path=override)


def test_cli_writes_a_real_tiny_checkpoint_report(checkpoint_fixture, tmp_path):
    checkpoint, _ = checkpoint_fixture
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    output = tmp_path / "evaluation.json"
    completed = subprocess.run(
        [sys.executable, str(script), "--checkpoint", str(checkpoint), "--split", "val", "--output", str(output)],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert completed.returncode == 0, completed.stderr
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["counts"]["samples"] == 2
    assert saved["dataset"]["split"] == "val"
    assert len(saved["examples"]) == 2
    assert "VALIDATION SPLIT" in completed.stdout
    assert "SYNTHETIC DATA" in completed.stdout
    assert "45 days" in completed.stdout and "120 days" in completed.stdout
