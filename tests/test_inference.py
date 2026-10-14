"""Explicit-input inference, safe loading and portable report invariants."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from forestmind.data.transforms import Normalize
from forestmind.inference.predict import load_sequence, predict_checkpoint, write_inference_report
from forestmind.inference.schemas import DRIVER_NAMES, HEALTH_NAMES, InferenceReport
from forestmind.models.forestmind_model import ForestMindModel


def example_report():
    return {
        "schema_version": 1, "kind": "forestmind_inference", "region_id": "xingu",
        "observed_through": "2025-06-01", "created_at": "2025-06-02T12:00:00+00:00",
        "checkpoint": {"name": "best.pt", "sha256": "a" * 64, "epoch": 0, "backbone": "custom_vit",
                       "training_source": "synthetic", "trained_on_synthetic": True},
        "input": {"name": "sample.npz", "sha256": "b" * 64, "synthetic": True, "timesteps": 2},
        "predictions": {
            "health": {"state": "Healthy", "score": 50.0, "class_scores": [
                {"name": name, "score": 0.6 if index == 0 else 0.1} for index, name in enumerate(HEALTH_NAMES)]},
            "drivers": [{"name": name, "score": 0.5, "strength": 0.1} for name in DRIVER_NAMES],
            "trajectory": [{"horizon_days": 30, "score": 49.5, "stddev": 80.0},
                           {"horizon_days": 180, "score": 45.5, "stddev": None}],
        }, "calibration": "uncalibrated", "caveats": ["Synthetic and uncalibrated demonstration."],
    }


@pytest.fixture(scope="module")
def config():
    return {
        "data": {"source": "synthetic"},
        "foundation": {"name": "custom_vit", "embedding_dim": 16,
                       "custom_vit": {"patch_embed_size": 8, "depth": 1, "num_heads": 2, "drop_rate": 0.0}},
        "fusion": {"num_heads": 2, "num_layers": 1},
        "temporal": {"depth": 1, "num_heads": 2, "dim_feedforward": 32, "max_seq_len": 4, "dropout": 0.0},
        "heads": {"trajectory": {"horizons": [30, 90], "predict_uncertainty": True}},
    }


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory, config):
    path = tmp_path_factory.mktemp("inference") / "checkpoint.pt"
    torch.manual_seed(4)
    torch.save({"model_state_dict": ForestMindModel(config).state_dict(), "config": config, "epoch": 0}, path)
    return path


def raw_sequence():
    return {
        "optical": np.full((2, 12, 16, 16), 0.2, dtype=np.float32),
        "sar": np.zeros((2, 2, 16, 16), dtype=np.float32),
        "timestamps": np.array([0, 31], dtype=np.float32),
        "observation_mask": np.array([True, True]), "optical_mask": np.array([True, True]),
        "sar_mask": np.array([False, False]), "quality_mask": np.ones((2, 16, 16), dtype=bool),
        "synthetic": np.array(True), "normalization": np.array("raw"),
        "observed_through": np.array("2025-06-01"),
        "acquisition_dates": np.array(["2025-05-01", "2025-06-01"]),
    }


@pytest.fixture
def sequence(tmp_path):
    path = tmp_path / "sequence.npz"
    np.savez(path, **raw_sequence())
    return path


def test_local_inference_is_reproducible_and_matches_raw_head_scores(checkpoint, sequence):
    report = predict_checkpoint(checkpoint, sequence, region_id="xingu", observed_through="2025-06-01", synthetic_input=True)
    repeated = predict_checkpoint(checkpoint, sequence, region_id="xingu", observed_through="2025-06-01", synthetic_input=True)
    assert report["predictions"] == repeated["predictions"]
    assert [item["horizon_days"] for item in report["predictions"]["trajectory"]] == [30, 90]
    assert report["checkpoint"]["name"] == checkpoint.name
    assert report["input"]["name"] == sequence.name
    assert report["input"]["normalization"] == "synthetic_defaults"
    assert report["input"]["timestamp_basis"] == "acquisition_dates"
    assert report["input"]["sar_timesteps"] == 0
    assert report["checkpoint"]["trained_on_synthetic"] is True
    assert report["calibration"] == "uncalibrated"
    assert "SYNTHETIC INPUT" in " ".join(report["caveats"])
    assert report["predictions"]["trajectory"][0]["stddev"] > 1  # score points, not normalized units
    assert len(report["checkpoint"]["sha256"]) == 64
    assert str(checkpoint.parent) not in json.dumps(report)
    InferenceReport.model_validate(report)


def test_real_inputs_refuse_synthetic_default_normalization(checkpoint, sequence):
    with pytest.raises(ValueError, match="explicit checkpoint data.normalization"):
        predict_checkpoint(checkpoint, sequence, region_id="xingu", observed_through="2025-06-01")


def test_real_input_declared_stats_and_unknown_training_provenance(tmp_path, checkpoint):
    content = torch.load(checkpoint, map_location="cpu", weights_only=True)
    normalizer = Normalize()
    content["config"]["data"] = {"source": "C:/private/training_area", "normalization": {
        key: getattr(normalizer, key).tolist() for key in ("optical_mean", "optical_std", "sar_mean", "sar_std")}}
    model_path = tmp_path / "local.pt"
    torch.save(content, model_path)
    raw = raw_sequence()
    raw["synthetic"] = np.array(False)
    data_path = tmp_path / "real.npz"
    np.savez(data_path, **raw)
    result = predict_checkpoint(model_path, data_path, region_id="xingu", observed_through="2025-06-01")
    assert result["input"]["normalization"] == "checkpoint_statistics"
    assert result["input"]["synthetic"] is False
    assert result["checkpoint"]["training_source"] == "training_area"
    assert result["checkpoint"]["trained_on_synthetic"] is None


@pytest.mark.parametrize("field,value,match", [
    ("timestamps", np.array([0, 0]), "strictly increase"),
    ("timestamps", np.array([1, 31]), "begin at zero"),
    ("timestamps", np.array([0, np.nan]), "finite"),
    ("timestamps", np.array([0, 32]), "acquisition_dates"),
    ("observed_through", np.array("2025-06-02"), "must match"),
    ("synthetic", np.array(False), "synthetic"),
    ("normalization", np.array("normalized"), "raw"),
    ("sar", np.zeros((2, 1, 16, 16)), "share the optical grid"),
    ("sar", np.ones((2, 2, 16, 16)), "zero filled"),
    ("optical_mask", np.array([1, 1]), "boolean"),
    ("observation_mask", np.array([False, True]), "union"),
    ("quality_mask", np.zeros((2, 16, 16), dtype=bool), "valid pixels"),
    ("climate_context", np.zeros((2, 4)), "context vector"),
])
def test_bad_sequences_fail_before_inference(tmp_path, config, field, value, match):
    raw = raw_sequence()
    raw[field] = value
    path = tmp_path / "invalid.npz"
    np.savez(path, **raw)
    with pytest.raises(ValueError, match=match):
        load_sequence(path, config, observed_through="2025-06-01", synthetic_input=True)


def test_missing_masks_and_hidden_pickle_are_rejected(tmp_path, config):
    raw = raw_sequence()
    del raw["optical_mask"]
    path = tmp_path / "missing.npz"
    np.savez(path, **raw)
    with pytest.raises(ValueError, match="explicit arrays"):
        load_sequence(path, config, observed_through="2025-06-01", synthetic_input=True)
    raw = raw_sequence()
    raw["unsafe_extra"] = np.array({"metadata": "object"}, dtype=object)
    np.savez(path, **raw)
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        load_sequence(path, config, observed_through="2025-06-01", synthetic_input=True)


def test_native_backbone_rejected_before_constructor_or_download(tmp_path, config, monkeypatch):
    import forestmind.inference.predict as module
    native = deepcopy(config)
    native["foundation"] = {"name": "prithvi", "prithvi": {"use_proxy": False}}
    path = tmp_path / "native.pt"
    torch.save({"config": native, "model_state_dict": {}}, path)
    monkeypatch.setattr(module, "ForestMindModel", lambda _: pytest.fail("Must not initialize a download-capable native model"))
    with pytest.raises(ValueError, match="offline inference"):
        predict_checkpoint(path, "unused.npz", region_id="xingu", observed_through="2025-06-01", synthetic_input=True)


def test_pickle_checkpoint_cannot_execute_arbitrary_payload(tmp_path):
    path = tmp_path / "unsafe.pt"
    torch.save({"model_state_dict": {}, "config": {}, "unsafe": Path("untrusted")}, path)
    import pickle
    with pytest.raises(pickle.UnpicklingError):
        predict_checkpoint(path, "unused.npz", region_id="xingu", observed_through="2025-06-01", synthetic_input=True)


def test_cli_saves_report_without_modifying_sources(checkpoint, sequence, tmp_path):
    from scripts.predict import main
    output = tmp_path / "report.json"
    assert main(["--checkpoint", str(checkpoint), "--input", str(sequence), "--region-id", "xingu",
                 "--observed-through", "2025-06-01", "--synthetic-input", "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["kind"] == "forestmind_inference"
    assert main(["--checkpoint", str(checkpoint), "--input", str(sequence), "--region-id", "xingu",
                 "--observed-through", "2025-06-01", "--synthetic-input", "--output", str(sequence)]) == 2
    with pytest.raises(ValueError, match="overwrite"):
        write_inference_report(result, checkpoint, protected_paths=[checkpoint])


def test_hardlink_report_output_cannot_replace_a_source(checkpoint, sequence, tmp_path, monkeypatch):
    from scripts import predict
    output = tmp_path / "checkpoint-alias.json"
    output.hardlink_to(checkpoint)
    before = checkpoint.read_bytes()
    monkeypatch.setattr(predict, "predict_checkpoint", lambda *a, **kw: pytest.fail("Reject output before inference"))
    assert predict.main(["--checkpoint", str(checkpoint), "--input", str(sequence), "--region-id", "xingu",
                         "--observed-through", "2025-06-01", "--synthetic-input", "--output", str(output)]) == 2
    with pytest.raises(ValueError, match="overwrite"):
        write_inference_report(example_report(), output, protected_paths=[checkpoint])
    assert checkpoint.read_bytes() == before


def test_report_cutoff_uses_utc_creation_day():
    report = example_report()
    report["created_at"] = "2025-06-01T00:30:00+14:00"  # Still May 31 in UTC.
    with pytest.raises(ValueError, match="cannot follow report creation"):
        InferenceReport.model_validate(report)


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(calibration="calibrated"),
    lambda r: r.update(schema_version=True),
    lambda r: r.update(observed_through="20250601"),
    lambda r: r.update(observed_through="2026-06-01"),
    lambda r: r.update(created_at="2025-06-02T12:00:00"),
    lambda r: r["input"].update(synthetic="false"),
    lambda r: r["input"].update(sha256="none"),
    lambda r: r["input"].update(name="C:\\private\\sample.npz"),
    lambda r: r["input"].update(timesteps=1, observed_timesteps=2),
    lambda r: r["checkpoint"].update(training_source="C:/private/train"),
    lambda r: r["checkpoint"].update(trained_on_synthetic=False),
    lambda r: r["predictions"]["health"].update(score=float("nan")),
    lambda r: r["predictions"]["health"].update(score=101),
    lambda r: r["predictions"]["health"].update(state="Cleared"),
    lambda r: r["predictions"]["health"]["class_scores"][0].update(score=0.7),
    lambda r: r["predictions"]["drivers"][0].update(name="Fabricated"),
    lambda r: r["predictions"]["trajectory"][1].update(horizon_days=30),
    lambda r: r["predictions"]["trajectory"][0].update(horizon_days=True),
    lambda r: r["predictions"]["trajectory"][0].update(stddev=-1),
])
def test_report_rejects_false_claims_nonfinite_paths_and_inconsistent_dimensions(mutate):
    report = example_report()
    mutate(report)
    with pytest.raises(ValueError):
        InferenceReport.model_validate(report)
