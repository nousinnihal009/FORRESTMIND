"""Offline native-loading contracts; tiny registry doubles are not pretrained weights."""

from copy import deepcopy
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

from forestmind.inference.predict import predict_checkpoint
from forestmind.models.foundation import native
from forestmind.models.forestmind_model import ForestMindModel


RUNTIME = {"terratorch": "1.2.13", "torch": "2.10.0", "torchvision": "0.25.0", "timm": "1.0.27"}


def native_config(name="terramind"):
    options = {"model_name": "terramind_v1_tiny" if name == "terramind" else "prithvi_eo_v2_300",
               "pretrained": True, "use_proxy": False, "freeze": True}
    if name == "terramind":
        options["sar_modality"] = "S1RTC"
    return {
        "data": {"source": "synthetic", "optical_units": "surface_reflectance_0_1", "sar_units": "gamma0_db",
                 "normalization": {"optical_mean": [.1] * 12, "optical_std": [.2] * 12,
                                   "sar_mean": [-10., -17.], "sar_std": [5., 6.]}},
        "foundation": {"name": name, "embedding_dim": 16, name: options, "native_runtime": dict(RUNTIME)},
        "fusion": {"num_heads": 2, "num_layers": 1},
        "temporal": {"depth": 1, "num_heads": 2, "dim_feedforward": 32, "max_seq_len": 4, "dropout": 0.},
        "heads": {"trajectory": {"horizons": [30, 90], "predict_uncertainty": True}},
    }


class TinyNativeBackbone(nn.Module):
    """A deliberately small interface double for round-trip state verification."""
    out_channels = [8]

    def __init__(self, name, calls):
        super().__init__()
        self.name, self.calls = name, calls
        self.optical = nn.Linear(12 if name.startswith("terra") else 6, 8)
        self.sar = nn.Linear(2, 8) if name.startswith("terra") else None

    def forward(self, values):
        if isinstance(values, dict):
            self.calls.append({"forward_modalities": sorted(values)})
            encoded = [self.optical(v.mean((-1, -2))) if key == "S2L2A" else self.sar(v.mean((-1, -2)))
                       for key, v in values.items()]
            output = torch.stack(encoded).mean(0)
        else:
            output = self.optical(values.mean((-1, -2, -3)))
        return [output[:, None]]


@pytest.fixture
def fake_native(monkeypatch):
    calls = []

    def build(name, **kwargs):
        # Any accidental pretrained=True in the inference path fails this test.
        assert kwargs["pretrained"] is False
        assert "ckpt_path" not in kwargs
        calls.append({"name": name, **kwargs})
        return TinyNativeBackbone(name, calls)

    package = types.ModuleType("terratorch")
    registry = types.ModuleType("terratorch.registry")
    registry.BACKBONE_REGISTRY = types.SimpleNamespace(build=build)
    package.registry = registry
    monkeypatch.setitem(sys.modules, "terratorch", package)
    monkeypatch.setitem(sys.modules, "terratorch.registry", registry)
    monkeypatch.setattr(native, "capture_native_runtime", lambda: dict(RUNTIME))
    return calls


def sample_sequence():
    return {"optical": np.full((2, 12, 16, 16), .2, np.float32),
            "sar": np.zeros((2, 2, 16, 16), np.float32), "timestamps": np.array([0., 31.]),
            "observation_mask": np.array([True, True]), "optical_mask": np.array([True, True]),
            "sar_mask": np.array([False, False]), "quality_mask": np.ones((2, 16, 16), bool),
            "normalization": np.asarray("raw"), "synthetic": np.asarray(True),
            "observed_through": np.asarray("2025-06-01"),
            "acquisition_dates": np.array(["2025-05-01", "2025-06-01"])}


@pytest.mark.parametrize("name", ["prithvi", "terramind"])
def test_native_complete_roundtrip_never_requests_pretrained_weights(tmp_path, fake_native, name):
    config = native_config(name)
    torch.manual_seed(11)
    model = ForestMindModel(native.offline_native_config(config))
    checkpoint, sequence = tmp_path / "model.pt", tmp_path / "sequence.npz"
    torch.save({"config": config, "model_state_dict": model.state_dict(), "epoch": 0}, checkpoint)
    np.savez(sequence, **sample_sequence())
    first = predict_checkpoint(checkpoint, sequence, region_id="example", observed_through="2025-06-01", synthetic_input=True)
    second = predict_checkpoint(checkpoint, sequence, region_id="example", observed_through="2025-06-01", synthetic_input=True)
    assert first["predictions"] == second["predictions"]
    assert first["checkpoint"]["backbone"] == name
    assert first["checkpoint"]["trained_on_synthetic"] is True
    assert first["calibration"] == "uncalibrated"
    assert "complete local ForestMind state" in " ".join(first["caveats"])
    assert all(call["pretrained"] is False for call in fake_native if "pretrained" in call)
    assert config["foundation"][name]["pretrained"] is True  # No mutation of provenance.


def test_terramind_rtc_is_never_relabelled_as_grd(fake_native):
    model = ForestMindModel(native.offline_native_config(native_config()))
    model.foundation.encode_modalities(torch.zeros(2, 12, 16, 16), torch.zeros(2, 2, 16, 16),
                                      torch.tensor([True, False]), torch.tensor([False, True]))
    assert fake_native[0]["modalities"] == ["S2L2A", "S1RTC"]
    assert {tuple(call["forward_modalities"]) for call in fake_native if "forward_modalities" in call} == {("S2L2A",), ("S1RTC",)}


@pytest.mark.parametrize("change,match", [
    (lambda c: c["foundation"].pop("native_runtime"), "exact foundation.native_runtime"),
    (lambda c: c["foundation"]["native_runtime"].update(terratorch="999"), "audited terratorch"),
    (lambda c: c["foundation"]["native_runtime"].update(torch="old"), "runtime differs"),
    (lambda c: c["foundation"]["terramind"].update(model_name="terramind_v1_tiny_tim"), "Unsupported native"),
    (lambda c: c["foundation"]["terramind"].update(model_name=["terramind_v1_tiny"]), "Unsupported native"),
    (lambda c: c["foundation"]["terramind"].update(ckpt_path="https://example.invalid/weights"), "custom constructor"),
    (lambda c: c["foundation"]["terramind"].update(pretrained="false"), "explicit boolean"),
    (lambda c: c["foundation"]["terramind"].update(freeze=1), "explicit boolean"),
    (lambda c: c["data"].update(optical_units="dn"), "optical_units"),
    (lambda c: c["data"].update(sar_units="sigma0_db"), "matching sar_modality"),
    (lambda c: c["data"].pop("normalization"), "normalization"),
    (lambda c: c["data"]["normalization"].update(sar_std=[0., 1.]), "normalization"),
])
def test_bad_native_contract_fails_before_constructor_or_input(tmp_path, fake_native, change, match):
    config = native_config()
    change(config)
    checkpoint = tmp_path / "invalid.pt"
    torch.save({"config": config, "model_state_dict": {"placeholder": torch.ones(1)}}, checkpoint)
    with pytest.raises(ValueError, match=match):
        predict_checkpoint(checkpoint, "nonexistent-input.npz", region_id="example", observed_through="2025-06-01", synthetic_input=True)
    assert fake_native == []


@pytest.mark.parametrize("mutation,match", [
    (lambda s: s.pop("bias"), "missing"),
    (lambda s: s.update(extra=torch.ones(1)), "unexpected"),
    (lambda s: s.update(weight=s["weight"].double()), "dtype"),
    (lambda s: s.update(weight=s["weight"][:, :1]), "shape"),
    (lambda s: s.update(weight=s["weight"] * float("nan")), "nonfinite"),
    (lambda s: s.update(weight=s["weight"] * float("inf")), "nonfinite"),
    (lambda s: s.update(weight="not a tensor"), "dense"),
])
def test_partial_corrupt_or_coerced_checkpoint_never_loads(mutation, match):
    model = nn.Linear(3, 2)
    original = deepcopy(model.state_dict())
    state = deepcopy(original)
    mutation(state)
    with pytest.raises(ValueError, match=match):
        native.load_complete_state(model, state)
    assert all(torch.equal(value, original[key]) for key, value in model.state_dict().items())


def test_complete_state_load_copies_all_parameters():
    model = nn.Linear(3, 2)
    state = {key: torch.full_like(value, .25) for key, value in model.state_dict().items()}
    native.load_complete_state(model, state)
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())


def test_prithvi_rejects_sar_instead_of_silently_ignoring_it():
    config, sample = native_config("prithvi"), sample_sequence()
    sample["sar_mask"][0] = True
    with pytest.raises(ValueError, match="silently discarded"):
        native.validate_native_sample(config, sample)
    sample["optical_mask"][0] = False
    with pytest.raises(ValueError, match="SAR-only"):
        native.validate_native_sample(config, sample)


def test_native_patch_geometry_checked_before_allocation():
    config, sample = native_config("prithvi"), sample_sequence()
    config["foundation"]["prithvi"]["model_name"] = "prithvi_eo_v2_600"
    with pytest.raises(ValueError, match="14-pixel"):
        native.validate_native_sample(config, sample)


def test_missing_runtime_is_actionable(monkeypatch):
    def missing(_):
        raise native.metadata.PackageNotFoundError("terratorch")
    monkeypatch.setattr(native.metadata, "version", missing)
    with pytest.raises(ValueError, match="requires installed terratorch"):
        native.capture_native_runtime()


def test_evaluation_rejects_observed_sar_before_native_prithvi_forward(tmp_path, fake_native, monkeypatch):
    from forestmind.training.evaluation import evaluate_checkpoint

    config = native_config("prithvi")
    config["device"] = "cpu"
    config["data"].update(patch_size=16, time_steps=2, num_train_samples=1,
                          num_val_samples=1, num_test_samples=1, batch_size=1, num_workers=0)
    model = ForestMindModel(native.offline_native_config(config))
    checkpoint = tmp_path / "native-evaluation.pt"
    torch.save({"config": config, "model_state_dict": model.state_dict()}, checkpoint)
    monkeypatch.setattr(ForestMindModel, "forward", lambda *args, **kwargs: pytest.fail("Input contract must fail before forward"))
    with pytest.raises(ValueError, match="Prithvi is optical-only"):
        evaluate_checkpoint(checkpoint)


def test_native_sample_validation_supports_tensor_batches():
    sample = {key: torch.from_numpy(value) for key, value in sample_sequence().items()
              if key in ("optical", "observation_mask", "optical_mask", "sar_mask")}
    native.validate_native_sample(native_config("prithvi"), sample)
    sample["sar_mask"][0] = True
    with pytest.raises(ValueError, match="silently discarded"):
        native.validate_native_sample(native_config("prithvi"), sample)


@pytest.mark.parametrize("explicit_masks", [True, False])
def test_training_rejects_sar_before_moving_native_prithvi_inputs(explicit_masks):
    from forestmind.training.trainer import ForestMindTrainer

    trainer = object.__new__(ForestMindTrainer)
    trainer.config = native_config("prithvi")
    sample = sample_sequence()
    sample["sar_mask"][0] = True
    batch = {key: torch.from_numpy(value).unsqueeze(0) for key, value in sample.items()
             if key in ("optical", "observation_mask", "optical_mask", "sar_mask")}
    if not explicit_masks:
        batch = {"optical": batch["optical"]}
    with pytest.raises(ValueError, match="silently discarded"):
        trainer._prepare_batch(batch)


def test_verification_network_guard_blocks_connections_and_restores_socket():
    import importlib.util
    from pathlib import Path
    import socket

    spec = importlib.util.spec_from_file_location("verify_native_runtime", Path(__file__).resolve().parents[1] / "scripts" / "verify_native_runtime.py")
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    original = socket.create_connection
    with diagnostic.deny_network() as attempts:
        with pytest.raises(RuntimeError, match="Network access is disabled"):
            socket.create_connection(("example.invalid", 443))
        assert len(attempts) == 1
    assert socket.create_connection is original
