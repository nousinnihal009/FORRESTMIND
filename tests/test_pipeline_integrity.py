"""Regression coverage for missing observations and scientific data contracts."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from forestmind.data.dataset import ForestMindDataset, RealDataTimeSeriesDataset, create_dataloaders
from forestmind.data.transforms import get_eval_transforms
from forestmind.models.forestmind_model import ForestMindModel
from forestmind.models.fusion.multimodal import MultimodalFusion
from forestmind.models.temporal.transformer import TemporalTransformer
from forestmind.training.losses import HealthLoss
from forestmind.training.metrics import ForestMindMetrics


CONFIG = {
    "foundation": {"name": "custom_vit", "embedding_dim": 32,
                   "custom_vit": {"depth": 1, "num_heads": 4, "drop_rate": 0.0}},
    "temporal": {"depth": 1, "num_heads": 4, "dropout": 0.0},
}


def cache(path, sensor="sentinel2", day=0, value=0.2):
    channels = 2 if sensor == "sentinel1" else 12
    np.savez(path, data=np.full((channels, 32, 32), value, dtype=np.float32),
             sensor=sensor, timestamp=1700000000.0 + day * 86400,
             bbox_west=1., bbox_east=2., bbox_south=3., bbox_north=4.)


def test_single_acquisition_does_not_fabricate_history_or_labels(tmp_path):
    cache(tmp_path / "scene_location.npz")
    sample = ForestMindDataset(str(tmp_path), config={"time_steps": 4},
                               transform=get_eval_transforms())[0]
    assert sample["observation_mask"].tolist() == [True, False, False, False]
    assert sample["sar_mask"].tolist() == [False] * 4
    assert not sample["label_valid"]
    assert "health_state" not in sample and "trajectory" not in sample
    assert torch.count_nonzero(sample["optical"][1:]) == 0
    assert torch.count_nonzero(sample["sar"]) == 0
    assert torch.count_nonzero(sample["timestamps"]) == 0


def test_real_sequences_use_metadata_order_and_never_future_sar(tmp_path):
    optical, sar = tmp_path / "optical", tmp_path / "sar"
    optical.mkdir()
    sar.mkdir()
    cache(optical / "z_early.npz", day=0, value=0.1)
    cache(optical / "a_late.npz", day=30, value=0.3)
    cache(sar / "before.npz", "sentinel1", day=-1, value=0.4)
    cache(sar / "future.npz", "sentinel1", day=31, value=0.9)
    sample = RealDataTimeSeriesDataset(optical, sar, time_steps=3)[0]
    assert sample["timestamps"].tolist() == [0, 30, 30]
    assert sample["observation_mask"].tolist() == [True, True, False]
    assert sample["sar_mask"].tolist() == [True, False, False]
    assert sample["sar"][0].mean() == pytest.approx(0.4)
    assert not sample["sar"][1].any()
    assert sample["optical"][0].mean() == pytest.approx(0.1)


def test_unsplit_disk_data_cannot_leak_into_evaluation(tmp_path):
    with pytest.raises(ValueError, match="explicit train/val/test"):
        create_dataloaders({"data": {"source": str(tmp_path)}})


def test_raw_real_data_cannot_train_on_placeholder_targets():
    with pytest.raises(ValueError, match="no supervised targets"):
        create_dataloaders({"data": {"source": "sentinel2"}})


def test_padding_is_ignored_end_to_end_even_if_it_contains_nan():
    torch.manual_seed(7)
    model = ForestMindModel(CONFIG).eval()
    optical = torch.rand(1, 2, 12, 32, 32)
    sar = torch.rand(1, 2, 2, 32, 32)
    times = torch.tensor([[0., 30.]])
    with torch.no_grad():
        expected = model(optical, sar, times)
        actual = model(
            torch.cat([optical, torch.full_like(optical[:, :1], float("nan"))], 1),
            torch.cat([sar, torch.full_like(sar[:, :1], float("nan"))], 1),
            torch.tensor([[0., 30., float("nan")]]),
            observation_mask=torch.tensor([[True, True, False]]),
        )
    for head, key in (("health", "score"), ("attribution", "probs"), ("trajectory", "mean")):
        torch.testing.assert_close(expected[head][key], actual[head][key], atol=1e-5, rtol=1e-5)


def test_missing_modality_cannot_influence_predictions():
    model = ForestMindModel(CONFIG).eval()
    optical = torch.rand(1, 2, 12, 32, 32)
    times = torch.tensor([[0., 30.]])
    with torch.no_grad():
        absent = model(optical, None, times)
        masked = model(optical, torch.full((1, 2, 2, 32, 32), float("nan")), times,
                       sar_mask=torch.zeros(1, 2, dtype=torch.bool))
    torch.testing.assert_close(absent["health"]["score"], masked["health"]["score"])


def test_all_missing_observations_are_rejected():
    model = ForestMindModel(CONFIG).eval()
    with pytest.raises(ValueError, match="at least one observed"):
        model(torch.zeros(1, 2, 12, 32, 32), None, torch.zeros(1, 2),
              observation_mask=torch.zeros(1, 2, dtype=torch.bool))


def test_temporal_transformer_rejects_time_reversal():
    model = TemporalTransformer(embed_dim=32, num_heads=4, depth=1)
    with pytest.raises(ValueError, match="chronological"):
        model(torch.randn(1, 3, 32), torch.tensor([[0., 30., 10.]]))


def test_causal_temporal_embeddings_cannot_see_later_observations():
    model = TemporalTransformer(embed_dim=32, num_heads=4, depth=2, causal=True,
                                dropout=0.0).eval()
    original = torch.randn(1, 3, 32)
    changed = original.clone()
    changed[:, 2] *= 100
    with torch.no_grad():
        first = model(original)[0]
        second = model(changed)[0]
    torch.testing.assert_close(first[:, :2], second[:, :2])


def test_fusion_does_not_invent_external_context():
    fusion = MultimodalFusion(embed_dim=32)
    embedding = torch.randn(2, 32)
    torch.testing.assert_close(fusion(embedding), embedding)


def test_empty_evaluation_has_no_fictitious_aggregate_score():
    assert ForestMindMetrics().compute().aggregate_score == 0


def test_health_regression_loss_uses_normalized_scale():
    loss = HealthLoss()(torch.zeros(1, 5), torch.tensor([100.]),
                        torch.tensor([0]), torch.tensor([0.]))
    assert loss["health_mse"].item() == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["prithvi", "terramind"])
def test_pretrained_download_failure_never_substitutes_proxy(monkeypatch, name):
    registry = ModuleType("terratorch.registry")
    registry.BACKBONE_REGISTRY = SimpleNamespace(build=Mock(side_effect=OSError("offline")))
    monkeypatch.setitem(sys.modules, "terratorch", ModuleType("terratorch"))
    monkeypatch.setitem(sys.modules, "terratorch.registry", registry)
    with pytest.raises(RuntimeError, match="No proxy was substituted"):
        ForestMindModel({"foundation": {"name": name, "embedding_dim": 32}})


@pytest.mark.parametrize("name", ["prithvi", "terramind"])
def test_native_pretrained_registry_input_contract(monkeypatch, name):
    class Backbone(nn.Module):
        out_channels = [8]

        def forward(self, inputs):
            self.last_inputs = inputs
            return [torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)]

    backbone = Backbone()
    registry = ModuleType("terratorch.registry")
    registry.BACKBONE_REGISTRY = SimpleNamespace(build=Mock(return_value=backbone))
    monkeypatch.setitem(sys.modules, "terratorch", ModuleType("terratorch"))
    monkeypatch.setitem(sys.modules, "terratorch.registry", registry)
    if name == "prithvi":
        from forestmind.models.foundation.prithvi import PrithviAdapter
        adapter = PrithviAdapter(target_dim=8)
    else:
        from forestmind.models.foundation.terramind import TerraMindAdapter
        adapter = TerraMindAdapter(target_dim=8)
    adapter.projection = nn.Identity()
    inputs = torch.arange(14.).reshape(1, 14, 1, 1).expand(1, 14, 32, 32)
    output = adapter(inputs)
    assert not adapter.is_proxy_mode
    assert registry.BACKBONE_REGISTRY.build.call_args.kwargs["pretrained"] is True
    if name == "prithvi":
        assert backbone.last_inputs.shape == (1, 6, 1, 32, 32)
        assert backbone.last_inputs[0, :, 0, 0, 0].tolist() == [1, 2, 3, 8, 10, 11]
        torch.testing.assert_close(output, torch.arange(8.).reshape(1, 8))
    else:
        assert set(backbone.last_inputs) == {"S2L2A", "S1GRD"}
        assert backbone.last_inputs["S2L2A"].shape == (1, 12, 32, 32)
        torch.testing.assert_close(output, torch.arange(8., 16.).reshape(1, 8))


def test_landsat_missing_bands_are_not_invented():
    from forestmind.data.adapters.harmonize import BandHarmonizer
    source = np.full((7, 3, 3), 0.2, dtype=np.float32)
    aligned = BandHarmonizer().harmonize_landsat(source)
    assert not aligned[[4, 5, 6, 8, 9]].any()
    np.testing.assert_equal(aligned[[0, 1, 2, 3, 7, 10, 11]], source)


def test_harmonizer_honors_target_subset():
    from forestmind.data.adapters.harmonize import BandHarmonizer
    harmonizer = BandHarmonizer(target_bands=["B04", "B08"])
    assert harmonizer.harmonize_landsat(np.zeros((7, 3, 3))).shape == (2, 3, 3)
    assert harmonizer.harmonize(np.zeros((12, 3, 3)), "sentinel2").shape == (2, 3, 3)


def test_pretrained_training_requires_checkpoint_normalization():
    with pytest.raises(ValueError, match="explicit data.normalization"):
        create_dataloaders({"foundation": {"name": "terramind"}})


def test_normalization_rejects_invalid_standard_deviation():
    from forestmind.data.transforms import Normalize
    with pytest.raises(ValueError, match="positive"):
        Normalize(sar_std=[0., 1.])


def test_synthetic_dataset_is_lazy_and_repeatable_by_index():
    dataset = ForestMindDataset(num_samples=100_000,
                               config={"patch_size": 32, "time_steps": 2})
    assert dataset.samples is None
    assert len(dataset) == 100_000
    first, repeat, different = dataset[3], dataset[3], dataset[4]
    np.testing.assert_equal(first["optical"], repeat["optical"])
    assert not np.array_equal(first["optical"], different["optical"])


def test_geographic_overlap_between_supervised_splits_is_rejected(tmp_path):
    for split in ("train", "val", "test"):
        directory = tmp_path / split
        directory.mkdir()
        np.savez(directory / "sample.npz", location_id="same-location",
                 health_state=0, health_score=80., drivers=np.zeros(8),
                 driver_strengths=np.zeros(8), trajectory=np.zeros(5))
    with pytest.raises(ValueError, match="Geographic leakage"):
        create_dataloaders({"data": {"source": str(tmp_path)}})


def test_cache_geometry_and_version_are_explicit(tmp_path):
    from datetime import datetime, timezone
    from forestmind.data.adapters.base import PREPROCESSING_VERSION, SceneMetadata
    from forestmind.data.adapters.sentinel2 import Sentinel2Adapter
    from forestmind.utils.geo import BoundingBox

    bbox = BoundingBox(west=1, south=2, east=3, north=4)
    scene = SceneMetadata("scene", "sentinel2", datetime(2024, 1, 1, tzinfo=timezone.utc), bbox)
    adapter = Sentinel2Adapter(cache_dir=tmp_path, patch_size=32)
    path = adapter.cache_path_for_scene(scene, bbox)
    adapter.save_patch_to_cache(path, np.ones((12, 32, 32)), scene, bbox,
                                preprocessing_version=PREPROCESSING_VERSION,
                                data_units="surface_reflectance_0_1")
    with np.load(path, allow_pickle=False) as record:
        assert str(record["preprocessing_version"]) == "forestmind-grid-v2"
        assert str(record["grid_crs"]) == "EPSG:4326"
        np.testing.assert_equal(record["grid_transform"], [0.0625, 0, 1, 0, -0.0625, 4])
        assert str(record["provider"]) == "microsoft-planetary-computer"
        assert str(record["scene_id"]) == "scene"


def test_different_grid_sizes_do_not_reuse_cache(tmp_path):
    from datetime import datetime
    from forestmind.data.adapters.base import SceneMetadata
    from forestmind.data.adapters.sentinel2 import Sentinel2Adapter
    from forestmind.utils.geo import BoundingBox

    bbox = BoundingBox(west=1, south=2, east=3, north=4)
    scene = SceneMetadata("scene", "sentinel2", datetime(2024, 1, 1), bbox)
    small = Sentinel2Adapter(cache_dir=tmp_path, patch_size=32)
    large = Sentinel2Adapter(cache_dir=tmp_path, patch_size=128)
    assert small.cache_path_for_scene(scene, bbox) != large.cache_path_for_scene(scene, bbox)
