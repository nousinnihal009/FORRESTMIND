"""
Tests for Phase 2: Real data adapters, band harmonization, and foundation models.

Tests are designed to run without internet access or real satellite data
by using mock objects and synthetic test data.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from forestmind.utils.geo import BoundingBox


# ─── Adapter Base Tests ───────────────────────────────────────────────────────

class TestAdapterBase:
    """Test the SatelliteAdapterBase interface and utilities."""

    def test_imports(self):
        from forestmind.data.adapters import (
            SatelliteAdapterBase,
            SceneMetadata,
            PatchData,
            TimeSeriesData,
            Sentinel2Adapter,
            Sentinel1Adapter,
            LandsatAdapter,
            BandHarmonizer,
        )

    def test_scene_metadata(self):
        from forestmind.data.adapters.base import SceneMetadata

        scene = SceneMetadata(
            scene_id="S2A_20230601T123456",
            sensor="sentinel2",
            datetime=datetime(2023, 6, 1, 12, 34, 56),
            bbox=BoundingBox(west=-63.5, south=-11.5, east=-62.5, north=-10.5),
            cloud_cover=15.3,
        )
        assert scene.date_str == "2023-06-01"
        assert scene.sensor == "sentinel2"
        assert scene.cloud_cover == 15.3

    def test_patch_data(self):
        from forestmind.data.adapters.base import PatchData

        patch = PatchData(
            data=np.random.rand(12, 128, 128).astype(np.float32),
            timestamp=0.0,
            datetime=datetime(2023, 6, 1),
            sensor="sentinel2",
            band_names=["B01", "B02", "B03", "B04", "B05", "B06",
                         "B07", "B08", "B8A", "B09", "B11", "B12"],
            bbox=BoundingBox(west=-63.5, south=-11.5, east=-62.5, north=-10.5),
        )
        assert patch.data.shape == (12, 128, 128)
        assert patch.sensor == "sentinel2"
        assert len(patch.band_names) == 12

    def test_cache_path_deterministic(self):
        from forestmind.data.adapters.base import SceneMetadata

        # Create a concrete subclass for testing
        from forestmind.data.adapters.sentinel2 import Sentinel2Adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = Sentinel2Adapter(cache_dir=tmpdir)
            bbox = BoundingBox(west=-63.5, south=-11.5, east=-62.5, north=-10.5)
            scene = SceneMetadata(
                scene_id="test_scene_001",
                sensor="sentinel2",
                datetime=datetime(2023, 6, 1),
                bbox=bbox,
            )

            path1 = adapter.cache_path_for_scene(scene, bbox)
            path2 = adapter.cache_path_for_scene(scene, bbox)
            assert path1 == path2  # deterministic

            # Different bbox → different path
            bbox2 = BoundingBox(west=-64.0, south=-12.0, east=-63.0, north=-11.0)
            path3 = adapter.cache_path_for_scene(scene, bbox2)
            assert path1 != path3

    def test_save_and_load_cache(self):
        from forestmind.data.adapters.base import SceneMetadata
        from forestmind.data.adapters.sentinel2 import Sentinel2Adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = Sentinel2Adapter(cache_dir=tmpdir)
            bbox = BoundingBox(west=-63.5, south=-11.5, east=-62.5, north=-10.5)
            scene = SceneMetadata(
                scene_id="cache_test",
                sensor="sentinel2",
                datetime=datetime(2023, 6, 1),
                bbox=bbox,
                cloud_cover=12.5,
            )

            # Create test data
            data = np.random.rand(12, 128, 128).astype(np.float32)
            quality_mask = np.ones((128, 128), dtype=bool)

            # Save
            path = adapter.cache_path_for_scene(scene, bbox)
            adapter.save_patch_to_cache(path, data, scene, bbox, quality_mask)

            # Load
            patch = adapter.load_patch_from_cache(path)
            assert patch.data.shape == (12, 128, 128)
            assert patch.sensor == "sentinel2"
            np.testing.assert_allclose(patch.data, data, atol=1e-6)
            assert patch.cloud_cover == pytest.approx(12.5)
            assert patch.quality_mask is not None
            assert patch.quality_mask.shape == (128, 128)


# ─── Sentinel-2 Adapter Tests ────────────────────────────────────────────────

class TestSentinel2Adapter:
    """Test Sentinel-2 adapter configuration and utilities."""

    def test_adapter_properties(self):
        from forestmind.data.adapters.sentinel2 import Sentinel2Adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = Sentinel2Adapter(cache_dir=tmpdir)
            assert adapter.sensor_name == "sentinel2"
            assert adapter.num_bands == 12
            assert len(adapter.band_names) == 12
            assert "B08" in adapter.band_names
            assert "B11" in adapter.band_names

    def test_ndvi_computation(self):
        from forestmind.data.adapters.sentinel2 import compute_ndvi

        # Create a patch with known values
        patch = np.zeros((12, 64, 64), dtype=np.float32)
        patch[3] = 0.04  # B04 (Red)
        patch[7] = 0.35  # B08 (NIR)

        ndvi = compute_ndvi(patch)
        assert ndvi.shape == (64, 64)
        # NDVI = (0.35 - 0.04) / (0.35 + 0.04) ≈ 0.795
        expected = (0.35 - 0.04) / (0.35 + 0.04 + 1e-8)
        np.testing.assert_allclose(ndvi, expected, atol=1e-4)

    def test_ndvi_temporal(self):
        from forestmind.data.adapters.sentinel2 import compute_ndvi

        patch = np.random.rand(6, 12, 64, 64).astype(np.float32) * 0.5
        ndvi = compute_ndvi(patch)
        assert ndvi.shape == (6, 64, 64)

    def test_evi_computation(self):
        from forestmind.data.adapters.sentinel2 import compute_evi

        patch = np.zeros((12, 64, 64), dtype=np.float32)
        patch[1] = 0.03  # B02 (Blue)
        patch[3] = 0.04  # B04 (Red)
        patch[7] = 0.35  # B08 (NIR)

        evi = compute_evi(patch)
        assert evi.shape == (64, 64)
        assert evi.min() >= -1.0
        assert evi.max() <= 1.0


# ─── Sentinel-1 Adapter Tests ────────────────────────────────────────────────

class TestSentinel1Adapter:
    """Test Sentinel-1 adapter configuration and utilities."""

    def test_adapter_properties(self):
        from forestmind.data.adapters.sentinel1 import Sentinel1Adapter

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = Sentinel1Adapter(cache_dir=tmpdir)
            assert adapter.sensor_name == "sentinel1"
            assert adapter.num_bands == 2
            assert adapter.band_names == ["VV", "VH"]

    def test_rfdi_computation(self):
        from forestmind.data.adapters.sentinel1 import compute_rfdi

        patch = np.array([
            [[0.6, 0.4], [0.3, 0.2]],  # VV
            [[0.4, 0.3], [0.1, 0.05]],  # VH
        ], dtype=np.float32)

        rfdi = compute_rfdi(patch)
        assert rfdi.shape == (2, 2)
        # RFDI = (VV - VH) / (VV + VH)
        expected_00 = (0.6 - 0.4) / (0.6 + 0.4 + 1e-8)
        np.testing.assert_allclose(rfdi[0, 0], expected_00, atol=1e-4)

    def test_vv_vh_ratio(self):
        from forestmind.data.adapters.sentinel1 import compute_vv_vh_ratio

        patch = np.array([
            [[0.6, 0.3]],  # VV
            [[0.4, 0.2]],  # VH
        ], dtype=np.float32)

        ratio = compute_vv_vh_ratio(patch)
        assert ratio.shape == (1, 2)
        np.testing.assert_allclose(ratio[0, 0], 0.6 / 0.4, atol=1e-4)


# ─── Landsat Adapter Tests ───────────────────────────────────────────────────

class TestLandsatAdapter:
    """Test Landsat adapter configuration."""

    def test_adapter_properties(self):
        from forestmind.data.adapters.landsat import LandsatAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = LandsatAdapter(cache_dir=tmpdir)
            assert adapter.sensor_name == "landsat"
            assert adapter.num_bands == 7
            assert len(adapter.band_names) == 7

    def test_platform_filter(self):
        from forestmind.data.adapters.landsat import LandsatAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = LandsatAdapter(cache_dir=tmpdir, platforms=["landsat-9"])
            assert adapter.platforms == ["landsat-9"]


# ─── Band Harmonization Tests ────────────────────────────────────────────────

class TestBandHarmonizer:
    """Test cross-sensor band harmonization."""

    def test_landsat_to_s2(self):
        from forestmind.data.adapters.harmonize import BandHarmonizer

        harmonizer = BandHarmonizer()

        # 7-band Landsat patch
        landsat = np.random.rand(7, 64, 64).astype(np.float32) * 0.3
        result = harmonizer.harmonize_landsat(landsat)

        assert result.shape == (12, 64, 64)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_sentinel2_passthrough(self):
        from forestmind.data.adapters.harmonize import BandHarmonizer

        harmonizer = BandHarmonizer()

        s2 = np.random.rand(12, 64, 64).astype(np.float32) * 0.3
        result = harmonizer.harmonize(s2, "sentinel2")

        assert result.shape == (12, 64, 64)
        np.testing.assert_array_equal(result, s2)

    def test_harmonize_wrong_bands(self):
        from forestmind.data.adapters.harmonize import BandHarmonizer

        harmonizer = BandHarmonizer()

        with pytest.raises(ValueError, match="Expected 7-band"):
            harmonizer.harmonize_landsat(np.zeros((5, 64, 64), dtype=np.float32))

    def test_harmonize_unknown_sensor(self):
        from forestmind.data.adapters.harmonize import BandHarmonizer

        harmonizer = BandHarmonizer()

        with pytest.raises(ValueError, match="Unsupported"):
            harmonizer.harmonize(np.zeros((12, 64, 64), dtype=np.float32), "unknown_sensor")

    def test_harmonize_time_series(self):
        from forestmind.data.adapters.harmonize import BandHarmonizer

        harmonizer = BandHarmonizer()

        # Mixed S2 + Landsat time series
        patches = [
            np.random.rand(12, 64, 64).astype(np.float32) * 0.3,  # S2
            np.random.rand(7, 64, 64).astype(np.float32) * 0.3,   # Landsat
            np.random.rand(12, 64, 64).astype(np.float32) * 0.3,  # S2
        ]
        sensors = ["sentinel2", "landsat8", "sentinel2"]

        result = harmonizer.harmonize_time_series(patches, sensors)
        assert result.shape == (3, 12, 64, 64)

    def test_convenience_function(self):
        from forestmind.data.adapters.harmonize import landsat_to_sentinel2

        # 3D input
        landsat = np.random.rand(7, 64, 64).astype(np.float32) * 0.3
        result = landsat_to_sentinel2(landsat)
        assert result.shape == (12, 64, 64)

        # 4D input
        landsat_ts = np.random.rand(6, 7, 64, 64).astype(np.float32) * 0.3
        result_ts = landsat_to_sentinel2(landsat_ts)
        assert result_ts.shape == (6, 12, 64, 64)

    def test_no_red_edge_interpolation(self):
        from forestmind.data.adapters.harmonize import BandHarmonizer

        harmonizer = BandHarmonizer(interpolate_red_edge=False)
        landsat = np.random.rand(7, 64, 64).astype(np.float32) * 0.3 + 0.1
        result = harmonizer.harmonize_landsat(landsat)

        # Red-edge bands (B05, B06, B07, B8A, B09) at indices 4,5,6,8,9
        # should be zero when interpolation is off
        for idx in [4, 5, 6, 8, 9]:
            assert np.all(result[idx] == 0.0), f"Band index {idx} should be zero"


# ─── Foundation Model Tests ──────────────────────────────────────────────────

class TestPrithviAdapter:
    """Test upgraded Prithvi adapter (proxy mode)."""

    def test_proxy_mode_initialization(self):
        from forestmind.models.foundation.prithvi import PrithviAdapter

        adapter = PrithviAdapter(target_dim=128, use_proxy=True)
        assert adapter.is_proxy_mode
        assert adapter.embedding_dim == 128
        assert "optical" in adapter.supports_modalities

    def test_proxy_mode_encoding(self):
        from forestmind.models.foundation.prithvi import PrithviAdapter

        adapter = PrithviAdapter(target_dim=128, use_proxy=True)

        # 14-channel input (12 optical + 2 SAR)
        x = torch.randn(2, 14, 128, 128)
        embedding = adapter.encode(x)
        assert embedding.shape == (2, 128)

    def test_proxy_mode_6_channel_input(self):
        from forestmind.models.foundation.prithvi import PrithviAdapter

        adapter = PrithviAdapter(target_dim=128, use_proxy=True)

        # 6-channel input (already HLS bands)
        x = torch.randn(2, 6, 128, 128)
        embedding = adapter.encode(x)
        assert embedding.shape == (2, 128)

    def test_freeze_unfreeze(self):
        from forestmind.models.foundation.prithvi import PrithviAdapter

        adapter = PrithviAdapter(target_dim=128, use_proxy=True)
        adapter.freeze()
        assert all(not p.requires_grad for p in adapter.parameters())
        adapter.unfreeze()
        assert all(p.requires_grad for p in adapter.parameters())

    def test_repr(self):
        from forestmind.models.foundation.prithvi import PrithviAdapter

        adapter = PrithviAdapter(target_dim=128, use_proxy=True)
        r = repr(adapter)
        assert "PROXY" in r


class TestTerraMindAdapter:
    """Test upgraded TerraMind adapter (proxy mode)."""

    def test_proxy_mode_initialization(self):
        from forestmind.models.foundation.terramind import TerraMindAdapter

        adapter = TerraMindAdapter(target_dim=128, use_proxy=True)
        assert adapter.is_proxy_mode
        assert adapter.embedding_dim == 128
        assert "optical" in adapter.supports_modalities
        assert "sar" in adapter.supports_modalities

    def test_proxy_mode_encoding(self):
        from forestmind.models.foundation.terramind import TerraMindAdapter

        adapter = TerraMindAdapter(target_dim=128, use_proxy=True)

        # 14-channel input (12 optical + 2 SAR)
        x = torch.randn(2, 14, 128, 128)
        embedding = adapter.encode(x)
        assert embedding.shape == (2, 128)

    def test_gradient_flow(self):
        from forestmind.models.foundation.terramind import TerraMindAdapter

        adapter = TerraMindAdapter(target_dim=128, use_proxy=True)
        adapter.train()

        x = torch.randn(2, 14, 64, 64, requires_grad=True)
        embedding = adapter.encode(x)
        loss = embedding.sum()
        loss.backward()

        # Check gradients flow through
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_repr(self):
        from forestmind.models.foundation.terramind import TerraMindAdapter

        adapter = TerraMindAdapter(target_dim=128, use_proxy=True)
        r = repr(adapter)
        assert "PROXY" in r


# ─── Dataset Extension Tests ─────────────────────────────────────────────────

class TestDatasetExtension:
    """Test the extended ForestMindDataset with real data sources."""

    def test_real_data_source_init_empty(self):
        """Real data source with no cached data should warn but not crash."""
        from forestmind.data.dataset import ForestMindDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            ds = ForestMindDataset(
                source="sentinel2",
                config={"cache_dir": tmpdir},
            )
            assert len(ds) == 0

    def test_real_data_source_with_cache(self):
        """Real data source with cached .npz files should load them."""
        from forestmind.data.dataset import ForestMindDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake cached data
            sentinel_dir = Path(tmpdir) / "sentinel2"
            sentinel_dir.mkdir()

            for i in range(3):
                data = np.random.rand(12, 128, 128).astype(np.float32) * 0.3
                np.savez_compressed(
                    sentinel_dir / f"scene_{i}_abc12345.npz",
                    data=data,
                    timestamp=float(1672531200 + i * 86400 * 30),
                    datetime_iso=f"2023-{i+1:02d}-01T00:00:00",
                    sensor="sentinel2",
                    band_names=["B01", "B02", "B03", "B04", "B05", "B06",
                                "B07", "B08", "B8A", "B09", "B11", "B12"],
                    bbox_west=-63.5,
                    bbox_south=-11.5,
                    bbox_east=-62.5,
                    bbox_north=-10.5,
                    cloud_cover=15.0,
                )

            ds = ForestMindDataset(
                source="sentinel2",
                config={"cache_dir": tmpdir},
            )
            assert len(ds) == 3

    def test_multi_source(self):
        """Multi-source should combine files from all sensor directories."""
        from forestmind.data.dataset import ForestMindDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            for sensor in ["sentinel2", "sentinel1"]:
                sensor_dir = Path(tmpdir) / sensor
                sensor_dir.mkdir()
                bands = 12 if sensor == "sentinel2" else 2
                data = np.random.rand(bands, 128, 128).astype(np.float32)
                np.savez_compressed(
                    sensor_dir / f"test_{sensor}_abc12345.npz",
                    data=data,
                    timestamp=1672531200.0,
                    datetime_iso="2023-01-01T00:00:00",
                    sensor=sensor,
                    band_names=[f"band_{i}" for i in range(bands)],
                    bbox_west=-63.5, bbox_south=-11.5,
                    bbox_east=-62.5, bbox_north=-10.5,
                    cloud_cover=-1.0,
                )

            ds = ForestMindDataset(
                source="multi",
                config={"cache_dir": tmpdir},
            )
            assert len(ds) == 2

    def test_synthetic_still_works(self):
        """Ensure synthetic data source is not broken by the extension."""
        from forestmind.data.dataset import ForestMindDataset
        from forestmind.data.transforms import get_eval_transforms

        ds = ForestMindDataset(
            source="synthetic",
            num_samples=5,
            transform=get_eval_transforms(),
            config={"patch_size": 128, "num_bands": 12, "sar_channels": 2, "time_steps": 12},
        )
        assert len(ds) == 5

        sample = ds[0]
        assert "optical" in sample
        assert "sar" in sample
        assert "timestamps" in sample


# ─── Integration Tests ────────────────────────────────────────────────────────

class TestPhase2Integration:
    """Integration tests for the full Phase 2 pipeline."""

    def test_harmonized_data_through_model(self):
        """Test that harmonized Landsat data flows through the full model."""
        from forestmind.data.adapters.harmonize import BandHarmonizer
        from forestmind.models.forestmind_model import ForestMindModel

        config = {
            "foundation": {"name": "custom_vit", "embedding_dim": 128,
                           "custom_vit": {"patch_embed_size": 16, "depth": 2,
                                          "num_heads": 4, "mlp_ratio": 2.0, "drop_rate": 0.0}},
            "fusion": {"method": "cross_attention", "num_heads": 2, "num_layers": 1, "dropout": 0.0},
            "temporal": {"depth": 2, "num_heads": 4, "dim_feedforward": 256,
                         "dropout": 0.0, "max_seq_len": 24, "use_continuous_pe": True},
            "heads": {"health": {"num_classes": 5, "continuous_score": True},
                      "attribution": {"num_drivers": 8, "multi_label": True},
                      "trajectory": {"horizons": [30, 180, 365, 1095, 1825],
                                     "predict_uncertainty": True}},
        }

        model = ForestMindModel(config)
        model.eval()

        # Simulate harmonized Landsat data
        harmonizer = BandHarmonizer()
        landsat_raw = np.random.rand(12, 7, 128, 128).astype(np.float32) * 0.3
        optical = np.stack([harmonizer.harmonize_landsat(landsat_raw[t]) for t in range(12)])
        optical_t = torch.from_numpy(optical).unsqueeze(0)  # (1, 12, 12, 128, 128)
        sar_t = torch.randn(1, 12, 2, 128, 128)
        timestamps = torch.arange(12, dtype=torch.float32).unsqueeze(0) * 30

        with torch.no_grad():
            outputs = model(optical_t, sar_t, timestamps)

        assert "health" in outputs
        assert "attribution" in outputs
        assert "trajectory" in outputs
        assert outputs["health"]["score"].shape == (1,)

    def test_prithvi_in_full_model(self):
        """Test ForestMind model with Prithvi adapter (proxy mode)."""
        from forestmind.models.forestmind_model import ForestMindModel

        config = {
            "foundation": {"name": "prithvi", "embedding_dim": 128,
                           "prithvi": {"model_name": "test", "freeze": False}},
            "fusion": {"method": "cross_attention", "num_heads": 2, "num_layers": 1, "dropout": 0.0},
            "temporal": {"depth": 2, "num_heads": 4, "dim_feedforward": 256,
                         "dropout": 0.0, "max_seq_len": 24, "use_continuous_pe": True},
            "heads": {"health": {"num_classes": 5, "continuous_score": True},
                      "attribution": {"num_drivers": 8, "multi_label": True},
                      "trajectory": {"horizons": [30, 180, 365, 1095, 1825],
                                     "predict_uncertainty": True}},
        }

        model = ForestMindModel(config)
        model.eval()

        optical = torch.randn(1, 6, 12, 128, 128)
        sar = torch.randn(1, 6, 2, 128, 128)
        timestamps = torch.arange(6, dtype=torch.float32).unsqueeze(0) * 30

        with torch.no_grad():
            outputs = model(optical, sar, timestamps)

        assert outputs["health"]["score"].shape == (1,)

    def test_terramind_in_full_model(self):
        """Test ForestMind model with TerraMind adapter (proxy mode)."""
        from forestmind.models.forestmind_model import ForestMindModel

        config = {
            "foundation": {"name": "terramind", "embedding_dim": 128,
                           "terramind": {"model_name": "test", "freeze": False}},
            "fusion": {"method": "cross_attention", "num_heads": 2, "num_layers": 1, "dropout": 0.0},
            "temporal": {"depth": 2, "num_heads": 4, "dim_feedforward": 256,
                         "dropout": 0.0, "max_seq_len": 24, "use_continuous_pe": True},
            "heads": {"health": {"num_classes": 5, "continuous_score": True},
                      "attribution": {"num_drivers": 8, "multi_label": True},
                      "trajectory": {"horizons": [30, 180, 365, 1095, 1825],
                                     "predict_uncertainty": True}},
        }

        model = ForestMindModel(config)
        model.eval()

        optical = torch.randn(1, 6, 12, 128, 128)
        sar = torch.randn(1, 6, 2, 128, 128)
        timestamps = torch.arange(6, dtype=torch.float32).unsqueeze(0) * 30

        with torch.no_grad():
            outputs = model(optical, sar, timestamps)

        assert outputs["health"]["score"].shape == (1,)
