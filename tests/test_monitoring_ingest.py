"""Offline acquisition-cache to strict dashboard-import integration checks."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient
import numpy as np
import pytest

from forestmind.api.app import create_app
from forestmind.api.schemas import ImportRegion
from forestmind.monitoring.ingest import (
    REQUIRED_PREPROCESSING_VERSION,
    prepare_cached_observations,
    write_prepared_import,
)


BOX = (-63.5, -11.5, -63.4, -11.4)


def save_cache(directory, filename="scene", *, acquired="2024-01-10T10:00:00+00:00",
               sensor="sentinel2", bbox=BOX, values=None, mask=None, updates=None, omit=()):
    directory.mkdir(parents=True, exist_ok=True)
    data = np.asarray(values if values is not None else [0.6, 0.2, 0.3, 0.1], dtype=np.float32)
    if data.ndim == 1:
        data = np.broadcast_to(data[:, None, None], (4, 2, 2)).copy()
    height, width = data.shape[1:]
    west, south, east, north = bbox
    fields = {
        "data": data, "datetime_iso": acquired,
        "timestamp": datetime.fromisoformat(acquired).timestamp(),
        "sensor": sensor, "band_names": ["B08", "B04", "B11", "B12"] if sensor == "sentinel2" else ["B5", "B4", "B6", "B7"],
        "bbox_west": west, "bbox_south": south, "bbox_east": east, "bbox_north": north,
        "quality_mask": np.ones((height, width), dtype=bool) if mask is None else np.asarray(mask),
        "scene_id": filename, "provider": "microsoft-planetary-computer",
        "collection": "sentinel-2-l2a" if sensor == "sentinel2" else "landsat-c2-l2",
        "preprocessing_version": REQUIRED_PREPROCESSING_VERSION,
        "data_units": "surface_reflectance_0_1", "grid_crs": "EPSG:4326",
        "grid_transform": [(east - west) / width, 0, west, 0, -(north - south) / height, north],
    }
    fields.update(updates or {})
    for key in omit:
        fields.pop(key, None)
    path = directory / (filename + ".npz")
    np.savez_compressed(path, **fields)
    return path


def prepare(root, **kwargs):
    return prepare_cached_observations(root, sensor=kwargs.pop("sensor", "sentinel2"), name="Observed AOI", **kwargs)


def test_monthly_aggregation_preserves_actual_dates_masks_and_gaps(tmp_path):
    cache = tmp_path / "sentinel2"
    save_cache(cache, "first", mask=[[1, 0], [0, 0]])
    save_cache(cache, "second", acquired="2024-01-29T12:30:00+00:00",
               values=[0.8, 0.2, 0.4, 0.2], mask=[[1, 1], [1, 1]])
    save_cache(cache, "march", acquired="2024-03-05T12:30:00+00:00")
    prepared = prepare(tmp_path)
    assert [row["date"] for row in prepared.payload["observations"]] == ["2024-01-29", "2024-03-05"]
    january = prepared.payload["observations"][0]
    assert january["ndvi"] == pytest.approx((0.5 + 4 * 0.6) / 5)
    assert january["ndmi"] == pytest.approx(1 / 3, abs=1e-6)
    assert january["valid_fraction"] == 5 / 8
    assert set(january) == {"date", "ndvi", "ndmi", "nbr", "valid_fraction"}
    assert prepared.payload["synthetic"] is False
    assert prepared.provenance["monthly_aggregation"][0]["acquisition_count"] == 2
    assert prepared.provenance["acquisitions"][0]["datetime_utc"] == "2024-01-10T10:00:00+00:00"
    assert len(prepared.provenance["acquisitions"][0]["sha256"]) == 64
    assert prepared.provenance["providers"] == ["microsoft-planetary-computer"]
    ImportRegion.model_validate(prepared.payload)


def test_named_band_order_not_channel_positions_controls_indices(tmp_path):
    save_cache(tmp_path, values=[0.1, 0.6, 0.2, 0.3],
               updates={"band_names": ["B12", "B08", "B04", "B11"]})
    row = prepare(tmp_path).payload["observations"][0]
    assert row["ndvi"] == pytest.approx(0.5, abs=1e-6)
    assert row["ndmi"] == pytest.approx(1 / 3, abs=1e-6)
    assert row["nbr"] == pytest.approx(5 / 7, abs=1e-6)


def test_missing_quality_data_is_never_promoted_to_clear_sky(tmp_path):
    save_cache(tmp_path, omit=["quality_mask"])
    with pytest.raises(ValueError, match="explicit cloud/quality mask"):
        prepare(tmp_path)


def test_clouded_nonfinite_and_zero_denominator_pixels_are_unavailable(tmp_path):
    data = np.broadcast_to(np.array([0.6, 0.2, 0.3, 0.1])[:, None, None], (4, 2, 2)).copy()
    data[0, 0, 0] = np.nan
    data[:, 0, 1] = 0
    data[1, 1, 0] = 2
    save_cache(tmp_path, values=data, mask=[[1, 1], [1, 0]])
    row = prepare(tmp_path).payload["observations"][0]
    assert row["valid_fraction"] == 0
    assert row["ndvi"] is None and row["ndmi"] is None and row["nbr"] is None
    json.dumps(row, allow_nan=False)


def test_landsat_family_and_platform_selection_do_not_mix_sentinel(tmp_path):
    save_cache(tmp_path / "landsat", "l8", sensor="landsat8")
    save_cache(tmp_path / "landsat", "l9", sensor="landsat9", acquired="2024-02-10T10:00:00+00:00")
    save_cache(tmp_path / "sentinel2", "s2")
    assert len(prepare(tmp_path, sensor="landsat").payload["observations"]) == 2
    assert len(prepare(tmp_path, sensor="landsat8").payload["observations"]) == 1
    assert prepare(tmp_path, sensor="landsat").payload["observations"][0]["ndvi"] == 0.5


def test_multiple_regions_require_explicit_selection(tmp_path):
    save_cache(tmp_path, "first")
    other = (-62.5, -10.5, -62.4, -10.4)
    save_cache(tmp_path, "second", bbox=other)
    with pytest.raises(ValueError, match="multiple AOIs"):
        prepare(tmp_path)
    selected = prepare(tmp_path, bbox=BOX)
    assert len(selected.provenance["acquisitions"]) == 1
    assert selected.payload["geometry"]["coordinates"][0][0] == [BOX[0], BOX[1]]


def test_unrelated_region_legacy_cache_does_not_block_selected_region(tmp_path):
    save_cache(tmp_path, "current")
    save_cache(tmp_path, "legacy", bbox=(-62.5, -10.5, -62.4, -10.4), omit=["preprocessing_version"])
    assert len(prepare(tmp_path, bbox=BOX).provenance["acquisitions"]) == 1


@pytest.mark.parametrize("updates,match", [
    ({"preprocessing_version": "unverified"}, "regenerate"),
    ({"data_units": "raw_DN"}, "optical units"),
    ({"grid_crs": "EPSG:32620"}, "EPSG:4326"),
    ({"grid_transform": [1, 0, 0, 0, 1, 0]}, "grid transform"),
    ({"quality_mask": [[1, 2], [1, 1]]}, "boolean or 0/1"),
    ({"quality_mask": [[True]]}, "spatial grid"),
    ({"band_names": ["B08", "B08", "B11", "B12"]}, "uniquely identify"),
    ({"band_names": ["B08", "B03", "B11", "B12"]}, "requires named bands"),
    ({"timestamp": 0}, "timestamp and datetime_iso disagree"),
])
def test_invalid_or_unverified_caches_fail_closed(tmp_path, updates, match):
    save_cache(tmp_path, updates=updates)
    with pytest.raises(ValueError, match=match):
        prepare(tmp_path)


def test_missing_and_ambiguous_acquisition_dates_are_rejected(tmp_path):
    save_cache(tmp_path, acquired="2024-01-01T12:00:00")
    with pytest.raises(ValueError, match="include its timezone"):
        prepare(tmp_path)


def test_month_assignment_uses_actual_utc_acquisition_time(tmp_path):
    save_cache(tmp_path, acquired="2024-02-01T00:30:00+02:00")
    assert prepare(tmp_path).payload["observations"][0]["date"] == "2024-01-31"


def test_duplicate_scene_ids_fail_instead_of_double_weighting(tmp_path):
    save_cache(tmp_path, "copy1", updates={"scene_id": "shared-scene"})
    save_cache(tmp_path, "copy2", updates={"scene_id": "shared-scene"})
    with pytest.raises(ValueError, match="Duplicate cached scene"):
        prepare(tmp_path)


def test_writer_and_cli_preserve_strict_payload_and_refuse_overwrite(tmp_path):
    cache = tmp_path / "cache"
    save_cache(cache)
    prepared = prepare(cache)
    output, manifest = write_prepared_import(prepared, tmp_path / "prepared.json")
    assert json.loads(output.read_text(encoding="utf-8")) == prepared.payload
    assert json.loads(manifest.read_text(encoding="utf-8")) == prepared.provenance
    with pytest.raises(FileExistsError):
        write_prepared_import(prepared, output)
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_observations.py"),
               "--cache-dir", str(cache), "--sensor", "sentinel2", "--name", "CLI AOI",
               "--region=" + ",".join(map(str, BOX)), "--output", str(tmp_path / "cli.json")]
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert "Prepared 1 observed months" in run.stdout
    ImportRegion.model_validate_json((tmp_path / "cli.json").read_text(encoding="utf-8"))


def test_actual_adapter_cache_contract_to_monitoring_api(tmp_path):
    from forestmind.data.adapters.base import PREPROCESSING_VERSION, SceneMetadata
    from forestmind.data.adapters.sentinel2 import Sentinel2Adapter
    from forestmind.utils.geo import BoundingBox
    bbox = BoundingBox(*BOX)
    adapter = Sentinel2Adapter(cache_dir=tmp_path / "actual-cache", patch_size=2)
    scene = SceneMetadata(scene_id="S2-actual-cache-contract", sensor="sentinel2",
                          datetime=datetime(2024, 1, 10, 10, tzinfo=timezone.utc), bbox=bbox,
                          stac_item={"collection": "sentinel-2-l2a"})
    data = np.full((12, 2, 2), 0.2, dtype=np.float32)
    data[7] = 0.6
    data[10] = 0.3
    data[11] = 0.1
    adapter.save_patch_to_cache(adapter.cache_path_for_scene(scene, bbox), data, scene, bbox,
                                np.ones((2, 2), dtype=bool),
                                preprocessing_version=PREPROCESSING_VERSION,
                                data_units="surface_reflectance_0_1")
    prepared = prepare(tmp_path / "actual-cache")
    with TestClient(create_app(tmp_path / "api.db", seed_demo=False)) as client:
        response = client.post("/api/v1/regions/import", json=prepared.payload)
        assert response.status_code == 201
        region = response.json()
        assert region["observations"][0]["ndvi"] == 0.5
        assert region["provenance"]["synthetic"] is False
        assert region["provenance"]["verified"] is False
        assert all(driver["evidence_score"] is None for driver in region["drivers"]
                   if driver["id"] in {"moisture", "heat", "fire", "cover", "structure"})
        assert region["forecast"]["points"] == []
        assert client.get("/api/v1/overview").json()["metrics"]["regions"] == 1


def test_empty_cache_and_wrong_sensor_do_not_create_dummy_observations(tmp_path):
    with pytest.raises(ValueError, match="No matching optical acquisitions"):
        prepare(tmp_path)
    with pytest.raises(ValueError, match="Unsupported optical sensor"):
        prepare(tmp_path, sensor="sentinel1")
