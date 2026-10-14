"""Acquisition-to-inference contracts: timing, georeferencing and no imputation."""

from datetime import datetime
import hashlib
import json

import numpy as np
import pytest

from forestmind.data.sequence import OPTICAL_BANDS, prepare_sequence, write_sequence
from forestmind.monitoring.ingest import REQUIRED_PREPROCESSING_VERSION


def cache(root, name, acquired, *, mask=None, updates=None):
    folder = root / "sentinel2"
    folder.mkdir(exist_ok=True)
    fields = {
        "data": np.full((12, 2, 2), 0.3, dtype=np.float32),
        "datetime_iso": acquired, "timestamp": datetime.fromisoformat(acquired).timestamp(),
        "sensor": "sentinel2", "band_names": OPTICAL_BANDS,
        "bbox_west": -55.1, "bbox_south": -3.5, "bbox_east": -55.08, "bbox_north": -3.48,
        "quality_mask": np.ones((2, 2), dtype=bool) if mask is None else np.asarray(mask),
        "scene_id": name, "provider": "offline-test", "collection": "sentinel-2-l2a",
        "preprocessing_version": REQUIRED_PREPROCESSING_VERSION, "data_units": "surface_reflectance_0_1",
        "grid_crs": "EPSG:4326", "grid_transform": [0.01, 0, -55.1, 0, -0.01, -3.48],
    }
    fields.update(updates or {})
    path = folder / f"{name}.npz"
    np.savez_compressed(path, **fields)
    return path


def test_sequence_preserves_acquisition_times_and_missing_sar(tmp_path):
    cache(tmp_path, "older", "2024-01-02T10:00:00+00:00")
    feb = cache(tmp_path, "feb", "2024-02-07T10:30:00+00:00")
    mar = cache(tmp_path, "mar", "2024-03-11T09:00:00+00:00", mask=[[1, 0], [1, 1]])
    cache(tmp_path, "future", "2024-04-07T10:00:00+00:00")
    result = prepare_sequence(tmp_path, through="2024-03-31", time_steps=2)
    arrays = result.arrays
    assert arrays["optical"].shape == (2, 12, 2, 2)
    assert arrays["timestamps"].tolist() == pytest.approx([0, 32.9375])
    assert arrays["observed_through"].item() == "2024-03-11"
    assert arrays["observation_mask"].all() and arrays["optical_mask"].all()
    assert not arrays["sar_mask"].any() and not arrays["sar"].any()
    assert not arrays["optical"][1, :, 0, 1].any()
    assert arrays["label_valid"].item() is False and arrays["synthetic"].item() is False
    assert [r["name"] for r in result.provenance["sources"]] == [feb.name, mar.name]
    assert result.provenance["requested_cutoff"] == "2024-03-31"


def test_sequence_does_not_pad_or_use_unusable_acquisitions(tmp_path):
    cache(tmp_path, "usable", "2024-01-02T10:00:00+00:00")
    cache(tmp_path, "cloudy", "2024-02-02T10:00:00+00:00", mask=[[0, 0], [0, 0]])
    result = prepare_sequence(tmp_path, through="2024-03-31", time_steps=12)
    assert len(result.arrays["timestamps"]) == 1
    assert result.provenance["skipped_for_quality"][0]["name"] == "cloudy.npz"
    assert result.provenance["observed_through"] == "2024-01-02"
    with pytest.raises(ValueError, match="No eligible"):
        prepare_sequence(tmp_path, through="2023-12-31")


def test_sequence_rejects_duplicate_times_and_legacy_cache(tmp_path):
    cache(tmp_path, "a", "2024-01-02T10:00:00+00:00")
    path = cache(tmp_path, "b", "2024-01-02T10:00:00+00:00")
    with pytest.raises(ValueError, match="Duplicate acquisition"):
        prepare_sequence(tmp_path, through="2024-03-31")
    path.unlink()
    cache(tmp_path, "legacy", "2024-02-02T10:00:00+00:00", updates={"preprocessing_version": "forestmind-grid-v2"})
    with pytest.raises(ValueError, match="Legacy or unverified"):
        prepare_sequence(tmp_path, through="2024-03-31")


def test_sequence_requires_complete_optical_bands_and_one_grid(tmp_path):
    cache(tmp_path, "a", "2024-01-02T10:00:00+00:00", updates={"band_names": list(reversed(OPTICAL_BANDS))})
    with pytest.raises(ValueError, match="canonical order"):
        prepare_sequence(tmp_path, through="2024-03-31")


def test_sequence_rejects_mixed_aois_but_can_select_one(tmp_path):
    cache(tmp_path, "a", "2024-01-02T10:00:00+00:00")
    cache(tmp_path, "b", "2024-02-02T10:00:00+00:00", updates={
        "bbox_west": -54.1, "bbox_east": -54.08,
        "grid_transform": [0.01, 0, -54.1, 0, -0.01, -3.48],
    })
    with pytest.raises(ValueError, match="Multiple AOIs"):
        prepare_sequence(tmp_path, through="2024-03-31")
    result = prepare_sequence(tmp_path, through="2024-03-31", bbox=(-55.1, -3.5, -55.08, -3.48))
    assert len(result.provenance["sources"]) == 1


def test_sequence_masks_nonfinite_pixels_in_any_band(tmp_path):
    pixels = np.full((12, 2, 2), 0.3, dtype=np.float32)
    pixels[0, 0, 0] = np.nan
    cache(tmp_path, "a", "2024-01-02T10:00:00+00:00", updates={"data": pixels})
    result = prepare_sequence(tmp_path, through="2024-03-31")
    assert result.provenance["sources"][0]["valid_fraction"] == 0.75
    assert np.isfinite(result.arrays["optical"]).all()
    assert not result.arrays["quality_mask"][0, 0, 0]


def test_sequence_serialization_is_pickle_free_and_hashes_match(tmp_path):
    path = cache(tmp_path, "a", "2024-01-02T10:00:00+00:00")
    prepared = prepare_sequence(tmp_path, through="2024-03-31")
    output, sidecar = write_sequence(prepared, tmp_path / "prepared" / "sequence.npz")
    with np.load(output, allow_pickle=False) as data:
        assert data["normalization"].item() == "raw"
        assert data["acquisition_dates"].tolist() == ["2024-01-02T10:00:00+00:00"]
        assert all(data[key].dtype.kind != "O" for key in data.files)
    manifest = json.loads(sidecar.read_text())
    assert manifest["sources"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["sequence_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize("kwargs", [{"time_steps": 0}, {"time_steps": True}, {"min_valid_fraction": float("nan")}, {"min_valid_fraction": -1}])
def test_invalid_sequence_selection_fails(tmp_path, kwargs):
    with pytest.raises(ValueError):
        prepare_sequence(tmp_path, through="2024-03-31", **kwargs)


def test_sequence_writer_preserves_source_cache_and_existing_provenance(tmp_path):
    source = cache(tmp_path, "a", "2024-01-02T10:00:00+00:00")
    original = source.read_bytes()
    prepared = prepare_sequence(tmp_path, through="2024-03-31")
    with pytest.raises(ValueError, match="already exists"):
        write_sequence(prepared, source)
    assert source.read_bytes() == original
    output = tmp_path / "sequence.npz"
    sidecar = output.with_suffix(".provenance.json")
    sidecar.write_text("existing provenance", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        write_sequence(prepared, output)
    assert not output.exists()
    assert sidecar.read_text() == "existing provenance"
