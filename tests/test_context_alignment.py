"""Measured-context alignment, leakage, units and independent missingness contracts."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from forestmind.data.context_alignment import (
    CLIMATE_FEATURE_NAMES, CLIMATE_FEATURE_UNITS, align_context,
)
from forestmind.data.sequence import OPTICAL_BANDS, add_sequence_context, prepare_sequence, write_sequence
from forestmind.monitoring.ingest import REQUIRED_PREPROCESSING_VERSION


BBOX = [-55.1, -3.5, -55.08, -3.48]
DATES = ["2024-02-10T10:00:00+00:00", "2024-03-10T10:00:00+00:00"]


def radar_scene(identity, acquired, *, mask=None, power=.01):
    mask = np.asarray(mask if mask is not None else [[True, True], [True, True]], dtype=bool)
    return {"id": identity, "datetime": acquired, "platform": "SENTINEL-1A", "relative_orbit": 39,
        "orbit_state": "descending", "instrument_mode": "IW", "polarizations": ["VV", "VH"],
        "valid_mask": mask.tolist(),
        "vv_linear_gamma0": [[power if valid else None for valid in row] for row in mask],
        "vh_linear_gamma0": [[power / 10 if valid else None for valid in row] for row in mask]}


def radar():
    return {"available": True, "collection": "sentinel-1-rtc", "units": "dB gamma0",
        "source_units": "positive linear gamma-naught power", "bbox_wgs84": list(BBOX),
        "grid_shape": [2, 2], "grid_crs": "EPSG:4326", "grid_pixel_size_degrees": [.01, .01],
        "scenes": [radar_scene("past", "2024-02-09T09:00:00Z", mask=[[True, False], [True, True]]),
                   radar_scene("future-for-first", "2024-02-11T09:00:00Z", power=.1)]}


def climate():
    return {"available": True, "dataset": "ERA5", "location_requested": [-55.09, -3.49],
        "start": "2024-02-03", "end": "2024-02-10", "raw_response": {
            "utc_offset_seconds": 0,
            "daily_units": {"temperature_2m_mean": "°C", "precipitation_sum": "mm",
                "et0_fao_evapotranspiration": "mm", "soil_moisture_0_to_7cm_mean": "m³/m³"},
            "daily": {"time": [f"2024-02-{day:02}" for day in range(3, 11)],
                "temperature_2m_mean": [20., 22., 24., 90., 90., 90., 90., 90.],
                "precipitation_sum": [1., 2., 3., 90., 90., 90., 90., 90.],
                "et0_fao_evapotranspiration": [2., 3., 4., 90., 90., 90., 90., 90.],
                "soil_moisture_0_to_7cm_mean": [.1, .2, .3, .9, .9, .9, .9, .9]}}}


def align(**kwargs):
    return align_context(acquisition_dates=DATES, bbox=BBOX, grid_shape=[2, 2], **kwargs)


def optical_cache(root, name, acquired):
    folder = root / "sentinel2"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / (name + ".npz"), data=np.full((12, 2, 2), .3, np.float32),
        datetime_iso=acquired, timestamp=datetime.fromisoformat(acquired).timestamp(),
        sensor="sentinel2", band_names=OPTICAL_BANDS,
        bbox_west=BBOX[0], bbox_south=BBOX[1], bbox_east=BBOX[2], bbox_north=BBOX[3],
        quality_mask=np.asarray([[False, True], [True, True]]), scene_id=name,
        provider="controlled-test-fixture", collection="sentinel-2-l2a",
        preprocessing_version=REQUIRED_PREPROCESSING_VERSION, data_units="surface_reflectance_0_1",
        grid_crs="EPSG:4326", grid_transform=[.01, 0, BBOX[0], 0, -.01, BBOX[3]])


def test_radar_matches_past_times_keeps_separate_pixels_and_zeros_missing():
    arrays, evidence = align(radar_context=radar())
    assert arrays["sar_mask"].tolist() == [True, False]
    assert arrays["sar_quality_mask"][0].tolist() == [[True, False], [True, True]]
    assert arrays["sar"][0, :, 0, 0].tolist() == pytest.approx([-20, -30])
    assert not arrays["sar"][0, :, 0, 1].any()
    assert not arrays["sar"][1].any() and not arrays["sar_quality_mask"][1].any()
    assert arrays["sar_acquisition_dates"].tolist() == ["2024-02-09T09:00:00+00:00", ""]
    assert arrays["sar_age_days"].tolist() == pytest.approx([25 / 24, -1])
    assert arrays["sar_units"].item() == "gamma0_db"
    assert evidence["radar"]["matches"][0]["scene_id"] == "past"
    assert evidence["available_modalities"] == ["optical", "sar"]


def test_future_radar_and_climate_measurements_cannot_change_earlier_inputs():
    radar1, climate1 = radar(), climate()
    first, _ = align(radar_context=radar1, climate_context=climate1, climate_window_days=3)
    radar1["scenes"][1]["vv_linear_gamma0"] = [[1e9] * 2] * 2
    climate1["raw_response"]["daily"]["precipitation_sum"][3:] = [10000.] * 5
    second, _ = align(radar_context=radar1, climate_context=climate1, climate_window_days=3)
    np.testing.assert_array_equal(first["sar"][0], second["sar"][0])
    np.testing.assert_array_equal(first["climate_context"][0], second["climate_context"][0])


def test_same_day_future_radar_hours_are_not_used():
    source = radar()
    source["scenes"] = [radar_scene("same-day-future", "2024-02-10T11:00:00Z")]
    arrays, _ = align(radar_context=source)
    assert not arrays["sar_mask"].any()


def test_radar_falls_back_to_older_usable_scene_and_never_clips_db():
    source = radar()
    source["scenes"] = [radar_scene("old", "2024-02-08T09:00:00Z", power=1e-9),
        radar_scene("bad", "2024-02-09T09:00:00Z", mask=[[False, False], [False, True]])]
    arrays, evidence = align(radar_context=source)
    assert evidence["radar"]["matches"][0]["scene_id"] == "old"
    assert arrays["sar"][0, 0, 0, 0] == pytest.approx(-90.)


@pytest.mark.parametrize("change,match", [
    ("bbox", "bounding box"), ("shape", "exact optical grid"), ("spacing", "grid spacing"),
    ("units", "calibrated"), ("orbit", "one platform"), ("mask", "boolean"),
    ("power", "positive"), ("hash", "hash changed"), ("duplicate", "Duplicate"),
    ("synthetic", "Synthetic"),
])
def test_radar_rejects_incompatible_evidence(change, match):
    context = radar()
    if change == "bbox": context["bbox_wgs84"][0] += .001
    if change == "shape": context["grid_shape"] = [4, 4]
    if change == "spacing": context["grid_pixel_size_degrees"] = [.001, .001]
    if change == "units": context["source_units"] = "sigma0"
    if change == "orbit": context["scenes"][1]["relative_orbit"] = 40
    if change == "mask": context["scenes"][0]["valid_mask"] = [[1, 0], [1, 1]]
    if change == "power": context["scenes"][0]["vv_linear_gamma0"][0][0] = 0
    if change == "hash": context["scenes"][0]["sample_sha256"] = "0" * 64
    if change == "duplicate": context["scenes"].append(deepcopy(context["scenes"][0]))
    if change == "synthetic": context["synthetic"] = True
    with pytest.raises(ValueError, match=match):
        align(radar_context=context)


def test_climate_uses_complete_past_window_and_preserves_physical_units():
    arrays, evidence = align(climate_context=climate(), climate_window_days=3)
    assert arrays["climate_context"][0].tolist() == pytest.approx([22, 6, 9, .2])
    assert arrays["climate_mask"].tolist() == [[True] * 4, [False] * 4]
    assert arrays["climate_coverage"][0].tolist() == [1.] * 4
    assert not arrays["climate_context"][1].any()
    assert arrays["climate_feature_names"].tolist() == list(CLIMATE_FEATURE_NAMES)
    assert arrays["climate_feature_units"].tolist() == list(CLIMATE_FEATURE_UNITS)
    assert evidence["climate"]["windows"][0]["end"] == "2024-02-05"
    assert evidence["climate"]["historical_availability_verified"] is False
    assert evidence["available_modalities"] == ["optical", "climate"]


def test_missing_climate_feature_does_not_become_zero_measurement_or_other_feature_missingness():
    context = climate()
    context["raw_response"]["daily"]["precipitation_sum"][0] = None
    arrays, _ = align(climate_context=context, climate_window_days=3)
    assert arrays["climate_mask"][0].tolist() == [True, False, True, True]
    assert arrays["climate_context"][0, 1] == 0
    assert arrays["climate_coverage"][0, 1] == pytest.approx(2 / 3)
    partial, _ = align(climate_context=context, climate_window_days=3, climate_min_coverage=.5)
    assert partial["climate_mask"][0, 1]
    assert partial["climate_context"][0, 1] == 5  # no extrapolation to three days


def test_unreported_dates_count_as_missing_and_zero_rain_is_valid():
    context = climate()
    for values in context["raw_response"]["daily"].values():
        del values[1]
    context["raw_response"]["daily"]["precipitation_sum"] = [0.] * 7
    arrays, _ = align(climate_context=context, climate_window_days=3)
    assert not arrays["climate_mask"][0].any()
    partial, _ = align(climate_context=context, climate_window_days=3, climate_min_coverage=.5)
    assert partial["climate_mask"][0, 1] and partial["climate_context"][0, 1] == 0


@pytest.mark.parametrize("change,match", [
    ("location", "centroid"), ("unit", "unit mismatch"), ("timezone", "UTC"),
    ("duplicate", "Duplicate"), ("bool", "finite numbers"), ("negative", "negative"),
    ("soil", "soil moisture"), ("monthly", "raw_response"), ("date", "collection window"),
])
def test_climate_rejects_mismatched_or_invalid_observations(change, match):
    context = climate()
    if change == "location": context["location_requested"] = [10., 20.]
    if change == "unit": context["raw_response"]["daily_units"]["temperature_2m_mean"] = "°F"
    if change == "timezone": context["raw_response"]["utc_offset_seconds"] = 3600
    if change == "duplicate": context["raw_response"]["daily"]["time"][1] = "2024-02-03"
    if change == "bool": context["raw_response"]["daily"]["temperature_2m_mean"][0] = True
    if change == "negative": context["raw_response"]["daily"]["precipitation_sum"][0] = -1
    if change == "soil": context["raw_response"]["daily"]["soil_moisture_0_to_7cm_mean"][0] = 2
    if change == "monthly": del context["raw_response"]
    if change == "date": context["end"] = "2024-02-05"
    with pytest.raises(ValueError, match=match):
        align(climate_context=context)


@pytest.mark.parametrize("kwargs", [
    {"radar_max_age_days": True}, {"radar_max_age_days": -1}, {"radar_min_valid_fraction": 2},
    {"climate_window_days": 1.5}, {"climate_lag_days": 0}, {"climate_lag_days": True},
    {"climate_min_coverage": float("nan")},
])
def test_invalid_alignment_policies_fail(kwargs):
    with pytest.raises(ValueError):
        align(**kwargs)


def test_unavailable_context_is_explicitly_masked():
    arrays, evidence = align(radar_context={"available": False}, climate_context={"available": False})
    assert evidence["available_modalities"] == ["optical"]
    assert not arrays["sar_mask"].any() and not arrays["climate_mask"].any()


def test_sequence_attachment_keeps_optical_and_original_evidence_unchanged(tmp_path):
    for index, acquired in enumerate(DATES):
        optical_cache(tmp_path, str(index), acquired)
    original = prepare_sequence(tmp_path, through="2024-03-10")
    source = radar()
    source_before = deepcopy(source)
    result = add_sequence_context(original, radar_context=source, climate_context=climate(), climate_window_days=3)
    assert source == source_before and "alignment" not in original.provenance
    assert original.provenance["sar_present"] is False and not original.arrays["sar_mask"].any()
    np.testing.assert_array_equal(result.arrays["optical"], original.arrays["optical"])
    assert not result.arrays["quality_mask"][0, 0, 0]
    assert result.arrays["sar_quality_mask"][0, 0, 0]  # radar can see an optical-invalid pixel
    assert result.provenance["available_modalities"] == ["optical", "sar", "climate"]
    assert result.provenance["alignment"]["parameters"]["climate_window_days"] == 3
    output, sidecar = write_sequence(result, tmp_path / "multimodal.npz")
    with np.load(output, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
    saved = json.loads(sidecar.read_text())
    assert saved["sequence_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="already has aligned"):
        add_sequence_context(result, radar_context=source)


def test_cli_attaches_local_context_without_network(tmp_path):
    optical_cache(tmp_path, "one", DATES[0])
    context_path = tmp_path / "climate.json"
    context_path.write_text(json.dumps(climate()), encoding="utf-8")
    output = tmp_path / "output.npz"
    process = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_sequence.py"),
        "--cache-dir", str(tmp_path), "--through", "2024-02-10", "--climate-context", str(context_path),
        "--climate-window-days", "3", "--output", str(output)], capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    assert "optical, climate" in process.stdout
    with np.load(output, allow_pickle=False) as archive:
        assert archive["climate_context"][0].tolist() == pytest.approx([22, 6, 9, .2])
