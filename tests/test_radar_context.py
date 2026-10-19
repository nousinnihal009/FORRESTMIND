from copy import deepcopy

import numpy as np
import pytest

from forestmind.monitoring.acquisition import geometry_for_bbox
import forestmind.monitoring.radar as radar


def geometry():
    return geometry_for_bbox([-55.1, -3.5, -55.08, -3.48])


def item(scene_id, acquired, *, orbit=39, platform="SENTINEL-1A", state="descending"):
    return {"id": scene_id, "collection": "sentinel-1-rtc", "properties": {
        "datetime": acquired, "sat:relative_orbit": orbit, "sat:orbit_state": state,
        "platform": platform, "sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"],
        "sar:resolution_range": 20, "sar:resolution_azimuth": 22,
        "sar:pixel_spacing_range": 10, "sar:pixel_spacing_azimuth": 10},
        "assets": {pol: {"href": "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/" + scene_id + "/" + pol + ".tiff",
                         "description": "Terrain-corrected gamma naught power", "raster:bands": [{"nodata": -32768, "spatial_resolution": 10.0}]} for pol in ("vv", "vh")}}


def pair():
    return [item("older", "2024-07-11T09:00:00Z"), item("latest", "2024-07-23T09:00:00Z")]


def test_pair_requires_matching_platform_orbit_pass_and_distinct_dates():
    scenes = pair() + [item("other-platform", "2024-07-27T09:00:00Z", platform="SENTINEL-1C"),
                      item("other-orbit", "2024-07-28T09:00:00Z", orbit=40),
                      item("other-pass", "2024-07-29T09:00:00Z", state="ascending"),
                      item("same-day", "2024-07-23T09:01:00Z")]
    chosen, _ = radar.select_pair(scenes, start="2024-07-01", end="2024-07-31")
    assert [row["id"] for row in chosen] == ["older", "same-day"]


def test_unsigned_metadata_and_outside_cutoff_rejection():
    scenes = pair()
    scenes[0]["assets"]["vv"]["href"] += "?sig=secret"
    scenes.append(item("future", "2024-08-01T00:00:00Z"))
    selected, rejected = radar.select_pair(scenes, start="2024-07-01", end="2024-07-31")
    assert "?" not in selected[0]["assets"]["vv"]["href"]
    assert rejected[0]["scene_id"] == "future"


@pytest.mark.parametrize("change", ["host", "mode", "polarization", "orbit", "calibration", "scale"])
def test_invalid_scene_metadata_cannot_create_pair(change):
    scenes = pair()
    latest = scenes[-1]
    if change == "host": latest["assets"]["vv"]["href"] = "https://evil.example/power.tif"
    if change == "mode": latest["properties"]["sar:instrument_mode"] = "EW"
    if change == "polarization": latest["properties"]["sar:polarizations"] = ["HH", "HV"]
    if change == "orbit": latest["properties"]["sat:relative_orbit"] = True
    if change == "calibration": latest["assets"]["vv"]["description"] = "Uncalibrated DN"
    if change == "scale": latest["assets"]["vv"]["raster:bands"][0]["scale"] = float("nan")
    with pytest.raises(ValueError, match="No pair"):
        radar.select_pair(scenes, start="2024-07-01", end="2024-07-31")


def test_actual_linear_power_conversion_unclipped_common_mask_and_provenance(monkeypatch):
    calls = []
    monkeypatch.setattr(radar, "_search", lambda *args: (pair(), {"request_url": "fixed", "returned_candidates": 2}))
    def read(scene, bbox, patch_size, timeout):
        calls.append(scene["id"])
        power = np.ones((2, patch_size, patch_size), dtype=np.float32)
        power[0] *= 10 if scene["id"] == "older" else 20
        power[1] *= .01 if scene["id"] == "older" else .005
        valid = np.ones((patch_size, patch_size), dtype=bool)
        if scene["id"] == "latest":
            valid[0, 0] = False
        return power, valid, []
    monkeypatch.setattr(radar, "_read_scene", read)
    result = radar.acquire_radar(geometry(), start="2024-01-01", end="2024-07-31", patch_size=16)
    assert calls == ["older", "latest"]
    assert result["comparison"]["vv_change_db"] == pytest.approx(3.0102999566, abs=1e-5)
    assert result["comparison"]["vh_change_db"] == pytest.approx(-3.0102999566, abs=1e-5)
    assert result["scenes"][0]["mean_vv_db"] == pytest.approx(10)
    assert result["comparison"]["common_valid_fraction"] == 255 / 256
    assert result["comparison"]["vv_change_map_db"][0][0] is None
    assert result["geometry"] == geometry() and result["geometry_sha256"]
    assert result["source_units"] == "positive linear gamma-naught power"
    assert result["scenes"][0]["native_resolution_range_m"] == 20
    assert result["scenes"][0]["provider_pixel_spacing_range_m"] == 10
    assert result["model_input"] is False and result["search_start"] == "2024-02-03"


def test_common_valid_threshold_abstains(monkeypatch):
    monkeypatch.setattr(radar, "_search", lambda *args: (pair(), {}))
    def read(scene, bbox, patch_size, timeout):
        mask = np.zeros((patch_size, patch_size), dtype=bool)
        mask[:4] = True
        return np.ones((2, patch_size, patch_size)), mask, []
    monkeypatch.setattr(radar, "_read_scene", read)
    with pytest.raises(ValueError, match="at least 40%"):
        radar.acquire_radar(geometry(), start="2024-07-01", end="2024-07-31", patch_size=16)


def test_export_preserves_small_positive_linear_power():
    values = np.asarray([[1e-9, .123456789]], dtype=np.float64)
    result = radar._map(values, np.asarray([[True, True]]), digits=None)
    assert result == [[1e-9, .123456789]]


@pytest.mark.parametrize("kwargs", [
    {"end": "2999-01-01"}, {"start": "2024-08-01"}, {"patch_size": 256}, {"patch_size": True},
    {"search_limit": 100}, {"timeout_seconds": float("inf")}, {"geometry": geometry_for_bbox([0, 0, 1, 1])},
])
def test_resource_bounds_before_network(monkeypatch, kwargs):
    monkeypatch.setattr(radar, "_search", lambda *args: pytest.fail("unbounded request reached provider"))
    with pytest.raises(ValueError):
        radar.acquire_radar(**{"geometry": geometry(), "start": "2024-07-01", "end": "2024-07-31", **kwargs})


def test_catalog_response_is_byte_and_candidate_bounded(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, amount):
            assert amount == radar.MAX_RESPONSE_BYTES + 1
            return b" " * amount
    class Opener:
        def open(self, *args, **kwargs): return Response()
    monkeypatch.setattr(radar, "build_opener", lambda *args: Opener())
    with pytest.raises(ValueError, match="4 MiB"):
        radar._search([-55.1, -3.5, -55.08, -3.48], "2024-07-01", "2024-07-31", 30, 10)
