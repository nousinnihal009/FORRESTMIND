from contextlib import contextmanager

import numpy as np
import pytest

import forestmind.monitoring.landcover as landcover
from forestmind.monitoring.acquisition import geometry_for_bbox


def geometry(bbox=None):
    return geometry_for_bbox(bbox or [-55.1, -3.5, -55.08, -3.48])


def test_official_tile_names_negative_coordinates_and_boundaries():
    assert landcover.tile_urls([-55.1, -3.5, -55.08, -3.48], 2021) == [
        landcover.HOST + "/v200/2021/map/ESA_WorldCover_10m_2021_v200_S06W057_Map.tif"]
    assert len(landcover.tile_urls([-.01, -.01, .01, .01], 2020)) == 4
    assert len(landcover.tile_urls([0, 0, .01, .01], 2020)) == 1
    assert len(landcover.tile_urls([2.99, 2.99, 3, 3], 2020)) == 1


@pytest.mark.parametrize("cutoff,year", [("2020-12-31", 2020), ("2021-12-30", 2020),
                                          ("2021-12-31", 2021), ("2026-10-17", 2021)])
def test_epoch_never_extends_past_observation_cutoff(cutoff, year):
    assert landcover.selected_epoch(cutoff) == year


@pytest.mark.parametrize("kwargs", [
    {"cutoff": "2020-06-01"}, {"patch_size": True}, {"patch_size": 129},
    {"timeout_seconds": float("nan")}, {"timeout_seconds": 61},
    {"geometry": geometry([0, 0, 1, 1])}, {"geometry": geometry([0, 86, .01, 86.01])},
])
def test_bounds_rejected_before_network(monkeypatch, kwargs):
    monkeypatch.setattr(landcover, "_read_tile", lambda *args: pytest.fail("invalid request performed network I/O"))
    with pytest.raises(ValueError):
        landcover.acquire_landcover(**{"geometry": geometry(), "cutoff": "2024-01-01", **kwargs})


def test_classification_keeps_nodata_unknown_and_mangroves_explicit(monkeypatch):
    def read(*args):
        values = np.full((16, 16), 10, dtype=np.float32)
        values[0, :4] = [0, 95, 40, 99]
        return values, np.ones((16, 16), dtype=bool), {"source_url": args[0]}
    monkeypatch.setattr(landcover, "_read_tile", read)
    result = landcover.acquire_landcover(geometry(), cutoff="2024-01-01", patch_size=16)
    assert result["forest"][0][:4] == [False, True, False, False]
    assert result["valid_mask"][0][:4] == [False, True, True, False]
    assert result["class_raster"][0][:4] == [0, 95, 40, 0]
    assert result["observed_at"] == "2021-12-31" and result["published_at"] == "2022-10-28"
    assert result["model_input"] is False and result["change_detection"]["available"] is False
    assert result["geometry"] == geometry() and result["native_resolution_m"] == 10
    assert result["valid_fraction"] == 254 / 256 and result["incomplete"]
    assert len(result["sha256"]) == 64


def test_tile_seam_mosaic_retains_partial_coverage_without_fabricating(monkeypatch):
    calls = []
    def read(url, *_):
        calls.append(url)
        if "W003" in url:
            raise OSError("Provider tile unavailable")
        values = np.full((16, 16), 10, dtype=np.float32)
        mask = np.zeros((16, 16), dtype=bool)
        mask[:, 8:] = True
        return values, mask, {"source_url": url}
    monkeypatch.setattr(landcover, "_read_tile", read)
    result = landcover.acquire_landcover(geometry([-.01, .01, .01, .03]), cutoff="2024-01-01", patch_size=16)
    assert len(calls) == 2 and len(result["failures"]) == 1
    assert result["valid_fraction"] == .5 and result["incomplete"]
    assert not result["valid_mask"][0][0] and result["forest"][0][-1]


def test_provider_failure_does_not_return_fake_empty_forest(monkeypatch):
    def fail(*_):
        raise OSError("Provider unavailable")
    monkeypatch.setattr(landcover, "_read_tile", fail)
    with pytest.raises(ValueError, match="no usable land-cover"):
        landcover.acquire_landcover(geometry(), cutoff="2024-01-01", patch_size=16)


def test_polygon_mask_restricts_the_same_selected_geometry(monkeypatch):
    triangle = {"type": "Polygon", "coordinates": [[[0, 0], [.01, 0], [0, .01], [0, 0]]]}
    monkeypatch.setattr(landcover, "_read_tile", lambda url, *args: (
        np.full((16, 16), 10), np.ones((16, 16), dtype=bool), {"source_url": url}))
    result = landcover.acquire_landcover(triangle, cutoff="2024-01-01", patch_size=16)
    assert result["geometry"] == triangle
    assert 0 < np.asarray(result["valid_mask"]).sum() < 256
    assert result["valid_fraction"] == 1


def test_real_categorical_reprojection_preserves_classes_and_exact_aoi(monkeypatch):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds
    values = np.full((32, 32), 10, dtype=np.uint8)
    values[:, 16:] = 40
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=32, height=32, count=1, dtype="uint8",
                         crs="EPSG:4326", transform=from_bounds(0, 0, .02, .02, 32, 32), nodata=0) as dataset:
            dataset.write(values, 1)
        @contextmanager
        def local(*args, **kwargs):
            with memory.open() as source:
                yield source
        monkeypatch.setattr(landcover, "open_raster", local)
        result = landcover.acquire_landcover(geometry([0, 0, .02, .02]), cutoff="2024-01-01", patch_size=16)
    assert result["tree_sample_fraction"] == .5
    assert result["valid_fraction"] == 1 and result["sources"][0]["source_crs"] == "EPSG:4326"
    assert result["grid_pixel_size_degrees"] == [.02 / 16, .02 / 16]
