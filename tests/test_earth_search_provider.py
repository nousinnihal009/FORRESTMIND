"""Offline provider contracts from Earth Search C1 Sentinel-2 metadata."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from forestmind.data.adapters.base import PREPROCESSING_VERSION, SceneMetadata
from forestmind.data.adapters import sentinel2
from forestmind.utils.geo import BoundingBox
from scripts import download_data


BOX = BoundingBox(-55.10, -3.50, -55.08, -3.48)
# Single-band EO names from public S2A_T21MYS_20240729T140134_L2A metadata.
OBSERVED_ASSETS = [
    ("coastal", "B01"), ("blue", "B02"), ("green", "B03"), ("red", "B04"),
    ("rededge1", "B05"), ("rededge2", "B06"), ("rededge3", "B07"), ("nir", "B08"),
    ("nir08", "B8A"), ("nir09", "B09"), ("swir16", "B11"), ("swir22", "B12"),
]


def scene(provider="element84-earth-search", collection="sentinel-2-c1-l2a"):
    return SceneMetadata("same-scene", "sentinel2", datetime(2024, 7, 29, tzinfo=timezone.utc), BOX,
                         stac_item={"collection": collection}, extra={"provider": provider})


def fake_item():
    assets = {
        key: SimpleNamespace(href=f"https://public.example/{key}.tif", roles=["data", "reflectance"],
                             extra_fields={"eo:bands": [{"name": band}],
                                           "raster:bands": [{"scale": 0.0001, "offset": -0.1}]})
        for key, band in OBSERVED_ASSETS
    }
    assets["scl"] = SimpleNamespace(href="https://public.example/scl.tif", roles=["data"], extra_fields={})
    return SimpleNamespace(assets=assets, properties={"s2:processing_baseline": "05.10"})


def stub_download(monkeypatch, item):
    monkeypatch.setitem(sys.modules, "pystac", SimpleNamespace(Item=SimpleNamespace(from_dict=lambda _: item)))
    monkeypatch.setitem(sys.modules, "rasterio", SimpleNamespace())
    calls = []

    @contextmanager
    def raster_open(href, timeout):
        calls.append((href, timeout))
        yield href

    def read(href, bbox, size, categorical=False):
        assert bbox == BOX
        values = np.full((size, size), 4, dtype=np.float32)
        valid = np.ones((size, size), dtype=bool)
        if href.endswith("/scl.tif"):
            assert categorical
            values[0, 0] = 8  # cloud
        else:
            assert not categorical
            index = [key for key, _ in OBSERVED_ASSETS].index(href.rsplit("/", 1)[1][:-4])
            values.fill(1000 + 100 * (index + 1))
            if index == 7:
                valid[1, 1] = False
        return values, valid

    monkeypatch.setattr(sentinel2, "open_raster", raster_open)
    monkeypatch.setattr(sentinel2, "read_aligned_band", read)
    return calls


@pytest.mark.parametrize("metadata", [{}, {"raster:bands": [{"scale": 0.0001}]},
                                      {"raster:bands": [{"offset": -0.1}]}])
def test_earth_search_never_falls_back_to_pc_baseline(metadata):
    with pytest.raises(ValueError, match="explicit"):
        sentinel2.reflectance_scaling(metadata, {"s2:processing_baseline": "05.10"}, require_explicit=True)


@pytest.mark.parametrize("provider", ["earth-search", "planetary-computer"])
def test_search_selects_provider_and_sorts_only_returned_subset(tmp_path, monkeypatch, provider):
    items = [SimpleNamespace(id=f"scene-{day}", bbox=[BOX.west, BOX.south, BOX.east, BOX.north],
                             datetime=datetime(2024, 7, day, tzinfo=timezone.utc),
                             properties={"eo:cloud_cover": 4}, to_dict=lambda: {}) for day in [29, 9]]
    catalog = Mock()
    catalog.search.return_value.items.return_value = iter(items)
    opened = Mock(return_value=catalog)
    monkeypatch.setattr(sentinel2, "open_catalog", opened)
    adapter = sentinel2.Sentinel2Adapter(cache_dir=tmp_path, provider=provider, timeout_seconds=9)
    scenes = adapter.search(BOX, "2024-07-01", "2024-07-31", max_results=2)
    opened.assert_called_once_with(adapter.catalog_url, 9)
    options = catalog.search.call_args.kwargs
    assert options["collections"] == [adapter.collection]
    assert options["max_items"] == 2
    assert ("sortby" in options) == (provider == "planetary-computer")
    assert [row.datetime.day for row in scenes] == [9, 29]
    assert all(row.extra["provider"] == adapter.provider_name for row in scenes)


def test_all_bands_scale_once_cloud_mask_and_provenance_survive(tmp_path, monkeypatch):
    item = fake_item()
    calls = stub_download(monkeypatch, item)
    adapter = sentinel2.Sentinel2Adapter(cache_dir=tmp_path, provider="earth-search", patch_size=4, timeout_seconds=11)
    source = scene()
    paths = adapter.download([source], BOX)
    assert not adapter.download_failures
    assert len(paths) == 1
    assert len(calls) == 13
    assert all(timeout == 11 for _, timeout in calls)
    assert calls[8][0].endswith("/nir08.tif")  # B8A, never broad B08
    assert calls[9][0].endswith("/nir09.tif")  # B09, never derived WVP
    with np.load(paths[0], allow_pickle=False) as cache:
        assert list(cache["band_names"]) == [band for _, band in OBSERVED_ASSETS]
        np.testing.assert_allclose(cache["data"][:, 2, 2], np.arange(1, 13) * 0.01, atol=2e-8)
        assert not cache["data"][:, 0, 0].any()
        assert not cache["data"][:, 1, 1].any()
        assert cache["quality_mask"].sum() == 14
        assert str(cache["provider"]) == "element84-earth-search"
        assert str(cache["collection"]) == "sentinel-2-c1-l2a"
        processing = json.loads(str(cache["processing_metadata_json"]))
        assert processing["band_scaling"]["B8A"]["source"] == "stac-raster-scale-offset"
        assert processing["invalid_scl_classes"] == sorted(sentinel2.SCL_INVALID)
    assert adapter.is_cached(source, BOX)
    adapter.download([source], BOX)
    assert len(calls) == 13  # Valid matching cache prevents further network requests.


@pytest.mark.parametrize("asset,band", [("blue", "B02"), ("scl", "SCL")])
def test_failed_band_keeps_cause_without_signed_credentials(tmp_path, monkeypatch, caplog, asset, band):
    stub_download(monkeypatch, fake_item())
    original_read = sentinel2.read_aligned_band

    def read(href, bbox, size, categorical=False):
        if href.endswith(f"/{asset}.tif"):
            try:
                raise OSError("HTTP 503 reading https://user:private-password@public.example/"
                              "band.tif?sig=private-token&se=tomorrow")
            except OSError as cause:
                raise RuntimeError("Read failed. See previous exception for details.") from cause
        return original_read(href, bbox, size, categorical=categorical)

    monkeypatch.setattr(sentinel2, "read_aligned_band", read)
    adapter = sentinel2.Sentinel2Adapter(cache_dir=tmp_path, provider="earth-search", patch_size=4)
    assert adapter.download([scene()], BOX) == []
    reason = adapter.download_failures[0]["reason"]
    assert reason.startswith(f"Band {band}:")
    assert "Read failed" in reason and "HTTP 503" in reason
    assert "OSError" in reason and "caused by" in reason
    assert "private-token" not in reason + caplog.text
    assert "private-password" not in reason + caplog.text
    assert not list(tmp_path.rglob("*.npz"))
    assert adapter.download_failures[0]["retryable"] is True


def test_band_error_chain_is_bounded_and_cycle_safe():
    outer = RuntimeError("Read failed")
    inner = OSError("decoder failed " + "x" * 5000)
    outer.__cause__ = inner
    inner.__cause__ = outer
    reason = sentinel2._band_failure_reason(outer, "B04")
    assert len(reason) <= 1000
    assert reason.count("Read failed") == 1
    assert reason.count("decoder failed") == 1
    assert "..." in reason


def test_band_error_removes_truncated_gdal_signed_filename_and_follows_context():
    try:
        raise OSError("private-signature-tail: band 1: IReadBlock failed at X offset 4")
    except OSError:
        try:
            raise RuntimeError("Read failed\nSee previous exception")
        except RuntimeError as error:
            reason = sentinel2._band_failure_reason(error, "SCL")
    assert "private-signature-tail" not in reason
    assert "IReadBlock failed at X offset 4" in reason
    assert "\n" not in reason


def test_band_error_does_not_restore_explicitly_suppressed_context():
    try:
        raise OSError("suppressed diagnostic")
    except OSError:
        try:
            raise RuntimeError("Read failed") from None
        except RuntimeError as error:
            reason = sentinel2._band_failure_reason(error, "B01")
    assert "suppressed diagnostic" not in reason


@pytest.mark.parametrize("problem", ["band_name", "missing_scale", "private_asset", "missing_collection",
                                    "missing_scl", "private_scl"])
def test_earth_search_invalid_metadata_fails_before_any_raster_read(tmp_path, monkeypatch, problem):
    item, source = fake_item(), scene()
    if problem == "band_name":
        item.assets["nir08"].extra_fields["eo:bands"][0]["name"] = "B08"
    elif problem == "missing_scale":
        item.assets["red"].extra_fields["raster:bands"][0].pop("scale")
    elif problem == "private_asset":
        item.assets["red"].href = "s3://private/band.tif"
    elif problem == "missing_scl":
        item.assets.pop("scl")
    elif problem == "private_scl":
        item.assets["scl"].href = "s3://private/scl.tif"
    else:
        source.stac_item = {}
    calls = stub_download(monkeypatch, item)
    adapter = sentinel2.Sentinel2Adapter(cache_dir=tmp_path, provider="earth-search", patch_size=4)
    assert adapter.download([source], BOX) == []
    assert len(adapter.download_failures) == 1
    assert calls == []


def test_cache_identity_and_reuse_enforce_provider_collection(tmp_path):
    adapter = sentinel2.Sentinel2Adapter(cache_dir=tmp_path, provider="earth-search", patch_size=4)
    original = scene()
    other_provider = scene(provider="microsoft-planetary-computer")
    other_collection = scene(collection="sentinel-2-l2a")
    paths = {adapter.cache_path_for_scene(row, BOX) for row in [original, other_provider, other_collection]}
    assert len(paths) == 3
    path = adapter.cache_path_for_scene(original, BOX)
    # Even if a wrong-provider file is copied to the intended path, reject it.
    adapter.save_patch_to_cache(path, np.ones((12, 4, 4)), other_provider, BOX,
                                np.ones((4, 4), dtype=bool), preprocessing_version=PREPROCESSING_VERSION)
    assert not adapter.is_cached(original, BOX)


def test_wrong_provider_scene_is_not_silently_reinterpreted(tmp_path, monkeypatch):
    calls = stub_download(monkeypatch, fake_item())
    adapter = sentinel2.Sentinel2Adapter(cache_dir=tmp_path, provider="earth-search", patch_size=4)
    assert adapter.download([scene(provider="microsoft-planetary-computer")], BOX) == []
    assert "does not match" in adapter.download_failures[0]["reason"]
    assert calls == []


@pytest.mark.parametrize("sensors", ["s1", "landsat", "s2,s1"])
def test_cli_rejects_unsupported_earth_search_sensor_before_network(monkeypatch, sensors):
    download = Mock()
    monkeypatch.setattr(download_data, "download_sentinel2", download)
    assert download_data.main(["--region=-55.10,-3.50,-55.08,-3.48", "--provider", "earth-search",
                               "--sensors", sensors]) == 2
    download.assert_not_called()


def test_cli_passes_explicit_provider(monkeypatch):
    download = Mock(return_value=1)
    monkeypatch.setattr(download_data, "download_sentinel2", download)
    assert download_data.main(["--region=-55.10,-3.50,-55.08,-3.48", "--provider", "earth-search",
                               "--sensors", "s2", "--max-scenes", "1"]) == 0
    assert download.call_args.kwargs["provider"] == "earth-search"


def test_unknown_provider_fails_before_cache_creation(tmp_path):
    with pytest.raises(ValueError, match="provider"):
        sentinel2.Sentinel2Adapter(cache_dir=tmp_path / "new", provider="automatic")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("failed_asset", ["nir08", "scl"])
def test_budgeted_retry_reuses_completed_bands_without_double_scaling_or_lost_masks(
        tmp_path, monkeypatch, failed_asset):
    calls = stub_download(monkeypatch, fake_item())
    original_read = sentinel2.read_aligned_band
    failed = False
    def read(href, bbox, size, categorical=False):
        nonlocal failed
        if href.endswith(f"/{failed_asset}.tif") and not failed:
            failed = True
            raise OSError("HTTP 503")
        return original_read(href, bbox, size, categorical=categorical)
    monkeypatch.setattr(sentinel2, "read_aligned_band", read)
    adapter = sentinel2.Sentinel2Adapter(cache_dir=tmp_path, provider="earth-search", patch_size=4)
    assert adapter.download([scene()], BOX) == []
    assert adapter.download_failures[0]["retryable"] is True
    assert not list(tmp_path.rglob("*.npz"))
    # The adapter never retries itself; only the caller's second attempt reads again.
    assert sum(href.endswith(f"/{failed_asset}.tif") for href, _ in calls) == 1
    paths = adapter.download([scene()], BOX)
    assert len(paths) == 1 and len(calls) == 14
    assert adapter._retry_bands == {} and adapter._retry_band_bytes == 0
    with np.load(paths[0], allow_pickle=False) as cache:
        np.testing.assert_allclose(cache["data"][:, 2, 2], np.arange(1, 13) * 0.01, atol=2e-8)
        assert cache["quality_mask"].sum() == 14
        assert not cache["data"][:, 0, 0].any()  # SCL cloud
        assert not cache["data"][:, 1, 1].any()  # Invalid optical sample


@pytest.mark.parametrize("change", ["source", "grid", "bounds", "date"])
def test_retry_bands_never_cross_source_or_grid_identity(tmp_path, monkeypatch, change):
    calls = stub_download(monkeypatch, fake_item())
    original_read = sentinel2.read_aligned_band
    failed = False
    def read(href, bbox, size, categorical=False):
        nonlocal failed
        if href.endswith("/nir08.tif") and not failed:
            failed = True
            raise OSError("HTTP 503")
        return original_read(href, BOX, size, categorical=categorical)
    monkeypatch.setattr(sentinel2, "read_aligned_band", read)
    adapter = sentinel2.Sentinel2Adapter(cache_dir=tmp_path, provider="earth-search", patch_size=4)
    source, box = scene(), BOX
    assert adapter.download([source], box) == []
    if change == "source":
        source.stac_item["source_version"] = "different"
    elif change == "grid":
        adapter.patch_size = 8
    elif change == "bounds":
        # A sub-microdegree change also matters even if the disk cache rounds bounds.
        box = BoundingBox(BOX.west + 1e-8, BOX.south, BOX.east, BOX.north)
    else:
        source.datetime = datetime(2024, 7, 30, tzinfo=timezone.utc)
    assert len(adapter.download([source], box)) == 1
    assert len(calls) == 22  # Nine original reads; all thirteen from the changed source/grid.


def test_permanent_read_failure_discards_partial_bands(tmp_path, monkeypatch):
    calls = stub_download(monkeypatch, fake_item())
    original_read = sentinel2.read_aligned_band
    def read(href, bbox, size, categorical=False):
        if href.endswith("/blue.tif"):
            raise OSError("HTTP 404")
        return original_read(href, bbox, size, categorical=categorical)
    monkeypatch.setattr(sentinel2, "read_aligned_band", read)
    adapter = sentinel2.Sentinel2Adapter(cache_dir=tmp_path, provider="earth-search", patch_size=4)
    assert adapter.download([scene()], BOX) == []
    assert adapter.download_failures[0]["retryable"] is False
    assert adapter._retry_bands == {} and len(calls) == 2


def test_retry_band_memory_is_bounded_and_cancellation_checks_cached_reads(tmp_path, monkeypatch):
    from forestmind.data.adapters.network import restricted_acquisition
    calls = stub_download(monkeypatch, fake_item())
    original_read = sentinel2.read_aligned_band
    def read(href, bbox, size, categorical=False):
        if href.endswith("/green.tif"):
            raise OSError("HTTP 503")
        return original_read(href, bbox, size, categorical=categorical)
    monkeypatch.setattr(sentinel2, "read_aligned_band", read)
    monkeypatch.setattr(sentinel2, "MAX_RETRY_BAND_BYTES", 80)
    adapter = sentinel2.Sentinel2Adapter(cache_dir=tmp_path, provider="earth-search", patch_size=4)
    source = scene()
    assert adapter.download([source], BOX) == []
    assert adapter._retry_band_bytes == 80 and len(adapter._retry_bands) == 1
    class Stopped(BaseException):
        pass
    checks = 0
    def check():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise Stopped()
    with restricted_acquisition(check):
        with pytest.raises(Stopped):
            adapter.download([source], BOX)
    assert checks == 2 and len(calls) == 3
