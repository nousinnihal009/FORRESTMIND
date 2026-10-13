"""Offline contracts for real-data scaling, bounded requests and CLI failures."""

from datetime import datetime, timedelta, timezone
import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from forestmind.data.adapters.network import AssetSigner, _RedactAccessQueries, open_catalog, public_error
from forestmind.data.adapters.sentinel2 import Sentinel2Adapter, reflectance_scaling
from scripts import download_data


@pytest.mark.parametrize("baseline,expected", [("03.01", 0.25), ("04.00", 0.15), ("05.10", 0.15)])
def test_sentinel2_baseline_controls_dn_conversion(baseline, expected):
    conversion = reflectance_scaling({}, {"s2:processing_baseline": baseline})
    assert 2500 * conversion["scale"] + conversion["offset"] == pytest.approx(expected)
    assert conversion["source"] == "planetary-computer-processing-baseline"


def test_explicit_stac_scaling_is_not_offset_twice():
    conversion = reflectance_scaling({"raster:bands": [{"scale": 0.0001, "offset": -0.1}]},
                                     {"s2:processing_baseline": "05.10"})
    assert 2500 * conversion["scale"] + conversion["offset"] == pytest.approx(0.15)
    assert conversion["source"] == "stac-raster-scale-offset"
    assert reflectance_scaling({"raster:bands": [{"scale": 0.0001, "offset": 0}]}, {})["offset"] == 0


@pytest.mark.parametrize("metadata,properties", [({}, {}), ({}, {"s2:processing_baseline": "nan"}),
    ({"raster:bands": [{"scale": 0, "offset": 0}]}, {}),
    ({"raster:bands": [{"scale": 0.1, "offset": float("nan")}]}, {})])
def test_ambiguous_or_invalid_reflectance_conversion_fails_closed(metadata, properties):
    with pytest.raises(ValueError):
        reflectance_scaling(metadata, properties)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_unbounded_adapter_timeout_is_rejected(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        Sentinel2Adapter(cache_dir=tmp_path, timeout_seconds=timeout)


def test_catalog_io_has_finite_timeouts_without_retries(monkeypatch):
    client, io = Mock(), Mock()
    monkeypatch.setitem(sys.modules, "pystac_client", SimpleNamespace(Client=client))
    monkeypatch.setitem(sys.modules, "pystac_client.stac_api_io", SimpleNamespace(StacApiIO=io))
    open_catalog("https://catalog.example", 8)
    io.assert_called_once_with(timeout=(8, 8), max_retries=0)
    client.open.assert_called_once_with("https://catalog.example", stac_io=io.return_value)


def test_asset_signing_is_bounded_cached_and_not_logged(monkeypatch):
    response = Mock()
    expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    response.json.return_value = {"token": "sp=r&sig=private-token", "msft:expiry": expiry}
    request = Mock(return_value=response)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=request))
    signer = AssetSigner(timeout_seconds=7)
    first = signer.sign("https://account.blob.core.windows.net/container/band.tif")
    signer.sign("https://account.blob.core.windows.net/container/other.tif")
    request.assert_called_once_with(
        "https://planetarycomputer.microsoft.com/api/sas/v1/token/account/container", timeout=(7, 7))
    assert "private-token" not in public_error(RuntimeError(f"Cannot read '{first}'"))
    assert signer.sign("https://public.example/band.tif") == "https://public.example/band.tif"


def test_partial_download_raises_and_retains_completed_count():
    adapter = SimpleNamespace(download=Mock(return_value=["cached.npz"]),
                              download_failures=[{"scene_id": "failed", "reason": "timed out"}])
    with pytest.raises(download_data.PartialDownloadError) as error:
        download_data.checked_download(adapter, ["scene"], "bbox")
    assert error.value.completed == 1
    assert "timed out" in str(error.value)


def test_gdal_signed_filename_diagnostics_are_redacted():
    record = logging.LogRecord("rasterio._err", logging.INFO, "", 1, "Cannot read %s", 
                               ("band.tif?st=today&sig=secret",), None)
    assert _RedactAccessQueries().filter(record)
    assert record.getMessage() == "Cannot read band.tif?[redacted]"


REQUEST = ["--region=-55.10,-3.50,-55.08,-3.48", "--sensors", "s2", "--max-scenes", "1"]


@pytest.mark.parametrize("result,expected", [(2, 0), (0, 1), (RuntimeError("network down"), 1),
    (ImportError("missing rasterio"), 1),
    (download_data.PartialDownloadError(1, [{"reason": "network down"}]), 1)])
def test_cli_failure_exit_codes(monkeypatch, result, expected):
    download = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
    monkeypatch.setattr(download_data, "download_sentinel2", download)
    assert download_data.main(REQUEST + ["--timeout", "9"]) == expected
    assert download.call_args.args[-1] == 9


@pytest.mark.parametrize("options", [
    ["--sensors", "typo"], ["--max-scenes", "0"], ["--timeout", "nan"],
    ["--region=nan,-1,1,1"], ["--region=1,1,-1,-1"],
    ["--start", "2025-10-01", "--end", "2024-01-01"], ["--max-cloud", "101"],
])
def test_bad_requests_never_start_network(monkeypatch, options):
    download = Mock()
    monkeypatch.setattr(download_data, "download_sentinel2", download)
    assert download_data.main(REQUEST + options) == 2
    download.assert_not_called()


def test_empty_dry_run_is_a_valid_search(monkeypatch):
    monkeypatch.setattr(download_data, "download_sentinel2", Mock(return_value=0))
    assert download_data.main(REQUEST + ["--dry-run"]) == 0


def test_old_scaling_cache_is_not_reused(tmp_path):
    from forestmind.data.adapters.base import SceneMetadata
    from forestmind.utils.geo import BoundingBox

    adapter = Sentinel2Adapter(cache_dir=tmp_path, patch_size=4)
    box = BoundingBox(-55.10, -3.50, -55.08, -3.48)
    scene = SceneMetadata("old", "sentinel2", datetime(2024, 7, 7), box)
    path = adapter.cache_path_for_scene(scene, box)
    adapter.save_patch_to_cache(path, np.ones((12, 4, 4)), scene, box, np.ones((4, 4), dtype=bool),
                                preprocessing_version="forestmind-grid-v2")
    assert not adapter.is_cached(scene, box)
