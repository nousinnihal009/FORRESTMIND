"""Offline contracts for real-data scaling, bounded requests and CLI failures."""

from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from forestmind.data.adapters.network import (
    AssetSigner, _RedactAccessQueries, is_retryable_error, open_catalog, open_raster,
    public_error, restricted_acquisition,
)
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


def test_raster_reads_have_bounded_timeouts_and_clean_up(monkeypatch):
    environment, opened = {}, []
    source = object()
    @contextmanager
    def env(**options):
        environment.update(options)
        yield
    @contextmanager
    def raster_open(href):
        opened.append(href)
        yield source
    monkeypatch.setitem(sys.modules, "rasterio", SimpleNamespace(Env=env, open=raster_open))
    error_logger = logging.getLogger("rasterio._err")
    previous_filters = list(error_logger.filters)
    with pytest.raises(RuntimeError, match="read failed"):
        with open_raster("https://public.example/band.tif", 7) as actual:
            assert actual is source
            assert len(error_logger.filters) == len(previous_filters) + 1
            raise RuntimeError("read failed")
    assert opened == ["https://public.example/band.tif"]
    assert environment["GDAL_HTTP_TIMEOUT"] == "7"
    assert environment["GDAL_HTTP_CONNECTTIMEOUT"] == "7"
    assert environment["GDAL_HTTP_MAX_RETRY"] == "0"
    assert error_logger.filters == previous_filters


@pytest.mark.parametrize("error,expected", [
    (TimeoutError("read timed out"), True),
    (ConnectionError("connection reset"), True),
    (OSError("offline"), True),
    (OSError("Read failed. See previous exception for details."), True),
    (RuntimeError("HTTP response code: 429"), True),
    (OSError("HTTP 503 reading raster"), True),
    (OSError("404 Client Error: Not Found"), False),
    (OSError("HTTP response code: 403"), False),
    (PermissionError("access denied"), False),
    (FileNotFoundError("cache missing"), False),
    (OSError(28, "No space left on device"), False),
    (OSError("not recognized as being in a supported file format"), False),
    (ValueError("timeout_seconds must be positive"), False),
    (ValueError("Scene has no valid pixels in the requested region"), False),
    (KeyError("B04"), False),
    (RuntimeError("unspecified processing failure"), False),
])
def test_retry_classification_keeps_transport_and_permanent_failures_distinct(error, expected):
    assert is_retryable_error(error) is expected


def test_retry_classification_reads_wrapped_http_response_and_handles_cycles():
    outer = RuntimeError("Read failed")
    inner = OSError("provider rejected request")
    inner.response = SimpleNamespace(status_code=503)
    outer.__cause__ = inner
    inner.__cause__ = outer
    assert is_retryable_error(outer)
    inner.response.status_code = 404
    assert not is_retryable_error(outer)


def test_retry_classification_does_not_restore_suppressed_context():
    try:
        raise TimeoutError("earlier operation timed out")
    except TimeoutError:
        try:
            raise ValueError("invalid metadata") from None
        except ValueError as error:
            assert not is_retryable_error(error)


def test_cancel_before_signing_never_requests_a_token(monkeypatch):
    request = Mock()
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=request))
    class Stopped(BaseException):
        pass
    def check():
        raise Stopped()
    with restricted_acquisition(check):
        with pytest.raises(Stopped):
            AssetSigner().sign("https://sentinel2l2a01.blob.core.windows.net/container/band.tif")
    request.assert_not_called()


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


def test_gdal_truncated_signed_filename_tail_is_discarded():
    record = logging.LogRecord("rasterio._err", logging.INFO, "", 1,
                               "GDAL signalled an error: err_no=1, msg='%s, band 1: IReadBlock failed at X offset 0, Y offset 2: TIFFReadEncodedTile() failed.'",
                               ("private-signature-tail%3D",), None)
    assert _RedactAccessQueries().filter(record)
    assert record.getMessage().startswith("[raster source] band 1: IReadBlock failed")
    assert "private-signature" not in record.getMessage()
    assert "X offset 0, Y offset 2" in record.getMessage()


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
