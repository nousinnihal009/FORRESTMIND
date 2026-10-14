"""Bounded network clients for public Planetary Computer assets."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit
import logging
import re


def open_catalog(url: str, timeout_seconds: float):
    from pystac_client import Client
    from pystac_client.stac_api_io import StacApiIO

    # Search does not need SAS signatures. Sign only assets actually read.
    timeout = (min(10.0, timeout_seconds), timeout_seconds)
    return Client.open(url, stac_io=StacApiIO(timeout=timeout, max_retries=0))


class AssetSigner:
    """Sign public Azure assets with finite HTTP timeouts and per-run caching.

    Uses the same public endpoint documented by Microsoft's SDK in
    planetary_computer/sas.py; its sign() path currently has no request timeout.
    Tokens remain in memory and are never written into cache provenance.
    """

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout_seconds = timeout_seconds
        self._tokens = {}

    def sign(self, href: str) -> str:
        import requests

        parsed = urlsplit(href)
        if not (parsed.hostname or "").endswith(".blob.core.windows.net"):
            return href
        account = parsed.hostname.split(".", 1)[0]
        container = parsed.path.lstrip("/").split("/", 1)[0]
        key = (account, container)
        token, expires = self._tokens.get(key, (None, None))
        if token is None or (expires - datetime.now(timezone.utc)).total_seconds() < 60:
            endpoint = f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}"
            response = requests.get(endpoint, timeout=(min(10., self.timeout_seconds), self.timeout_seconds))
            response.raise_for_status()
            body = response.json()
            token = body["token"]
            expires = datetime.fromisoformat(body["msft:expiry"].replace("Z", "+00:00"))
            self._tokens[key] = (token, expires)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, token, parsed.fragment))


@contextmanager
def open_raster(href: str, timeout_seconds: float = 30.0):
    import rasterio

    # GDAL may log a signed filename before raising. Redact at its Python
    # logging boundary, not only in the adapter's final exception handler.
    error_logger = logging.getLogger("rasterio._err")
    token_filter = _RedactAccessQueries()
    error_logger.addFilter(token_filter)
    try:
        with rasterio.Env(
            GDAL_HTTP_CONNECTTIMEOUT=str(min(10., timeout_seconds)),
            GDAL_HTTP_TIMEOUT=str(timeout_seconds), GDAL_HTTP_MAX_RETRY="0",
            # Range requests run sequentially: concurrent COG ranges can be
            # truncated by intermediary transports even when single ranges work.
            # https://gdal.org/en/stable/user/configoptions.html
            GDAL_HTTP_MULTIRANGE="SERIAL",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.jp2",
        ):
            with rasterio.open(href) as source:
                yield source
    finally:
        error_logger.removeFilter(token_filter)


def public_error(error: Exception) -> str:
    """Keep useful error text while removing signed access query strings."""
    return re.sub(r"((?:https?://[^\s?'\"]+)|(?:[^\s?'\"]+\.(?:tif|tiff|jp2)))\?[^\s'\"]+",
                  r"\1?[redacted]", str(error), flags=re.IGNORECASE)


class _RedactAccessQueries(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = public_error(record.getMessage())
        record.args = ()
        return True
