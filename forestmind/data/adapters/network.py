"""Bounded network clients for public Planetary Computer assets."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit
import logging
import re
import errno


_restricted = ContextVar("forestmind_restricted_acquisition", default=False)
_acquisition_check = ContextVar("forestmind_acquisition_check", default=None)
CATALOG_HOSTS = {"earth-search.aws.element84.com", "planetarycomputer.microsoft.com"}
ASSET_HOSTS = {"e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com",
               "sentinel-cogs.s3.us-west-2.amazonaws.com", "sentinel-cogs.s3.amazonaws.com",
               "sentinel2l2a01.blob.core.windows.net", "sentinel2l2a02.blob.core.windows.net",
               "sentinel1rtc.blob.core.windows.net", "sentinel1euwestrtc.blob.core.windows.net",
               "esa-worldcover.s3.eu-central-1.amazonaws.com", "landsateuwest.blob.core.windows.net"}


def check_acquisition() -> None:
    """Check the caller's cancellation/deadline even when a read is reused."""
    if _acquisition_check.get():
        _acquisition_check.get()()


def is_retryable_error(error: Exception) -> bool:
    """Recognize transient transport failures without retrying bad inputs.

    Rasterio wraps HTTP/read exceptions, so inspect the bounded cause chain.
    Unclassified I/O failures remain eligible for bounded caller retries;
    classification is a recovery policy, not proof of the underlying cause.
    This helper performs no retries; callers retain their own attempt budgets.
    """
    chain, seen = [], set()
    current = error
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)

    # Explicit HTTP outcomes take precedence over generic wrapper wording.
    statuses = []
    for current in chain:
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            statuses.append(status)
        statuses.extend(int(value) for value in re.findall(
            r"\bHTTP(?:\s+response)?(?:\s+(?:error|status|code))?\s*[:=]?\s*([45]\d\d)\b|"
            r"\b([45]\d\d)(?=\s+(?:Client|Server)\s+Error)",
            str(current), flags=re.IGNORECASE))
    if statuses:
        return all(status in {408, 425, 429, 500, 502, 503, 504} for status in statuses)
    for current in chain:
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        # requests/urllib3 timeouts do not consistently subclass built-in
        # TimeoutError; messages also carry GDAL's libcurl transport failures.
        if re.search(r"\b(?:timed?\s*out|timeout|connection\s+(?:reset|aborted|refused)|"
                     r"remote\s+end\s+closed|temporary\s+failure\s+in\s+name\s+resolution|"
                     r"could\s+not\s+resolve\s+host|couldn't\s+connect|"
                     r"failed\s+to\s+connect|empty\s+reply\s+from\s+server)\b",
                     str(current), flags=re.IGNORECASE):
            return True
    return False


def validate_destination(href: str, *, catalog=False) -> str:
    parsed = urlsplit(href)
    if (parsed.scheme != "https" or parsed.hostname not in (CATALOG_HOSTS if catalog else ASSET_HOSTS)
            or parsed.username or parsed.password or parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError("Acquisition destination is outside the documented public provider allowlist")
    return href


@contextmanager
def restricted_acquisition(check=None):
    """Web workers restrict provider destinations; research CLI keeps explicit local inputs."""
    token = _restricted.set(True)
    check_token = _acquisition_check.set(check)
    try:
        yield
    finally:
        _restricted.reset(token)
        _acquisition_check.reset(check_token)


def open_catalog(url: str, timeout_seconds: float):
    check_acquisition()
    if _restricted.get():
        validate_destination(url, catalog=True)
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

        check_acquisition()
        if _restricted.get():
            validate_destination(href)

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

    check_acquisition()
    if _restricted.get():
        validate_destination(href)

    # GDAL may log a signed filename before raising. Redact at its Python
    # logging boundary, not only in the adapter's final exception handler.
    error_logger = logging.getLogger("rasterio._err")
    token_filter = _RedactAccessQueries()
    error_logger.addFilter(token_filter)
    try:
        with rasterio.Env(
            GDAL_HTTP_CONNECTTIMEOUT=str(min(10., timeout_seconds)),
            GDAL_HTTP_TIMEOUT=str(timeout_seconds), GDAL_HTTP_MAX_RETRY="0",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.jp2",
        ):
            with rasterio.open(href) as source:
                yield source
            check_acquisition()
    finally:
        error_logger.removeFilter(token_filter)


def public_error(error: Exception) -> str:
    """Keep useful error text while removing signed access query strings."""
    return re.sub(r"((?:https?://[^\s?'\"]+)|(?:[^\s?'\"]+\.(?:tif|tiff|jp2)))\?[^\s'\"]+",
                  r"\1?[redacted]", str(error), flags=re.IGNORECASE)


class _RedactAccessQueries(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = public_error(record.getMessage())
        # GDAL sometimes truncates a signed filename to its final characters,
        # losing the URL and query separator needed by public_error(). Keep the
        # actionable decoder/block diagnostic but discard its source prefix.
        block_error = re.search(r"band \d+: IReadBlock failed\b", message)
        record.msg = "[raster source] " + message[block_error.start():] if block_error else message
        record.args = ()
        return True
