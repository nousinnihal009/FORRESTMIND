"""Bounded ERA5 contextual reanalysis acquisition from a fixed public endpoint."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import math
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from forestmind.monitoring.spatial import geometry_bbox

SOURCE = "https://open-meteo.com/en/docs/historical-weather-api"
VARIABLES = {"temperature_2m_mean": "°C", "precipitation_sum": "mm",
             "et0_fao_evapotranspiration": "mm", "soil_moisture_0_to_7cm_mean": "m³/m³"}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Climate provider redirected; refusing an unapproved destination")


def aggregate_climate(response: dict, *, start: str, end: str) -> list[dict]:
    """Monthly statistics retain per-variable counts; no gap filling or anomalies."""
    days = response.get("daily", {})
    times = days.get("time", [])
    if not times or len(times) > 3660 or len(times) != len(set(times)):
        raise ValueError("Climate response has no data, duplicate dates, or exceeds the ten-year limit")
    for variable, expected in VARIABLES.items():
        unit = response.get("daily_units", {}).get(variable)
        if unit != expected:
            raise ValueError(f"Climate unit mismatch for {variable}: expected {expected}")
        if len(days.get(variable, [])) != len(times):
            raise ValueError(f"Climate dates and {variable} counts disagree")
    groups = defaultdict(list)
    for index, value in enumerate(times):
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value or not start <= value <= end:
            raise ValueError("Climate response contains a date outside the requested cutoff")
        groups[value[:7]].append(index)
    output = []
    for month, indices in sorted(groups.items()):
        row = {"month": month, "reported_days": len(indices), "quality": {}}
        for variable in VARIABLES:
            values = [days[variable][i] for i in indices if days[variable][i] is not None]
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError("Climate response contains a nonfinite or nonnumeric measurement")
            row[variable] = (sum(values) if variable in {"precipitation_sum", "et0_fao_evapotranspiration"}
                             else sum(values) / len(values)) if values else None
            row["quality"][variable] = {"valid_days": len(values), "missing_days": len(indices) - len(values)}
        output.append(row)
    return output


def acquire_climate(geometry: dict, *, start: str, end: str, timeout: int = 20) -> dict:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or (last - first).days > 3653 or first.year < 1940:
        raise ValueError("ERA5 requests require an ordered date window from 1940 onward, at most ten years")
    west, south, east, north = geometry_bbox(geometry)
    latitude, longitude = (south + north) / 2, (west + east) / 2
    query = urlencode({"latitude": latitude, "longitude": longitude, "start_date": start, "end_date": end,
                       "daily": ",".join(VARIABLES), "models": "era5", "timezone": "GMT"})
    url = "https://archive-api.open-meteo.com/v1/archive?" + query
    request = Request(url, headers={"User-Agent": "ForestMind-local-research/0.6", "Accept": "application/json"})
    with build_opener(NoRedirect()).open(request, timeout=timeout) as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("Climate response exceeds the 4 MiB limit")
    response = json.loads(raw)
    monthly = aggregate_climate(response, start=start, end=end)
    return {"available": True, "provider": "Open-Meteo", "dataset": "ERA5", "source_url": SOURCE,
            "request_url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "response_sha256": hashlib.sha256(raw).hexdigest(), "raw_response": response,
            "start": start, "end": end, "location_requested": [longitude, latitude],
            "grid_location": [response.get("longitude"), response.get("latitude")],
            "spatial_resolution": "ERA5 0.25 degrees (approximately 25 km)", "units": VARIABLES,
            "monthly": monthly, "role": "context_only", "model_input": False,
            "license": "CC BY 4.0; free hosted API is for non-commercial use, subject to provider limits",
            "limitations": ["Model-based reanalysis at the selected area's centroid; not an in-situ measurement or a forest-only spatial average.",
                            "No anomaly, drought index, causal driver or model embedding is fabricated from this short context window.",
                            "ERA5 can lag by five days; missing days remain missing, and monthly sums may be partial."]}
