"""Bounded ERA5 contextual reanalysis acquisition from a fixed public endpoint."""
from __future__ import annotations

from collections import defaultdict
from calendar import monthrange
from copy import deepcopy
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
WATER_BALANCE_METHOD = "paired-daily-precipitation-minus-reference-et0-v1"
WATER_BALANCE_DESCRIPTION = (
    "Sum of daily precipitation minus reference ET₀ on dates with both values. "
    "ET₀ represents atmospheric demand for a well-watered reference grass surface, "
    "not measured forest evapotranspiration. This climatic balance is not an ecosystem "
    "water budget, a drought diagnosis, a standardized anomaly or a causal driver score."
)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Climate provider redirected; refusing an unapproved destination")


def aggregate_climate(response: dict, *, start: str, end: str) -> list[dict]:
    """Monthly raw summaries and paired-day climatic balance, without gap filling."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first.isoformat() != start or last.isoformat() != end or first > last or (last - first).days > 3653:
        raise ValueError("Climate aggregation requires an ordered ISO date window of at most ten years")
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
    cursor = first.replace(day=1)
    while cursor <= last:
        groups[cursor.strftime("%Y-%m")] = []
        cursor = date(cursor.year + 1, 1, 1) if cursor.month == 12 else cursor.replace(month=cursor.month + 1)
    for index, value in enumerate(times):
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value or not start <= value <= end:
            raise ValueError("Climate response contains a date outside the requested cutoff")
        groups[value[:7]].append(index)
    output = []
    for month, indices in sorted(groups.items()):
        month_first = date.fromisoformat(month + "-01")
        calendar_days = monthrange(month_first.year, month_first.month)[1]
        month_last = month_first.replace(day=calendar_days)
        period_start, period_end = max(first, month_first), min(last, month_last)
        expected = (period_end - period_start).days + 1
        row = {"month": month, "reported_days": len(indices), "requested_days": expected,
               "unreported_days": expected - len(indices), "quality": {}}
        for variable in VARIABLES:
            values = [days[variable][i] for i in indices if days[variable][i] is not None]
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError("Climate response contains a nonfinite or nonnumeric measurement")
            row[variable] = (sum(values) if variable in {"precipitation_sum", "et0_fao_evapotranspiration"}
                             else sum(values) / len(values)) if values else None
            row["quality"][variable] = {"valid_days": len(values), "missing_days": len(indices) - len(values),
                                        "missing_requested_days": expected - len(values)}
        paired = [i for i in indices if days["precipitation_sum"][i] is not None
                  and days["et0_fao_evapotranspiration"][i] is not None]
        balance = sum(days["precipitation_sum"][i] - days["et0_fao_evapotranspiration"][i]
                      for i in paired) if paired else None
        row["water_balance"] = {
            "method": WATER_BALANCE_METHOD, "units": "mm", "sum_mm": balance,
            "mean_daily_mm": balance / len(paired) if paired else None,
            "mean_daily_units": "mm/day", "paired_days": len(paired), "expected_days": expected,
            "missing_pair_days": expected - len(paired), "unreported_days": expected - len(indices),
            "paired_dates": sorted(times[i] for i in paired),
            "period_start": period_start.isoformat(), "period_end": period_end.isoformat(),
            "complete_requested_period": len(paired) == expected,
            "complete_calendar_month": len(paired) == calendar_days,
            "partial_calendar_month": period_start != month_first or period_end != month_last,
        }
        output.append(row)
    return output


def refresh_climate_diagnostics(context: dict) -> dict:
    """Recompute derived context from saved daily values; never alter its source.

    The caller may save this new context for a new immutable analysis snapshot.
    Legacy contexts lacking daily evidence remain unchanged, rather than deriving
    a balance by subtracting monthly totals with potentially different masks.
    """
    result = deepcopy(context)
    if not result.get("available") or not isinstance(result.get("raw_response"), dict):
        return result
    result["monthly"] = aggregate_climate(result["raw_response"], start=result["start"], end=result["end"])
    result["diagnostics"] = {"water_balance": {
        "method": WATER_BALANCE_METHOD, "definition": WATER_BALANCE_DESCRIPTION,
        "source_url": SOURCE, "units": "mm", "role": "context_only", "model_input": False,
        "missing_data": "Only same-day valid pairs contribute; absent dates and either missing input remain missing."
    }}
    return result


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
    return refresh_climate_diagnostics({"available": True, "provider": "Open-Meteo", "dataset": "ERA5", "source_url": SOURCE,
            "request_url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "response_sha256": hashlib.sha256(raw).hexdigest(), "raw_response": response,
            "start": start, "end": end, "location_requested": [longitude, latitude],
            "grid_location": [response.get("longitude"), response.get("latitude")],
            "spatial_resolution": "ERA5 0.25 degrees (approximately 25 km)", "units": VARIABLES,
            "role": "context_only", "model_input": False,
            "license": "CC BY 4.0; free hosted API is for non-commercial use, subject to provider limits",
            "limitations": ["Model-based reanalysis at the selected area's centroid; not an in-situ measurement or a forest-only spatial average.",
                            "No anomaly, drought index, causal driver or model embedding is fabricated from this short context window.",
                            "ERA5 can lag by five days; missing days remain missing, and monthly sums may be partial."]})
