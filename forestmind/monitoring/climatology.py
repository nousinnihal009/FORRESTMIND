"""Frozen, complete-month ERA5 reference departures; no fitted drought index.

References and observations are separate inputs. No current observation fits its
own reference, no missing day is imputed, and these diagnostics are never added
to a model's existing four-feature climate input contract.
"""
from __future__ import annotations

from calendar import monthrange
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from urllib.parse import parse_qs, urlparse

from forestmind.monitoring.climate import SOURCE, VARIABLES


METHOD = "frozen-complete-calendar-month-era5-v1"
MIN_REFERENCE_YEARS = 30
MAX_CONTEXTS = 12
MAX_JSON_BYTES = 16 * 1024**2
FEATURES = {
    "temperature_mean_c": ("degC", "temperature_2m_mean", "mean"),
    "precipitation_sum_mm": ("mm", "precipitation_sum", "sum"),
    "soil_moisture_mean_m3_m3": ("m3/m3", "soil_moisture_0_to_7cm_mean", "mean"),
    "precipitation_minus_reference_et0_mm": ("mm", None, "paired_sum"),
}
POLICY = {
    "minimum_reference_years_per_calendar_month_and_feature": MIN_REFERENCE_YEARS,
    "daily_coverage": "Every calendar day and each required measurement must be present; no imputation or partial-sum extrapolation",
    "standard_deviation": "Sample standard deviation (ddof=1) across complete same-calendar-month yearly aggregates",
    "quantiles": "Linear interpolation at (n-1)*q of sorted yearly monthly aggregates",
    "future_data": "The frozen reference ends before every target month; observed context ends on or before the explicit cutoff",
}
LIMITATIONS = [
    "Reanalysis at one requested centroid and provider grid point; not in-situ forest measurements.",
    "Standardized departures are not SPI, SPEI, calibrated probabilities, or validated ecological drought diagnoses.",
    "Temperature exceedance compares monthly mean daily temperature with the reference monthly 90th percentile; it is not a heatwave or daily-maximum index.",
    "Reference ET0 describes reference-grass atmospheric demand, not actual forest water use; P minus ET0 is a climatic balance.",
    "Thirty complete reference years is this implementation's conservative eligibility rule, not WMO-method certification or ecological validation.",
    "Observation-date separation does not establish historical product availability: ERA5 vintages may have been revised after a historical cutoff.",
    "Hashes establish local artifact consistency, not provider authenticity, source rights, or independent scientific validation.",
]


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("Climate artifacts must contain finite JSON values") from error


def artifact_hash(value: dict) -> str:
    """Canonical digest excluding only the artifact's own digest field."""
    return hashlib.sha256(_canonical({k: v for k, v in value.items() if k != "artifact_sha256"})).hexdigest()


def _seal(value):
    value["artifact_sha256"] = artifact_hash(value)
    return value


def _date(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must use YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must use YYYY-MM-DD") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must use YYYY-MM-DD")
    return parsed


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, not a boolean")
    return value


def _coordinates(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} requires longitude and latitude")
    longitude, latitude = [_number(v, name) for v in value]
    if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        raise ValueError(f"{name} is outside WGS84 bounds")
    return [longitude, latitude]


def _digest_field(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} requires a lowercase SHA256")


def _context(context):
    """Validate the existing saved ERA5 contract and retain raw source identity."""
    if not isinstance(context, dict) or context.get("available") is not True:
        raise ValueError("An available saved ERA5 context is required")
    if context.get("synthetic", False) is not False:
        raise ValueError("Synthetic context cannot establish an observed climate reference")
    if (context.get("provider") != "Open-Meteo" or context.get("dataset") != "ERA5"
            or context.get("source_url") != SOURCE or context.get("units") != VARIABLES):
        raise ValueError("Climate source, dataset and units must match the saved Open-Meteo ERA5 contract")
    start, end = _date(context.get("start"), "Context start"), _date(context.get("end"), "Context end")
    if start.year < 1940 or start > end or (end - start).days > 3653:
        raise ValueError("Each source context must span at most ten years from 1940 onward")
    requested = _coordinates(context.get("location_requested"), "Requested location")
    grid = _coordinates(context.get("grid_location"), "Provider grid location")
    raw = context.get("raw_response")
    if not isinstance(raw, dict) or raw.get("utc_offset_seconds") != 0 or raw.get("timezone") not in {"GMT", "UTC"}:
        raise ValueError("Saved daily ERA5 evidence must explicitly use UTC/GMT")
    if _coordinates([raw.get("longitude"), raw.get("latitude")], "Raw provider grid location") != grid:
        raise ValueError("Raw response and declared provider grid location differ")
    elevation = _number(raw.get("elevation"), "Provider elevation")
    for field in ("license", "spatial_resolution"):
        if not isinstance(context.get(field), str) or not context[field].strip():
            raise ValueError(f"Context {field} must be documented")
    _digest_field(context.get("response_sha256"), "Provider response hash")
    try:
        retrieved = datetime.fromisoformat(context["retrieved_at"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError("A timezone-aware retrieval timestamp is required") from error
    if retrieved.tzinfo is None:
        raise ValueError("A timezone-aware retrieval timestamp is required")
    parsed_url = urlparse(context.get("request_url", ""))
    if (parsed_url.scheme != "https" or parsed_url.netloc != "archive-api.open-meteo.com"
            or parsed_url.path != "/v1/archive" or parsed_url.fragment):
        raise ValueError("Saved request must identify the fixed historical ERA5 provider endpoint")
    query = parse_qs(parsed_url.query, strict_parsing=True)
    expected = {"models": "era5", "timezone": "GMT", "start_date": start.isoformat(), "end_date": end.isoformat()}
    if any(query.get(k) != [v] for k, v in expected.items()):
        raise ValueError("Saved request dates, model and timezone differ from the context")
    try:
        query_location = [float(query["longitude"][0]), float(query["latitude"][0])]
        query_variables = query["daily"][0].split(",")
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError("Saved request requires location and daily variable definitions") from error
    if (any(len(query.get(key, [])) != 1 for key in ("longitude", "latitude", "daily"))
            or query_location != requested or set(query_variables) != set(VARIABLES)
            or len(query_variables) != len(VARIABLES)):
        raise ValueError("Saved request location or variables differ from the context")
    daily, units = raw.get("daily"), raw.get("daily_units")
    if not isinstance(daily, dict) or not isinstance(units, dict) or any(units.get(k) != v for k, v in VARIABLES.items()):
        raise ValueError("Raw daily units differ from the documented climate units")
    times = daily.get("time")
    if not isinstance(times, list) or not 1 <= len(times) <= 3660:
        raise ValueError("A source context requires 1–3660 daily records")
    days = [_date(value, "Daily date") for value in times]
    if any(a >= b for a, b in zip(days, days[1:])):
        raise ValueError("Daily dates must be unique and strictly increasing")
    if any(not start <= day <= end for day in days):
        raise ValueError("Daily evidence lies outside the declared context window")
    for variable in VARIABLES:
        values = daily.get(variable)
        if not isinstance(values, list) or len(values) != len(days):
            raise ValueError("Daily dates and measurement counts disagree")
        for value in values:
            if value is None:
                continue
            _number(value, variable)
            if variable in {"precipitation_sum", "et0_fao_evapotranspiration"} and value < 0:
                raise ValueError("Precipitation and reference ET0 cannot be negative")
            if variable == "soil_moisture_0_to_7cm_mean" and not 0 <= value <= 1:
                raise ValueError("Volumetric soil moisture must lie in [0, 1]")
    identity = {"provider": "Open-Meteo", "dataset": "ERA5", "source_url": SOURCE,
                "location_requested": requested, "grid_location": grid, "elevation_m": elevation,
                "units": VARIABLES.copy(), "timezone": "UTC", "license": context["license"],
                "spatial_resolution": context["spatial_resolution"]}
    provenance = {"start": start.isoformat(), "end": end.isoformat(), "request_url": context["request_url"],
                  "retrieved_at": context["retrieved_at"], "provider_response_sha256": context["response_sha256"],
                  "raw_response_sha256": hashlib.sha256(_canonical(raw)).hexdigest(),
                  "context_sha256": hashlib.sha256(_canonical(context)).hexdigest()}
    rows = {day: {key: daily[key][i] for key in VARIABLES} for i, day in enumerate(days)}
    return start, end, identity, provenance, rows


def _merge(contexts):
    if not isinstance(contexts, (list, tuple)) or not 1 <= len(contexts) <= MAX_CONTEXTS:
        raise ValueError(f"Supply 1–{MAX_CONTEXTS} bounded source contexts")
    parts = sorted((_context(value) for value in contexts), key=lambda part: part[0])
    identity, rows, sources, previous_end = parts[0][2], {}, [], None
    for start, end, source_identity, provenance, daily in parts:
        if source_identity != identity:
            raise ValueError("Sources must use exactly the same provider, grid, requested location, units, resolution and license")
        if previous_end is not None and start <= previous_end:
            raise ValueError("Source context windows overlap; supply disjoint frozen histories")
        previous_end = end
        rows.update(daily)
        sources.append(provenance)
    return parts[0][0], parts[-1][1], identity, sources, rows


def _months(start, end):
    cursor = start.replace(day=1)
    while cursor <= end:
        yield cursor
        cursor = date(cursor.year + 1, 1, 1) if cursor.month == 12 else cursor.replace(month=cursor.month + 1)


def _aggregate(rows, month):
    days = [date(month.year, month.month, day) for day in range(1, monthrange(month.year, month.month)[1] + 1)]
    result = {}
    for name, (unit, variable, operation) in FEATURES.items():
        required = [variable] if variable else ["precipitation_sum", "et0_fao_evapotranspiration"]
        valid = [day for day in days if day in rows and all(rows[day][key] is not None for key in required)]
        complete = len(valid) == len(days)
        value = None
        if complete:
            values = [(rows[day][required[0]] - rows[day][required[1]]) if operation == "paired_sum"
                      else rows[day][variable] for day in days]
            try:
                value = math.fsum(values) / (len(days) if operation == "mean" else 1)
            except (OverflowError, ValueError) as error:
                raise ValueError("Monthly climate aggregation exceeds finite numeric bounds") from error
            _number(value, "Monthly aggregate")
        result[name] = {"value": value, "unit": unit, "complete": complete,
                        "valid_days": len(valid), "expected_days": len(days),
                        "unreported_days": sum(day not in rows for day in days),
                        "missing_measurement_days": sum(day in rows and any(rows[day][k] is None for k in required) for day in days)}
    return result


def _quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left = int(position)
    return ordered[left] + (ordered[min(left + 1, len(ordered) - 1)] - ordered[left]) * (position - left)


def _reference_stats(samples, years, unit):
    values = [sample["value"] for sample in samples]
    available = len(samples) >= MIN_REFERENCE_YEARS
    mean = statistics.mean(values) if available else None
    sd = statistics.stdev(values) if available else None
    for value in (mean, sd):
        if value is not None:
            _number(value, "Reference statistic")
    return {"unit": unit, "available": available, "complete_year_count": len(samples),
            "samples": samples, "missing_years": sorted(set(years) - {sample["year"] for sample in samples}),
            "mean": mean, "sample_standard_deviation": sd,
            "p10": _quantile(values, .1) if available else None,
            "p90": _quantile(values, .9) if available else None,
            "reason": None if available else f"Need at least {MIN_REFERENCE_YEARS} complete reference years for this calendar month and feature; {len(samples)} available."}


def freeze_climatology(contexts: list[dict], *, baseline_start: str, baseline_end: str) -> dict:
    """Freeze ≥30-year reference windows; unavailable feature/months stay explicit."""
    start, end = _reference_dates(baseline_start, baseline_end)
    source_start, source_end, identity, sources, rows = _merge(contexts)
    if source_start < start or source_end > end:
        raise ValueError("Every reference context must lie inside the declared frozen baseline")
    yearly = {month: _aggregate(rows, month) for month in _months(start, end)}
    years = list(range(start.year, end.year + 1))
    months = []
    for calendar_month in range(1, 13):
        variables = {}
        for name, (unit, _, _) in FEATURES.items():
            samples = [{"year": year, "value": yearly[date(year, calendar_month, 1)][name]["value"]}
                       for year in years if yearly[date(year, calendar_month, 1)][name]["complete"]]
            variables[name] = _reference_stats(samples, years, unit)
        months.append({"calendar_month": calendar_month, "variables": variables})
    return _seal({"schema_version": 1, "kind": "forestmind_frozen_monthly_climatology", "method": METHOD,
                  "baseline": {"start": baseline_start, "end": baseline_end, "calendar_years": len(years)},
                  "policy": POLICY.copy(), "source_contract": identity, "sources": sources, "months": months,
                  "available_feature_months": sum(v["available"] for m in months for v in m["variables"].values()),
                  "historical_availability_verified": False, "model_input": False,
                  "limitations": LIMITATIONS.copy()})


def _reference_dates(start, end):
    start, end = _date(start, "Baseline start"), _date(end, "Baseline end")
    if (start.year < 1940 or (start.month, start.day) != (1, 1) or (end.month, end.day) != (12, 31)
            or not MIN_REFERENCE_YEARS <= end.year - start.year + 1 <= 60):
        raise ValueError("The frozen baseline must span 30–60 complete calendar years from 1940 onward")
    return start, end


def validate_climatology(reference: dict) -> None:
    """Validate integrity and recompute statistics from stored annual samples."""
    if not isinstance(reference, dict) or reference.get("artifact_sha256") != artifact_hash(reference):
        raise ValueError("Frozen climatology artifact hash mismatch")
    if (reference.get("schema_version") != 1 or reference.get("kind") != "forestmind_frozen_monthly_climatology"
            or reference.get("method") != METHOD or reference.get("policy") != POLICY):
        raise ValueError("Unsupported frozen climatology schema, method or completeness policy")
    baseline = reference.get("baseline", {})
    start, end = _reference_dates(baseline.get("start"), baseline.get("end"))
    years = list(range(start.year, end.year + 1))
    if baseline.get("calendar_years") != len(years):
        raise ValueError("Reference calendar-year count differs from its dates")
    months = reference.get("months")
    if not isinstance(months, list) or len(months) != 12:
        raise ValueError("Reference requires all twelve calendar months")
    count = 0
    for month, record in enumerate(months, 1):
        if record.get("calendar_month") != month or set(record.get("variables", {})) != set(FEATURES):
            raise ValueError("Reference month/feature definitions disagree")
        for name, (unit, _, _) in FEATURES.items():
            stats = record["variables"][name]
            samples = stats.get("samples")
            if not isinstance(samples, list) or len(samples) > len(years):
                raise ValueError("Reference samples must be bounded annual observations")
            used = []
            for sample in samples:
                if (not isinstance(sample, dict) or set(sample) != {"year", "value"}
                        or type(sample["year"]) is not int or sample["year"] not in years):
                    raise ValueError("Reference sample lies outside the declared baseline")
                _number(sample["value"], "Reference aggregate")
                used.append(sample["year"])
            if used != sorted(set(used)):
                raise ValueError("Reference samples require unique ordered years")
            if stats != _reference_stats(samples, years, unit):
                raise ValueError("Frozen reference statistics disagree with the preserved yearly aggregates")
            count += stats["available"]
    if reference.get("available_feature_months") != count:
        raise ValueError("Reference availability count differs from its validated samples")
    if reference.get("historical_availability_verified") is not False or reference.get("model_input") is not False:
        raise ValueError("Reference cannot assert historical availability or implicit neural inputs")


def analyze_climate(contexts: list[dict], reference: dict, *, cutoff: str) -> dict:
    """Describe complete months after the frozen reference, up to a strict cutoff."""
    validate_climatology(reference)
    cutoff_date = _date(cutoff, "Observation cutoff")
    start, end, identity, sources, rows = _merge(contexts)
    reference_end = _date(reference["baseline"]["end"], "Baseline end")
    if identity != reference.get("source_contract"):
        raise ValueError("Observed climate and frozen reference use different source, location, grid or units")
    if start <= reference_end:
        raise ValueError("Every observation window must begin after the frozen reference ends")
    if end > cutoff_date:
        raise ValueError("Observed context contains a window after the explicit cutoff; future values are rejected")
    if start > cutoff_date or (cutoff_date - start).days > 3653:
        raise ValueError("Analysis must cover an ordered window of at most ten years")
    output = []
    for month in _months(start, cutoff_date):
        measurements = _aggregate(rows, month)
        variables = {}
        for name, observed in measurements.items():
            stats = reference["months"][month.month - 1]["variables"][name]
            available = observed["complete"] and stats["available"]
            reason = ("A complete calendar month with every required daily measurement is needed."
                      if not observed["complete"] else stats["reason"])
            departure = observed["value"] - stats["mean"] if available else None
            sd = stats["sample_standard_deviation"]
            standardized = departure / sd if available and sd > 0 else None
            if standardized is not None:
                _number(standardized, "Standardized departure")
            variables[name] = {**observed, "available": available, "reason": reason,
                "reference_years": stats["complete_year_count"], "reference_mean": stats["mean"],
                "reference_sample_standard_deviation": sd, "departure": departure,
                "standardized_departure": standardized, "standardized_units": "reference sample standard deviations",
                "standardized_reason": "Reference variance is zero; a standardized departure is undefined." if available and sd == 0 else reason}
        temperature = variables["temperature_mean_c"]
        threshold = reference["months"][month.month - 1]["variables"]["temperature_mean_c"]["p90"]
        output.append({"month": month.strftime("%Y-%m"), "variables": variables,
            "warm_month": {"available": temperature["available"], "threshold_c": threshold,
                "above_reference_p90": temperature["value"] > threshold if temperature["available"] else None,
                "reason": temperature["reason"], "definition": "Monthly mean daily 2 m temperature strictly above the same-calendar-month reference 90th percentile; descriptive, not a heatwave."}})
    return _seal({"schema_version": 1, "kind": "forestmind_monthly_climate_departures", "method": METHOD,
                  "reference_sha256": reference["artifact_sha256"], "baseline": reference["baseline"].copy(),
                  "cutoff": cutoff, "start": start.isoformat(), "observed_through": end.isoformat(),
                  "source_contract": identity, "sources": sources, "months": output,
                  "historical_availability_verified": False, "model_input": False,
                  "limitations": LIMITATIONS.copy()})


def read_json(path: str | Path) -> dict:
    """Bounded JSON with duplicate keys and nonfinite literals rejected."""
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Climate JSON exceeds the 16 MiB limit")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Climate JSON must contain one object")
    return value


def write_new_artifact(value: dict, path: str | Path) -> Path:
    """Write a new hashed artifact; existing files are never overwritten."""
    if value.get("artifact_sha256") != artifact_hash(value):
        raise ValueError("Refusing to write an artifact whose hash does not match its contents")
    rendered = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)
    return path
