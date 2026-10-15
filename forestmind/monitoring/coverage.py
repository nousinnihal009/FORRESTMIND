"""Calendar coverage and software readiness for prepared observation histories.

Missing months are report entries only: they are never inserted into the source
observations or passed to the monitoring methods as fabricated measurements.
"""

from __future__ import annotations

from datetime import date

from forestmind.api.schemas import ImportRegion
from forestmind.monitoring.analysis import forecast_summary, indicator_score, resilience_summary


def _bound(value: str | None, default: str, name: str) -> date:
    if value is None:
        return date.fromisoformat(default)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a calendar date formatted YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} must be a calendar date formatted YYYY-MM-DD") from None
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must be a calendar date formatted YYYY-MM-DD")
    return parsed


def assess_history(payload: dict, start: str | None = None, end: str | None = None) -> dict:
    """Validate a strict import payload and assess an inclusive calendar window.

    Bounds default to the first and last source observation dates. Explicit
    bounds may extend past the source span; rows outside them are excluded.
    Months intersecting a partial-month window are included, but only actual
    observation dates inside the window count toward coverage. A low-coverage
    row takes precedence over missing indices in the mutually exclusive counts.

    Availability exactly follows the existing analysis methods on the included
    observations. The separate ``*_ready_at_requested_end`` flags also require
    a usable observation in the requested ending month. Neither is evidence of
    predictive validity, freshness beyond that month, or ecological stability.
    """
    prepared = ImportRegion.model_validate(payload)
    source_rows = [row.model_dump() for row in prepared.observations]
    first = _bound(start, source_rows[0]["date"], "start")
    last = _bound(end, source_rows[-1]["date"], "end")
    if last < first:
        raise ValueError("end must be on or after start")
    start_text, end_text = first.isoformat(), last.isoformat()
    rows = [
        {**row, "health_score": indicator_score(row)}
        for row in source_rows if start_text <= row["date"] <= end_text
    ]
    by_month = {row["date"][:7]: row for row in rows}
    months = []
    longest = trailing = 0
    # Integer month offsets avoid advancing datetime.date beyond year 9999.
    first_month = (first.year - 1) * 12 + first.month - 1
    last_month = (last.year - 1) * 12 + last.month - 1
    for offset in range(first_month, last_month + 1):
        year, month = divmod(offset, 12)
        key = f"{year + 1:04d}-{month + 1:02d}"
        row = by_month.get(key)
        if row is None:
            state = "missing"
        elif row["health_score"] is not None:
            state = "usable"
        elif row["valid_fraction"] < 0.4:
            state = "low_coverage"
        else:
            state = "missing_indices"
        months.append({
            "month": key, "state": state,
            "date": row["date"] if row is not None else None,
            "valid_fraction": row["valid_fraction"] if row is not None else None,
        })
        trailing = trailing + 1 if state == "usable" else 0
        longest = max(longest, trailing)

    counts = {
        "expected_months": len(months), "total_observations": len(source_rows),
        "observations": len(rows), "excluded_observations": len(source_rows) - len(rows),
        **{state: sum(item["state"] == state for item in months)
           for state in ("usable", "low_coverage", "missing_indices", "missing")},
    }
    forecast = forecast_summary(rows)
    resilience = resilience_summary(rows)
    forecast_available = bool(forecast["points"])
    resilience_available = resilience["status"] != "insufficient_data"
    ending_usable = months[-1]["state"] == "usable"
    last_usable = next((row["date"] for row in reversed(rows) if row["health_score"] is not None), None)

    reasons = []
    if counts["excluded_observations"]:
        reasons.append(f"{counts['excluded_observations']} source observations fall outside the inclusive requested dates.")
    if counts["missing"]:
        reasons.append(f"{counts['missing']} requested calendar months have no observation inside the requested dates.")
    if counts["low_coverage"]:
        reasons.append(f"{counts['low_coverage']} months have valid-pixel fraction below the current 0.4 minimum.")
    if counts["missing_indices"]:
        reasons.append(f"{counts['missing_indices']} months meet the coverage minimum but lack NDVI, NDMI, or NBR.")
    if not forecast_available:
        reasons.append(forecast["caveat"])
    if not resilience_available:
        reasons.extend(resilience["evidence"])
    if not ending_usable:
        reasons.append(
            f"The requested ending month ({months[-1]['month']}) has no usable observation. "
            "Any available historical diagnostics apply only through their recorded observation dates."
        )

    return {
        "schema_version": 1, "name": prepared.name,
        "source": prepared.source, "synthetic": prepared.synthetic,
        "input_span": {"start": source_rows[0]["date"], "end": source_rows[-1]["date"]},
        "requested_span": {"start": start_text, "end": end_text},
        "observed_span": {"start": rows[0]["date"] if rows else None,
                          "end": rows[-1]["date"] if rows else None},
        "months": months, "counts": counts,
        "longest_usable_run": longest, "trailing_usable_run": trailing,
        "baseline_forecast_available": forecast_available,
        "resilience_available": resilience_available,
        "readiness_through": {
            "baseline_forecast": rows[-1]["date"] if forecast_available else None,
            "resilience": last_usable if resilience_available else None,
        },
        "requested_end_month_usable": ending_usable,
        "baseline_forecast_ready_at_requested_end": forecast_available and ending_usable,
        "resilience_ready_at_requested_end": resilience_available and ending_usable,
        "minimum_valid_fraction": 0.4,
        "reasons": reasons,
        "caveat": (
            "Readiness checks software minimums for the current illustrative forecast and "
            "descriptive resilience diagnostics only. It does not establish predictive validity, "
            "calibration, ecological stability, or a tipping-point warning. Source observations "
            "and provenance are not independently verified by this report."
        ),
    }
