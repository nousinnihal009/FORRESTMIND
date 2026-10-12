"""Rolling-origin evaluation of the public, uncalibrated monitoring baselines.

Targets are future observations of the same fixed spectral indicator, not field
labels of ecosystem health. A synthetic run checks software behavior only.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json
import math
from typing import Any

from forestmind.monitoring.analysis import (
    METHOD, RANGES, WEIGHTS, forecast_summary, indicator_score,
)


HORIZON_DAYS = {1: 30, 6: 180, 12: 365}
METHODS = ("persistence", "seasonal_naive", "damped_trend")


def _month_number(value: str) -> int:
    parsed = date.fromisoformat(value)
    return parsed.year * 12 + parsed.month - 1


def _month_label(value: int) -> str:
    year, month = divmod(value, 12)
    return f"{year:04d}-{month + 1:02d}"


def _metrics(records: list[dict]) -> dict:
    scored = [row for row in records if row["status"] == "scored"]
    errors = [row["prediction"] - row["target"] for row in scored]
    bounded = [row for row in scored if row["lower"] is not None and row["upper"] is not None]
    n = len(scored)
    return {
        "n": n,
        "mae": sum(abs(error) for error in errors) / n if n else None,
        "rmse": math.sqrt(sum(error * error for error in errors) / n) if n else None,
        "eligible_predictions": len(records),
        "skipped_missing_target": sum(row["status"] == "missing_target" for row in records),
        "skipped_unusable_target": sum(row["status"] == "unusable_target" for row in records),
        "outside_observation_period": sum(row["status"] == "outside_observation_period" for row in records),
        "n_bounds": len(bounded),
        "coverage": (sum(row["lower"] <= row["target"] <= row["upper"] for row in bounded)
                     / len(bounded)) if bounded else None,
        "mean_interval_width": (sum(row["upper"] - row["lower"] for row in bounded)
                                / len(bounded)) if bounded else None,
        "coverage_interpretation": "Empirical coverage of uncalibrated residual envelopes; no nominal coverage guarantee.",
        "score_unit": "heuristic indicator points (0–100)",
    }


def backtest_prepared(
    payload: dict,
    *,
    horizons_months: tuple[int, ...] | list[int] = (1, 6, 12),
) -> dict[str, Any]:
    """Evaluate a prepared import JSON using only data available at each cutoff.

    The same strict import contract as the application is used, including at
    most one dated observation per calendar month. Install ``forestmind[web]``
    for its Pydantic validator. All methods share the existing forecast's
    eligibility gate: a usable latest observation and 12 consecutive usable
    monthly observations. No fitting, normalizing or imputation uses targets.

    Forecast-summary's nominal 30/180/365-day predictions are aligned to the
    observed month at calendar offsets 1/6/12. Nominal forecast dates and actual
    target dates are both retained; missing months are never shifted or filled.
    """
    try:
        from forestmind.api.schemas import ImportRegion
    except ImportError as exc:
        raise ImportError("Prepared backtesting requires the import validator: install forestmind[web]") from exc

    horizons = list(horizons_months)
    if (not horizons or any(type(month) is not int or month not in HORIZON_DAYS for month in horizons)
            or horizons != sorted(set(horizons))):
        raise ValueError("horizons_months must be a nonempty increasing subset of [1, 6, 12]")
    prepared = ImportRegion.model_validate(payload).model_dump()
    canonical = json.dumps(prepared, sort_keys=True, separators=(",", ":"), allow_nan=False)
    rows = [{**row, "health_score": indicator_score(row)} for row in prepared["observations"]]
    by_month = {_month_number(row["date"]): row for row in rows}
    last_month = _month_number(rows[-1]["date"])
    records: list[dict] = []
    cutoffs: list[dict] = []

    for index, origin in enumerate(rows):
        # This slice is the complete information boundary. Even though future
        # scores are computed by a fixed formula, no future row enters a model.
        history = rows[:index + 1]
        forecast = forecast_summary(history)
        points = {point["horizon_days"]: point for point in forecast["points"]}
        eligible = bool(points)
        recent = [row for row in history if row["health_score"] is not None][-12:]
        cutoffs.append({
            "cutoff_date": origin["date"], "eligible": eligible,
            "available_observations": len(history),
            "training_start_date": recent[0]["date"] if recent else None,
            "training_end_date": origin["date"] if eligible else None,
            "training_observations": len(recent),
            "reason": None if eligible else forecast["caveat"],
        })
        if not eligible:
            continue
        origin_month = _month_number(origin["date"])
        for horizon in horizons:
            target_month = origin_month + horizon
            target_row = by_month.get(target_month)
            target = target_row["health_score"] if target_row else None
            if target_month > last_month:
                status = "outside_observation_period"
            elif target_row is None:
                status = "missing_target"
            elif target is None:
                status = "unusable_target"
            else:
                status = "scored"
            point = points[HORIZON_DAYS[horizon]]
            seasonal_reference = by_month[target_month - 12]
            # This assertion guards future changes to supported horizons or
            # eligibility; seasonal naive must never reach forward of origin.
            if seasonal_reference["date"] > origin["date"]:
                raise RuntimeError("Seasonal reference is after the prediction cutoff")
            predictions = {
                "persistence": (origin["health_score"], None, None, origin["date"]),
                "seasonal_naive": (seasonal_reference["health_score"], None, None,
                                   seasonal_reference["date"]),
                "damped_trend": (point["score"], point["lower"], point["upper"], None),
            }
            for method, (prediction, lower, upper, reference_date) in predictions.items():
                records.append({
                    "method": method, "cutoff_date": origin["date"],
                    "training_start_date": recent[0]["date"],
                    "training_end_date": origin["date"], "training_observations": 12,
                    "reference_date": reference_date,
                    "horizon_months": horizon,
                    "nominal_forecast_days": HORIZON_DAYS[horizon],
                    "nominal_forecast_date": point["date"],
                    "target_month": _month_label(target_month),
                    "target_date": target_row["date"] if target_row else None,
                    "actual_horizon_days": (
                        (date.fromisoformat(target_row["date"]) - date.fromisoformat(origin["date"])).days
                        if target_row else None
                    ),
                    "prediction": prediction, "target": target,
                    "lower": lower, "upper": upper, "status": status,
                })

    synthetic = prepared["synthetic"]
    return {
        "schema_version": 1,
        "evaluation": "rolling_origin_monitoring_baselines",
        "region": {"name": prepared["name"], "country": prepared["country"]},
        "provenance": {
            "source": prepared["source"], "notes": prepared["notes"],
            "synthetic": synthetic, "verified": False,
            "input_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "input_hash_basis": "Canonical schema-validated prepared JSON, including provenance and geometry",
            "first_observation": rows[0]["date"], "last_observation": rows[-1]["date"],
            "observation_count": len(rows),
            "reference": {
                "target": "indicator_score from the held-out dated spectral observation",
                "method": METHOD, "weights": deepcopy(WEIGHTS), "ranges": deepcopy(RANGES),
                "independent_field_validation": False,
            },
        },
        "synthetic_warning": (
            "SYNTHETIC DATA: this run checks implementation behavior and does not establish real-world forecast performance."
            if synthetic else None
        ),
        "protocol": {
            "horizons_months": horizons, "minimum_consecutive_usable_months": 12,
            "origin_schedule": "Every observed month meeting the existing forecast_summary eligibility gate",
            "information_boundary": "All predictions use observations dated at or before their cutoff only",
            "target_alignment": "Exact future calendar month; use its actual dated observation without imputation or nearest-month substitution",
            "forecast_alignment": "Existing 30/180/365-day damped forecasts are evaluated at 1/6/12 calendar-month observations; actual and nominal dates are retained",
            "aggregation": "Observation-weighted pooling across cutoff/horizon pairs; not an average of horizon metrics",
            "bounds_calibrated": False,
        },
        "methods": {
            method: {
                "aggregate": _metrics([row for row in records if row["method"] == method]),
                "by_horizon": {
                    str(horizon): _metrics([row for row in records
                                            if row["method"] == method and row["horizon_months"] == horizon])
                    for horizon in horizons
                },
            } for method in METHODS
        },
        "cutoffs": cutoffs, "records": records,
        "limitations": [
            "Targets are a fixed, uncalibrated spectral indicator; this is not validation against ecological health, forest loss, or field measurements.",
            "Adjacent origins and overlapping forecast horizons are dependent; no independent-sample confidence intervals are reported.",
            "Prepared inputs may already contain processing leakage; cloud masking, compositing and any upstream normalization must be audited independently.",
            "No tuning is performed here; any later choice or tuning based on these results needs a separate untouched test period or region.",
            "Persistence and seasonal naive supply no interval bounds; their coverage is unavailable rather than zero.",
            "Source provenance is user supplied and not independently verified.",
        ],
    }
