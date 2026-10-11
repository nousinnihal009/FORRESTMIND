"""Interpretable monitoring baselines, deliberately separate from untrained models.

These summaries are screening indicators. They are not validated ecological health
measurements, calibrated probabilities, causal attribution, or tipping predictions.
"""

from __future__ import annotations

from datetime import date, timedelta
import math
from typing import Any

import numpy as np


METHOD = "transparent-indicator-baseline-v1"
DISCLAIMER = (
    "Research screening only. Indicator scores and scenario sensitivities are heuristic; "
    "forecasts are uncalibrated. No causal attribution or ecological tipping-point "
    "prediction is established. Validate against field observations before decisions."
)
WEIGHTS = {"ndvi": 0.4, "ndmi": 0.35, "nbr": 0.25}
RANGES = {"ndvi": (0.2, 0.9), "ndmi": (-0.1, 0.6), "nbr": (0.0, 0.8)}


def rounded(value: float | None, digits: int = 2) -> float | None:
    return round(float(value), digits) if value is not None and math.isfinite(value) else None


def condition(score: float | None) -> str:
    if score is None:
        return "insufficient_data"
    if score >= 75:
        return "stable"
    if score >= 58:
        return "watch"
    if score >= 40:
        return "stressed"
    return "critical"


def indicator_score(observation: dict) -> float | None:
    if observation["valid_fraction"] < 0.4:
        return None
    if any(observation.get(key) is None for key in WEIGHTS):
        return None
    total = sum(
        WEIGHTS[key] * np.clip((observation[key] - low) / (high - low), 0, 1)
        for key, (low, high) in RANGES.items()
    )
    return rounded(100 * total)


def _monthly_regular(observations: list[dict]) -> bool:
    dates = [date.fromisoformat(row["date"]) for row in observations]
    return all(
        (right.year - left.year) * 12 + right.month - left.month == 1
        for left, right in zip(dates, dates[1:])
    )


def _series_trend(rows: list[dict]) -> float | None:
    if len(rows) < 6:
        return None
    dates = [date.fromisoformat(row["date"]) for row in rows]
    x = np.array([(day - dates[0]).days / 365.25 for day in dates])
    if x[-1] < 0.4:
        return None
    return float(np.polyfit(x, [row["health_score"] for row in rows], 1)[0])


def resilience_summary(observations: list[dict]) -> dict:
    usable = [row for row in observations if row["health_score"] is not None]
    result: dict[str, Any] = {
        "status": "insufficient_data", "score": None, "ar1": None,
        "variance": None, "trend": None, "evidence": [],
        "caveat": "Autocorrelation and variance are descriptive diagnostics, not validated "
        "tipping probabilities or a time to ecological transition.",
        "calibration_status": "uncalibrated", "time_to_tipping": None,
    }
    if len(usable) < 36 or not _monthly_regular(usable):
        result["evidence"] = [
            "At least 36 consecutive usable monthly observations are required for "
            "the season-adjusted descriptive diagnostics."
        ]
        return result
    values = np.array([row["health_score"] for row in usable], dtype=float)
    x = np.arange(len(values), dtype=float)
    detrended = values - np.polyval(np.polyfit(x, values, 1), x)
    months = np.array([date.fromisoformat(row["date"]).month for row in usable])
    residual = detrended.copy()
    for month in range(1, 13):
        residual[months == month] -= np.mean(detrended[months == month])
    previous, latest = residual[-36:-18], residual[-18:]

    def ar1(values: np.ndarray) -> float | None:
        if np.std(values[:-1]) < 1e-8 or np.std(values[1:]) < 1e-8:
            return None
        return float(np.corrcoef(values[:-1], values[1:])[0, 1])

    current_ar, previous_ar = ar1(latest), ar1(previous)
    current_var, previous_var = float(np.var(latest)), float(np.var(previous))
    rising = current_ar is not None and previous_ar is not None and (
        current_ar > previous_ar + 0.1 and current_var > previous_var * 1.25
    )
    result.update({
        "status": "review_signal" if rising else "no_joint_increase",
        "ar1": rounded(current_ar, 3), "variance": rounded(current_var, 3),
        "trend": "Joint increase" if rising else "No joint increase detected",
        "evidence": [
            "Linear trend and calendar-month climatology removed before analysis.",
            f"Compared the latest and preceding 18-month residual windows ({len(usable)} usable months).",
            "An increase in both indicators warrants review; its absence does not establish stability.",
        ],
    })
    return result


def forecast_summary(observations: list[dict]) -> dict:
    usable = [row for row in observations if row["health_score"] is not None]
    historical = [{"date": row["date"], "score": row["health_score"]} for row in observations]
    result = {
        "method": "Damped historical trend with empirical residual envelope",
        "calibrated": False, "points": [], "historical": historical,
        "caveat": "Illustrative indicator extrapolation. Bounds are heuristic residual envelopes, "
        "not calibrated confidence intervals. Disturbances and interventions are not predicted.",
    }
    if (len(usable) < 12 or observations[-1]["health_score"] is None
            or not _monthly_regular(usable[-12:])):
        result["caveat"] = (
            "A usable latest observation and at least 12 consecutive usable monthly observations "
            "are required. No forecast has been produced."
        )
        return result
    recent = usable[-12:]
    slope = _series_trend(recent)
    if slope is None:
        return result
    recent_days = np.array([
        (date.fromisoformat(row["date"]) - date.fromisoformat(recent[0]["date"])).days / 365.25
        for row in recent
    ])
    recent_scores = np.array([row["health_score"] for row in recent])
    residual = recent_scores - np.polyval(np.polyfit(recent_days, recent_scores, 1), recent_days)
    spread = max(float(np.std(residual)), 2.5)
    last_date = date.fromisoformat(observations[-1]["date"])
    current = float(observations[-1]["health_score"])
    for horizon in (30, 180, 365):
        years = horizon / 365.25
        score = float(np.clip(current + slope * (1 - math.exp(-years)) * 0.65, 0, 100))
        envelope = 1.96 * spread * math.sqrt(1 + horizon / 90)
        result["points"].append({
            "horizon_days": horizon, "date": (last_date + timedelta(days=horizon)).isoformat(),
            "score": rounded(score), "lower": rounded(max(0, score - envelope)),
            "upper": rounded(min(100, score + envelope)),
        })
    return result


def driver_summary(observations: list[dict]) -> list[dict]:
    """Rank indicator evidence without assigning causal contribution shares."""
    recent = observations[-3:]
    previous = observations[-15:-3]

    def average(rows: list[dict], key: str) -> float | None:
        values = [row[key] for row in rows if row.get(key) is not None and row["valid_fraction"] >= 0.4]
        return float(np.mean(values)) if values else None

    precipitation = average(recent, "precipitation_anomaly")
    temperature = average(recent, "temperature_anomaly")
    fire = average(recent, "fire_anomaly")
    sar = average(recent, "sar_change")
    cover = average(recent, "forest_cover")
    past_cover = average(previous, "forest_cover")
    nbr = average(recent, "nbr")
    past_nbr = average(previous, "nbr")
    specs = [
        ("moisture", "Moisture stress", None if precipitation is None else max(0, -precipitation) / 2.5,
         "Precipitation anomaly unavailable." if precipitation is None else f"Recent precipitation anomaly: {precipitation:+.2f} standard deviations."),
        ("heat", "Heat stress", None if temperature is None else max(0, temperature) / 3,
         "Temperature anomaly unavailable." if temperature is None else f"Recent temperature anomaly: {temperature:+.2f} °C."),
        ("fire", "Fire disturbance", None if fire is None else max(0, fire) / 3,
         "Independent fire observations unavailable; NBR alone does not identify a fire." if fire is None else f"Recent supplied fire anomaly: {fire:+.2f} standard deviations; not independently verified."),
        ("cover", "Canopy loss", None if cover is None or past_cover is None else max(0, past_cover - cover) / 0.1,
         "Comparable forest-cover observations unavailable." if cover is None or past_cover is None else f"Recent cover differs from the preceding baseline by {(cover - past_cover) * 100:+.2f} percentage points."),
        ("structure", "Structural disturbance", None if sar is None else abs(sar) / 4,
         "SAR observations unavailable." if sar is None else f"Recent supplied SAR backscatter change: {sar:+.2f} dB; moisture and geometry can confound it."),
        ("spectral", "Spectral disturbance", None if nbr is None or past_nbr is None else max(0, past_nbr - nbr) / 0.25,
         "Comparable NBR observations unavailable." if nbr is None or past_nbr is None else f"NBR differs from its preceding baseline by {nbr - past_nbr:+.3f}; this signal is not specific to fire or logging."),
    ]
    output = [{
        "id": key, "name": name,
        "evidence_score": rounded(min(1, value) * 100) if value is not None else None,
        "direction": "candidate_pressure" if value is not None and value > 0.1 else "unresolved",
        "evidence": evidence, "confidence": None,
        "interpretation": "Heuristic evidence strength; not causal contribution or probability.",
    } for key, name, value, evidence in specs]
    return sorted(output, key=lambda row: row["evidence_score"] if row["evidence_score"] is not None else -1, reverse=True)


def analyze_region(region: dict, observations: list[dict]) -> dict:
    rows = [{**row, "health_score": indicator_score(row)} for row in observations]
    usable = [row for row in rows if row["health_score"] is not None]
    current = rows[-1]["health_score"]
    status = condition(current)
    baseline = [row["health_score"] for row in rows[-15:-3] if row["health_score"] is not None]
    recent = [row["health_score"] for row in rows[-3:] if row["health_score"] is not None]
    change = rounded(float(np.mean(recent) - np.mean(baseline))) if baseline and recent else None
    coverage = float(np.mean([row["valid_fraction"] for row in rows[-3:]]))
    alerts = []
    last_date = rows[-1]["date"]
    if status in {"stressed", "critical"}:
        alerts.append({
            "id": f"{region['id']}-condition", "severity": "high" if status == "critical" else "medium",
            "title": "Elevated indicator stress", "message": f"The latest heuristic condition score is {current}/100. "
            "Review source imagery and field observations before interpreting ecological change.",
            "date": last_date, "status": "active",
        })
    if change is not None and change <= -5:
        alerts.append({
            "id": f"{region['id']}-decline", "severity": "medium", "title": "Recent condition decline",
            "message": f"Recent three-observation average is {abs(change):.1f} points below the preceding "
            "baseline. Seasonality can contribute; this comparison is not a significance test.",
            "date": last_date, "status": "active",
        })
    if current is None or coverage < 0.65:
        alerts.append({
            "id": f"{region['id']}-coverage", "severity": "info", "title": "Observation coverage gap",
            "message": "Clouds, missing bands, or low valid-pixel fraction limit the latest assessment.",
            "date": last_date, "status": "active",
        })
    first_date, end_date = date.fromisoformat(rows[0]["date"]), date.fromisoformat(last_date)
    expected_months = (end_date.year - first_date.year) * 12 + end_date.month - first_date.month + 1
    result = {
        **region, "status": status, "health_score": current, "health_change": change,
        "confidence": None, "quality": "Insufficient observations" if current is None else (
            "High coverage" if coverage >= 0.8 else "Partial coverage"),
        "latest_date": last_date, "alert_count": len(alerts), "observations": rows,
        "health": {
            "score": current, "status": status, "change": change,
            "trend_per_year": rounded(_series_trend(usable[-12:])), "coverage": rounded(coverage, 3),
            "components": [{"name": key.upper(), "value": rows[-1].get(key), "unit": "index"} for key in WEIGHTS],
            "method": METHOD, "weights": WEIGHTS, "reference_ranges": RANGES,
            "interpretation": "0–100 heuristic spectral condition index; thresholds and mappings are not field calibrated.",
        },
        "data_quality": {"observation_count": len(rows), "usable_observations": len(usable),
                         "expected_months": expected_months, "missing_months": max(0, expected_months - len(rows)),
                         "minimum_valid_fraction": 0.4, "calibrated_confidence": None},
        "forecast": forecast_summary(rows), "drivers": driver_summary(rows),
        "resilience": resilience_summary(rows), "alerts": alerts,
        "disclaimer": DISCLAIMER,
    }
    return result


def sensitivity_scenario(analysis: dict, parameters: dict) -> dict:
    """Apply explicitly chosen sensitivity coefficients, never learned effects."""
    coefficients = {"fire_reduction": 0.04, "pressure_reduction": 0.05,
                    "restoration": 0.03, "climate_stress": -0.08}
    points = analysis["forecast"]["points"]
    annual_shift = sum(parameters[key] * value for key, value in coefficients.items())
    scenario = []
    for point in points:
        shift = annual_shift * point["horizon_days"] / 365
        scenario.append({**point, **{
            field: rounded(float(np.clip(point[field] + shift, 0, 100)))
            for field in ("score", "lower", "upper")
        }})
    return {
        "region_id": analysis["id"], "parameters": parameters,
        "baseline": points, "scenario": scenario,
        "delta": rounded(scenario[-1]["score"] - points[-1]["score"]) if points else None,
        "method": "User-controlled linear sensitivity illustration", "calibrated": False,
        "assumptions": [
            "Slider values are percentage-point scenario inputs (0–100), not planned hectares or measured treatment.",
            "Annual score-point sensitivity per input point: fire +0.04, pressure +0.05, restoration +0.03, climate −0.08.",
            "Effects accumulate linearly over the requested horizon; interactions, costs and feasibility are not modeled.",
        ],
        "caveat": "Illustrative sensitivity only. Coefficients are assumptions, not estimated intervention "
        "effects, causal predictions, or management recommendations.",
    }
