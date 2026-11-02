"""Explicit five-year indicator sensitivity, separate from ecological forecasts."""
from datetime import date, timedelta
import math

COEFFICIENTS = {"fire_reduction": .04, "pressure_reduction": .05,
                "restoration": .03, "climate_stress": -.08}
DEFAULTS = [("none", "No additional intervention", {}),
            ("conservation", "Conservation assumption", {"pressure_reduction": 20., "fire_reduction": 10.}),
            ("restoration", "Restoration assumption", {"restoration": 30.}),
            ("fire_prevention", "Fire-prevention assumption", {"fire_reduction": 40.}),
            ("climate", "Greater climate-stress assumption", {"climate_stress": 25.})]


def _parameters(values):
    if not isinstance(values, dict) or set(values) - set(COEFFICIENTS):
        raise ValueError("Unknown scenario input; use the four documented assumptions")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100 for value in values.values()):
        raise ValueError("Scenario inputs must be finite numbers from 0 to 100")
    return {field: float(values.get(field, 0.)) for field in COEFFICIENTS}


def validate_parameter_ranges(parameters, ranges):
    values = _parameters(parameters)
    if not isinstance(ranges, dict) or not ranges or set(ranges) - set(COEFFICIENTS):
        raise ValueError("Assumption ranges require one or more documented scenario inputs")
    result = {}
    for field, central in values.items():
        bounds = ranges.get(field, {"lower": central, "upper": central})
        if not isinstance(bounds, dict) or set(bounds) != {"lower", "upper"}:
            raise ValueError("Each assumption range requires lower and upper values")
        low, high = bounds["lower"], bounds["upper"]
        if (any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100 for v in (low, high))
                or not low <= central <= high):
            raise ValueError("Every range must satisfy 0 <= lower <= selected assumption <= upper <= 100")
        result[field] = {"lower": float(low), "upper": float(high)}
    return result


def scenario_projection(analysis: dict, parameters: dict | None = None, *, parameter_ranges: dict | None = None) -> dict:
    normalized = _parameters(parameters) if parameters is not None else None
    if parameter_ranges is not None and normalized is None:
        raise ValueError("Assumption ranges require an explicit central scenario")
    ranges = validate_parameter_ranges(normalized, parameter_ranges) if parameter_ranges is not None else None
    forecast = analysis["forecast"]
    points = forecast["points"]
    current = analysis.get("health_score", analysis.get("condition", {}).get("score"))
    result = {"available": False, "method": "five-year-damped-indicator-sensitivity-v1", "causal": False,
              "scientifically_validated": False, "baseline": [], "comparisons": [],
              "assumptions": ["The uncalibrated one-year indicator trajectory supplies a damped trend, extended to five years.",
                              "Scenario inputs are assumed 0–100 levels, not measured treatment, protected hectares or predicted intervention effects.",
                              "Annual score-point sensitivity per input point: fire +0.04, pressure +0.05, restoration +0.03, climate −0.08.",
                              "Input shifts accumulate linearly; the score is constrained to 0–100. Costs, policy feasibility and interactions are not modeled."],
              "limitations": ["This is hypothetical arithmetic sensitivity, not a trained five-year ecological forecast or event probability.",
                              "No calibrated five-year uncertainty interval or causal intervention effect is available.",
                              "Identical starting data and cutoff are used for every comparison; future disturbances are unknown."]}
    if current is None or not points:
        return {**result, "reason": forecast["caveat"]}
    annual = next((p for p in points if p["horizon_days"] == 365), None)
    if not annual:
        return {**result, "reason": "A one-year indicator baseline is required to anchor sensitivity comparisons."}
    start = date.fromisoformat(analysis.get("latest_date", analysis.get("manifest", {}).get("observed_through")))
    one_year_shift = annual["score"] - current
    baseline = [{"horizon_days": days, "date": (start + timedelta(days=days)).isoformat(),
                 "score": round(max(0, min(100, current + one_year_shift * (1 - math.exp(-days / 365.25)) / (1 - math.exp(-365 / 365.25)))), 2)}
                for days in (0, 365, 730, 1095, 1460, 1825)]
    choices = DEFAULTS if normalized is None else [("custom", "User-defined assumptions", normalized)]
    comparisons = []
    for key, name, values in choices:
        parameters = _parameters(values)
        shift = sum(parameters[field] * coefficient for field, coefficient in COEFFICIENTS.items())
        trajectory = [{**point, "score": round(max(0, min(100, point["score"] + shift * point["horizon_days"] / 365)), 2)} for point in baseline]
        comparisons.append({"id": key, "name": name, "parameters": parameters, "points": trajectory,
                            "five_year_difference": round(trajectory[-1]["score"] - baseline[-1]["score"], 2)})
        if ranges is not None:
            minimum = sum(coefficient * ranges[field]["lower" if coefficient >= 0 else "upper"] for field, coefficient in COEFFICIENTS.items())
            maximum = sum(coefficient * ranges[field]["upper" if coefficient >= 0 else "lower"] for field, coefficient in COEFFICIENTS.items())
            for point, base in zip(trajectory, baseline):
                point["sensitivity_lower"] = round(max(0., min(100., base["score"] + minimum * base["horizon_days"] / 365)), 2)
                point["sensitivity_upper"] = round(max(0., min(100., base["score"] + maximum * base["horizon_days"] / 365)), 2)
            comparisons[-1]["parameter_ranges"] = ranges
            comparisons[-1]["uncertainty"] = {
                "kind": "explicit_assumption_range", "calibrated": False, "confidence_level": None,
                "definition": "Exact arithmetic minimum and maximum across the supplied input ranges with fixed coefficients. This is not a probability or confidence interval; uncertainty in coefficients, baseline and ecological mechanisms is not included."}
    return {**result, "available": True, "reason": "Hypothetical five-year sensitivity is enabled by a usable one-year indicator baseline.",
            "baseline": baseline, "comparisons": comparisons, "coefficients": COEFFICIENTS}
