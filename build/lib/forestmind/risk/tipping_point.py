"""Transparent condition screening with explicit gaps in ecological evidence.

The historical API name is retained for compatibility. This module cannot infer
irreversible transitions, spatial fragmentation, drought causation, calibrated
probabilities or tipping dates from a one-dimensional health trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from forestmind.risk.critical_slowing import (
    CSDResult, _health_array, _time_axis, analyze_critical_slowing_down,
)


@dataclass
class TippingPointAssessment:
    """Uncalibrated screening index, not a probability of ecosystem collapse."""

    risk_score: float | None
    risk_category: str
    csd_score: float | None
    resilience_score: float | None
    fragmentation_score: float | None
    momentum_score: float | None
    drought_sensitivity_score: float | None
    csd_result: CSDResult | None = None
    time_to_tipping: float | None = None
    confidence: float | None = None
    contributing_factors: list[str] = field(default_factory=list)
    calibration_status: str = "uncalibrated_heuristic"
    limitations: list[str] = field(default_factory=list)
    data_quality: dict = field(default_factory=dict)
    component_weights: dict[str, float] = field(default_factory=dict)


def _compute_resilience(health_series: np.ndarray) -> float | None:
    """Immediate rebound proxy after observed dips; not ecological resilience."""
    if len(health_series) < 4:
        return None
    diffs = np.diff(health_series)
    rebounds = []
    for before, after in zip(diffs[:-1], diffs[1:]):
        if np.isfinite(before) and np.isfinite(after) and before < -0.02:
            rebounds.append(float(np.clip(after / abs(before), 0, 1)))
    return float(np.mean(rebounds)) if rebounds else None


def _compute_momentum(health_series: np.ndarray, time_axis: np.ndarray | None = None) -> float | None:
    """Magnitude of fitted decline across the observed span, with gaps retained."""
    valid = np.isfinite(health_series)
    if valid.sum() < 4:
        return None
    axis = np.arange(len(health_series), dtype=float) if time_axis is None else time_axis
    x = axis[valid]
    x = (x - x[0]) / (x[-1] - x[0])
    slope = np.polyfit(x, health_series[valid], 1)[0]
    return float(np.clip(-slope, 0, 1))


def _compute_drought_sensitivity(
    health_series: np.ndarray, drought_indices: list[int] | None = None,
) -> float | None:
    """Observed drop coinciding with supplied drought events, without attribution."""
    if drought_indices is None:
        return None
    drops = []
    for index in drought_indices:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(health_series):
            raise ValueError("drought_indices must contain valid integer observation indices")
        if index == 0:
            continue
        before = health_series[max(0, index - 2):index]
        before = before[np.isfinite(before)]
        during = health_series[index]
        if len(before) and np.isfinite(during) and before.mean() > 0.01:
            drops.append(float(np.clip((before.mean() - during) / before.mean(), 0, 1)))
    return float(np.mean(drops)) if drops else None


def assess_tipping_point(
    health_series: np.ndarray,
    config: dict | None = None,
    *,
    timestamps: np.ndarray | list | None = None,
    season_period: int | None = None,
) -> TippingPointAssessment:
    """Screen health observations while leaving unsupported evidence unavailable.

    Optional config ``fragmentation_score`` must come from an independently
    computed spatial metric; ``drought_indices`` must identify observed events.
    Neither is reconstructed from generic health variability. Legacy variance
    and autocorrelation weights map to rebound and decline proxies respectively.
    """
    config = config or {}
    values = _health_array(health_series)
    axis, regular = _time_axis(timestamps, len(values))
    valid = np.isfinite(values)
    csd_config = config.get("csd", {})
    period = season_period if season_period is not None else csd_config.get("season_period")
    csd = analyze_critical_slowing_down(
        values, window_size=csd_config.get("window_size", 6),
        ar1_threshold=csd_config.get("ar1_threshold", 0.8),
        timestamps=timestamps, season_period=period,
    )
    limitations = [
        "Risk categories are screening priorities, not calibrated probabilities or evidence of irreversible change.",
        "No validated model is available for time to tipping or assessment confidence; both remain unavailable.",
        "Health scores may themselves be model estimates rather than field measurements.",
    ]
    limitations.extend(csd.limitations)
    resilience = _compute_resilience(values)
    if resilience is not None:
        limitations.append("resilience_score is an immediate rebound proxy, not validated ecological resilience.")
    else:
        limitations.append("No usable perturbation/recovery observations; resilience is unavailable.")
    momentum = _compute_momentum(values, axis)
    if period is not None:
        limitations.append("Decline and rebound proxies use the original series and may retain seasonal effects.")
    fragmentation = config.get("fragmentation_score")
    if fragmentation is not None:
        fragmentation = float(fragmentation)
        if not np.isfinite(fragmentation) or not 0 <= fragmentation <= 1:
            raise ValueError("fragmentation_score must be finite and lie in [0, 1]")
        limitations.append("Fragmentation uses a caller-supplied spatial score whose validation is external.")
    else:
        limitations.append("Spatial fragmentation requires a spatial forest mask and is unavailable.")
    drought = _compute_drought_sensitivity(values, config.get("drought_indices"))
    if drought is None:
        limitations.append("Drought response requires observed drought-event timing and is unavailable.")
    else:
        limitations.append("Drought-coincident decline does not establish drought causation.")
    components = {
        "condition": 1 - float(values[valid][-1]) if valid.any() else None,
        "csd": csd.csd_score if csd.status == "available" else None,
        "resilience": 1 - resilience if resilience is not None else None,
        "momentum": momentum,
        "fragmentation": fragmentation,
        "drought_sensitivity": drought,
    }
    weights = {"condition": 0.35, "csd": 0.15, "resilience": 0.1,
               "momentum": 0.3, "fragmentation": 0.05, "drought_sensitivity": 0.05}
    aliases = {"variance": "resilience", "autocorrelation": "momentum"}
    for name, weight in config.get("risk_weights", {}).items():
        name = aliases.get(name, name)
        if name not in weights:
            raise ValueError(f"Unknown risk weight: {name}")
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("risk weights must be finite and non-negative")
        weights[name] = weight
    available = {name: weights[name] for name, value in components.items()
                 if value is not None and weights[name] > 0}
    weight_sum = sum(available.values())
    normalized = {name: weight / weight_sum for name, weight in available.items()}
    if valid.sum() < 4 or not weight_sum:
        score, category = None, "Insufficient data"
        limitations.append("At least four finite health observations and one nonzero component weight are required.")
    else:
        score = float(sum(components[name] * weight for name, weight in normalized.items()))
        category = next((label for threshold, label in
                         [(0.2, "Low"), (0.4, "Moderate"), (0.6, "Elevated"), (0.8, "High")]
                         if score < threshold), "Critical")
    factors = []
    if csd.is_warning:
        factors.append("Rising persistence and variance warrant investigation; no tipping transition is established")
    if momentum is not None and momentum > 0.1:
        factors.append("Health observations show a declining fitted trend")
    if resilience is not None and resilience < 0.3:
        factors.append("Limited immediate rebound following observed health dips")
    if fragmentation is not None and fragmentation > 0.5:
        factors.append("Caller-supplied spatial fragmentation indicator is elevated")
    if drought is not None and drought > 0.3:
        factors.append("Health declines coincide with supplied drought-event observations")
    return TippingPointAssessment(
        risk_score=score, risk_category=category,
        csd_score=components["csd"], resilience_score=resilience,
        fragmentation_score=fragmentation, momentum_score=momentum,
        drought_sensitivity_score=drought, csd_result=csd,
        contributing_factors=factors, limitations=limitations,
        data_quality={"observations": len(values), "valid_observations": int(valid.sum()),
                      "missing_observations": int((~valid).sum()),
                      "regular_sampling": regular if timestamps is not None else None,
                      "csd_status": csd.status},
        component_weights=normalized,
    )
