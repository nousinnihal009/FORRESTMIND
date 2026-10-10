"""
Tipping-Point Estimator for ForestMind.

Combines multiple ecological indicators into a unified tipping-point
risk score. Goes beyond Critical Slowing Down by incorporating:
  - Vegetation resilience metrics
  - Spatial fragmentation analysis
  - Drought sensitivity assessment
  - Degradation momentum

The output is a continuous risk score [0, 1] with interpretive categories:
  0.0-0.2: Low risk — ecosystem appears resilient
  0.2-0.4: Moderate risk — some warning signs
  0.4-0.6: Elevated risk — multiple indicators active
  0.6-0.8: High risk — approaching potential tipping point
  0.8-1.0: Critical risk — imminent or ongoing regime shift
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from forestmind.risk.critical_slowing import CSDResult, analyze_critical_slowing_down


@dataclass
class TippingPointAssessment:
    """Comprehensive tipping-point risk assessment."""
    # Risk score
    risk_score: float                    # Combined risk [0, 1]
    risk_category: str                   # Human-readable category
    
    # Component scores
    csd_score: float                     # Critical slowing down [0, 1]
    resilience_score: float              # Vegetation resilience [0, 1] (1 = resilient)
    fragmentation_score: float           # Spatial fragmentation [0, 1] (1 = fragmented)
    momentum_score: float                # Degradation momentum [0, 1] (1 = accelerating)
    drought_sensitivity_score: float     # Drought sensitivity [0, 1] (1 = sensitive)
    
    # Detailed indicators
    csd_result: CSDResult | None = None
    time_to_tipping: float | None = None  # Estimated days until critical threshold
    confidence: float = 0.0               # Confidence in the assessment [0, 1]
    
    # Contributing factors
    contributing_factors: list[str] = field(default_factory=list)


def _compute_resilience(health_series: np.ndarray) -> float:
    """Estimate vegetation resilience from health trajectory.
    
    Resilience is measured by how quickly the system recovers from
    negative perturbations. Higher values = more resilient.
    """
    if len(health_series) < 4:
        return 0.5

    # Find recovery events: dips followed by partial recovery
    diffs = np.diff(health_series)
    recoveries = []

    for i in range(len(diffs) - 1):
        if diffs[i] < -0.02:  # significant dip
            if diffs[i + 1] > 0:  # followed by recovery
                recovery_ratio = min(1.0, abs(diffs[i + 1]) / abs(diffs[i]))
                recoveries.append(recovery_ratio)

    if not recoveries:
        # No perturbation events observed — use stability as proxy
        stability = 1.0 - np.std(health_series) * 5  # low variance = stable
        return float(np.clip(stability, 0.0, 1.0))

    return float(np.clip(np.mean(recoveries), 0.0, 1.0))


def _compute_fragmentation(health_series: np.ndarray) -> float:
    """Estimate spatial fragmentation from temporal health indicators.
    
    In a full system, this would use spatial data. Here we use temporal
    variability as a proxy — high variability suggests patchy, fragmented
    ecosystem state.
    """
    if len(health_series) < 3:
        return 0.0

    # Coefficient of variation as fragmentation proxy
    mean_h = np.mean(health_series)
    if mean_h < 0.01:
        return 1.0  # fully degraded

    cv = np.std(health_series) / mean_h
    # High CV → patchy → fragmented
    fragmentation = float(np.clip(cv * 3, 0.0, 1.0))
    return fragmentation


def _compute_momentum(health_series: np.ndarray) -> float:
    """Compute degradation momentum — is degradation accelerating?
    
    Fits a quadratic to the health trajectory. Negative second derivative
    means the decline is accelerating (concave down with negative slope).
    """
    n = len(health_series)
    if n < 4:
        return 0.0

    x = np.arange(n, dtype=np.float64)
    coeffs = np.polyfit(x, health_series, deg=2)
    # coeffs[0] = quadratic term, coeffs[1] = linear term

    # If quadratic is negative AND linear is negative → accelerating decline
    if coeffs[0] < 0 and coeffs[1] < 0:
        # Normalize acceleration to [0, 1]
        acceleration = abs(coeffs[0]) * n * n  # scale by sequence length
        return float(np.clip(acceleration * 10, 0.0, 1.0))
    elif coeffs[1] < 0:
        # Linear decline but not accelerating
        decline_rate = abs(coeffs[1]) * n
        return float(np.clip(decline_rate * 2, 0.0, 0.5))
    else:
        return 0.0  # improving or stable


def _compute_drought_sensitivity(
    health_series: np.ndarray,
    drought_indices: list[int] | None = None,
) -> float:
    """Estimate how sensitive the ecosystem is to drought events.
    
    If drought indices are provided, measures the health impact during
    those periods. Otherwise estimates from general variability patterns.
    """
    if len(health_series) < 4:
        return 0.0

    if drought_indices and len(drought_indices) > 0:
        # Measure health drop during drought events
        sensitivities = []
        for idx in drought_indices:
            if 0 < idx < len(health_series):
                pre_drought = health_series[max(0, idx - 2): idx].mean()
                during_drought = health_series[idx]
                if pre_drought > 0.01:
                    sensitivity = max(0, (pre_drought - during_drought) / pre_drought)
                    sensitivities.append(sensitivity)
        if sensitivities:
            return float(np.clip(np.mean(sensitivities) * 3, 0.0, 1.0))

    # Proxy: sensitivity from negative excursions
    diffs = np.diff(health_series)
    negative_diffs = diffs[diffs < 0]
    if len(negative_diffs) == 0:
        return 0.0

    avg_drop = abs(np.mean(negative_diffs))
    max_drop = abs(np.min(negative_diffs))

    sensitivity = avg_drop * 5 + max_drop * 2
    return float(np.clip(sensitivity, 0.0, 1.0))


def _estimate_time_to_tipping(
    health_series: np.ndarray,
    tipping_threshold: float = 0.3,
    days_per_step: float = 30.0,
) -> float | None:
    """Estimate time (in days) until health drops below tipping threshold.
    
    Uses quadratic extrapolation from the observed trajectory.
    Returns None if the ecosystem is above threshold and stable.
    """
    current_health = health_series[-1]
    if current_health <= tipping_threshold:
        return 0.0  # already below threshold

    n = len(health_series)
    if n < 3:
        return None

    x = np.arange(n, dtype=np.float64)
    coeffs = np.polyfit(x, health_series, deg=2)

    # Find when polynomial crosses the threshold
    # coeffs[0]*t^2 + coeffs[1]*t + coeffs[2] = threshold
    a, b, c = coeffs[0], coeffs[1], coeffs[2] - tipping_threshold
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        return None  # no crossing

    if a == 0:
        if b == 0:
            return None
        t_cross = -c / b
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        # Take the future crossing (> current position)
        future_crossings = [t for t in [t1, t2] if t > n - 1]
        if not future_crossings:
            return None
        t_cross = min(future_crossings)

    steps_until = t_cross - (n - 1)
    if steps_until <= 0:
        return 0.0

    return float(steps_until * days_per_step)


def assess_tipping_point(
    health_series: np.ndarray,
    config: dict | None = None,
) -> TippingPointAssessment:
    """Perform comprehensive tipping-point risk assessment.
    
    Args:
        health_series: (T,) array of health scores [0, 1] over time.
        config: Risk engine configuration.
        
    Returns:
        TippingPointAssessment with risk score and detailed indicators.
    """
    config = config or {}
    risk_weights = config.get("risk_weights", {})

    # ─── Component Analysis ────────────────────────────────
    # 1. Critical Slowing Down
    csd_cfg = config.get("csd", {})
    csd_result = analyze_critical_slowing_down(
        health_series,
        window_size=csd_cfg.get("window_size", 6),
        ar1_threshold=csd_cfg.get("ar1_threshold", 0.8),
    )

    # 2. Vegetation resilience
    resilience = _compute_resilience(health_series)

    # 3. Spatial fragmentation
    fragmentation = _compute_fragmentation(health_series)

    # 4. Degradation momentum
    momentum = _compute_momentum(health_series)

    # 5. Drought sensitivity
    drought_sensitivity = _compute_drought_sensitivity(health_series)

    # ─── Combined Risk Score ───────────────────────────────
    w_csd = risk_weights.get("csd", 0.25)
    w_var = risk_weights.get("variance", 0.20)
    w_ac = risk_weights.get("autocorrelation", 0.20)
    w_frag = risk_weights.get("fragmentation", 0.15)
    w_drought = risk_weights.get("drought_sensitivity", 0.20)

    risk_score = (
        w_csd * csd_result.csd_score
        + w_var * (1.0 - resilience)  # low resilience = high risk
        + w_ac * momentum
        + w_frag * fragmentation
        + w_drought * drought_sensitivity
    )
    risk_score = float(np.clip(risk_score, 0.0, 1.0))

    # ─── Risk Category ─────────────────────────────────────
    if risk_score < 0.2:
        category = "Low"
    elif risk_score < 0.4:
        category = "Moderate"
    elif risk_score < 0.6:
        category = "Elevated"
    elif risk_score < 0.8:
        category = "High"
    else:
        category = "Critical"

    # ─── Contributing Factors ──────────────────────────────
    factors = []
    if csd_result.csd_score > 0.5:
        factors.append("Critical slowing down detected")
    if csd_result.ar1_trend > 0.3:
        factors.append("Rising autocorrelation (slowing recovery)")
    if csd_result.variance_trend > 0.3:
        factors.append("Increasing variance (weakening stability)")
    if resilience < 0.3:
        factors.append("Low vegetation resilience")
    if fragmentation > 0.5:
        factors.append("High ecosystem fragmentation")
    if momentum > 0.5:
        factors.append("Accelerating degradation")
    if drought_sensitivity > 0.5:
        factors.append("High drought sensitivity")

    # ─── Time to Tipping ───────────────────────────────────
    time_to_tipping = _estimate_time_to_tipping(health_series)

    # ─── Confidence ────────────────────────────────────────
    # Higher confidence with longer time series and more indicator agreement
    series_length_factor = min(1.0, len(health_series) / 12)
    indicator_agreement = 1.0 - np.std([
        csd_result.csd_score, 1 - resilience, fragmentation, momentum, drought_sensitivity
    ])
    confidence = float(np.clip(0.5 * series_length_factor + 0.5 * indicator_agreement, 0.0, 1.0))

    return TippingPointAssessment(
        risk_score=risk_score,
        risk_category=category,
        csd_score=csd_result.csd_score,
        resilience_score=resilience,
        fragmentation_score=fragmentation,
        momentum_score=momentum,
        drought_sensitivity_score=drought_sensitivity,
        csd_result=csd_result,
        time_to_tipping=time_to_tipping,
        confidence=confidence,
        contributing_factors=factors,
    )
