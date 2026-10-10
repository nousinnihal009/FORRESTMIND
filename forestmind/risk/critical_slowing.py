"""
Critical Slowing Down (CSD) detector for ecological tipping points.

Implements early-warning indicators from dynamical systems theory:

1. **Rising AR(1) coefficient**: As a system approaches a bifurcation point,
   its recovery rate from perturbations slows down. This manifests as
   increasing lag-1 autocorrelation in the time series.

2. **Rising variance**: Near a tipping point, the system's "restoring force"
   weakens, causing fluctuations to amplify.

3. **Rising temporal autocorrelation**: Similar to AR(1) but measured
   differently — the system becomes more "memory-full" as it approaches
   the critical transition.

References:
  - Scheffer et al. (2009) "Early-warning signals for critical transitions" Nature
  - Dakos et al. (2012) "Methods for Detecting Early Warnings" PLoS ONE
  - Verbesselt et al. (2016) "Remotely sensed resilience of tropical forests" Nature Climate Change
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CSDResult:
    """Results from Critical Slowing Down analysis."""
    ar1_coefficient: float          # AR(1) autoregressive coefficient
    ar1_trend: float                # Kendall tau trend of AR(1) over time
    variance: float                 # Current variance
    variance_trend: float           # Kendall tau trend of variance over time
    autocorrelation: float          # Lag-1 autocorrelation
    autocorrelation_trend: float    # Trend of autocorrelation
    recovery_rate: float            # Estimated recovery rate (1 - AR1)
    csd_score: float                # Combined CSD indicator [0, 1]
    is_warning: bool                # Whether CSD indicators suggest approaching tipping point


def _kendall_tau_trend(series: np.ndarray) -> float:
    """Compute Kendall tau trend statistic for a time series.
    
    Returns a value in [-1, 1] where positive values indicate
    an increasing trend (which for CSD indicators = approaching tipping point).
    """
    n = len(series)
    if n < 3:
        return 0.0

    concordant = 0
    discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = series[j] - series[i]
            if diff > 0:
                concordant += 1
            elif diff < 0:
                discordant += 1

    denom = n * (n - 1) / 2
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def _rolling_ar1(series: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling AR(1) coefficient over a time series.
    
    AR(1) coefficient is the correlation between x(t) and x(t-1).
    """
    n = len(series)
    if n < window + 1:
        return np.array([0.0])

    ar1_values = []
    for i in range(window, n):
        window_data = series[i - window: i]
        if len(window_data) < 3:
            ar1_values.append(0.0)
            continue
        x = window_data[:-1]
        y = window_data[1:]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            ar1_values.append(0.0)
        else:
            corr = np.corrcoef(x, y)[0, 1]
            ar1_values.append(float(corr) if not np.isnan(corr) else 0.0)

    return np.array(ar1_values)


def _rolling_variance(series: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling variance over a time series."""
    n = len(series)
    if n < window:
        return np.array([np.var(series)])

    variances = []
    for i in range(window, n + 1):
        variances.append(np.var(series[i - window: i]))

    return np.array(variances)


def _detrend(series: np.ndarray) -> np.ndarray:
    """Remove linear trend from time series (for CSD analysis).
    
    CSD indicators must be computed on detrended residuals,
    otherwise the trend itself would inflate variance/autocorrelation.
    """
    n = len(series)
    if n < 2:
        return series.copy()

    x = np.arange(n, dtype=np.float64)
    coeffs = np.polyfit(x, series, deg=1)
    trend = np.polyval(coeffs, x)
    return (series - trend).astype(np.float32)


def analyze_critical_slowing_down(
    health_series: np.ndarray,
    window_size: int = 6,
    ar1_threshold: float = 0.8,
) -> CSDResult:
    """Analyze a health time series for Critical Slowing Down indicators.
    
    Args:
        health_series: (T,) array of health scores over time.
        window_size: Rolling window size for indicator computation.
        ar1_threshold: AR(1) threshold for tipping-point warning.
        
    Returns:
        CSDResult with all indicators and combined score.
    """
    # Detrend the series
    residuals = _detrend(health_series)

    # Rolling AR(1)
    ar1_series = _rolling_ar1(residuals, window=window_size)
    ar1_current = float(ar1_series[-1]) if len(ar1_series) > 0 else 0.0
    ar1_trend = _kendall_tau_trend(ar1_series) if len(ar1_series) >= 3 else 0.0

    # Rolling variance
    var_series = _rolling_variance(residuals, window=window_size)
    var_current = float(var_series[-1]) if len(var_series) > 0 else 0.0
    var_trend = _kendall_tau_trend(var_series) if len(var_series) >= 3 else 0.0

    # Lag-1 autocorrelation (full series)
    if len(residuals) >= 3:
        x, y = residuals[:-1], residuals[1:]
        if np.std(x) > 1e-8 and np.std(y) > 1e-8:
            ac = float(np.corrcoef(x, y)[0, 1])
            ac = ac if not np.isnan(ac) else 0.0
        else:
            ac = 0.0
    else:
        ac = 0.0

    # Rolling autocorrelation for trend
    ac_series = _rolling_ar1(residuals, window=window_size)  # same as AR(1) for lag-1
    ac_trend = _kendall_tau_trend(ac_series) if len(ac_series) >= 3 else 0.0

    # Recovery rate: how quickly the system bounces back from perturbations
    # Estimated as 1 - AR(1): lower = slower recovery = closer to tipping
    recovery_rate = max(0.0, 1.0 - abs(ar1_current))

    # Combined CSD score [0, 1]
    # Higher = more warning signs of approaching tipping point
    csd_components = [
        np.clip(ar1_current, 0, 1),           # High AR(1) → tipping
        np.clip((ar1_trend + 1) / 2, 0, 1),   # Rising AR(1) trend → tipping
        np.clip((var_trend + 1) / 2, 0, 1),   # Rising variance → tipping
        np.clip((ac_trend + 1) / 2, 0, 1),    # Rising autocorrelation → tipping
        np.clip(1 - recovery_rate, 0, 1),      # Low recovery → tipping
    ]
    csd_score = float(np.mean(csd_components))

    # Warning threshold
    is_warning = ar1_current > ar1_threshold or csd_score > 0.65

    return CSDResult(
        ar1_coefficient=ar1_current,
        ar1_trend=ar1_trend,
        variance=var_current,
        variance_trend=var_trend,
        autocorrelation=ac,
        autocorrelation_trend=ac_trend,
        recovery_rate=recovery_rate,
        csd_score=csd_score,
        is_warning=is_warning,
    )
