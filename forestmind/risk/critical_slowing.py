"""Exploratory early-warning statistics, not proof of an ecological tipping point.

Rolling statistics follow the general workflow in Dakos et al. (2012),
https://doi.org/10.1371/journal.pone.0041010. The combined score and thresholds
are uncalibrated heuristics. Sampling, seasonality and detrending affect results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CSDResult:
    """Diagnostic statistics; inspect ``status`` before interpreting values."""

    ar1_coefficient: float = 0.0
    ar1_trend: float = 0.0
    variance: float = 0.0
    variance_trend: float = 0.0
    autocorrelation: float = 0.0
    autocorrelation_trend: float = 0.0
    recovery_rate: float = 0.0  # Legacy dimensionless persistence proxy, not a measured rate.
    csd_score: float = 0.0
    is_warning: bool = False
    status: str = "insufficient_data"
    n_observations: int = 0
    n_valid: int = 0
    window_size: int = 6
    seasonal_adjustment: bool = False
    limitations: list[str] = field(default_factory=list)


def _health_array(series: np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("health_series must be a one-dimensional array")
    valid = values[np.isfinite(values)]
    if np.any((valid < 0) | (valid > 1)):
        raise ValueError("finite health values must lie in [0, 1]")
    return values


def _time_axis(timestamps: np.ndarray | list | None, n: int) -> tuple[np.ndarray, bool]:
    if timestamps is None:
        return np.arange(n, dtype=float), True
    raw = np.asarray(timestamps)
    if raw.ndim != 1 or len(raw) != n:
        raise ValueError("timestamps must have the same length as health_series")
    if np.issubdtype(raw.dtype, np.number):
        axis = raw.astype(float)
    else:
        try:
            dates = raw.astype("datetime64[ns]")
        except (ValueError, TypeError) as exc:
            raise ValueError("timestamps must be numeric or valid dates") from exc
        if np.any(np.isnat(dates)):
            raise ValueError("timestamps must not contain missing dates")
        axis = dates.astype(np.int64).astype(float) / 86_400_000_000_000
    if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
        raise ValueError("timestamps must be finite, unique and increasing")
    axis = axis - axis[0] if n else axis
    gaps = np.diff(axis)
    regular = len(gaps) < 2 or bool(np.allclose(gaps, gaps[0], rtol=0.01, atol=1e-8))
    return axis, regular


def _kendall_tau_trend(series: np.ndarray) -> float:
    """Kendall tau-a trend; a descriptive statistic, without a significance test."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 3 or np.ptp(series) < 1e-12:
        return 0.0
    signs = sum(float(np.sign(series[i + 1:] - series[i]).sum()) for i in range(n - 1))
    return signs / (n * (n - 1) / 2)


def _ar1(series: np.ndarray) -> float:
    """OLS lag-one coefficient with intercept (distinct from correlation)."""
    x, y = series[:-1], series[1:]
    if len(x) < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    centered = x - x.mean()
    return float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))


def _correlation(series: np.ndarray) -> float:
    if len(series) < 3 or min(np.std(series[:-1]), np.std(series[1:])) < 1e-10:
        return 0.0
    return float(np.clip(np.corrcoef(series[:-1], series[1:])[0, 1], -1, 1))


def _rolling_ar1(series: np.ndarray, window: int) -> np.ndarray:
    return np.asarray([_ar1(series[i - window:i]) for i in range(window, len(series) + 1)])


def _rolling_variance(series: np.ndarray, window: int) -> np.ndarray:
    return np.asarray([np.var(series[i - window:i]) for i in range(window, len(series) + 1)])


def _detrend(series: np.ndarray, season_period: int | None = None) -> np.ndarray:
    """Fit trend and optional seasonal fixed effects jointly to avoid trend leakage."""
    if len(series) < 2:
        return series.copy()
    x = np.linspace(-1, 1, len(series))
    columns = [np.ones(len(series)), x]
    if season_period:
        phases = np.arange(len(series)) % season_period
        columns.extend((phases == phase).astype(float) for phase in range(1, season_period))
    design = np.column_stack(columns)
    residuals = series - design @ np.linalg.lstsq(design, series, rcond=None)[0]
    residuals[np.abs(residuals) < 1e-10] = 0
    return residuals


def analyze_critical_slowing_down(
    health_series: np.ndarray,
    window_size: int = 6,
    ar1_threshold: float = 0.8,
    *,
    timestamps: np.ndarray | list | None = None,
    season_period: int | None = None,
) -> CSDResult:
    """Compute exploratory statistics only for complete, evenly sampled histories.

    Missing observations are never removed or interpolated into apparent temporal
    adjacency. Withhold diagnostics if gaps, irregular sampling or short seasonal
    baselines make this method inapplicable. ``season_period`` is observations per
    seasonal cycle; at least three cycles are required for seasonal adjustment.
    """
    values = _health_array(health_series)
    if not isinstance(window_size, (int, np.integer)) or window_size < 2:
        raise ValueError("window_size must be an integer of at least 2")
    if not np.isfinite(ar1_threshold) or not 0 < ar1_threshold <= 1:
        raise ValueError("ar1_threshold must lie in (0, 1]")
    if season_period is not None and (
        not isinstance(season_period, (int, np.integer)) or season_period < 2
    ):
        raise ValueError("season_period must be an integer of at least 2")
    _, regular = _time_axis(timestamps, len(values))
    result = CSDResult(
        n_observations=len(values), n_valid=int(np.isfinite(values).sum()),
        window_size=window_size,
        limitations=[
            "Exploratory, uncalibrated indicators; neither a tipping probability nor a tipping date.",
            "Overlapping windows are dependent; trends have not been significance-tested.",
        ],
    )
    if timestamps is None:
        result.limitations.append("Equal observation spacing is assumed; no timestamps supplied.")
    if not regular:
        result.status = "irregular_sampling"
        result.limitations.append("Aggregate observations onto a regular time grid before CSD analysis.")
        return result
    if result.n_valid != len(values):
        result.status = "missing_data"
        result.limitations.append("CSD withheld because the time series contains non-finite observations.")
        return result
    minimum = max(8, window_size + 2, 3 * season_period if season_period else 0)
    if window_size < 3 or len(values) < minimum:
        result.limitations.append(f"At least {minimum} observations and window_size >= 3 are required.")
        return result
    if season_period is None:
        result.limitations.append("Seasonality was not removed; seasonal cycles can confound these indicators.")
    result.seasonal_adjustment = season_period is not None
    residuals = _detrend(values, season_period)
    ar1_series = _rolling_ar1(residuals, window_size)
    variance_series = _rolling_variance(residuals, window_size)
    correlation_series = np.asarray([
        _correlation(residuals[i - window_size:i]) for i in range(window_size, len(values) + 1)
    ])
    result.status = "available"
    result.ar1_coefficient = float(ar1_series[-1])
    result.ar1_trend = _kendall_tau_trend(ar1_series)
    result.variance = float(variance_series[-1])
    result.variance_trend = _kendall_tau_trend(variance_series)
    result.autocorrelation = _correlation(residuals)
    result.autocorrelation_trend = _kendall_tau_trend(correlation_series)
    result.recovery_rate = float(1 - np.clip(result.ar1_coefficient, 0, 1))
    # Positive trends only: a flat series gets zero, not a spurious 0.3 score.
    # AR1 and lag-one correlation describe similar evidence and are not double-counted.
    result.csd_score = float(
        (max(0, result.ar1_trend) + max(0, result.variance_trend)) / 2
    )
    result.is_warning = bool(
        result.ar1_coefficient > ar1_threshold
        and result.ar1_trend > 0.3 and result.variance_trend > 0.3
    )
    result.limitations.append("recovery_rate is a dimensionless persistence proxy, not measured recovery.")
    return result
