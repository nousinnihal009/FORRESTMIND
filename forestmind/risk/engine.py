"""Orchestrate condition screening and explicitly hypothetical scenarios.

Predictions, diagnostic statistics and scenario assumptions remain separate;
none of these outputs establishes ecological irreversibility or policy effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from forestmind.risk.critical_slowing import _health_array
from forestmind.risk.intervention import (
    InterventionAnalysis, horizon_label, resolve_horizons, run_intervention_analysis,
)
from forestmind.risk.tipping_point import TippingPointAssessment, assess_tipping_point


HEALTH_STATE_NAMES = ["Healthy", "Stressed", "Degraded", "Severely Degraded", "Cleared"]
DRIVER_NAMES = [
    "Agriculture", "Cattle Pasture", "Mining", "Logging",
    "Road Construction", "Wildfire", "Drought", "Urban Expansion",
]


@dataclass
class ForestMindReport:
    timestamp: str
    location: str | None = None
    health_state: str = ""
    health_score: float = 0.0
    health_confidence: float | None = None
    active_drivers: list[dict] = field(default_factory=list)
    primary_driver: str | None = None
    trajectory_horizons: list[str] = field(default_factory=list)
    trajectory_scores: list[float] = field(default_factory=list)
    trajectory_uncertainty: list[float] = field(default_factory=list)
    tipping_point: TippingPointAssessment | None = None
    interventions: InterventionAnalysis | None = None
    early_warnings: list[str] = field(default_factory=list)
    alert_level: str = "NONE"
    limitations: list[str] = field(default_factory=list)
    calibration_status: str = "uncalibrated"

    def summary(self) -> str:
        """Render results without converting diagnostic scores into scientific claims."""
        lines = ["=" * 60, "        FORESTMIND INTELLIGENCE REPORT", "=" * 60,
                 f"  Generated: {self.timestamp}"]
        if self.location:
            lines.append(f"  Location: {self.location}")
        lines.extend(["", f"  Screening alert: {self.alert_level}"])
        lines.extend(f"    -> {warning}" for warning in self.early_warnings)
        lines.extend(["", "--- CURRENT STATE ---",
                      f"  Forest Health: {self.health_state} ({self.health_score:.1f}/100)"])
        if self.health_confidence is not None:
            lines.append(f"  Model classification score: {self.health_confidence:.1f}% (uncalibrated)")
        if self.active_drivers:
            lines.append("\n--- CANDIDATE PRESSURES (NOT CAUSAL ATTRIBUTION) ---")
            for driver in self.active_drivers:
                lines.append(f"  {driver.get('name', 'Unknown')}: {driver.get('strength', 0):.1f}/100")
        if self.trajectory_scores:
            lines.append("\n--- MODEL TRAJECTORY ---")
            uncertainty = self.trajectory_uncertainty or [0] * len(self.trajectory_scores)
            for label, score, spread in zip(self.trajectory_horizons, self.trajectory_scores, uncertainty):
                spread_text = f" (model spread {spread:.1f}; uncalibrated)" if spread else ""
                lines.append(f"  {label}: {score:.1f}/100{spread_text}")
        if self.tipping_point:
            assessment = self.tipping_point
            score = f"{assessment.risk_score:.2f}" if assessment.risk_score is not None else "unavailable"
            lines.extend(["\n--- EXPLORATORY ECOLOGICAL SCREENING ---",
                          f"  Heuristic index: {score} ({assessment.risk_category})",
                          "  Tipping probability, tipping date and calibrated confidence: unavailable"])
            lines.extend(f"    - {factor}" for factor in assessment.contributing_factors)
        if self.interventions and self.interventions.recommended:
            result = self.interventions.recommended
            label = self.interventions.horizons[-1]
            lines.extend(["\n--- HIGHEST RANKED HYPOTHETICAL SCENARIO ---",
                          f"  {result.scenario.name}", f"  {result.scenario.description}",
                          f"  Conditional {label} health: {result.projected_health_100[-1]:.1f}/100",
                          f"  Assumed difference: +{result.improvement_vs_baseline[-1] * 100:.1f} points",
                          "  Scenario ranking is not a validated intervention recommendation."])
        lines.append("\n--- LIMITATIONS ---")
        lines.extend(f"  - {limitation}" for limitation in self.limitations)
        if self.tipping_point:
            lines.extend(f"  - {limitation}" for limitation in self.tipping_point.limitations)
        lines.append("=" * 60)
        return "\n".join(lines)


class EcologicalRiskEngine:
    """Combine model outputs with descriptive diagnostics and scenario sensitivity."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def analyze(
        self,
        health_history: np.ndarray,
        health_state: int,
        health_score: float,
        health_confidence: float | None,
        active_drivers: list[dict],
        driver_strengths: np.ndarray,
        trajectory_scores: np.ndarray,
        trajectory_uncertainty: np.ndarray | None = None,
        location: str | None = None,
        *,
        horizon_days: list[int] | None = None,
        timestamps: np.ndarray | list | None = None,
        season_period: int | None = None,
    ) -> ForestMindReport:
        """Analyze historical observations, without appending model forecasts to CSD.

        Explicit horizon_days override configuration. If none are supplied, the
        configured horizon_days/horizons or the legacy five horizons are used.
        Histories may contain gaps; diagnostics disclose rather than fill them.
        """
        history = _health_array(health_history)
        trajectory = np.asarray(trajectory_scores, dtype=float)
        if (trajectory.ndim != 1 or not len(trajectory) or not np.all(np.isfinite(trajectory))
                or np.any((trajectory < 0) | (trajectory > 100))):
            raise ValueError("trajectory_scores must be a finite nonempty one-dimensional array in [0, 100]")
        if not np.isfinite(health_score) or not 0 <= health_score <= 100:
            raise ValueError("health_score must be finite and lie in [0, 100]")
        if health_confidence is not None and (
            not np.isfinite(health_confidence) or not 0 <= health_confidence <= 100
        ):
            raise ValueError("health_confidence must be unavailable or finite and lie in [0, 100]")
        uncertainty = np.asarray(trajectory_uncertainty, dtype=float) if trajectory_uncertainty is not None else None
        if uncertainty is not None and (
            uncertainty.shape != trajectory.shape or not np.all(np.isfinite(uncertainty))
            or np.any(uncertainty < 0)
        ):
            raise ValueError("trajectory_uncertainty must be finite, non-negative and match trajectory_scores")
        configured = self.config.get("horizon_days", self.config.get("horizons"))
        if configured is None:
            configured = self.config.get("heads", {}).get("trajectory", {}).get("horizons")
        days = resolve_horizons(horizon_days if horizon_days is not None else configured, len(trajectory))
        labels = [horizon_label(day) for day in days]
        risk_config = self.config.get("risk_engine", self.config)
        assessment_config = dict(risk_config.get("tipping_point", {}))
        assessment_config["csd"] = {
            **risk_config.get("csd", {}), **assessment_config.get("csd", {}),
        }
        assessment = assess_tipping_point(history, assessment_config,
                                         timestamps=timestamps, season_period=season_period)
        interventions = run_intervention_analysis(
            baseline_trajectory=trajectory / 100, driver_strengths=driver_strengths,
            current_health=health_score / 100, horizon_days=days,
            config=risk_config.get("intervention", {}),
        )
        warnings = []
        alert_level = "NONE"
        if assessment.risk_category in {"High", "Critical"}:
            warnings.append("Several screening indicators warrant prompt field investigation; irreversibility is not established")
            alert_level = "WARNING" if assessment.risk_category == "High" else "CRITICAL"
        elif assessment.risk_category == "Elevated":
            warnings.append("Exploratory condition indicators are elevated")
            alert_level = "WATCH"
        if trajectory[-1] < 30:
            warnings.append(f"Model projection at {labels[-1]} indicates low health ({trajectory[-1]:.0f}/100); validation required")
            if alert_level in {"NONE", "WATCH"}:
                alert_level = "WARNING"
        elif trajectory[-1] < 50:
            warnings.append(f"Model projection at {labels[-1]} indicates reduced health ({trajectory[-1]:.0f}/100); validation required")
            if alert_level == "NONE":
                alert_level = "WATCH"
        if assessment.csd_result and assessment.csd_result.is_warning:
            warnings.append("Increasing residual persistence and variance warrant investigation; CSD alone does not establish a tipping point")
            if alert_level == "NONE":
                alert_level = "WATCH"
        if len(history) >= 3 and np.all(np.isfinite(history[-3:])):
            recent_change = history[-1] - history[-3]
            if recent_change < -0.15:
                warnings.append(f"Recent health decrease: {recent_change * 100:.1f} points over the last three observations")
                if alert_level in {"NONE", "WATCH"}:
                    alert_level = "WARNING"
        if isinstance(health_state, str):
            state_label = health_state
        elif isinstance(health_state, (int, np.integer)) and 0 <= int(health_state) < len(HEALTH_STATE_NAMES):
            state_label = HEALTH_STATE_NAMES[int(health_state)]
        else:
            state_label = "Unknown"
        primary = max(active_drivers, key=lambda driver: driver.get("strength", 0)).get("name") if active_drivers else None
        return ForestMindReport(
            timestamp=datetime.now(timezone.utc).isoformat(), location=location,
            health_state=state_label, health_score=float(health_score),
            health_confidence=float(health_confidence) if health_confidence is not None else None,
            active_drivers=active_drivers, primary_driver=primary,
            trajectory_horizons=labels, trajectory_scores=trajectory.tolist(),
            trajectory_uncertainty=uncertainty.tolist() if uncertainty is not None else [],
            tipping_point=assessment, interventions=interventions,
            early_warnings=warnings, alert_level=alert_level,
            limitations=[
                "Model scores and uncertainty have not been calibrated by this risk engine.",
                "Candidate pressures do not establish the causes of forest change.",
                "Intervention outputs are hypothetical scenario calculations, not causal forecasts.",
            ],
        )
