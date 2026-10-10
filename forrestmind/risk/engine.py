"""
Ecological Risk Engine — the core intelligence layer of ForestMind.

Orchestrates all risk analysis components:
  1. Model predictions (health, attribution, trajectory)
  2. Critical Slowing Down analysis
  3. Tipping-point risk assessment
  4. Intervention simulation

Produces a comprehensive ForestMindReport that answers:
  - What is the current state of this forest?
  - Why is it changing?
  - What will happen next?
  - How close is it to irreversible degradation?
  - What can we do about it?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from forestmind.risk.critical_slowing import CSDResult, analyze_critical_slowing_down
from forestmind.risk.intervention import (
    InterventionAnalysis,
    run_intervention_analysis,
)
from forestmind.risk.tipping_point import TippingPointAssessment, assess_tipping_point


HEALTH_STATE_NAMES = ["Healthy", "Stressed", "Degraded", "Severely Degraded", "Cleared"]
DRIVER_NAMES = [
    "Agriculture", "Cattle Pasture", "Mining", "Logging",
    "Road Construction", "Wildfire", "Drought", "Urban Expansion",
]


@dataclass
class ForestMindReport:
    """Comprehensive forest intelligence report."""
    # Metadata
    timestamp: str
    location: str | None = None
    
    # Current State
    health_state: str = ""
    health_score: float = 0.0
    health_confidence: float = 0.0
    
    # Cause Attribution
    active_drivers: list[dict] = field(default_factory=list)
    primary_driver: str | None = None
    
    # Future Trajectory
    trajectory_horizons: list[str] = field(default_factory=list)
    trajectory_scores: list[float] = field(default_factory=list)
    trajectory_uncertainty: list[float] = field(default_factory=list)
    
    # Tipping Point Risk
    tipping_point: TippingPointAssessment | None = None
    
    # Intervention Analysis
    interventions: InterventionAnalysis | None = None
    
    # Early Warning
    early_warnings: list[str] = field(default_factory=list)
    alert_level: str = "NONE"  # NONE, WATCH, WARNING, CRITICAL
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "        FORESTMIND INTELLIGENCE REPORT",
            "=" * 60,
            f"  Generated: {self.timestamp}",
        ]
        if self.location:
            lines.append(f"  Location:  {self.location}")
        lines.append("")
        
        # Alert banner
        if self.alert_level != "NONE":
            alert_symbols = {"WATCH": "[!]", "WARNING": "[!!]", "CRITICAL": "[!!!]"}
            symbol = alert_symbols.get(self.alert_level, "[?]")
            lines.append(f"  {symbol} ALERT LEVEL: {self.alert_level}")
            for warning in self.early_warnings:
                lines.append(f"    -> {warning}")
            lines.append("")
        
        # Current state
        lines.extend([
            "--- CURRENT STATE -------------------------------------------",
            f"  Forest Health: {self.health_state} ({self.health_score:.1f}/100)",
            f"  Confidence:    {self.health_confidence:.1f}%",
            "",
        ])
        
        # Drivers
        if self.active_drivers:
            lines.append("--- DEGRADATION DRIVERS ---------------------")
            for d in self.active_drivers:
                filled = int(d["strength"] / 10)
                bar = "#" * filled + "." * (10 - filled)
                lines.append(f"  {d['name']:20s} [{bar}] {d['strength']:.0f}%")
            lines.append("")
        
        # Trajectory
        if self.trajectory_scores:
            lines.append("--- PREDICTED TRAJECTORY --------------------")
            lines.append(f"  Current health: {self.health_score:.1f}")
            for horizon, score, unc in zip(
                self.trajectory_horizons, self.trajectory_scores, self.trajectory_uncertainty or [0]*5
            ):
                arrow = "v" if score < self.health_score else "^" if score > self.health_score else "="
                unc_str = f" +/-{unc:.1f}" if unc > 0 else ""
                lines.append(f"  {horizon:>10s}  {arrow} {score:.1f}{unc_str}")
            lines.append("")
        
        # Tipping point
        if self.tipping_point:
            tp = self.tipping_point
            lines.extend([
                "--- TIPPING-POINT RISK -------------------------",
                f"  Risk Score:    {tp.risk_score:.2f} ({tp.risk_category})",
                f"  Confidence:    {tp.confidence:.0%}",
            ])
            if tp.time_to_tipping is not None:
                if tp.time_to_tipping == 0:
                    lines.append("  Time to tipping: !! ALREADY PAST THRESHOLD")
                else:
                    years = tp.time_to_tipping / 365
                    lines.append(f"  Time to tipping: ~{years:.1f} years ({tp.time_to_tipping:.0f} days)")
            if tp.contributing_factors:
                lines.append("  Contributing factors:")
                for f in tp.contributing_factors:
                    lines.append(f"    - {f}")
            lines.append("")
        
        # Interventions
        if self.interventions and self.interventions.recommended:
            rec = self.interventions.recommended
            lines.extend([
                "--- RECOMMENDED INTERVENTION -------------------",
                f"  {rec.scenario.name}",
                f"  {rec.scenario.description}",
                f"  Projected 5yr health: {rec.projected_health_100[-1]:.1f}%",
                f"  Improvement:          +{rec.improvement_vs_baseline[-1]*100:.1f}pp",
                f"  Cost effectiveness:   {rec.cost_effectiveness:.3f}",
                "",
            ])
        
        lines.append("=" * 60)
        return "\n".join(lines)


class EcologicalRiskEngine:
    """Orchestrates full ecological risk analysis.
    
    Takes model predictions and performs post-hoc risk analysis
    including tipping-point assessment and intervention simulation.
    
    Args:
        config: Risk engine configuration.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def analyze(
        self,
        health_history: np.ndarray,
        health_state: int,
        health_score: float,
        health_confidence: float,
        active_drivers: list[dict],
        driver_strengths: np.ndarray,
        trajectory_scores: np.ndarray,
        trajectory_uncertainty: np.ndarray | None = None,
        location: str | None = None,
    ) -> ForestMindReport:
        """Run full ecological risk analysis.
        
        Args:
            health_history: (T,) historical health scores.
            health_state: Predicted health state (0-4).
            health_score: Predicted health score (0-100).
            health_confidence: Confidence in prediction (0-100).
            active_drivers: List of active driver dicts.
            driver_strengths: (8,) driver strength array.
            trajectory_scores: (H,) predicted future health scores (0-100).
            trajectory_uncertainty: (H,) prediction uncertainty.
            location: Optional location identifier.
            
        Returns:
            ForestMindReport with comprehensive analysis.
        """
        # ─── Tipping-Point Assessment ─────────────────────
        # Use the full health series (history + predicted future)
        extended_series = np.concatenate([
            health_history,
            trajectory_scores / 100.0,  # normalize to [0, 1]
        ])
        
        tp_config = self.config.get("tipping_point", {})
        tp_assessment = assess_tipping_point(health_history, tp_config)

        # ─── Intervention Simulation ──────────────────────
        int_config = self.config.get("intervention", {})
        trajectory_normalized = trajectory_scores / 100.0

        interventions = run_intervention_analysis(
            baseline_trajectory=trajectory_normalized,
            driver_strengths=driver_strengths,
            current_health=health_score / 100.0,
            config=int_config,
        )

        # ─── Early Warnings ──────────────────────────────
        early_warnings = []
        alert_level = "NONE"

        # Check tipping-point risk
        if tp_assessment.risk_category == "Critical":
            early_warnings.append(
                "CRITICAL: Ecosystem shows strong signs of approaching irreversible degradation"
            )
            alert_level = "CRITICAL"
        elif tp_assessment.risk_category == "High":
            early_warnings.append(
                "WARNING: Multiple tipping-point indicators are elevated"
            )
            alert_level = "WARNING"
        elif tp_assessment.risk_category == "Elevated":
            early_warnings.append(
                "WATCH: Some early-warning indicators are active"
            )
            alert_level = "WATCH"

        # Check trajectory
        if len(trajectory_scores) > 0:
            predicted_5yr = trajectory_scores[-1]
            if predicted_5yr < 30:
                early_warnings.append(
                    f"5-year projection shows severe degradation ({predicted_5yr:.0f}/100)"
                )
                if alert_level == "NONE":
                    alert_level = "WARNING"
            elif predicted_5yr < 50:
                early_warnings.append(
                    f"5-year projection shows significant decline ({predicted_5yr:.0f}/100)"
                )
                if alert_level == "NONE":
                    alert_level = "WATCH"

        # Check CSD-specific warnings
        if tp_assessment.csd_result and tp_assessment.csd_result.is_warning:
            early_warnings.append(
                f"Critical slowing down detected (AR1={tp_assessment.csd_result.ar1_coefficient:.2f})"
            )

        # Check rapid decline
        if len(health_history) >= 3:
            recent_change = health_history[-1] - health_history[-3]
            if recent_change < -0.15:
                early_warnings.append(
                    f"Rapid recent decline: {recent_change*100:.1f}pp over last observations"
                )
                if alert_level in ("NONE", "WATCH"):
                    alert_level = "WARNING"

        # Primary driver
        primary_driver = None
        if active_drivers:
            primary_driver = max(active_drivers, key=lambda d: d.get("strength", 0))["name"]

        # ─── Build Report ─────────────────────────────────
        horizon_names = ["30 days", "6 months", "1 year", "3 years", "5 years"]

        report = ForestMindReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            location=location,
            health_state=HEALTH_STATE_NAMES[health_state] if 0 <= health_state < 5 else "Unknown",
            health_score=health_score,
            health_confidence=health_confidence,
            active_drivers=active_drivers,
            primary_driver=primary_driver,
            trajectory_horizons=horizon_names[:len(trajectory_scores)],
            trajectory_scores=trajectory_scores.tolist(),
            trajectory_uncertainty=trajectory_uncertainty.tolist() if trajectory_uncertainty is not None else [],
            tipping_point=tp_assessment,
            interventions=interventions,
            early_warnings=early_warnings,
            alert_level=alert_level,
        )

        return report
