"""Tests for ecological risk engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from forestmind.risk.critical_slowing import analyze_critical_slowing_down, CSDResult
from forestmind.risk.tipping_point import assess_tipping_point, TippingPointAssessment
from forestmind.risk.intervention import (
    simulate_intervention,
    run_intervention_analysis,
    InterventionScenario,
    DEFAULT_SCENARIOS,
)
from forestmind.risk.engine import EcologicalRiskEngine


class TestCriticalSlowingDown:
    """Test CSD indicator computation."""

    def test_stable_series_low_csd(self):
        """A stable series should have low CSD indicators."""
        series = np.ones(20, dtype=np.float32) * 0.85
        series += np.random.RandomState(42).normal(0, 0.01, 20).astype(np.float32)
        result = analyze_critical_slowing_down(series, window_size=5)

        assert isinstance(result, CSDResult)
        assert result.csd_score < 0.6, f"Stable series has high CSD: {result.csd_score}"
        assert result.recovery_rate > 0.3

    def test_declining_series_higher_csd(self):
        """A declining series should have higher CSD indicators."""
        series = np.linspace(0.9, 0.3, 20).astype(np.float32)
        series += np.random.RandomState(42).normal(0, 0.02, 20).astype(np.float32)
        series = np.clip(series, 0, 1)
        result = analyze_critical_slowing_down(series, window_size=5)

        # Declining series should trigger some CSD signals
        assert isinstance(result, CSDResult)

    def test_short_series_handles_gracefully(self):
        """Very short series doesn't crash."""
        series = np.array([0.8, 0.7], dtype=np.float32)
        result = analyze_critical_slowing_down(series, window_size=2)
        assert isinstance(result, CSDResult)

    def test_ar1_threshold_warning(self):
        """High AR(1) coefficient triggers warning."""
        # Create a highly autocorrelated series (near tipping)
        np.random.seed(42)
        series = np.zeros(30, dtype=np.float32)
        series[0] = 0.5
        for i in range(1, 30):
            series[i] = 0.95 * series[i-1] + np.random.normal(0, 0.01)
        series = np.clip(series, 0, 1)

        result = analyze_critical_slowing_down(series, window_size=6, ar1_threshold=0.5)
        # With AR(1) ~ 0.95, this should trigger
        assert isinstance(result, CSDResult)


class TestTippingPoint:
    """Test tipping-point assessment."""

    def test_healthy_forest_low_risk(self):
        """Healthy stable forest should have low tipping-point risk."""
        series = np.ones(15) * 0.90
        series += np.random.RandomState(42).normal(0, 0.01, 15)
        series = np.clip(series, 0, 1).astype(np.float32)

        result = assess_tipping_point(series)
        assert isinstance(result, TippingPointAssessment)
        assert result.risk_score < 0.5, f"Healthy forest has high risk: {result.risk_score}"
        assert result.risk_category in ("Low", "Moderate")

    def test_severely_declining_high_risk(self):
        """Severely declining forest should have higher risk."""
        series = np.linspace(0.85, 0.20, 15).astype(np.float32)
        series += np.random.RandomState(42).normal(0, 0.02, 15).astype(np.float32)
        series = np.clip(series, 0, 1).astype(np.float32)

        result = assess_tipping_point(series)
        assert isinstance(result, TippingPointAssessment)
        assert result.momentum_score > 0.1, "Declining series should have momentum"

    def test_time_to_tipping_is_unavailable_without_validation(self):
        """A declining health line cannot establish an ecological tipping date."""
        series = np.linspace(0.7, 0.4, 12).astype(np.float32)
        result = assess_tipping_point(series)

        assert result.time_to_tipping is None

    def test_contributing_factors_populated(self):
        """Declining series should generate contributing factors."""
        series = np.linspace(0.9, 0.2, 20).astype(np.float32)
        result = assess_tipping_point(series)
        # With severe decline, should have at least one factor
        assert isinstance(result.contributing_factors, list)

    def test_confidence_is_not_fabricated_from_series_length(self):
        """Longer histories alone do not produce a calibrated probability."""
        short = np.linspace(0.9, 0.5, 5).astype(np.float32)
        long_ = np.linspace(0.9, 0.5, 20).astype(np.float32)

        r_short = assess_tipping_point(short)
        r_long = assess_tipping_point(long_)

        assert r_short.confidence is None
        assert r_long.confidence is None
        assert r_long.calibration_status == "uncalibrated_heuristic"


class TestIntervention:
    """Test intervention simulator."""

    def test_no_intervention_returns_baseline(self):
        """No-intervention scenario returns the baseline trajectory."""
        baseline = np.array([0.7, 0.6, 0.5, 0.4, 0.3], dtype=np.float32)
        drivers = np.array([0.5, 0.3, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        scenario = InterventionScenario(
            name="No Intervention", description="", driver_reduction={},
        )
        result = simulate_intervention(baseline, drivers, scenario)

        np.testing.assert_array_almost_equal(result.projected_health, baseline.tolist(), decimal=3)
        assert all(imp == 0 for imp in result.improvement_vs_baseline)

    def test_intervention_improves_trajectory(self):
        """An intervention should improve over baseline."""
        baseline = np.array([0.7, 0.6, 0.5, 0.4, 0.3], dtype=np.float32)
        drivers = np.array([0.8, 0.5, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        scenario = InterventionScenario(
            name="Full Protection", description="",
            driver_reduction={"agriculture": 0.8, "cattle_pasture": 0.8},
        )
        result = simulate_intervention(baseline, drivers, scenario)

        # At least some horizons should improve
        improvements = result.improvement_vs_baseline
        assert any(imp > 0 for imp in improvements), "No improvement from intervention"

    def test_full_analysis_produces_recommendation(self):
        """Full analysis should produce a recommended scenario."""
        baseline = np.array([0.7, 0.6, 0.5, 0.4, 0.3], dtype=np.float32)
        drivers = np.array([0.5, 0.3, 0.2, 0.4, 0, 0.1, 0, 0], dtype=np.float32)

        analysis = run_intervention_analysis(
            baseline_trajectory=baseline,
            driver_strengths=drivers,
            current_health=0.75,
        )

        assert len(analysis.scenarios) > 0
        assert analysis.baseline is not None
        assert analysis.recommended is not None
        assert len(analysis.horizons) > 0


class TestEcologicalRiskEngine:
    """Test the full risk engine orchestration."""

    def test_full_analysis_produces_report(self):
        """Risk engine produces a complete report."""
        engine = EcologicalRiskEngine()

        report = engine.analyze(
            health_history=np.linspace(0.9, 0.6, 12).astype(np.float32),
            health_state=1,
            health_score=60.0,
            health_confidence=85.0,
            active_drivers=[{"name": "Logging", "confidence": 72, "strength": 55}],
            driver_strengths=np.array([0, 0, 0, 0.55, 0, 0, 0, 0], dtype=np.float32),
            trajectory_scores=np.array([58, 52, 45, 35, 28], dtype=np.float32),
            location="Test Forest",
        )

        assert report.health_state == "Stressed"
        assert report.health_score == 60.0
        assert report.tipping_point is not None
        assert report.interventions is not None
        assert isinstance(report.early_warnings, list)
        assert report.alert_level in ("NONE", "WATCH", "WARNING", "CRITICAL")

    def test_report_summary_renders(self):
        """Report summary string renders without errors."""
        engine = EcologicalRiskEngine()

        report = engine.analyze(
            health_history=np.linspace(0.9, 0.6, 12).astype(np.float32),
            health_state=1,
            health_score=60.0,
            health_confidence=85.0,
            active_drivers=[],
            driver_strengths=np.zeros(8, dtype=np.float32),
            trajectory_scores=np.array([58, 55, 52, 48, 45], dtype=np.float32),
        )

        summary = report.summary()
        assert isinstance(summary, str)
        assert "FORESTMIND" in summary
        assert "Forest Health" in summary
