"""Regressions for scientific boundaries, missingness and scenario invariants."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from forestmind.risk.critical_slowing import (
    _rolling_ar1, analyze_critical_slowing_down,
)
from forestmind.risk.engine import EcologicalRiskEngine
from forestmind.risk.intervention import (
    InterventionScenario, run_intervention_analysis, simulate_intervention,
)
from forestmind.risk.tipping_point import assess_tipping_point


def test_constant_history_has_no_artificial_early_warning():
    result = analyze_critical_slowing_down(np.full(48, 0.8), season_period=12)
    assert result.status == "available"
    assert result.csd_score == 0
    assert result.variance == 0
    assert not result.is_warning


@pytest.mark.parametrize("values,status", [
    ([], "insufficient_data"), ([0.8], "insufficient_data"),
    ([0.8, 0.7], "insufficient_data"),
    ([np.nan] * 10, "missing_data"),
    ([0.8] * 10 + [np.inf], "missing_data"),
])
def test_unusable_histories_are_withheld_and_json_safe(values, status):
    result = analyze_critical_slowing_down(np.asarray(values))
    assert result.status == status
    assert not result.is_warning
    json.dumps(asdict(result), allow_nan=False)
    assessment = assess_tipping_point(np.asarray(values))
    json.dumps(asdict(assessment), allow_nan=False)
    assert assessment.time_to_tipping is None
    assert assessment.confidence is None


def test_missing_observations_are_not_compressed_into_adjacent_samples():
    series = np.linspace(0.9, 0.5, 20)
    series[8] = np.nan
    result = assess_tipping_point(series)
    assert result.csd_result.status == "missing_data"
    assert result.csd_score is None
    assert result.data_quality["missing_observations"] == 1
    assert result.momentum_score == pytest.approx(0.4)


def test_irregular_sampling_withholds_csd():
    result = analyze_critical_slowing_down(np.linspace(0.9, 0.6, 10),
                                          timestamps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 25])
    assert result.status == "irregular_sampling"
    assert not result.is_warning


def test_seasonal_signal_and_linear_decline_do_not_create_residual_warning():
    time = np.arange(60)
    series = 0.8 - time * 0.003 + 0.1 * np.sin(2 * np.pi * time / 12)
    result = analyze_critical_slowing_down(series, window_size=12, season_period=12)
    assert result.seasonal_adjustment
    assert result.csd_score == 0
    assert result.variance < 1e-15
    assert not result.is_warning


def test_one_season_is_insufficient_for_seasonal_resilience_analysis():
    result = analyze_critical_slowing_down(np.full(12, 0.8), season_period=12)
    assert result.status == "insufficient_data"


def test_rolling_ar1_includes_last_observation_and_is_regression_coefficient():
    series = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32])
    values = _rolling_ar1(series, 4)
    assert len(values) == 3
    np.testing.assert_allclose(values, 2.0)
    series[-1] = 0
    changed = _rolling_ar1(series, 4)
    assert changed[-1] != values[-1]
    np.testing.assert_array_equal(changed[:-1], values[:-1])


@pytest.mark.parametrize("timestamps", [[0] * 10, list(range(9)), [0, 1, np.nan] + list(range(3, 10))])
def test_invalid_timestamps_are_rejected(timestamps):
    with pytest.raises(ValueError, match="timestamps"):
        analyze_critical_slowing_down(np.full(10, 0.8), timestamps=timestamps)


def test_health_series_cannot_create_spatial_or_drought_evidence():
    result = assess_tipping_point(np.linspace(0.9, 0.1, 40))
    assert result.fragmentation_score is None
    assert result.drought_sensitivity_score is None
    assert "fragmentation" not in result.component_weights
    assert "drought_sensitivity" not in result.component_weights
    assert sum(result.component_weights.values()) == pytest.approx(1.0)
    assert result.time_to_tipping is None
    assert result.confidence is None


def test_explicit_external_evidence_is_accepted():
    result = assess_tipping_point(np.linspace(0.9, 0.2, 20), {
        "fragmentation_score": 0.7, "drought_indices": [10, 15],
    })
    assert result.fragmentation_score == 0.7
    assert result.drought_sensitivity_score > 0


def test_short_history_is_not_labelled_low_risk():
    result = assess_tipping_point(np.array([0.9, 0.8]))
    assert result.risk_score is None
    assert result.risk_category == "Insufficient data"


def test_no_recovery_events_leave_resilience_unknown():
    result = assess_tipping_point(np.full(20, 0.8))
    assert result.resilience_score is None


def test_zero_pressure_preserves_mixed_baseline_exactly():
    baseline = np.array([0.8, 0.9, 0.6])
    result = simulate_intervention(baseline, np.zeros(8),
                                  InterventionScenario("Protect", "", {"logging": 1}),
                                  [30, 365, 900], current_health=0.7)
    np.testing.assert_array_equal(result.projected_health, baseline)
    assert result.improvement_vs_baseline == [0, 0, 0]


def test_delay_uses_real_current_health_and_preserves_recovery():
    result = simulate_intervention(np.array([0.6, 0.5, 0.9]),
                                  np.array([0, 0, 0, 1, 0, 0, 0, 0]),
                                  InterventionScenario("Protect", "", {"logging": 1}, 1, 30),
                                  [30, 210, 365], current_health=0.8)
    assert result.projected_health == pytest.approx([0.6, 0.8, 0.9])
    assert "1 year" in result.summary
    assert "5yr" not in result.summary
    assert result.evidence_status == "hypothetical_sensitivity_analysis"


def test_empty_scenarios_still_return_actual_baseline_and_no_recommendation():
    result = run_intervention_analysis(np.array([0.7, 0.5]), np.zeros(8), 0.8,
                                       scenarios=[], horizon_days=[60, 120])
    assert result.baseline.projected_health == [0.7, 0.5]
    assert result.recommended is None
    assert result.horizons == ["60 days", "120 days"]


def test_configured_aliases_costs_and_delays_are_applied():
    result = run_intervention_analysis(
        np.array([0.5, 0.4]), np.array([0, 0, 0, 0, 1, 1, 0, 0]), 0.8,
        horizon_days=[90, 500], config={"scenarios": [{
            "name": "Custom", "driver_reduction": {"road": 0.8, "fire": 0.7},
            "cost_relative": 4, "implementation_delay_days": 180,
        }]},
    )
    custom = result.recommended
    assert custom.scenario.driver_reduction == {"road_construction": 0.8, "wildfire": 0.7}
    assert custom.scenario.cost_relative == 4
    assert custom.improvement_vs_baseline[0] == 0
    assert custom.improvement_vs_baseline[1] > 0


@pytest.mark.parametrize("days", [[30], [30, 30], [90, 30], [0, 30], [30.5, 90]])
def test_bad_horizons_are_never_silently_truncated(days):
    with pytest.raises(ValueError, match="horizon_days"):
        run_intervention_analysis(np.array([0.8, 0.6]), np.zeros(8), 0.9, horizon_days=days)


@pytest.mark.parametrize("reductions", [{"unknown": 0.5}, {"fire": 1.1}, {"fire": np.nan}])
def test_bad_scenario_assumptions_fail_explicitly(reductions):
    with pytest.raises(ValueError):
        InterventionScenario("Invalid", "", reductions)


def test_engine_propagates_horizons_and_csd_configuration():
    engine = EcologicalRiskEngine({"horizon_days": [45, 120], "csd": {"window_size": 10}})
    report = engine.analyze(
        health_history=np.linspace(0.9, 0.6, 20), health_state=1,
        health_score=60, health_confidence=85, active_drivers=[],
        driver_strengths=np.zeros(8), trajectory_scores=np.array([55, 25]),
    )
    assert report.trajectory_horizons == ["45 days", "120 days"]
    assert report.interventions.horizons == report.trajectory_horizons
    assert report.tipping_point.csd_result.window_size == 10
    summary = report.summary()
    assert "120 days" in summary
    assert "5-year" not in summary
    assert "5yr" not in summary
    assert "uncalibrated" in summary
    json.dumps(asdict(report), allow_nan=False)
