"""Headless save contracts for unavailable and uncalibrated scientific outputs."""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pytest

from forestmind.risk.critical_slowing import CSDResult
from forestmind.risk.intervention import run_intervention_analysis
from forestmind.risk.tipping_point import TippingPointAssessment, assess_tipping_point
from forestmind.utils.visualization import (
    plot_driver_attribution,
    plot_health_trajectory,
    plot_intervention_comparison,
    plot_tipping_point_dashboard,
)


@pytest.fixture(autouse=True)
def headless_figures(monkeypatch):
    def no_gui():
        raise AssertionError("Saving a plot must not open a GUI")
    monkeypatch.setattr(plt, "show", no_gui)
    yield
    plt.close("all")


def figure_text(figure):
    return "\n".join(item.get_text() for item in figure.findobj(Text))


def assert_saved(path):
    assert path.exists() and path.stat().st_size > 1000
    pixels = plt.imread(path)
    assert pixels.shape[0] > 100 and pixels.shape[1] > 100


@pytest.mark.parametrize("history", [[], [0.8, 0.7], [np.nan] * 12])
def test_unavailable_risk_dashboard_saves_without_zero_component_bars(tmp_path, history):
    assessment = assess_tipping_point(np.asarray(history))
    output = tmp_path / "unavailable.png"
    plot_tipping_point_dashboard(assessment, np.asarray(history), save_path=output)
    assert_saved(output)
    figure = plt.gcf()
    text = figure_text(figure)
    assert "Unavailable" in text
    assert "Diagnostics unavailable" in text
    assert "uncalibrated" in text
    assert len(figure.axes[1].patches) == 0
    assert len(figure.axes[2].patches) == 0
    assert len(figure.axes[3].lines) == 1  # History only, no invented threshold or date.


def test_partial_components_remain_distinct_from_available_zero(tmp_path):
    assessment = TippingPointAssessment(
        risk_score=0.2, risk_category="Moderate", csd_score=None,
        resilience_score=None, fragmentation_score=None, momentum_score=0.0,
        drought_sensitivity_score=None,
    )
    output = tmp_path / "partial.png"
    plot_tipping_point_dashboard(assessment, np.linspace(0.9, 0.8, 12), save_path=output)
    assert_saved(output)
    components = plt.gcf().axes[1]
    assert len(components.patches) == 1
    assert components.patches[0].get_width() == 0
    assert sum(item.get_text() == "Unavailable" for item in components.texts) == 4


def test_csd_raw_signed_trends_handle_individually_missing_values(tmp_path):
    assessment = assess_tipping_point(np.linspace(0.9, 0.5, 20))
    assessment.csd_result = CSDResult(
        status="available", ar1_coefficient=1.2, ar1_trend=-0.4,
        variance_trend=None, autocorrelation_trend=np.nan, recovery_rate=0.2,
    )
    output = tmp_path / "csd.png"
    plot_tipping_point_dashboard(assessment, np.linspace(0.9, 0.5, 20), save_path=output)
    assert_saved(output)
    diagnostics = plt.gcf().axes[2]
    widths = [bar.get_width() for bar in diagnostics.patches]
    assert widths == pytest.approx([1.2, -0.4, 0.2])
    assert sum(item.get_text() == "Unavailable" for item in diagnostics.texts) == 2
    assert diagnostics.get_xlim()[1] >= 1.2


def test_legacy_tipping_date_is_not_rendered_as_validated_forecast(tmp_path):
    assessment = assess_tipping_point(np.linspace(0.9, 0.5, 20))
    assessment.time_to_tipping = 365.0
    output = tmp_path / "legacy.png"
    plot_tipping_point_dashboard(assessment, np.linspace(0.9, 0.5, 20), save_path=output)
    assert_saved(output)
    text = figure_text(plt.gcf())
    assert "Tipping threshold" not in text
    assert "Est. tipping" not in text
    assert len(plt.gcf().axes[3].lines) == 1


@pytest.mark.parametrize("history", [np.array([]), np.array([0.7, 0.6])])
def test_health_projection_saves_without_history_or_tipping_threshold(tmp_path, history):
    output = tmp_path / "trajectory.png"
    plot_health_trajectory(
        history, predicted_trajectory=np.array([0.5, 0.4]),
        predicted_uncertainty=np.array([0.05, 0.1]),
        horizon_labels=["45 days", "120 days"], save_path=output,
    )
    assert_saved(output)
    figure = plt.gcf()
    text = figure_text(figure)
    assert "45 days" in text and "120 days" in text
    assert "Model spread (uncalibrated)" in text
    assert "Tipping threshold" not in text
    assert len(figure.axes[0].lines) == 2


def test_projection_rejects_mismatched_horizon_labels_before_plotting(tmp_path):
    with pytest.raises(ValueError, match="horizon_labels"):
        plot_health_trajectory(np.array([0.7]), np.array([0.6, 0.5]),
                               horizon_labels=["30 days"], save_path=tmp_path / "bad.png")


def test_intervention_plot_preserves_custom_horizons_and_conditional_labels(tmp_path):
    analysis = run_intervention_analysis(
        np.array([0.7, 0.5]), np.array([0, 0, 0, 1, 0, 0, 0, 0]), 0.9,
        horizon_days=[45, 120],
    )
    output = tmp_path / "scenario.png"
    plot_intervention_comparison(analysis, save_path=output)
    assert_saved(output)
    figure = plt.gcf()
    assert [tick.get_text() for tick in figure.axes[0].get_xticklabels()] == ["45 days", "120 days"]
    text = figure_text(figure)
    assert "Highest assumed benefit/cost" in text
    assert "not validated causal intervention effects" in text
    assert "Recommended" not in text
    assert "5yr" not in text and "5 years" not in text


def test_intervention_plot_handles_empty_scenario_collection_with_baseline(tmp_path):
    analysis = run_intervention_analysis(np.array([0.7, 0.5]), np.zeros(8), 0.9,
                                         scenarios=[], horizon_days=[45, 120])
    analysis = replace(analysis, scenarios=[])
    output = tmp_path / "baseline.png"
    plot_intervention_comparison(analysis, save_path=output)
    assert_saved(output)
    assert len(plt.gcf().axes[0].patches) == 2
    assert "No active intervention scenarios" in figure_text(plt.gcf())


def test_intervention_plot_rejects_truncated_horizon_arrays(tmp_path):
    analysis = run_intervention_analysis(np.array([0.7, 0.5]), np.zeros(8), 0.9,
                                         horizon_days=[45, 120])
    analysis.horizons = ["45 days"]
    with pytest.raises(ValueError, match="configured horizon"):
        plot_intervention_comparison(analysis, save_path=tmp_path / "bad.png")


def test_driver_scores_are_not_presented_as_calibrated_causal_probabilities(tmp_path):
    output = tmp_path / "drivers.png"
    plot_driver_attribution(["Logging", "Drought"], np.array([0.7, 0.5]), save_path=output)
    assert_saved(output)
    text = figure_text(plt.gcf())
    assert "uncalibrated" in text
    assert "not causal contribution" in text
    assert "Detection Probability" not in text
