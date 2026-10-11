"""Behavioral checks for the public monitoring baseline and its scientific limits."""

from copy import deepcopy

import pytest

from forestmind.monitoring.analysis import analyze_region, indicator_score, sensitivity_scenario
from forestmind.monitoring.demo import demo_regions


def test_demo_is_deterministic_and_honestly_labeled():
    first, second = demo_regions(), demo_regions()
    assert first == second
    assert len(first) == 4
    for region, observations in first:
        result = analyze_region(region, observations)
        assert result["provenance"]["synthetic"] is True
        assert result["confidence"] is None
        assert result["resilience"]["score"] is None
        assert result["resilience"]["time_to_tipping"] is None
        assert result["forecast"]["calibrated"] is False
        assert len(observations) == 48
        assert {point["horizon_days"] for point in result["forecast"]["points"]} == {30, 180, 365}
        assert all(0 <= point["lower"] <= point["score"] <= point["upper"] <= 100
                   for point in result["forecast"]["points"])


def test_low_coverage_and_missing_bands_never_become_healthy_defaults():
    region, rows = demo_regions()[0]
    missing = deepcopy(rows)
    for row in missing:
        row["ndvi"] = None
    result = analyze_region(region, missing)
    assert result["health_score"] is None
    assert result["status"] == "insufficient_data"
    assert result["forecast"]["points"] == []
    assert result["resilience"]["ar1"] is None
    assert result["data_quality"]["usable_observations"] == 0
    assert any(alert["id"].endswith("coverage") for alert in result["alerts"])
    assert indicator_score({**rows[-1], "valid_fraction": 0.39}) is None


def test_short_series_cannot_claim_resilience_or_forecast():
    region, rows = demo_regions()[0]
    result = analyze_region(region, rows[-8:])
    assert result["resilience"]["status"] == "insufficient_data"
    assert result["resilience"]["score"] is None
    assert result["forecast"]["points"] == []
    assert "required" in result["forecast"]["caveat"]


def test_gaps_are_exposed_and_prevent_lag_diagnostics():
    region, rows = demo_regions()[0]
    result = analyze_region(region, rows[:40] + rows[41:])
    assert result["data_quality"]["missing_months"] == 1
    assert result["resilience"]["status"] == "insufficient_data"
    assert result["forecast"]["points"] == []


def test_missing_driver_observations_remain_unresolved():
    region, rows = demo_regions()[1]
    for row in rows:
        row["fire_anomaly"] = None
        row["precipitation_anomaly"] = None
    result = analyze_region(region, rows)
    fire = next(driver for driver in result["drivers"] if driver["id"] == "fire")
    moisture = next(driver for driver in result["drivers"] if driver["id"] == "moisture")
    assert fire["evidence_score"] is None
    assert moisture["evidence_score"] is None
    assert fire["direction"] == "unresolved"
    assert "does not identify" in fire["evidence"]


def test_sensitivity_is_zero_without_inputs_and_exposes_assumptions():
    region, rows = demo_regions()[1]
    analysis = analyze_region(region, rows)
    inputs = {"fire_reduction": 0, "pressure_reduction": 0, "restoration": 0, "climate_stress": 0}
    neutral = sensitivity_scenario(analysis, inputs)
    assert neutral["baseline"] == neutral["scenario"]
    assert neutral["delta"] == 0
    assert neutral["calibrated"] is False
    improving = sensitivity_scenario(analysis, {**inputs, "restoration": 50})
    worsening = sensitivity_scenario(analysis, {**inputs, "climate_stress": 50})
    assert improving["delta"] > 0 > worsening["delta"]
    assert "not estimated intervention" in improving["caveat"]


def test_flat_observations_do_not_generate_spurious_autocorrelation():
    region, rows = demo_regions()[0]
    for row in rows:
        row.update(ndvi=0.8, ndmi=0.5, nbr=0.7)
    result = analyze_region(region, rows)
    assert result["resilience"]["ar1"] is None
    assert result["resilience"]["variance"] == pytest.approx(0)

