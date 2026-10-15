"""Coverage reports preserve gaps and use the existing method readiness gates."""

from copy import deepcopy
import math

import pytest
from pydantic import ValidationError

from forestmind.monitoring.analysis import forecast_summary, indicator_score, resilience_summary
from forestmind.monitoring.coverage import assess_history


def payload_for(dates, *, synthetic=False):
    return {
        "name": "Coverage test AOI", "country": "Brazil", "source": "Prepared test measurements",
        "synthetic": synthetic,
        "geometry": {"type": "Polygon", "coordinates": [
            [[-55.1, -3.5], [-55.08, -3.5], [-55.08, -3.48], [-55.1, -3.48], [-55.1, -3.5]]
        ]},
        "observations": [
            {"date": day, "ndvi": 0.7 + 0.04 * math.sin(index), "ndmi": 0.4,
             "nbr": 0.6, "valid_fraction": 0.9}
            for index, day in enumerate(dates)
        ],
    }


def monthly_dates(length, start_year=2020):
    return [f"{start_year + index // 12:04d}-{index % 12 + 1:02d}-01" for index in range(length)]


def test_classifies_missing_quality_and_index_gaps_without_source_mutation():
    payload = payload_for(["2024-01-01", "2024-02-01", "2024-04-01", "2024-05-01"])
    payload["observations"][1].update(valid_fraction=0.3999, ndvi=None)
    payload["observations"][2]["ndmi"] = None
    payload["observations"][3]["valid_fraction"] = 0.4
    original = deepcopy(payload)
    report = assess_history(payload)
    assert payload == original
    assert [row["state"] for row in report["months"]] == [
        "usable", "low_coverage", "missing", "missing_indices", "usable"
    ]
    assert report["months"][2] == {
        "month": "2024-03", "state": "missing", "date": None, "valid_fraction": None,
    }
    assert report["counts"] == {
        "expected_months": 5, "total_observations": 4, "observations": 4,
        "excluded_observations": 0, "usable": 2, "low_coverage": 1,
        "missing_indices": 1, "missing": 1,
    }
    assert report["longest_usable_run"] == report["trailing_usable_run"] == 1
    assert "ndvi" not in report["months"][0]


@pytest.mark.parametrize("synthetic", [True, False])
def test_preserves_source_and_synthetic_provenance(synthetic):
    payload = payload_for(["2024-02-29"], synthetic=synthetic)
    report = assess_history(payload)
    assert report["synthetic"] is synthetic
    assert report["source"] == payload["source"]
    assert report["name"] == payload["name"]
    assert "software minimums" in report["caveat"]
    assert "does not establish predictive validity" in report["caveat"]


def test_exact_date_bounds_exclude_rows_but_keep_intersecting_partial_months():
    payload = payload_for(["2024-01-31", "2024-02-29", "2024-03-31"])
    report = assess_history(payload, start="2024-02-15", end="2024-03-15")
    assert report["input_span"] == {"start": "2024-01-31", "end": "2024-03-31"}
    assert report["observed_span"] == {"start": "2024-02-29", "end": "2024-02-29"}
    assert report["requested_span"] == {"start": "2024-02-15", "end": "2024-03-15"}
    assert report["counts"]["excluded_observations"] == 2
    assert [row["state"] for row in report["months"]] == ["usable", "missing"]
    assert report["trailing_usable_run"] == 0


@pytest.mark.parametrize("start,end", [("2023-01-01", "2023-02-28"), ("2024-02-01", "2024-02-29")])
def test_empty_selected_window_has_no_readiness_or_fabricated_observations(start, end):
    report = assess_history(payload_for(["2024-01-01"]), start=start, end=end)
    assert report["observed_span"] == {"start": None, "end": None}
    assert report["counts"]["observations"] == 0
    assert report["counts"]["missing"] == report["counts"]["expected_months"]
    assert report["baseline_forecast_available"] is False
    assert report["resilience_available"] is False
    assert report["trailing_usable_run"] == report["longest_usable_run"] == 0


def test_month_runs_reset_on_missing_or_unusable_records():
    payload = payload_for(monthly_dates(14))
    payload["observations"].pop(7)
    payload["observations"][3]["valid_fraction"] = 0.2
    report = assess_history(payload)
    assert report["longest_usable_run"] == 6
    assert report["trailing_usable_run"] == 6
    assert report["baseline_forecast_available"] is False


@pytest.mark.parametrize("length,forecast_ready,resilience_ready", [
    (1, False, False), (11, False, False), (12, True, False),
    (35, True, False), (36, True, True), (48, True, True),
])
def test_readiness_agrees_with_existing_methods(length, forecast_ready, resilience_ready):
    payload = payload_for(monthly_dates(length))
    report = assess_history(payload)
    scored = [{**row, "health_score": indicator_score(row)} for row in payload["observations"]]
    assert report["baseline_forecast_available"] == bool(forecast_summary(scored)["points"]) == forecast_ready
    assert report["resilience_available"] == (resilience_summary(scored)["status"] != "insufficient_data") == resilience_ready
    assert report["baseline_forecast_ready_at_requested_end"] == forecast_ready
    assert report["resilience_ready_at_requested_end"] == resilience_ready


def test_stale_ending_month_does_not_present_historical_readiness_as_current():
    payload = payload_for(monthly_dates(36))
    report = assess_history(payload, end="2023-02-28")
    assert report["baseline_forecast_available"] is True
    assert report["resilience_available"] is True
    assert report["readiness_through"] == {"baseline_forecast": "2022-12-01", "resilience": "2022-12-01"}
    assert report["baseline_forecast_ready_at_requested_end"] is False
    assert report["resilience_ready_at_requested_end"] is False
    assert report["requested_end_month_usable"] is False
    assert report["longest_usable_run"] == 36
    assert report["trailing_usable_run"] == 0
    assert report["counts"]["missing"] == 2


def test_existing_resilience_gate_handles_unusable_latest_without_overstating_current_readiness():
    payload = payload_for(monthly_dates(37))
    payload["observations"][-1]["ndvi"] = None
    report = assess_history(payload)
    assert report["resilience_available"] is True
    assert report["readiness_through"]["resilience"] == "2022-12-01"
    assert report["baseline_forecast_available"] is False
    assert report["resilience_ready_at_requested_end"] is False


def test_resilience_does_not_ignore_older_gaps_when_long_recent_run_exists():
    payload = payload_for(monthly_dates(49))
    payload["observations"].pop(1)
    report = assess_history(payload)
    assert report["trailing_usable_run"] == 47
    assert report["baseline_forecast_available"] is True
    assert report["resilience_available"] is False


def test_readiness_uses_only_included_history():
    report = assess_history(payload_for(monthly_dates(48)), start="2023-02-01")
    assert report["counts"]["observations"] == 11
    assert report["baseline_forecast_available"] is False
    assert report["resilience_available"] is False


@pytest.mark.parametrize("bounds", [
    {"start": "20240201"}, {"end": "2023-02-29"}, {"start": "2024-1-01"},
    {"start": "2024-02-02", "end": "2024-02-01"}, {"end": 2024},
    {"start": "0001-01-01", "end": "9999-12-31"},
])
def test_invalid_or_excessive_bounds_raise_clear_error(bounds):
    with pytest.raises(ValueError, match="YYYY-MM-DD|on or after|1200 calendar months"):
        assess_history(payload_for(["2024-02-01"]), **bounds)


def test_maximum_span_and_last_supported_calendar_month():
    report = assess_history(payload_for(["9999-12-31"]), start="9900-01-01")
    assert report["counts"]["expected_months"] == 1200
    assert report["months"][-1]["month"] == "9999-12"


def test_strict_import_schema_is_applied_before_analysis():
    payload = payload_for(["2024-01-01"])
    payload["observations"][0]["ndvi"] = "0.6"
    with pytest.raises(ValidationError):
        assess_history(payload)
    payload = payload_for(["2024-01-01", "2024-01-20"])
    with pytest.raises(ValidationError, match="at most one"):
        assess_history(payload)
