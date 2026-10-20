"""Paired climate diagnostics retain dates, nulls and source evidence."""
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json

import pytest

from forestmind.monitoring import climate
from test_worldwide_web import _workflow


def daily_response(dates, rain, et0):
    return {"daily_units": climate.VARIABLES.copy(), "daily": {
        "time": dates, "precipitation_sum": rain, "et0_fao_evapotranspiration": et0,
        "temperature_2m_mean": [20.] * len(dates), "soil_moisture_0_to_7cm_mean": [.2] * len(dates),
    }}


def test_balance_uses_only_same_day_pairs_and_does_not_subtract_unpaired_totals():
    response = daily_response(["2024-02-01", "2024-02-02", "2024-02-03", "2024-02-04"],
                              [10., None, 5., 0.], [None, 7., 2., 0.])
    original = deepcopy(response)
    row = climate.aggregate_climate(response, start="2024-02-01", end="2024-02-04")[0]
    balance = row["water_balance"]
    assert row["precipitation_sum"] - row["et0_fao_evapotranspiration"] == 6.
    assert balance["sum_mm"] == 3. and balance["mean_daily_mm"] == 1.5
    assert balance["paired_dates"] == ["2024-02-03", "2024-02-04"]
    assert balance["paired_days"] == balance["missing_pair_days"] == 2
    assert balance["units"] == "mm" and balance["mean_daily_units"] == "mm/day"
    assert not balance["complete_requested_period"] and balance["partial_calendar_month"]
    assert response == original


def test_disjoint_valid_days_do_not_produce_zero_balance():
    row = climate.aggregate_climate(daily_response(["2024-01-01", "2024-01-02"], [5., None], [None, 5.]),
                                    start="2024-01-01", end="2024-01-02")[0]
    assert row["precipitation_sum"] == row["et0_fao_evapotranspiration"] == 5.
    assert row["water_balance"]["sum_mm"] is None
    assert row["water_balance"]["mean_daily_mm"] is None
    assert row["water_balance"]["paired_days"] == 0


def test_leap_month_complete_and_requested_partial_periods_are_distinct():
    dates = [(date(2024, 2, 1) + timedelta(days=i)).isoformat() for i in range(29)]
    full = climate.aggregate_climate(daily_response(dates, [1.] * 29, [2.] * 29),
                                    start=dates[0], end=dates[-1])[0]["water_balance"]
    assert full["sum_mm"] == -29. and full["expected_days"] == 29
    assert full["complete_calendar_month"] and full["complete_requested_period"]
    partial = climate.aggregate_climate(daily_response(dates[-2:], [1., 1.], [2., 2.]),
                                       start=dates[-2], end=dates[-1])[0]["water_balance"]
    assert partial["expected_days"] == 2 and partial["sum_mm"] == -2.
    assert partial["complete_requested_period"] and not partial["complete_calendar_month"]
    assert partial["partial_calendar_month"]


def test_unreported_days_and_absent_months_stay_missing():
    rows = climate.aggregate_climate(daily_response(["2024-01-31", "2024-03-01"], [1., 1.], [3., 3.]),
                                     start="2024-01-31", end="2024-03-02")
    assert [r["month"] for r in rows] == ["2024-01", "2024-02", "2024-03"]
    february = rows[1]
    assert february["reported_days"] == 0 and february["unreported_days"] == 29
    assert february["precipitation_sum"] is None
    assert february["quality"]["precipitation_sum"]["missing_requested_days"] == 29
    assert february["water_balance"]["sum_mm"] is None
    assert february["water_balance"]["missing_pair_days"] == 29
    assert not february["water_balance"]["complete_calendar_month"]
    assert rows[2]["water_balance"]["missing_pair_days"] == 1
    assert rows[2]["quality"]["precipitation_sum"]["missing_requested_days"] == 1


def test_saved_context_derivation_preserves_source_and_is_idempotent():
    context = {"available": True, "start": "2024-02-01", "end": "2024-02-02",
               "retrieved_at": "2026-10-01T00:00:00+00:00", "response_sha256": "recorded-raw-hash",
               "geometry_sha256": "fixed-area", "monthly": [{"old": "summary"}],
               "raw_response": daily_response(["2024-02-01", "2024-02-02"], [2., 2.], [3., None])}
    original = deepcopy(context)
    derived = climate.refresh_climate_diagnostics(context)
    assert context == original and derived is not context
    for key in ("start", "end", "retrieved_at", "response_sha256", "geometry_sha256", "raw_response"):
        assert derived[key] == original[key]
    assert derived["monthly"][0]["water_balance"]["sum_mm"] == -1.
    assert derived["diagnostics"]["water_balance"]["model_input"] is False
    assert climate.refresh_climate_diagnostics(derived) == derived
    legacy = {"available": True, "monthly": [{"precipitation_sum": 10., "et0_fao_evapotranspiration": 5.}]}
    assert climate.refresh_climate_diagnostics(legacy) == legacy  # no daily pairing evidence


@pytest.mark.parametrize("rain,et0", [([True], [1.]), ([1.], [float("nan")]), ([float("inf")], [1.])])
def test_invalid_raw_measurements_are_not_used_for_diagnostics(rain, et0):
    with pytest.raises(ValueError, match="nonnumeric"):
        climate.aggregate_climate(daily_response(["2024-01-01"], rain, et0),
                                  start="2024-01-01", end="2024-01-01")


def test_acquisition_retains_raw_hash_and_adds_context_diagnostic(monkeypatch):
    response = daily_response(["2024-01-01"], [3.], [1.])
    raw = json.dumps(response).encode()

    class Stream:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return raw

    class Opener:
        def open(self, request, timeout):
            assert request.full_url.startswith("https://archive-api.open-meteo.com/v1/archive?")
            assert "models=era5" in request.full_url
            return Stream()

    monkeypatch.setattr(climate, "build_opener", lambda *_: Opener())
    geometry = {"coordinates": [[[0., 0.], [.01, 0.], [.01, .01], [0., .01], [0., 0.]]]}
    result = climate.acquire_climate(geometry, start="2024-01-01", end="2024-01-01")
    assert result["raw_response"] == response
    assert result["response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["monthly"][0]["water_balance"]["sum_mm"] == 2.
    assert result["model_input"] is False


def test_climate_ui_preserves_missing_balance_and_coverage_interpretation():
    row = climate.aggregate_climate(daily_response(["2024-02-28", "2024-02-29"], [1., 1.], [2., 2.]),
                                    start="2024-02-28", end="2024-02-29")[0]
    output = _workflow("renderClimateContext(" + json.dumps({"available": True, "start": "2024-02-28",
                       "end": "2024-02-29", "monthly": [row]}) + ")")
    assert "Paired P − ET₀ (mm)" in output and "2 / 2 paired days · partial month" in output
    assert "not measured forest water use or a validated drought index" in output
    missing = _workflow("climateBalanceMarkup({water_balance:{sum_mm:null,paired_days:0,expected_days:29}})")
    assert "—" in missing and "0 / 29 paired days" in missing and "partial sum; missing days" in missing
