"""Leakage, date alignment and metric contracts for monitoring backtests."""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from forestmind.monitoring.analysis import forecast_summary, indicator_score
from forestmind.monitoring.evaluation import backtest_prepared


def prepared_payload(n=36, *, seasonal=False, synthetic=False):
    rows = []
    for index in range(n):
        year, month = divmod(2020 * 12 + index, 12)
        fraction = 0.5 + (0.2 * math.sin(month / 12 * 2 * math.pi) if seasonal else 0)
        rows.append({
            "date": f"{year:04d}-{month + 1:02d}-01", "valid_fraction": 0.9,
            "ndvi": 0.2 + fraction * 0.7, "ndmi": -0.1 + fraction * 0.7,
            "nbr": fraction * 0.8,
        })
    return {
        "name": "Evaluation fixture", "country": "Brazil", "source": "Prepared test observations",
        "synthetic": synthetic, "notes": "Unit-test fixture; not field measurements",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        "observations": rows,
    }


def rows_for(result, *, cutoff=None, horizon=None, method=None):
    return [row for row in result["records"]
            if (cutoff is None or row["cutoff_date"] == cutoff)
            and (horizon is None or row["horizon_months"] == horizon)
            and (method is None or row["method"] == method)]


def test_constant_series_produces_known_counts_and_zero_errors():
    result = backtest_prepared(prepared_payload())
    for method in result["methods"].values():
        assert [method["by_horizon"][key]["n"] for key in ("1", "6", "12")] == [24, 19, 13]
        assert method["aggregate"]["n"] == 56
        assert method["aggregate"]["mae"] == 0
        assert method["aggregate"]["rmse"] == 0
    assert result["methods"]["persistence"]["aggregate"]["coverage"] is None
    assert result["methods"]["seasonal_naive"]["aggregate"]["n_bounds"] == 0
    assert result["methods"]["damped_trend"]["aggregate"]["coverage"] == 1
    assert result["methods"]["damped_trend"]["aggregate"]["n_bounds"] == 56
    assert result["protocol"]["bounds_calibrated"] is False
    json.dumps(result, allow_nan=False)


def test_seasonal_naive_uses_previous_year_calendar_month_only():
    result = backtest_prepared(prepared_payload(seasonal=True))
    assert result["methods"]["seasonal_naive"]["aggregate"]["rmse"] == 0
    assert result["methods"]["persistence"]["by_horizon"]["1"]["mae"] > 0
    for row in rows_for(result, method="seasonal_naive"):
        assert row["reference_date"] <= row["cutoff_date"]
        assert row["reference_date"][5:7] == row["target_month"][5:7]
        assert int(row["reference_date"][:4]) + 1 == int(row["target_month"][:4])


def test_future_target_changes_do_not_change_prior_predictions_or_bounds():
    original = prepared_payload(seasonal=True)
    altered = deepcopy(original)
    for row in altered["observations"][18:]:
        row.update(ndvi=0.9, ndmi=0.6, nbr=0.8)
    first = backtest_prepared(original)
    second = backtest_prepared(altered)
    columns = ("cutoff_date", "method", "horizon_months", "prediction", "lower", "upper")
    prior_first = [{key: row[key] for key in columns} for row in first["records"]
                   if row["cutoff_date"] < "2021-07-01"]
    prior_second = [{key: row[key] for key in columns} for row in second["records"]
                    if row["cutoff_date"] < "2021-07-01"]
    assert prior_first == prior_second
    assert first["provenance"]["input_sha256"] != second["provenance"]["input_sha256"]
    assert rows_for(first, cutoff="2020-12-01", horizon=12)[0]["target"] != rows_for(
        second, cutoff="2020-12-01", horizon=12)[0]["target"]


def test_damped_forecast_is_exactly_existing_cutoff_only_implementation():
    payload = prepared_payload(seasonal=True)
    result = backtest_prepared(payload)
    history = [{**row, "health_score": indicator_score(row)} for row in payload["observations"][:12]]
    expected = forecast_summary(history)["points"]
    actual = rows_for(result, method="damped_trend", cutoff="2020-12-01")
    for record, point in zip(actual, expected):
        assert record["prediction"] == point["score"]
        assert record["lower"] == point["lower"]
        assert record["upper"] == point["upper"]
        assert record["nominal_forecast_date"] == point["date"]
        assert record["training_end_date"] == "2020-12-01"


def test_calendar_target_is_actual_dated_month_not_nominal_day_or_index():
    result = backtest_prepared(prepared_payload())
    record = rows_for(result, method="damped_trend", cutoff="2021-01-01", horizon=1)[0]
    assert record["target_date"] == "2021-02-01"
    assert record["nominal_forecast_date"] == "2021-01-31"
    assert record["actual_horizon_days"] == 31
    assert record["nominal_forecast_days"] == 30
    assert record["status"] == "scored"


def test_missing_target_month_is_skipped_without_nearest_observation_substitution():
    payload = prepared_payload()
    payload["observations"].pop(12)  # January 2021 is absent; February must not replace it.
    result = backtest_prepared(payload)
    for record in rows_for(result, cutoff="2020-12-01", horizon=1):
        assert record["target_month"] == "2021-01"
        assert record["target_date"] is None
        assert record["target"] is None
        assert record["status"] == "missing_target"
    for method in result["methods"].values():
        assert method["by_horizon"]["1"]["skipped_missing_target"] == 1
    resumed = [row for row in result["cutoffs"] if row["cutoff_date"] == "2022-01-01"][0]
    assert resumed["eligible"]  # Twelve consecutive usable observations after the gap.


def test_low_coverage_target_is_not_scored_and_invalidates_subsequent_origins():
    payload = prepared_payload()
    payload["observations"][12]["valid_fraction"] = 0.1
    result = backtest_prepared(payload)
    for record in rows_for(result, cutoff="2020-12-01", horizon=1):
        assert record["status"] == "unusable_target"
        assert record["target_date"] == "2021-01-01"
        assert record["target"] is None
    invalid = [row for row in result["cutoffs"] if "2021-01-01" <= row["cutoff_date"] <= "2021-12-01"]
    assert all(not row["eligible"] for row in invalid)


@pytest.mark.parametrize("n", [1, 11, 12])
def test_insufficient_future_or_history_returns_null_metrics_not_zero_error(n):
    result = backtest_prepared(prepared_payload(n))
    for method in result["methods"].values():
        assert method["aggregate"]["n"] == 0
        assert method["aggregate"]["mae"] is None
        assert method["aggregate"]["rmse"] is None
        assert method["aggregate"]["coverage"] is None


def test_pooled_errors_weight_actual_sample_counts():
    result = backtest_prepared(prepared_payload(seasonal=True))
    scores = result["methods"]["persistence"]
    horizons = list(scores["by_horizon"].values())
    expected_mae = sum(row["mae"] * row["n"] for row in horizons) / sum(row["n"] for row in horizons)
    expected_rmse = math.sqrt(sum(row["rmse"] ** 2 * row["n"] for row in horizons)
                              / sum(row["n"] for row in horizons))
    assert scores["aggregate"]["mae"] == pytest.approx(expected_mae)
    assert scores["aggregate"]["rmse"] == pytest.approx(expected_rmse)


def test_provenance_and_synthetic_warning_are_preserved_without_validation_claims():
    payload = prepared_payload(synthetic=True)
    original = deepcopy(payload)
    result = backtest_prepared(payload)
    assert payload == original
    assert result["provenance"]["synthetic"] is True
    assert result["provenance"]["verified"] is False
    assert result["provenance"]["source"] == payload["source"]
    assert result["provenance"]["reference"]["independent_field_validation"] is False
    assert "does not establish real-world" in result["synthetic_warning"]
    assert result == backtest_prepared(payload)


@pytest.mark.parametrize("horizons", [[], [6, 1], [1, 1], [True], [3], [1.0]])
def test_bad_horizons_fail_explicitly(horizons):
    with pytest.raises(ValueError, match="horizons_months"):
        backtest_prepared(prepared_payload(), horizons_months=horizons)


def test_invalid_or_duplicate_prepared_dates_are_not_silently_reordered():
    payload = prepared_payload()
    payload["observations"][1]["date"] = payload["observations"][0]["date"]
    with pytest.raises(ValueError):
        backtest_prepared(payload)


def test_cli_writes_strict_json_with_file_hash_and_rejects_overwriting_input(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "backtest_monitoring.py"
    specification = importlib.util.spec_from_file_location("monitoring_backtest_cli", script)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    source = tmp_path / "prepared.json"
    output = tmp_path / "backtest.json"
    source.write_text(json.dumps(prepared_payload(synthetic=True)), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script), "--input", str(source), "--output", str(output),
         "--horizons", "1", "12"], capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["protocol"]["horizons_months"] == [1, 12]
    assert len(result["provenance"]["input_file_sha256"]) == 64
    assert result["provenance"]["input_file"] == str(source.resolve())
    assert "SYNTHETIC DATA" in completed.stderr
    with pytest.raises(SystemExit) as exc:
        module.main(["--input", str(source), "--output", str(source)])
    assert exc.value.code == 2
