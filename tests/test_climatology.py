"""Controlled fixtures verify calculations/contracts, never climate skill or real labels."""
from calendar import monthrange
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
import math
import statistics
from urllib.parse import urlencode

import pytest

from forestmind.monitoring import climatology as cm
from forestmind.monitoring.climate import SOURCE, VARIABLES
from scripts.climate_diagnostics import main


def context(start, end, *, temperature=None):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    dates = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    daily = {"time": [day.isoformat() for day in dates]}
    daily["temperature_2m_mean"] = [21. + day.year - 1991 if temperature is None else temperature for day in dates]
    daily["precipitation_sum"] = [(10. + day.year - 1991) / monthrange(day.year, day.month)[1] for day in dates]
    daily["et0_fao_evapotranspiration"] = [5. / monthrange(day.year, day.month)[1] for day in dates]
    daily["soil_moisture_0_to_7cm_mean"] = [.1 + .01 * (day.year - 1991) for day in dates]
    response = {"longitude": 76.5, "latitude": 11.5, "elevation": 800.,
                "utc_offset_seconds": 0, "timezone": "GMT", "daily": daily,
                "daily_units": {"time": "iso8601", **VARIABLES}}
    encoded = json.dumps(response).encode()
    query = {"longitude": 76.53, "latitude": 11.59, "start_date": start, "end_date": end,
             "models": "era5", "timezone": "GMT", "daily": ",".join(VARIABLES)}
    return {"available": True, "synthetic": False, "provider": "Open-Meteo", "dataset": "ERA5",
            "source_url": SOURCE, "request_url": "https://archive-api.open-meteo.com/v1/archive?" + urlencode(query),
            "start": start, "end": end, "units": VARIABLES.copy(), "raw_response": response,
            "response_sha256": hashlib.sha256(encoded).hexdigest(),
            "retrieved_at": "2026-11-03T00:00:00+00:00", "location_requested": [76.53, 11.59],
            "grid_location": [76.5, 11.5], "license": "CONTROLLED SOFTWARE TEST FIXTURE; not real source evidence",
            "spatial_resolution": "ERA5 0.25 degrees"}


@pytest.fixture(scope="module")
def histories():
    return [context(f"{year}-01-01", f"{year + 9}-12-31") for year in (1991, 2001, 2011)]


@pytest.fixture(scope="module")
def reference(histories):
    return cm.freeze_climatology(histories, baseline_start="1991-01-01", baseline_end="2020-12-31")


def test_frozen_reference_matches_analytic_monthly_statistics_without_mutation(histories, reference):
    before = deepcopy(histories)
    repeated = cm.freeze_climatology(list(reversed(histories)), baseline_start="1991-01-01", baseline_end="2020-12-31")
    assert repeated == reference and histories == before
    assert reference["available_feature_months"] == 48
    january = reference["months"][0]["variables"]
    assert january["temperature_mean_c"]["mean"] == 35.5
    assert january["temperature_mean_c"]["sample_standard_deviation"] == pytest.approx(statistics.stdev(range(21, 51)))
    assert january["temperature_mean_c"]["p90"] == pytest.approx(47.1)
    assert january["precipitation_sum_mm"]["mean"] == pytest.approx(24.5)
    assert january["precipitation_minus_reference_et0_mm"]["mean"] == pytest.approx(19.5)
    assert january["soil_moisture_mean_m3_m3"]["mean"] == pytest.approx(.245)
    assert january["temperature_mean_c"]["samples"] == [{"year": y, "value": float(y - 1970)} for y in range(1991, 2021)]
    assert reference["sources"][0]["start"] == "1991-01-01"
    assert len(reference["sources"][0]["raw_response_sha256"]) == 64
    cm.validate_climatology(json.loads(json.dumps(reference)))


def test_apply_uses_frozen_reference_not_observed_fitting(reference):
    before = deepcopy(reference)
    observation = context("2021-01-01", "2021-01-31", temperature=60.)
    report = cm.analyze_climate([observation], reference, cutoff="2021-01-31")
    january = report["months"][0]
    measured = january["variables"]["temperature_mean_c"]
    assert measured["value"] == 60. and measured["departure"] == 24.5
    assert measured["standardized_departure"] == pytest.approx(24.5 / statistics.stdev(range(21, 51)))
    assert january["warm_month"]["above_reference_p90"] is True
    observation["raw_response"]["daily"]["temperature_2m_mean"] = [0.] * 31
    other = cm.analyze_climate([observation], reference, cutoff="2021-01-31")
    assert other["months"][0]["warm_month"]["above_reference_p90"] is False
    assert other["months"][0]["variables"]["temperature_mean_c"]["reference_mean"] == 35.5
    assert reference == before and report["reference_sha256"] == reference["artifact_sha256"]
    assert not report["model_input"] and not report["historical_availability_verified"]


def test_incomplete_reference_is_feature_specific_and_not_extrapolated(histories):
    incomplete = deepcopy(histories)
    first = incomplete[0]["raw_response"]["daily"]
    first["temperature_2m_mean"][0] = None
    frozen = cm.freeze_climatology(incomplete, baseline_start="1991-01-01", baseline_end="2020-12-31")
    january = frozen["months"][0]["variables"]
    assert january["temperature_mean_c"]["complete_year_count"] == 29
    assert january["temperature_mean_c"]["missing_years"] == [1991]
    assert january["temperature_mean_c"]["mean"] is None
    assert january["precipitation_sum_mm"]["available"]
    assert frozen["available_feature_months"] == 47
    result = cm.analyze_climate([context("2021-01-01", "2021-01-31")], frozen, cutoff="2021-01-31")
    assert result["months"][0]["variables"]["temperature_mean_c"]["standardized_departure"] is None
    assert "29 available" in result["months"][0]["warm_month"]["reason"]
    assert result["months"][0]["variables"]["precipitation_sum_mm"]["available"]


def test_missing_source_years_remain_unavailable_and_short_window_is_rejected():
    source = context("2020-01-01", "2020-12-31")
    frozen = cm.freeze_climatology([source], baseline_start="1991-01-01", baseline_end="2020-12-31")
    assert frozen["available_feature_months"] == 0
    assert frozen["months"][0]["variables"]["temperature_mean_c"]["complete_year_count"] == 1
    with pytest.raises(ValueError, match="30–60"):
        cm.freeze_climatology([source], baseline_start="2020-01-01", baseline_end="2020-12-31")


def test_partial_target_month_missing_days_and_feature_masks(reference):
    target = context("2024-01-01", "2024-02-28")
    target["raw_response"]["daily"]["temperature_2m_mean"][0] = None
    report = cm.analyze_climate([target], reference, cutoff="2024-02-28")
    january, february = report["months"]
    assert january["variables"]["temperature_mean_c"]["missing_measurement_days"] == 1
    assert january["variables"]["temperature_mean_c"]["value"] is None
    assert january["variables"]["precipitation_sum_mm"]["available"]
    assert all(not value["available"] for value in february["variables"].values())
    assert february["variables"]["precipitation_sum_mm"]["expected_days"] == 29
    assert february["variables"]["precipitation_sum_mm"]["unreported_days"] == 1
    assert february["warm_month"]["above_reference_p90"] is None


def test_complete_leap_month_and_wholly_missing_month(reference):
    february = context("2024-02-01", "2024-02-29")
    result = cm.analyze_climate([february], reference, cutoff="2024-02-29")
    rain = result["months"][0]["variables"]["precipitation_sum_mm"]
    assert rain["available"] and rain["valid_days"] == rain["expected_days"] == 29
    assert rain["value"] == pytest.approx(43.)
    gapped = cm.analyze_climate([context("2024-01-01", "2024-01-31"), context("2024-03-01", "2024-03-31")],
                                reference, cutoff="2024-03-31")
    missing = gapped["months"][1]["variables"]["precipitation_sum_mm"]
    assert missing["value"] is None and missing["unreported_days"] == 29 and missing["valid_days"] == 0


def test_genuine_zero_rain_and_missing_pair_are_distinct(reference):
    target = context("2021-01-01", "2021-01-31")
    target["raw_response"]["daily"]["precipitation_sum"] = [0.] * 31
    target["raw_response"]["daily"]["et0_fao_evapotranspiration"][3] = None
    result = cm.analyze_climate([target], reference, cutoff="2021-01-31")["months"][0]["variables"]
    assert result["precipitation_sum_mm"]["available"] and result["precipitation_sum_mm"]["value"] == 0.
    assert result["precipitation_minus_reference_et0_mm"]["value"] is None
    assert result["precipitation_minus_reference_et0_mm"]["valid_days"] == 30


def test_zero_reference_variance_withholds_only_standardization(histories):
    constant = deepcopy(histories)
    for source in constant:
        daily = source["raw_response"]["daily"]
        daily["temperature_2m_mean"] = [20.] * len(daily["time"])
    frozen = cm.freeze_climatology(constant, baseline_start="1991-01-01", baseline_end="2020-12-31")
    report = cm.analyze_climate([context("2021-01-01", "2021-01-31", temperature=20.)], frozen, cutoff="2021-01-31")
    value = report["months"][0]["variables"]["temperature_mean_c"]
    assert value["available"] and value["departure"] == 0. and value["standardized_departure"] is None
    assert "zero" in value["standardized_reason"]
    assert report["months"][0]["warm_month"]["above_reference_p90"] is False  # strict exceedance


@pytest.mark.parametrize("start,end,cutoff,error", [
    ("2020-12-31", "2021-01-31", "2021-01-31", "after the frozen"),
    ("2021-01-01", "2021-02-01", "2021-01-31", "after the explicit cutoff"),
    ("2021-01-01", "2021-01-31", "2032-01-31", "at most ten years"),
])
def test_strict_reference_and_forecast_cutoffs(reference, start, end, cutoff, error):
    with pytest.raises(ValueError, match=error):
        cm.analyze_climate([context(start, end)], reference, cutoff=cutoff)


def test_reference_rejects_overlap_and_outside_reference_context(histories):
    with pytest.raises(ValueError, match="overlap"):
        cm.freeze_climatology([histories[0], histories[0]], baseline_start="1991-01-01", baseline_end="2020-12-31")
    with pytest.raises(ValueError, match="inside the declared"):
        cm.freeze_climatology([context("2021-01-01", "2021-01-31")], baseline_start="1991-01-01", baseline_end="2020-12-31")


@pytest.mark.parametrize("change,error", [
    (lambda c: c.update(synthetic=True), "Synthetic"),
    (lambda c: c.update(dataset="ERA5-Land"), "source, dataset and units"),
    (lambda c: c["units"].update(temperature_2m_mean="K"), "source, dataset and units"),
    (lambda c: c["raw_response"]["daily_units"].update(temperature_2m_mean="K"), "Raw daily units"),
    (lambda c: c["raw_response"].update(utc_offset_seconds=19800), "UTC/GMT"),
    (lambda c: c.update(grid_location=[77., 11.5]), "grid location differ"),
    (lambda c: c.update(location_requested=[77., 11.59]), "location or variables"),
    (lambda c: c.update(response_sha256="unknown"), "SHA256"),
    (lambda c: c.update(retrieved_at="2026-11-03"), "timezone-aware"),
    (lambda c: c.update(request_url="https://example.org/archive"), "fixed historical"),
    (lambda c: c["raw_response"]["daily"]["time"].__setitem__(1, "2021-01-01"), "unique and strictly"),
    (lambda c: c["raw_response"]["daily"]["time"].__setitem__(0, "2020-12-31"), "outside the declared"),
    (lambda c: c["raw_response"]["daily"]["precipitation_sum"].__setitem__(0, True), "finite number"),
    (lambda c: c["raw_response"]["daily"]["precipitation_sum"].__setitem__(0, math.nan), "finite number"),
    (lambda c: c["raw_response"]["daily"]["precipitation_sum"].__setitem__(0, -1.), "cannot be negative"),
    (lambda c: c["raw_response"]["daily"]["soil_moisture_0_to_7cm_mean"].__setitem__(0, 2.), "soil moisture"),
    (lambda c: c["raw_response"]["daily"]["precipitation_sum"].pop(), "counts disagree"),
])
def test_corrupt_source_contracts_rejected_before_calculation(reference, change, error):
    target = context("2021-01-01", "2021-01-31")
    change(target)
    with pytest.raises(ValueError, match=error):
        cm.analyze_climate([target], reference, cutoff="2021-01-31")


def test_same_grid_and_source_required_even_when_context_is_individually_valid(reference, histories):
    target = context("2021-01-01", "2021-01-31")
    target["grid_location"][0] = target["raw_response"]["longitude"] = 76.75
    with pytest.raises(ValueError, match="different source, location, grid"):
        cm.analyze_climate([target], reference, cutoff="2021-01-31")
    mixed = deepcopy(histories)
    mixed[-1]["license"] = "Different documented source rights"
    with pytest.raises(ValueError, match="exactly the same"):
        cm.freeze_climatology(mixed, baseline_start="1991-01-01", baseline_end="2020-12-31")


def test_reference_integrity_and_recomputed_statistics(reference):
    changed = deepcopy(reference)
    changed["months"][0]["variables"]["temperature_mean_c"]["mean"] = 100.
    with pytest.raises(ValueError, match="hash mismatch"):
        cm.validate_climatology(changed)
    changed["artifact_sha256"] = cm.artifact_hash(changed)
    with pytest.raises(ValueError, match="statistics disagree"):
        cm.validate_climatology(changed)
    changed = deepcopy(reference)
    changed["months"][0]["variables"]["temperature_mean_c"]["samples"][0]["year"] = 2021
    changed["artifact_sha256"] = cm.artifact_hash(changed)
    with pytest.raises(ValueError, match="outside the declared"):
        cm.validate_climatology(changed)
    changed = deepcopy(reference)
    changed["model_input"] = True
    changed["artifact_sha256"] = cm.artifact_hash(changed)
    with pytest.raises(ValueError, match="implicit neural"):
        cm.validate_climatology(changed)


@pytest.mark.parametrize("change,error", [
    (lambda r: r.update(schema_version=True), "Unsupported"),
    (lambda r: r.update(baseline=None), "explicit start and end"),
    (lambda r: r.update(source_contract=None), "source contract"),
    (lambda r: r.update(sources=[]), "provenance list"),
    (lambda r: r["sources"][0].update(end="2021-01-01"), "source windows"),
    (lambda r: r["months"].__setitem__(0, None), "month/feature"),
    (lambda r: r["months"][0]["variables"].update(temperature_mean_c=None), "statistics must contain"),
    (lambda r: r["policy"].update(minimum_reference_years_per_calendar_month_and_feature=5), "completeness policy"),
])
def test_malformed_reference_rejected_even_if_rehashed(reference, change, error):
    changed = deepcopy(reference)
    change(changed)
    changed["artifact_sha256"] = cm.artifact_hash(changed)
    with pytest.raises(ValueError, match=error):
        cm.validate_climatology(changed)


def test_artifact_roundtrip_and_exclusive_output(reference, tmp_path):
    output = tmp_path / "reference.json"
    cm.write_new_artifact(reference, output)
    original = output.read_bytes()
    restored = cm.read_json(output)
    assert restored == reference
    cm.validate_climatology(restored)
    with pytest.raises(FileExistsError):
        cm.write_new_artifact(reference, output)
    assert output.read_bytes() == original


@pytest.mark.parametrize("value,error", [('{"a":1,"a":2}', "Duplicate JSON"), ('{"a":NaN}', "Nonfinite"), ('[]', "one object")])
def test_strict_json_input(value, error, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(value, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        cm.read_json(path)


def test_json_read_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "MAX_JSON_BYTES", 32)
    path = tmp_path / "large.json"
    path.write_bytes(b" " * 33)
    with pytest.raises(ValueError, match="limit"):
        cm.read_json(path)


def test_cli_freeze_analyze_preserve_sources_and_refuse_overwrites(histories, tmp_path):
    inputs, arguments = [], []
    for i, source in enumerate(histories):
        path = tmp_path / f"reference-source-{i}.json"
        path.write_text(json.dumps(source), encoding="utf-8")
        inputs.append(path)
        arguments += ["--context", str(path)]
    original_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs]
    frozen = tmp_path / "frozen.json"
    command = ["freeze", *arguments, "--baseline-start", "1991-01-01", "--baseline-end", "2020-12-31", "--output", str(frozen)]
    assert main(command) == 0
    reference_bytes = frozen.read_bytes()
    assert main(command) == 2 and frozen.read_bytes() == reference_bytes
    observed = tmp_path / "selected-area.json"
    observed.write_text(json.dumps({"kind": "forestmind_selected_area_analysis", "climate": context("2021-01-01", "2021-01-31")}), encoding="utf-8")
    output = tmp_path / "departures.json"
    assert main(["analyze", "--reference", str(frozen), "--context", str(observed), "--cutoff", "2021-01-31", "--output", str(output)]) == 0
    result = cm.read_json(output)
    assert result["months"][0]["variables"]["temperature_mean_c"]["available"]
    assert result["input_files"][0]["file_sha256"] == hashlib.sha256(observed.read_bytes()).hexdigest()
    assert result["reference_sha256"] == cm.read_json(frozen)["artifact_sha256"]
    assert original_hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs]
    assert frozen.read_bytes() == reference_bytes


def test_cli_invalid_inputs_do_not_create_output(reference, tmp_path):
    source = tmp_path / "too-short.json"
    source.write_text(json.dumps(context("2020-01-01", "2020-01-31")), encoding="utf-8")
    output = tmp_path / "invalid.json"
    assert main(["freeze", "--context", str(source), "--baseline-start", "2020-01-01", "--baseline-end", "2020-12-31", "--output", str(output)]) == 2
    assert not output.exists()
