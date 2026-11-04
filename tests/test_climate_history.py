"""Controlled provider fixtures test acquisition contracts, not climate validity."""
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

import pytest

from forestmind.data import climate_history as ch
from forestmind.monitoring.climate import SOURCE, VARIABLES
from forestmind.monitoring.climatology import artifact_hash
from scripts.build_climate_history import main

BBOX = [76.5238, 11.5889, 76.5438, 11.6089]


def provider(geometry, *, start, end, timeout):
    west, south = geometry["coordinates"][0][0]
    east, north = geometry["coordinates"][0][2]
    lon, lat = (west + east) / 2, (south + north) / 2
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    dates = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    daily = {"time": [d.isoformat() for d in dates]}
    daily.update({"temperature_2m_mean": [20. + .01 * (d.year - 1991) for d in dates],
                  "precipitation_sum": [2. + .01 * (d.year - 1991) for d in dates],
                  "soil_moisture_0_to_7cm_mean": [.3 for d in dates],
                  "et0_fao_evapotranspiration": [1. for d in dates]})
    raw = {"longitude": 76.5, "latitude": 11.5, "elevation": 800., "utc_offset_seconds": 0,
           "timezone": "GMT", "daily": daily, "daily_units": {"time": "iso8601", **VARIABLES}}
    query = {"longitude": lon, "latitude": lat, "start_date": start, "end_date": end,
             "models": "era5", "timezone": "GMT", "daily": ",".join(VARIABLES)}
    return {"available": True, "synthetic": False, "provider": "Open-Meteo", "dataset": "ERA5",
            "source_url": SOURCE, "request_url": "https://archive-api.open-meteo.com/v1/archive?" + urlencode(query),
            "start": start, "end": end, "units": VARIABLES.copy(), "raw_response": raw,
            "response_sha256": hashlib.sha256(json.dumps(raw).encode()).hexdigest(),
            "retrieved_at": "2026-11-04T00:00:00+00:00", "location_requested": [lon, lat],
            "grid_location": [76.5, 11.5], "license": "SOFTWARE TEST FIXTURE; no actual source evidence",
            "spatial_resolution": "ERA5 0.25 degrees"}


@pytest.fixture(autouse=True)
def local_provider(monkeypatch):
    monkeypatch.setattr(ch, "acquire_climate", provider)
    monkeypatch.setattr(ch, "sleep", lambda seconds: None)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_build_complete_and_resume_uses_identical_sources_without_network(tmp_path, monkeypatch):
    output = tmp_path / "build"
    report = ch.build_climate_history(BBOX, output)
    assert report["status"] == "complete" and report["requests_this_invocation"] == 4
    assert report["available_reference_feature_months"] == 48
    assert report["observed_months"] == 60 and report["available_observed_feature_months"] == 240
    before = {p.name: p.read_bytes() for p in output.glob("*.json") if p.name != "progress.json"}
    monkeypatch.setattr(ch, "acquire_climate", lambda *a, **kw: pytest.fail("Cached source redownloaded"))
    resumed = ch.build_climate_history(BBOX, output, resume=True)
    assert resumed["status"] == "complete" and resumed["requests_this_invocation"] == 0
    assert before == {p.name: p.read_bytes() for p in output.glob("*.json") if p.name != "progress.json"}
    assert not (output / ".build.lock").exists()
    with pytest.raises(FileExistsError):
        ch.build_climate_history(BBOX, output)


def test_request_budget_resume_and_source_hash_recheck(tmp_path, monkeypatch):
    output = tmp_path / "build"
    report = ch.build_climate_history(BBOX, output, max_requests=1)
    assert report["accepted_windows"] == 1 and report["status"] == "incomplete"
    assert not (output / "reference.json").exists()
    first = output / report["windows"][0]["file"]
    source = read(first)
    source["raw_response"]["daily"]["temperature_2m_mean"][0] += 1
    source["artifact_sha256"] = artifact_hash(source)
    first.write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(ch, "acquire_climate", lambda *a, **kw: pytest.fail("Network before cache audit"))
    with pytest.raises(ValueError, match="accepted source hash"):
        ch.build_climate_history(BBOX, output, resume=True)


def test_resume_rejects_changed_plan_and_missing_accepted_sources(tmp_path):
    output = tmp_path / "build"
    report = ch.build_climate_history(BBOX, output, max_requests=1)
    with pytest.raises(ValueError, match="frozen plan"):
        ch.build_climate_history(BBOX, output, resume=True, observed_end="2024-12-31")
    (output / report["windows"][0]["file"]).unlink()
    with pytest.raises(ValueError, match="source file is missing"):
        ch.build_climate_history(BBOX, output, resume=True)


def test_retry_budget_is_persisted_across_invocations(tmp_path, monkeypatch):
    calls = []
    def timeout(*args, **kwargs):
        calls.append(kwargs["start"])
        raise URLError("Controlled test timeout")
    monkeypatch.setattr(ch, "acquire_climate", timeout)
    output = tmp_path / "build"
    first = ch.build_climate_history(BBOX, output, max_requests=1)
    assert first["windows"][0]["attempts"] == 1
    second = ch.build_climate_history(BBOX, output, max_requests=1, resume=True)
    assert second["windows"][0]["attempts"] == 2 and second["windows"][0]["status"] == "failed"
    ch.build_climate_history(BBOX, output, max_requests=1, resume=True)
    assert calls == ["1991-01-01", "1991-01-01", "2001-01-01"]


@pytest.mark.parametrize("status", [401, 403, 429])
def test_denial_or_rate_limit_stops_further_requests(tmp_path, monkeypatch, status):
    def denied(*args, **kwargs):
        raise HTTPError("https://archive-api.open-meteo.com/v1/archive", status, "controlled refusal", {}, None)
    monkeypatch.setattr(ch, "acquire_climate", denied)
    report = ch.build_climate_history(BBOX, tmp_path / "build")
    assert report["requests_this_invocation"] == 1
    assert report["windows"][0]["error"]["http_status"] == status
    assert all(row["attempts"] == 0 for row in report["windows"][1:])


def test_grid_mismatch_does_not_replace_existing_sources(tmp_path, monkeypatch):
    output = tmp_path / "build"
    report = ch.build_climate_history(BBOX, output, max_requests=1)
    path = output / report["windows"][0]["file"]
    before = path.read_bytes()
    def shifted(*args, **kwargs):
        context = provider(*args, **kwargs)
        context["grid_location"][0] += 1
        context["raw_response"]["longitude"] += 1
        return context
    monkeypatch.setattr(ch, "acquire_climate", shifted)
    result = ch.build_climate_history(BBOX, output, max_requests=1, resume=True)
    assert result["windows"][1]["status"] == "failed"
    assert "different source/grid" in result["windows"][1]["error"]["reason"]
    assert path.read_bytes() == before


def test_missing_measurements_remain_missing_in_final_reference(tmp_path, monkeypatch):
    def incomplete(*args, **kwargs):
        context = provider(*args, **kwargs)
        if kwargs["start"] == "1991-01-01":
            context["raw_response"]["daily"]["precipitation_sum"][0] = None
        return context
    monkeypatch.setattr(ch, "acquire_climate", incomplete)
    output = tmp_path / "build"
    result = ch.build_climate_history(BBOX, output)
    assert result["status"] == "complete" and result["available_reference_feature_months"] == 46
    january = read(output / "departures.json")["months"][0]["variables"]
    assert not january["precipitation_sum_mm"]["available"]
    assert january["precipitation_sum_mm"]["standardized_departure"] is None


def test_corrupt_product_is_refused_without_download(tmp_path, monkeypatch):
    output = tmp_path / "build"
    ch.build_climate_history(BBOX, output)
    (output / "departures.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ch, "acquire_climate", lambda *a, **kw: pytest.fail("Cached source redownloaded"))
    with pytest.raises(ValueError, match="hash mismatch"):
        ch.build_climate_history(BBOX, output, resume=True)


@pytest.mark.parametrize("changes", [
    {"baseline_start": "1992-01-01"}, {"observed_start": "2020-01-01"},
    {"observed_end": "2030-12-31"}, {"observed_start": "2021-01-02"},
    {"max_requests": True}, {"max_requests": 17}, {"timeout": float("nan")}, {"resume": "yes"},
])
def test_invalid_bounds_do_not_create_artifacts(tmp_path, changes):
    output = tmp_path / "build"
    with pytest.raises(ValueError):
        ch.build_climate_history(BBOX, output, **changes)
    assert not output.exists()


def test_cli_roundtrip_and_existing_output_refusal(tmp_path, capsys):
    area = tmp_path / "area.json"
    area.write_text(json.dumps({"bbox": BBOX}), encoding="utf-8")
    output = tmp_path / "build"
    args = ["--area", str(area), "--output", str(output)]
    assert main(args) == 0
    assert '"observed_months": 60' in capsys.readouterr().out
    assert main(args) == 2
    assert main(args + ["--resume"]) == 0
