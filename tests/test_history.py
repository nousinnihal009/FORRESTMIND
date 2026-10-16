"""Bounded history acquisition: real dates, failure provenance, and safe resume."""

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pytest

from forestmind.data.adapters.base import BoundingBox, PREPROCESSING_VERSION, SceneMetadata
from forestmind.data.adapters.sentinel2 import Sentinel2Adapter
from forestmind.data.history import (
    HistoryRequest, create_history_plan, execute_history_plan, load_history_plan, month_windows,
)
import forestmind.data.history as history
from scripts.build_history import main


BBOX = (-55.1, -3.5, -55.08, -3.48)


def request(**kwargs):
    defaults = dict(provider="earth-search", bbox=BBOX, start="2024-01-01", end="2024-02-29",
                    patch_size=2, max_downloads=6, min_valid_fraction=.6)
    defaults.update(kwargs)
    return HistoryRequest(**defaults)


def scene(name, acquired="2024-01-12T10:30:00+00:00", cloud=5):
    item = {"type": "Feature", "stac_version": "1.0.0", "id": name,
            "collection": "sentinel-2-c1-l2a", "bbox": list(BBOX),
            "geometry": {"type": "Polygon", "coordinates": [[[-55.1, -3.5], [-55.08, -3.5],
                         [-55.08, -3.48], [-55.1, -3.48], [-55.1, -3.5]]]},
            "properties": {"datetime": acquired, "eo:cloud_cover": cloud},
            "assets": {"red": {"href": "https://example.test/B04.tif", "roles": ["data"]}},
            "links": []}
    return SceneMetadata(name, "sentinel2", datetime.fromisoformat(acquired), BoundingBox(*BBOX),
                         cloud_cover=cloud, stac_item=item, extra={"provider": "element84-earth-search"})


class OfflineAdapter(Sentinel2Adapter):
    def __init__(self, *, fixture, **kwargs):
        super().__init__(**kwargs)
        self.fixture = fixture

    def search(self, bbox, start_date, end_date, max_cloud_cover, max_results):
        self.fixture["searches"].append((start_date, end_date, max_cloud_cover, max_results, self.provider))
        result = self.fixture["scenes"].get(start_date[:7], [])
        if isinstance(result, Exception):
            raise result
        return result

    def download(self, scenes, bbox):
        assert len(scenes) == 1
        selected = scenes[0]
        self.fixture["downloads"].append(selected.scene_id)
        outcome = self.fixture.get("outcomes", {}).get(selected.scene_id, "valid")
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome == "failed":
            self.download_failures = [{"scene_id": selected.scene_id,
                                       "reason": "Timed out https://example.test/a.tif?sig=topsecret"}]
            return []
        data = np.full((12, self.patch_size, self.patch_size), .3, dtype=np.float32)
        data[7] = .6
        data[3] = .1
        mask = np.ones((self.patch_size, self.patch_size), dtype=bool)
        if outcome == "cloudy":
            mask[:] = False
            mask[0, 0] = True
        if outcome == "bad_other_band":
            data[0, 0, 0] = np.nan
        path = self.cache_path_for_scene(selected, bbox)
        self.save_patch_to_cache(path, data, selected, bbox, quality_mask=mask,
                                 preprocessing_version=PREPROCESSING_VERSION,
                                 data_units="surface_reflectance_0_1", processing_metadata={"fixture": True})
        return [path]


def fixture(scenes=None, outcomes=None):
    state = {"scenes": scenes or {"2024-01": [scene("jan")],
                                  "2024-02": [scene("feb", "2024-02-14T10:00:00+00:00")]},
             "outcomes": outcomes or {}, "searches": [], "downloads": []}
    def factory(**kwargs):
        return OfflineAdapter(fixture=state, **kwargs)
    return state, factory


def plan_and_execute(tmp_path, req=None, *, state_factory=None):
    state, factory = state_factory or fixture()
    plan_path, report_path, cache = tmp_path / "plan.json", tmp_path / "report.json", tmp_path / "cache"
    plan = create_history_plan(req or request(), plan_path, adapter_factory=factory)
    report = execute_history_plan(plan_path, report_path, cache, adapter_factory=factory)
    return plan, report, state, factory, plan_path, report_path, cache


def test_month_windows_are_utc_inclusive_and_clipped_across_leap_day():
    windows = month_windows("2024-01-19", "2024-03-02")
    assert windows == [
        {"month": "2024-01", "start": "2024-01-19T00:00:00+00:00", "end": "2024-01-31T23:59:59.999999+00:00"},
        {"month": "2024-02", "start": "2024-02-01T00:00:00+00:00", "end": "2024-02-29T23:59:59.999999+00:00"},
        {"month": "2024-03", "start": "2024-03-01T00:00:00+00:00", "end": "2024-03-02T23:59:59.999999+00:00"},
    ]


def test_month_windows_support_the_last_representable_month_without_overflow():
    assert month_windows("9999-12-30", "9999-12-31") == [
        {"month": "9999-12", "start": "9999-12-30T00:00:00+00:00", "end": "9999-12-31T23:59:59.999999+00:00"},
    ]


@pytest.mark.parametrize("values", [
    {"start": "2024-1-01"}, {"start": "2024-02-30"}, {"start": "2025-01-01"},
    {"start": "2000-01-01"}, {"provider": "automatic"}, {"patch_size": True},
    {"patch_size": 2049}, {"candidates_per_month": 21}, {"candidates_per_month": 0},
    {"search_limit": 2}, {"search_limit": 101}, {"max_downloads": -1},
    {"max_attempts": 0}, {"timeout_seconds": float("nan")}, {"timeout_seconds": 0},
    {"max_cloud_cover": 101}, {"min_valid_fraction": 0}, {"bbox": [0, 0, 0, 1]},
])
def test_request_rejects_invalid_or_unbounded_arguments(values):
    with pytest.raises((ValueError, TypeError)):
        request(**values)


def test_plan_is_deterministic_bounded_and_strips_credentials(tmp_path):
    selected = scene("early", "2024-01-02T10:00:00+00:00", 1)
    selected.stac_item["assets"]["red"]["href"] = "https://user:password@example.test/B04.tif?sig=secret#fragment"
    selected.stac_item["properties"]["access_token"] = "do-not-persist"
    selected.stac_item["properties"]["description"] = "Source https://example.test/x?api-key=hidden"
    state, factory = fixture({"2024-01": [scene("b", cloud=2), scene("a", cloud=2), selected,
                                         scene("late", "2024-01-20T10:00:00+00:00", 1)]})
    plan = create_history_plan(request(end="2024-01-31", candidates_per_month=3, search_limit=4),
                               tmp_path / "plan.json", adapter_factory=factory)
    assert [row["scene_id"] for row in plan["months"][0]["candidates"]] == ["early", "late", "a"]
    assert state["searches"][0] == ("2024-01-01T00:00:00+00:00", "2024-01-31T23:59:59.999999+00:00", 30., 4, "earth-search")
    text = (tmp_path / "plan.json").read_text()
    assert "secret" not in text and "password" not in text and "do-not-persist" not in text and "hidden" not in text
    assert plan["months"][0]["candidates"][0]["stac_item"]["assets"]["red"]["href"] == "https://example.test/B04.tif"
    assert load_history_plan(tmp_path / "plan.json")[0] == plan
    assert state["downloads"] == []


@pytest.mark.parametrize("scheme", ["HTTPS", "hTtPs", "HTTP", "hTtP"])
def test_signed_urls_are_sanitized_regardless_of_scheme_case(tmp_path, scheme):
    selected = scene("signed")
    selected.stac_item["assets"]["red"]["href"] = f"{scheme}://user:password@example.test:8443/B04.tif?sig=secret#fragment"
    _, factory = fixture({"2024-01": [selected]})
    path = tmp_path / "plan.json"
    plan = create_history_plan(request(end="2024-01-31"), path, adapter_factory=factory)
    href = plan["months"][0]["candidates"][0]["stac_item"]["assets"]["red"]["href"]
    assert href == f"{scheme.lower()}://example.test:8443/B04.tif"
    assert all(secret not in path.read_text() for secret in ("user", "password", "secret", "fragment"))
    assert load_history_plan(path)[0] == plan


@pytest.mark.parametrize("existing", [False, True])
def test_atomic_json_enforces_readable_size_before_mutating_destination(tmp_path, monkeypatch, existing):
    path = tmp_path / "nested" / "report.json"
    if existing:
        path.parent.mkdir()
        path.write_bytes(b"preserve existing report")
    monkeypatch.setattr(history, "MAX_JSON_BYTES", 64)
    with pytest.raises(ValueError, match="limit"):
        history._atomic_json(path, {"multibyte": "\u00e9" * 50})
    if existing:
        assert path.read_bytes() == b"preserve existing report"
        assert list(path.parent.iterdir()) == [path]
    else:
        assert not path.parent.exists()
    history._atomic_json(path, {"small": "ok"})
    assert history._read_json(path) == {"small": "ok"}
    path.write_bytes(b" " * 65)
    with pytest.raises(ValueError, match="limit"):
        history._read_json(path)


def test_invalid_out_of_window_or_provider_candidates_are_recorded_and_excluded(tmp_path):
    wrong_provider = scene("wrong-provider")
    wrong_provider.extra["provider"] = "microsoft-planetary-computer"
    state, factory = fixture({"2024-01": [scene("outside", "2024-02-01T00:00:00+00:00"),
                                         wrong_provider, scene("../../escape"), scene("good")]})
    plan = create_history_plan(request(end="2024-01-31"), tmp_path / "plan.json", adapter_factory=factory)
    month = plan["months"][0]
    assert [item["scene_id"] for item in month["candidates"]] == ["good"]
    assert len(month["discarded"]) == 3


def test_plan_refuses_overwrite_and_execute_rejects_tampering(tmp_path):
    state, factory = fixture()
    path = tmp_path / "plan.json"
    plan = create_history_plan(request(), path, adapter_factory=factory)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        create_history_plan(request(), path, adapter_factory=factory)
    assert path.read_bytes() == before
    plan["request"]["max_downloads"] = 99
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="hash mismatch"):
        execute_history_plan(path, tmp_path / "report.json", tmp_path / "cache", adapter_factory=factory)
    assert state["downloads"] == []


def test_cloudy_candidate_is_rejected_and_missing_month_is_not_filled(tmp_path):
    state, factory = fixture({"2024-01": [scene("cloudy", cloud=1), scene("clear", cloud=2)], "2024-02": []},
                             {"cloudy": "cloudy"})
    _, report, *_ = plan_and_execute(tmp_path, state_factory=(state, factory))
    assert report["status"] == "incomplete"
    assert len(report["accepted_cache_paths"]) == 1
    assert report["months"][0]["accepted"]["scene_id"] == "clear"
    assert report["months"][0]["attempts"][0]["status"] == "rejected"
    assert "0.250000" in report["months"][0]["attempts"][0]["reason"]
    assert report["months"][1]["accepted"] is None
    assert report["months"][1]["reason"] == "No eligible catalog candidates"
    assert state["downloads"] == ["cloudy", "clear"]


def test_download_budget_is_preserved_and_resume_continues_remaining_months(tmp_path):
    _, report, state, factory, plan_path, report_path, cache = plan_and_execute(tmp_path, request(max_downloads=1))
    assert report["status"] == "incomplete" and report["runs"][0]["downloads"] == 1
    assert state["downloads"] == ["jan"]
    resumed = execute_history_plan(plan_path, report_path, cache, resume=True, adapter_factory=factory)
    assert resumed["status"] == "complete" and len(resumed["runs"]) == 2
    assert state["downloads"] == ["jan", "feb"]
    assert len(resumed["months"][0]["attempts"]) == 1
    complete = execute_history_plan(plan_path, report_path, cache, resume=True, adapter_factory=factory)
    assert complete["runs"][-1]["attempts"] == 0 and complete["runs"][-1]["downloads"] == 0
    assert state["downloads"] == ["jan", "feb"]


def test_failed_reads_count_toward_budget_and_resume_uses_next_candidate(tmp_path):
    state, factory = fixture({"2024-01": [scene("fail", cloud=1), scene("clear", cloud=2)]}, {"fail": "failed"})
    _, report, _, _, plan_path, report_path, cache = plan_and_execute(
        tmp_path, request(end="2024-01-31", max_downloads=1), state_factory=(state, factory))
    assert report["status"] == "incomplete" and report["runs"][0]["downloads"] == 1
    assert "topsecret" not in report_path.read_text()
    resumed = execute_history_plan(plan_path, report_path, cache, resume=True, adapter_factory=factory)
    assert resumed["status"] == "complete" and state["downloads"] == ["fail", "clear"]


def test_missing_accepted_cache_is_not_trusted_and_is_downloaded_again(tmp_path):
    _, report, state, factory, plan_path, report_path, cache = plan_and_execute(tmp_path, request(end="2024-01-31"))
    Path(report["accepted_cache_paths"][0]).unlink()
    resumed = execute_history_plan(plan_path, report_path, cache, resume=True, adapter_factory=factory)
    assert resumed["status"] == "complete" and state["downloads"] == ["jan", "jan"]
    assert [item["status"] for item in resumed["months"][0]["attempts"]] == ["accepted", "invalidated", "accepted"]
    assert resumed["runs"][1]["attempts"] == 1 and resumed["runs"][1]["downloads"] == 1
    audit = resumed["months"][0]["attempts"][1]
    assert audit["stage"] == "resume_validation" and audit["run_index"] == 1
    assert "download_attempted" not in audit
    again = execute_history_plan(plan_path, report_path, cache, resume=True, adapter_factory=factory)
    assert again["runs"][-1]["attempts"] == 0 and state["downloads"] == ["jan", "jan"]


def test_altered_accepted_cache_is_rejected_without_overwriting_it(tmp_path):
    state, factory = fixture({"2024-01": [scene("a", cloud=1), scene("b", cloud=2)]})
    _, report, _, _, plan_path, report_path, cache = plan_and_execute(
        tmp_path, request(end="2024-01-31"), state_factory=(state, factory))
    path = Path(report["accepted_cache_paths"][0])
    altered = path.read_bytes() + b"altered cache tail"
    path.write_bytes(altered)
    resumed = execute_history_plan(plan_path, report_path, cache, resume=True, adapter_factory=factory)
    assert resumed["status"] == "complete" and resumed["months"][0]["accepted"]["scene_id"] == "b"
    assert "hash changed" in resumed["months"][0]["attempts"][1]["reason"]
    assert path.read_bytes() == altered and state["downloads"] == ["a", "b"]


def test_interruption_has_durable_attempt_and_resume_retries_it(tmp_path):
    state, factory = fixture({"2024-01": [scene("jan")]}, {"jan": KeyboardInterrupt()})
    plan_path, report_path, cache = tmp_path / "plan.json", tmp_path / "report.json", tmp_path / "cache"
    create_history_plan(request(end="2024-01-31"), plan_path, adapter_factory=factory)
    with pytest.raises(KeyboardInterrupt):
        execute_history_plan(plan_path, report_path, cache, adapter_factory=factory)
    interrupted = json.loads(report_path.read_text())
    assert interrupted["runs"][0]["attempts"] == 1 and interrupted["runs"][0]["downloads"] == 1
    assert interrupted["months"][0]["attempts"][0]["status"] == "running"
    assert "finished_at" not in interrupted["months"][0]["attempts"][0]
    assert interrupted["report_sha256"] == history._digest({k: v for k, v in interrupted.items() if k != "report_sha256"})
    state["outcomes"].clear()
    resumed = execute_history_plan(plan_path, report_path, cache, resume=True, adapter_factory=factory)
    assert resumed["status"] == "complete" and state["downloads"] == ["jan", "jan"]
    assert resumed["months"][0]["attempts"][0]["status"] == "interrupted"
    assert resumed["runs"][0]["status"] == "interrupted" and resumed["runs"][0]["stop_reason"] == "interrupted"
    assert [attempt["run_index"] for attempt in resumed["months"][0]["attempts"]] == [0, 1]
    assert execute_history_plan(plan_path, report_path, cache, resume=True, adapter_factory=factory)["status"] == "complete"


def test_resume_rejects_changed_report_hash_before_adapter_or_file_mutation(tmp_path):
    _, report, state, _, plan_path, report_path, cache = plan_and_execute(tmp_path)
    report["runs"][0]["downloads"] = -100
    report_path.write_text(json.dumps(report))
    before = report_path.read_bytes()
    def forbidden_factory(**kwargs):
        pytest.fail("Invalid progress must be rejected before constructing an adapter")
    with pytest.raises(ValueError, match="report hash mismatch"):
        execute_history_plan(plan_path, report_path, cache, resume=True, adapter_factory=forbidden_factory)
    assert report_path.read_bytes() == before and state["downloads"] == ["jan", "feb"]


@pytest.mark.parametrize("target, key, value", [
    ("report", "schema_version", 1), ("report", "status", "invented"),
    ("report", "months", {}), ("report", "runs", []), ("report", "accepted_cache_paths", []),
    ("run", "downloads", -100), ("run", "downloads", False), ("run", "downloads", 1),
    ("run", "attempts", 1), ("run", "attempt_limit", 100), ("run", "download_limit", 100),
    ("run", "run_index", 2), ("run", "status", "imaginary"), ("run", "stop_reason", "budget_exhausted"),
    ("month", "status", "running"), ("month", "accepted", None), ("month", "attempts", {}),
    ("attempt", "status", "imaginary"), ("attempt", "status", "invalidated"),
    ("attempt", "run_index", 99), ("attempt", "run_index", True), ("attempt", "stage", "unknown"),
    ("attempt", "stage", "resume_validation"), ("attempt", "download_attempted", "false"),
    ("attempt", "candidate_sha256", "0" * 64), ("attempt", "scene_id", []),
    ("attempt", "valid_fraction", -1), ("attempt", "cache_sha256", "bad"),
    ("accepted", "valid_samples", False), ("accepted", "valid_fraction", .7),
    ("accepted", "path", "relative.npz"), ("accepted", "sha256", "0" * 64),
])
def test_resume_rejects_structurally_invalid_reports_even_with_recomputed_digest(tmp_path, target, key, value):
    _, report, state, _, plan_path, report_path, cache = plan_and_execute(tmp_path)
    targets = {"report": report, "run": report["runs"][0], "month": report["months"][0],
               "attempt": report["months"][0]["attempts"][0], "accepted": report["months"][0]["accepted"]}
    targets[target][key] = value
    report["report_sha256"] = history._digest({k: v for k, v in report.items() if k != "report_sha256"})
    report_path.write_text(json.dumps(report))
    before = report_path.read_bytes()
    def forbidden_factory(**kwargs):
        pytest.fail("Malformed progress must be rejected before constructing an adapter")
    with pytest.raises(ValueError):
        execute_history_plan(plan_path, report_path, cache, resume=True, adapter_factory=forbidden_factory)
    assert report_path.read_bytes() == before and state["downloads"] == ["jan", "feb"]


def test_cached_scene_reuse_obeys_attempt_limit_without_network(tmp_path):
    state, factory = fixture()
    _, report, _, _, plan_path, _, cache = plan_and_execute(tmp_path, state_factory=(state, factory))
    existing_downloads = list(state["downloads"])
    next_plan = tmp_path / "cache-only-plan.json"
    create_history_plan(request(max_attempts=1, max_downloads=0), next_plan, adapter_factory=factory)
    first = execute_history_plan(next_plan, tmp_path / "cache-only-report.json", cache, adapter_factory=factory)
    assert first["status"] == "incomplete" and first["runs"][0]["attempts"] == 1
    assert first["runs"][0]["downloads"] == 0 and state["downloads"] == existing_downloads
    second = execute_history_plan(next_plan, tmp_path / "cache-only-report.json", cache, resume=True, adapter_factory=factory)
    assert second["status"] == "complete" and state["downloads"] == existing_downloads


def test_all_optical_bands_are_validated_not_only_index_bands(tmp_path):
    state, factory = fixture({"2024-01": [scene("bad")]}, {"bad": "bad_other_band"})
    _, report, *_ = plan_and_execute(tmp_path, request(end="2024-01-31"), state_factory=(state, factory))
    assert report["status"] == "incomplete" and report["accepted_cache_paths"] == []
    assert "invalid reflectance" in report["months"][0]["attempts"][0]["reason"]


@pytest.mark.parametrize("updates, reason", [
    ({"provider": "another-provider"}, "identity"),
    ({"source_metadata_sha256": "0" * 64}, "metadata hash"),
    ({"scene_id": "not-the-planned-scene"}, "identity"),
    ({"grid_transform": [1, 0, -55.1, 0, -1, -3.48]}, "grid transform"),
    ({"band_names": ["B12", "B11", "B09", "B8A", "B08", "B07", "B06", "B05", "B04", "B03", "B02", "B01"]}, "band order"),
])
def test_existing_caches_must_satisfy_the_complete_frozen_source_contract(tmp_path, updates, reason):
    _, report, state, factory, _, _, cache = plan_and_execute(tmp_path, request(end="2024-01-31"))
    path = Path(report["accepted_cache_paths"][0])
    with np.load(path, allow_pickle=False) as original:
        fields = {key: original[key] for key in original.files}
    fields.update(updates)
    np.savez_compressed(path, **fields)
    altered = path.read_bytes()
    fresh_plan = tmp_path / "cache-only-plan.json"
    create_history_plan(request(end="2024-01-31", max_downloads=0), fresh_plan, adapter_factory=factory)
    result = execute_history_plan(fresh_plan, tmp_path / "fresh-report.json", cache, adapter_factory=factory)
    assert result["status"] == "incomplete" and result["accepted_cache_paths"] == []
    assert reason in result["months"][0]["attempts"][0]["reason"]
    assert path.read_bytes() == altered and state["downloads"] == ["jan"]


def test_resume_rejects_different_request_or_cache_identity(tmp_path):
    _, _, _, factory, plan_path, report_path, cache = plan_and_execute(tmp_path)
    with pytest.raises(ValueError, match="cache directory"):
        execute_history_plan(plan_path, report_path, tmp_path / "other-cache", resume=True, adapter_factory=factory)
    new_plan = tmp_path / "new-plan.json"
    create_history_plan(request(min_valid_fraction=.8), new_plan, adapter_factory=factory)
    with pytest.raises(ValueError, match="frozen plan"):
        execute_history_plan(new_plan, report_path, cache, resume=True, adapter_factory=factory)
    with pytest.raises(FileExistsError, match="--resume"):
        execute_history_plan(plan_path, report_path, cache, adapter_factory=factory)
    with pytest.raises(ValueError, match="must not overwrite"):
        execute_history_plan(plan_path, plan_path, cache, adapter_factory=factory)


def test_catalog_failure_is_persisted_and_other_months_still_execute(tmp_path):
    state, factory = fixture({"2024-01": OSError("Catalog unavailable"),
                             "2024-02": [scene("feb", "2024-02-10T10:00:00+00:00")]})
    plan, report, *_ = plan_and_execute(tmp_path, state_factory=(state, factory))
    assert plan["months"][0]["status"] == "search_failed"
    assert report["status"] == "incomplete" and len(report["accepted_cache_paths"]) == 1
    assert report["months"][0]["reason"] == "Catalog search failed"


def test_cli_preflights_output_overwrite_before_any_execution(tmp_path, monkeypatch):
    output = tmp_path / "plan.json"
    output.write_text("keep me")
    called = []
    monkeypatch.setattr("scripts.build_history.execute_history_plan", lambda *a, **kw: called.append(True))
    with pytest.raises(SystemExit) as error:
        main(["execute", "--plan", str(output), "--report", str(tmp_path / "report.json"),
              "--cache-dir", str(tmp_path / "cache"), "--output", str(output), "--name", "Forest"])
    assert error.value.code == 2 and called == [] and output.read_text() == "keep me"


def test_cli_incomplete_returns_nonzero_and_exports_only_accepted_scenes(tmp_path, monkeypatch):
    state, factory = fixture({"2024-01": [scene("cloudy", cloud=1), scene("clear", cloud=2)], "2024-02": []},
                             {"cloudy": "cloudy"})
    plan_path, report_path, cache = tmp_path / "plan.json", tmp_path / "report.json", tmp_path / "cache"
    create_history_plan(request(), plan_path, adapter_factory=factory)
    real_execute = execute_history_plan
    monkeypatch.setattr("scripts.build_history.execute_history_plan",
                        lambda *a, **kw: real_execute(*a, **kw, adapter_factory=factory))
    output = tmp_path / "import.json"
    result = main(["execute", "--plan", str(plan_path), "--report", str(report_path),
                   "--cache-dir", str(cache), "--output", str(output), "--name", "History pilot"])
    assert result == 2
    payload = json.loads(output.read_text())
    provenance = json.loads(output.with_suffix(".provenance.json").read_text())
    assert len(payload["observations"]) == 1 and payload["observations"][0]["valid_fraction"] == 1
    assert provenance["acquisitions"][0]["scene_id"] == "clear" and len(provenance["acquisitions"]) == 1
    assert provenance["history_selection"]["missing_months"] == ["2024-02"]
    assert len(provenance["history_selection"]["report_sha256"]) == 64
    assert payload["synthetic"] is False


def test_cli_no_accepted_month_does_not_fabricate_an_import(tmp_path, monkeypatch):
    state, factory = fixture({"2024-01": []})
    plan_path = tmp_path / "plan.json"
    create_history_plan(request(end="2024-01-31"), plan_path, adapter_factory=factory)
    monkeypatch.setattr("scripts.build_history.execute_history_plan",
                        lambda *a, **kw: execute_history_plan(*a, **kw, adapter_factory=factory))
    output = tmp_path / "import.json"
    assert main(["execute", "--plan", str(plan_path), "--report", str(tmp_path / "report.json"),
                 "--cache-dir", str(tmp_path / "cache"), "--output", str(output), "--name", "Empty"]) == 2
    assert not output.exists()
