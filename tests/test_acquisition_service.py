"""Exercise the actual history/ingest path with a deterministic offline provider."""
import json
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from forestmind.api.acquisition import install_acquisition
from forestmind.api.store import Store
from forestmind.monitoring.acquisition import (
    AcquisitionConflict, AcquisitionService, LIMITS, validate_request,
)
from test_history import fixture, scene


def request(**changes):
    return {"name": "Local satellite area", "country": "Brazil", "provider": "earth-search",
            "bbox": [-55.1, -3.5, -55.08, -3.48], "start": "2024-01-01", "end": "2024-02-29",
            "patch_size": 32, "max_downloads": 1, **changes}


def service(tmp_path, state_factory=None):
    state, factory = state_factory or fixture()
    worker = AcquisitionService(Store(tmp_path / "test.db"), adapter_factory=factory)
    return worker, state, factory


@pytest.mark.parametrize("changes", [
    {"bbox": [-180, -1, 180, 1]}, {"bbox": [0, 0, 1, 1]},
    {"bbox": [1, 1, 0, 0]}, {"bbox": [0, 86, .01, 86.01]},
    {"start": "2000-01-01"}, {"end": "2999-01-01"}, {"start": "2015-06-23"},
    {"provider": "planetary-computer"}, {"patch_size": 512}, {"max_downloads": 25},
    {"max_attempts": 73}, {"timeout_seconds": 120},
])
def test_interactive_bounds(changes):
    with pytest.raises((ValueError, TypeError)):
        validate_request(request(**changes))


def test_real_history_ingestion_budget_resume_dedupe_and_atomic_snapshot(tmp_path):
    worker, state, _ = service(tmp_path)
    submitted = worker.submit(request())
    duplicate = worker.submit(request())
    assert duplicate["id"] == submitted["id"] and duplicate["deduplicated"]
    assert state["searches"] == []  # HTTP submission does no network work.
    assert worker.run_once()
    first = worker.job(submitted["id"])
    assert first["status"] == "partial" and first["region_id"]
    region_id = first["region_id"]
    assert len(worker.store.region(region_id)[1]) == 1
    assert first["download_attempts"] == 1
    assert state["downloads"] == ["jan"]
    worker.resume(first["id"])
    assert worker.run_once()
    final = worker.job(first["id"])
    assert final["status"] == "completed" and final["region_id"] == region_id
    assert len(worker.store.region(region_id)[1]) == 2
    assert state["downloads"] == ["jan", "feb"]
    assert len(state["searches"]) == 2  # frozen plan reused, no catalog reselection
    area = worker.area(region_id)
    assert len(area["snapshots"]) == 1
    snapshot = area["snapshots"][0]
    assert len(snapshot["accepted_cache_paths"]) == 2
    assert snapshot["payload"]["synthetic"] is False
    assert worker.store.analysis(region_id)["provenance"]["mode"] == "satellite_acquisition"
    assert worker.store.analysis(region_id)["forecast"]["points"] == []
    repeated = worker.submit(request())
    assert repeated["id"] == final["id"] and repeated["deduplicated"]
    assert not worker.run_once()


def test_resume_conflicts_with_replacement_job_for_cancelled_request(tmp_path):
    worker, _, _ = service(tmp_path)
    old = worker.submit(request())
    worker.cancel(old["id"])
    replacement = worker.submit(request())
    assert old["id"] != replacement["id"]
    with pytest.raises(AcquisitionConflict, match="identical request"):
        worker.resume(old["id"])


def test_refresh_skips_saved_months_and_preserves_original_records(tmp_path):
    worker, state, _ = service(tmp_path)
    job = worker.submit(request(max_downloads=2))
    worker.run_once()
    region_id = worker.job(job["id"])["region_id"]
    before = worker.store.region(region_id)[1]
    # An overlapping refresh must acquire March only and keep earlier observations.
    state["scenes"]["2024-03"] = [scene("mar", "2024-03-14T10:00:00+00:00")]
    refresh = worker.refresh(region_id, {"end": "2024-03-31"})
    with pytest.raises(AcquisitionConflict, match="active"):
        worker.refresh(region_id, {"end": "2024-04-30"})
    worker.run_once()
    assert worker.job(refresh["id"])["status"] == "completed"
    after = worker.store.region(region_id)[1]
    assert after[:2] == before and len(after) == 3
    assert state["downloads"] == ["jan", "feb", "mar"]
    assert len(state["searches"]) == 3  # saved months do not consume catalog requests
    again = worker.refresh(region_id, {"end": "2024-03-31"})
    worker.run_once()
    assert worker.job(again["id"])["status"] == "completed"
    assert worker.store.region(region_id)[1] == after
    assert state["downloads"] == ["jan", "feb", "mar"]
    assert len(state["searches"]) == 3


def test_restart_recovers_claimed_job_and_resumes_frozen_progress(tmp_path):
    worker, state, factory = service(tmp_path)
    job = worker.submit(request())
    worker.run_once()
    worker.resume(job["id"])
    with worker.store.connection() as db:
        saved = worker.job(job["id"], progress=False)
        saved.update(status="running", phase="acquiring")
        worker._save(db, saved)
    restarted = AcquisitionService(Store(tmp_path / "test.db"), adapter_factory=factory)
    restarted.recover()
    assert restarted.job(job["id"])["status"] == "queued"
    assert restarted.job(job["id"])["recovery_count"] == 1
    restarted.run_once()
    assert restarted.job(job["id"])["status"] == "completed"
    assert state["downloads"] == ["jan", "feb"]
    assert len(state["searches"]) == 2


def test_queued_cancellation_is_durable_and_no_provider_work(tmp_path):
    worker, state, _ = service(tmp_path)
    job = worker.submit(request())
    assert worker.cancel(job["id"])["status"] == "cancelled"
    assert not worker.run_once()
    assert not state["searches"] and not state["downloads"]
    assert worker.store.list_regions() == []


def test_running_cancellation_never_partially_changes_saved_area(tmp_path):
    worker, state, factory = service(tmp_path)
    job = worker.submit(request())
    def cancel_factory(**kwargs):
        adapter = factory(**kwargs)
        original = adapter.download
        def cancel_after_download(*args, **kw):
            result = original(*args, **kw)
            worker.cancel(job["id"])
            return result
        adapter.download = cancel_after_download
        return adapter
    worker.adapter_factory = cancel_factory
    worker.run_once()
    assert worker.job(job["id"])["status"] == "cancelled"
    assert worker.store.list_regions() == []
    worker.adapter_factory = factory
    worker.resume(job["id"])
    worker.run_once()
    assert worker.job(job["id"])["status"] == "completed"
    assert state["downloads"] == ["jan", "feb"]


def test_transient_failure_retries_have_shared_download_budget(tmp_path, monkeypatch):
    worker, state, _ = service(tmp_path, fixture(outcomes={"jan": "failed"}))
    monkeypatch.setattr(worker.stop_event, "wait", lambda _: False)
    job = worker.submit(request(max_downloads=2))
    worker.run_once()
    result = worker.job(job["id"])
    assert state["downloads"] == ["jan", "jan"]
    assert result["download_attempts"] == 2
    assert result["status"] == "partial" and result["region_id"] is None
    assert worker.store.list_regions() == []


def test_provider_failure_retains_no_fabricated_data(tmp_path, monkeypatch):
    worker, state, _ = service(tmp_path, fixture({"2024-01": OSError("offline"), "2024-02": []}))
    monkeypatch.setattr(worker.stop_event, "wait", lambda _: False)
    job = worker.submit(request())
    worker.run_once()
    result = worker.job(job["id"])
    assert result["status"] == "failed" and "No usable" in result["error"]
    assert len(state["searches"]) == 4  # three bounded attempts for failed January
    assert state["downloads"] == [] and worker.store.list_regions() == []


def test_storage_and_queue_limits(tmp_path, monkeypatch):
    worker, _, _ = service(tmp_path)
    monkeypatch.setitem(LIMITS, "max_storage_bytes", 1)
    with pytest.raises(AcquisitionConflict, match="storage"):
        worker.submit(request())
    monkeypatch.setitem(LIMITS, "max_storage_bytes", 2 * 1024 ** 3)
    monkeypatch.setitem(LIMITS, "max_queued_jobs", 1)
    worker.submit(request())
    with pytest.raises(AcquisitionConflict, match="queue"):
        worker.submit(request(name="Second area"))


def test_schedules_persist_and_enqueue_only_one_due_refresh(tmp_path):
    worker, _, factory = service(tmp_path)
    job = worker.submit(request(max_downloads=2))
    worker.run_once()
    region_id = worker.job(job["id"])["region_id"]
    schedule = worker.set_schedule(region_id, True, 7)
    schedule["next_run_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with worker.store.connection() as db:
        db.execute("UPDATE acquisition_schedules SET payload=? WHERE region_id=?", (json.dumps(schedule), region_id))
    restarted = AcquisitionService(Store(tmp_path / "test.db"), adapter_factory=factory)
    restarted.tick_schedules()
    first = restarted.schedule(region_id)
    assert first["last_job_id"] != job["id"] and first["last_error"] is None
    restarted.tick_schedules()
    assert restarted.schedule(region_id)["last_job_id"] == first["last_job_id"]
    assert len(restarted.jobs(region_id)) == 2
    assert restarted.set_schedule(region_id, False, 7)["next_run_at"] is None


def test_api_validation_status_and_refresh_missing_area(tmp_path):
    app = FastAPI()
    store = Store(tmp_path / "test.db")
    install_acquisition(app, store)
    client = TestClient(app)  # no lifespan: queue stays idle for request assertions
    assert client.post("/api/v1/acquisition/jobs", json=request(provider="automatic")).status_code == 422
    assert client.post("/api/v1/acquisition/jobs", json=request(max_downloads=True)).status_code == 422
    assert client.post("/api/v1/acquisition/jobs", json=request(bbox=[0, 0, 1, 1])).status_code == 422
    response = client.post("/api/v1/acquisition/jobs", json=request())
    assert response.status_code == 202
    job_id = response.json()["id"]
    assert client.get("/api/v1/acquisition/jobs").json()["jobs"][0]["id"] == job_id
    assert client.post(f"/api/v1/acquisition/jobs/{job_id}/cancel").json()["status"] == "cancelled"
    assert client.post("/api/v1/regions/missing/refresh").status_code == 404
    assert client.get("/api/v1/acquisition/capabilities").json()["limits"]["concurrency"] == 1


def test_partial_catalog_plan_resumes_without_repeating_completed_search(tmp_path):
    worker, state, factory = service(tmp_path)
    job = worker.submit(request(max_downloads=2))
    def cancel_factory(**kwargs):
        adapter = factory(**kwargs)
        original = adapter.search
        def cancel_after_search(*args, **kw):
            result = original(*args, **kw)
            worker.cancel(job["id"])
            return result
        adapter.search = cancel_after_search
        return adapter
    worker.adapter_factory = cancel_factory
    worker.run_once()
    assert worker.job(job["id"])["status"] == "cancelled"
    assert len(state["searches"]) == 1
    worker.adapter_factory = factory
    worker.resume(job["id"])
    worker.run_once()
    assert worker.job(job["id"])["status"] == "completed"
    assert len(state["searches"]) == 2
    assert state["downloads"] == ["jan", "feb"]


def test_worker_lease_prevents_duplicate_process_owner_and_shutdown_releases(tmp_path):
    worker, _, factory = service(tmp_path)
    worker.start()
    second = AcquisitionService(Store(tmp_path / "test.db"), adapter_factory=factory)
    try:
        with pytest.raises(RuntimeError, match="Another acquisition worker"):
            second.start()
    finally:
        worker.stop()
    second.start()
    second.stop()
    assert not second.thread.is_alive()


def test_tampered_plan_does_not_corrupt_saved_observations(tmp_path):
    worker, _, _ = service(tmp_path)
    job = worker.submit(request())
    worker.run_once()
    result = worker.job(job["id"])
    before = worker.store.region(result["region_id"])
    plan_path = worker.root / job["id"] / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["request"]["bbox"][0] = 0
    plan_path.write_text(json.dumps(plan))
    worker.resume(job["id"])
    worker.run_once()
    assert worker.job(job["id"])["status"] == "failed"
    assert worker.store.region(result["region_id"]) == before
