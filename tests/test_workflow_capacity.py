"""Bounded persistence and freshness retain immutable evidence at capacity."""
from copy import deepcopy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from forestmind.api.workflow import install_workflow
from forestmind.monitoring.workflow import WorkflowCapacityError, WorkflowService, digest
from test_workflow import saved


@pytest.fixture
def service(saved):
    value = WorkflowService(saved[0])
    yield value
    value.executor.shutdown(wait=True)


def test_aggregate_queue_deduplicates_and_refuses_extra_jobs(service, monkeypatch):
    monkeypatch.setattr(service.executor, "submit", lambda *a: None)
    first = service.submit("one")
    assert service.submit("one")["id"] == first["id"]
    with pytest.raises(ValueError, match="different collection settings"):
        service.submit("one", include_climate=True)
    for name in ("two", "three", "four"):
        service.submit(name)
    with pytest.raises(ValueError, match="queue is full"):
        service.submit("five")
    assert service.capacity()["jobs"] == 4


def test_job_history_limit_preserves_history(service, monkeypatch):
    monkeypatch.setattr("forestmind.monitoring.workflow.MAX_WORKFLOW_JOBS", 1)
    service._save_job({"id": "finished", "region_id": "observed", "status": "completed"})
    with pytest.raises(WorkflowCapacityError, match="history limit"):
        service.submit("observed")
    assert service.job("finished")["status"] == "completed"


def test_snapshot_quota_stops_before_collection_and_preserves_export(service, saved, monkeypatch):
    original = service.run("observed")
    monkeypatch.setattr("forestmind.monitoring.workflow.MAX_REGION_SNAPSHOTS", 1)
    monkeypatch.setattr("forestmind.monitoring.climate.acquire_climate", lambda *a, **k: pytest.fail("Capacity must be checked before network collection"))
    with pytest.raises(WorkflowCapacityError, match="snapshot limit"):
        service.run("observed", include_climate=True)
    assert service.latest("observed")["manifest"] == original["manifest"]
    assert service.capacity()["snapshots"] == 1


def test_metadata_quota_checks_utf8_and_context_replacement(service, monkeypatch):
    service.save_context("observed", "test", {"text": "original"})
    used = service.capacity()["metadata_bytes"]
    monkeypatch.setattr("forestmind.monitoring.workflow.MAX_WORKFLOW_METADATA_BYTES", used + 1)
    with pytest.raises(WorkflowCapacityError, match="metadata storage"):
        service.save_context("observed", "test", {"text": "世界" * 100})
    assert service._context("observed", "test") == {"text": "original"}
    service.save_context("observed", "test", {"text": "x"})
    assert service.capacity()["metadata_bytes"] < used


def test_reserved_job_completion_can_finish_at_storage_limit(service, monkeypatch):
    monkeypatch.setattr(service.executor, "submit", lambda *a: None)
    job = service.submit("observed")
    capacity = service.capacity()
    monkeypatch.setattr("forestmind.monitoring.workflow.MAX_WORKFLOW_METADATA_BYTES", capacity["metadata_bytes"] + capacity["reserved_job_completion_bytes"])
    with pytest.raises(WorkflowCapacityError, match="metadata storage"):
        service.save_context("observed", "test", {"text": "cannot fit"})
    service._save_job({**job, "status": "running", "started_at": "2026-10-19T00:00:00+00:00"})
    service._save_job({**job, "status": "failed", "error": "Quota reached; prior evidence preserved.", "completed_at": "2026-10-19T00:00:01+00:00"})
    assert service.job(job["id"])["status"] == "failed"
    assert service.capacity()["reserved_job_completion_bytes"] == 0


def test_artifact_quota_never_deletes_existing_files(service, monkeypatch):
    artifact = service.root / "prior-evidence.npz"
    artifact.write_bytes(b"bounded evidence")
    monkeypatch.setattr("forestmind.monitoring.workflow.MAX_WORKFLOW_ARTIFACT_BYTES", 10)
    with pytest.raises(WorkflowCapacityError, match="artifact storage"):
        service.run("observed")
    assert artifact.read_bytes() == b"bounded evidence"


def test_registered_file_and_software_changes_mark_saved_snapshot_stale(service, tmp_path, monkeypatch):
    model = tmp_path / "missing-compatible-configuration.pt"
    model.write_bytes(b"ineligible fixture")
    service.registry = [{"id": "fixture", "path": str(model)}]
    before = service.run("observed")
    assert not service.latest("observed")["stale"]
    model.write_bytes(b"changed ineligible fixture")
    assert service.latest("observed")["stale"]
    assert service.latest("observed")["manifest"] == before["manifest"]
    service.run("observed")
    monkeypatch.setattr("forestmind.monitoring.workflow.__version__", "changed-version")
    assert service.latest("observed")["stale"]


def test_unavailable_radar_recollection_retains_successful_context(service, saved, monkeypatch):
    radar = {"available": True, "end": "2024-02-02", "geometry_sha256": digest(saved[1]["geometry"]), "source": "controlled prior context"}
    service.save_context("observed", "radar", radar)
    monkeypatch.setattr("forestmind.monitoring.radar.acquire_radar", lambda *a, **k: {"available": False, "reason": "No eligible pair"})
    result = service.run("observed", include_radar=True)
    assert result["radar"] == radar
    assert result["collection_errors"]["radar"] == "No eligible pair"
    assert not service.latest("observed")["stale"]


def test_capacity_api_and_admission_conflict_preserve_snapshots(saved, monkeypatch):
    app = FastAPI()
    service = install_workflow(app, saved[0])
    snapshot = service.run("observed")
    monkeypatch.setattr("forestmind.monitoring.workflow.MAX_REGION_SNAPSHOTS", 1)
    with TestClient(app) as client:
        assert client.get("/api/v1/workflow/storage").json()["snapshots"] == 1
        response = client.post("/api/v1/regions/observed/workflow", json={})
        assert response.status_code == 409 and "snapshot limit" in response.json()["detail"]
        exported = client.get("/api/v1/regions/observed/workflow/export", params={"manifest_id": snapshot["manifest"]["id"]})
        assert exported.status_code == 200 and exported.json() == snapshot


def test_scenario_capacity_rolls_back_without_deleting_prior_results(service, monkeypatch):
    result = {"id": "one", "region_id": "observed", "evidence": "controlled scenario"}
    service.save_scenario(result)
    monkeypatch.setattr("forestmind.monitoring.workflow.MAX_WORKFLOW_METADATA_BYTES", service.capacity()["metadata_bytes"])
    with pytest.raises(WorkflowCapacityError, match="metadata storage"):
        service.save_scenario({**result, "id": "two"})
    with service.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM scenarios").fetchone()[0] == 1
