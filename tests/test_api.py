"""Local application end-to-end contracts, strict import, and persistence."""

from copy import deepcopy
import csv
import importlib
import io

from fastapi.testclient import TestClient
import pytest

from forestmind.api.app import create_app


@pytest.fixture
def app(tmp_path):
    return create_app(tmp_path / "monitoring.db")


@pytest.fixture
def client(app):
    with TestClient(app) as active:
        yield active


def test_demo_overview_and_details(client):
    overview = client.get("/api/v1/overview")
    assert overview.status_code == 200
    assert overview.json()["mode"] == "synthetic_demo"
    assert overview.json()["metrics"]["regions"] == 4
    assert overview.json()["as_of"] == "2026-09-01"
    assert overview.headers["x-content-type-options"] == "nosniff"
    for summary in overview.json()["regions"]:
        region = client.get(f"/api/v1/regions/{summary['id']}").json()
        assert len(region["observations"]) == 48
        assert region["provenance"]["synthetic"] is True
        assert region["confidence"] is None
        for endpoint in ("health", "forecast", "drivers", "alerts"):
            assert client.get(f"/api/v1/regions/{summary['id']}/{endpoint}").status_code == 200


def test_analysis_job_and_scenario_persist_after_restart(client, app):
    response = client.post("/api/v1/analysis", json={"region_id": "xingu"})
    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    job_id = response.json()["id"]
    job = client.get(f"/api/v1/analysis/{job_id}").json()
    assert job["status"] == "completed"
    assert job["started_at"] and job["completed_at"]
    scenario = client.post("/api/v1/scenarios", json={
        "region_id": "xingu", "restoration": 60, "pressure_reduction": 50,
    })
    assert scenario.status_code == 201
    saved = scenario.json()
    assert saved["delta"] > 0
    with TestClient(create_app(app.state.store.path)) as restarted:
        assert restarted.get(f"/api/v1/analysis/{job_id}").json() == job
        assert restarted.get(f"/api/v1/scenarios/{saved['id']}").json() == saved
        assert restarted.get("/api/v1/alerts").json()["alerts"]


def test_analysis_failure_is_recorded(client, monkeypatch):
    module = importlib.import_module("forestmind.api.app")
    def broken(*_args):
        raise RuntimeError("Artificial test failure")
    monkeypatch.setattr(module, "analyze_region", broken)
    response = client.post("/api/v1/analysis", json={"region_id": "xingu"})
    job = client.get("/api/v1/analysis/" + response.json()["id"]).json()
    assert job["status"] == "failed"
    assert "failed" in job["error"].lower()
    assert job["completed_at"] is not None
    assert client.get("/api/v1/regions/xingu").json()["health_score"] is not None


def test_restart_recovers_orphan_jobs(app):
    app.state.store.save_job({"id": "interrupted", "region_id": "xingu", "status": "running", "result": None})
    restarted = create_app(app.state.store.path)
    job = restarted.state.store.job("interrupted")
    assert job["status"] == "failed"
    assert "restarted" in job["error"]


def test_real_import_and_exports_preserve_provenance(client, app):
    payload = client.get("/api/v1/import-example").json()
    payload.update(name="User AOI", source="Prepared station observations", synthetic=False)
    response = client.post("/api/v1/regions/import", json=payload)
    assert response.status_code == 201
    imported = response.json()
    assert imported["provenance"]["mode"] == "imported_observations"
    assert imported["provenance"]["verified"] is False
    assert imported["provenance"]["synthetic"] is False
    assert client.get("/api/v1/overview").json()["mode"] == "mixed"
    region_id = imported["id"]
    exported = client.get(f"/api/v1/regions/{region_id}/export")
    assert exported.json() == imported
    assert "attachment" in exported.headers["content-disposition"]
    csv_export = client.get(f"/api/v1/regions/{region_id}/export?format=csv")
    rows = list(csv.DictReader(io.StringIO(csv_export.text)))
    assert len(rows) == 48
    assert rows[0]["provenance_mode"] == "imported_observations"
    with TestClient(create_app(app.state.store.path)) as restarted:
        assert restarted.get(f"/api/v1/regions/{region_id}").json() == imported


def test_example_is_not_mislabeled_as_real_when_imported(client):
    example = client.get("/api/v1/import-example").json()
    imported = client.post("/api/v1/regions/import", json=example).json()
    assert imported["provenance"]["synthetic"] is True
    assert imported["provenance"]["mode"] == "synthetic_import"


@pytest.mark.parametrize("field,value", [
    ("ndvi", 1.2), ("ndmi", -1.5), ("nbr", "0.8"), ("forest_cover", 95),
    ("valid_fraction", True), ("valid_fraction", -0.1), ("date", "2026-13-01"),
    ("date", "20260901"), ("unrecognized_column", 2),
])
def test_import_rejects_invalid_measurements(client, field, value):
    payload = client.get("/api/v1/import-example").json()
    payload["observations"][0][field] = value
    assert client.post("/api/v1/regions/import", json=payload).status_code == 422


def test_import_rejects_duplicate_dates_and_bad_geometry(client):
    payload = client.get("/api/v1/import-example").json()
    payload["observations"][1]["date"] = payload["observations"][0]["date"]
    assert client.post("/api/v1/regions/import", json=payload).status_code == 422
    payload = client.get("/api/v1/import-example").json()
    payload["geometry"]["coordinates"][0][-1] = [0, 0]
    assert client.post("/api/v1/regions/import", json=payload).status_code == 422
    payload["geometry"]["coordinates"] = [[[0, 0], [3, 0], [0, 2], [2, 2], [0, 0]]]
    assert client.post("/api/v1/regions/import", json=payload).status_code == 422


def test_short_import_does_not_manufacture_predictions(client):
    payload = client.get("/api/v1/import-example").json()
    payload["observations"] = [{"date": "2026-09-01", "valid_fraction": 0.2}]
    response = client.post("/api/v1/regions/import", json=payload)
    assert response.status_code == 201
    result = response.json()
    assert result["status"] == "insufficient_data"
    assert result["health_score"] is None
    assert result["forecast"]["points"] == []


def test_csv_protects_user_supplied_source_from_formula_execution(client):
    payload = client.get("/api/v1/import-example").json()
    payload["source"] = "=HYPERLINK(\"https://example.com\")"
    result = client.post("/api/v1/regions/import", json=payload).json()
    csv_export = client.get(f"/api/v1/regions/{result['id']}/export?format=csv")
    rows = list(csv.DictReader(io.StringIO(csv_export.text)))
    assert rows[0]["source"].startswith("'=")


def test_unknown_regions_and_invalid_scenarios(client):
    assert client.get("/api/v1/regions/unknown").status_code == 404
    assert client.post("/api/v1/analysis", json={"region_id": "unknown"}).status_code == 404
    assert client.post("/api/v1/scenarios", json={"region_id": "xingu", "restoration": 101}).status_code == 422
    assert client.post("/api/v1/scenarios", json={"region_id": "xingu", "restoration": "50"}).status_code == 422
    assert client.get("/api/v1/regions/xingu/export?format=exe").status_code == 422
    assert client.get("/api/v1/analysis/unknown").status_code == 404


def test_empty_workspace_can_receive_imports(tmp_path):
    with TestClient(create_app(tmp_path / "empty.db", seed_demo=False)) as client:
        assert client.get("/api/v1/overview").json()["metrics"]["regions"] == 0
        assert client.get("/api/v1/overview").json()["mode"] == "empty"
