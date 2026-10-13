"""Model-run imports remain auditable, persistent and separate from baselines."""

import json
import sqlite3

from fastapi.testclient import TestClient
import pytest

from forestmind.api.app import create_app
from test_inference import example_report


def test_model_run_import_preserves_report_and_baseline_on_restart(tmp_path):
    database = tmp_path / "runs.db"
    with TestClient(create_app(database)) as client:
        baseline = client.get("/api/v1/regions/xingu").json()
        payload = example_report()
        response = client.post("/api/v1/regions/xingu/model-runs", json=payload)
        assert response.status_code == 201
        run = response.json()
        assert run["verification"] == "user_provided_unverified"
        assert run["report"]["predictions"] == payload["predictions"]
        assert run["report"]["checkpoint"]["trained_on_synthetic"] is True
        assert run["imported_at"].endswith("+00:00")
        assert client.get("/api/v1/regions/xingu").json() == baseline
        assert client.get("/api/v1/regions/xingu/model-runs").json() == {"runs": [run]}
        assert client.get(f"/api/v1/regions/xingu/model-runs/{run['id']}").json() == run
        assert client.get(f"/api/v1/regions/tapajos/model-runs/{run['id']}").status_code == 404
    with TestClient(create_app(database)) as restarted:
        assert restarted.get("/api/v1/regions/xingu/model-runs").json() == {"runs": [run]}
        assert restarted.get("/api/v1/regions/xingu").json() == baseline


def test_model_run_import_rejects_region_mismatch_unknown_and_paths(tmp_path):
    app = create_app(tmp_path / "runs.db")
    with TestClient(app) as client:
        assert client.get("/api/v1/regions/unknown/model-runs").status_code == 404
        assert client.get("/api/v1/regions/xingu/model-runs/unknown").status_code == 404
        report = example_report()
        assert client.post("/api/v1/regions/unknown/model-runs", json=report).status_code == 404
        assert client.post("/api/v1/regions/tapajos/model-runs", json=report).status_code == 422
        report["checkpoint"]["name"] = "../../model.pt"
        assert client.post("/api/v1/regions/xingu/model-runs", json=report).status_code == 422
        with pytest.raises(sqlite3.IntegrityError):
            app.state.store.save_model_run("unknown", {"id": "foreign-key-test"})


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_bad_model_scores_return_usable_422(tmp_path, nonfinite):
    with TestClient(create_app(tmp_path / "runs.db")) as client:
        report = example_report()
        report["predictions"]["health"]["score"] = nonfinite
        response = client.post("/api/v1/regions/xingu/model-runs", content=json.dumps(report),
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 422
        assert response.json()["detail"]
