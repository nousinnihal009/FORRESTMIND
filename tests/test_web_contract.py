"""Contracts that connect the interactive dashboard to the local service."""

from fastapi.testclient import TestClient

from forestmind.api.app import create_app


def test_dashboard_assets_and_evaluation_endpoint(tmp_path):
    with TestClient(create_app(tmp_path / "workspace.db")) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert 'id="scenario-form"' in page.text
        assert 'data-detail="observations"' in page.text
        for filename, mime in (("app.js", "javascript"), ("styles.css", "text/css"),
                               ("mark.svg", "image/svg+xml")):
            asset = client.get("/static/" + filename)
            assert asset.status_code == 200
            assert mime in asset.headers["content-type"]
            assert asset.headers["cache-control"] == "no-cache"
        response = client.get("/api/v1/regions/xingu/evaluation")
        assert response.status_code == 200
        result = response.json()
        assert result["synthetic_warning"]
        assert set(result["methods"]) == {"persistence", "seasonal_naive", "damped_trend"}
        assert all(method["aggregate"]["n"] > 0 for method in result["methods"].values())
        assert client.get("/api/v1/regions/missing/evaluation").status_code == 404


def test_scenario_export_retains_synthetic_provenance(tmp_path):
    with TestClient(create_app(tmp_path / "workspace.db")) as client:
        result = client.post("/api/v1/scenarios", json={"region_id": "xingu"}).json()
        assert result["provenance"]["synthetic"] is True
        assert result["delta"] == 0
