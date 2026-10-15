"""Read-only coverage API validates windows and retains imported provenance."""

from fastapi.testclient import TestClient
import pytest

from forestmind.api.app import create_app
from forestmind.monitoring.coverage import assess_history


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "coverage.db")) as active:
        yield active


def test_demo_coverage_agrees_with_saved_methods_and_is_read_only(client):
    before = client.get("/api/v1/regions/tapajos").json()
    response = client.get("/api/v1/regions/tapajos/coverage")
    assert response.status_code == 200
    report = response.json()
    assert report["synthetic"] is True
    assert report["counts"]["observations"] == len(before["observations"])
    assert report["baseline_forecast_available"] == bool(before["forecast"]["points"])
    assert report["resilience_available"] == (before["resilience"]["status"] != "insufficient_data")
    assert response.headers["cache-control"] == "no-store"
    assert client.get("/api/v1/regions/tapajos").json() == before


def test_imported_single_month_report_preserves_payload_source(client):
    payload = client.get("/api/v1/import-example").json()
    payload.update(synthetic=False, source="Sentinel-2 prepared test data")
    payload["observations"] = [{"date": "2024-07-29", "valid_fraction": 0.4,
                                "ndvi": 0.7, "ndmi": 0.4, "nbr": 0.6}]
    saved = client.post("/api/v1/regions/import", json=payload).json()
    response = client.get(f"/api/v1/regions/{saved['id']}/coverage", params={
        "start": "2024-07-01", "end": "2025-06-30",
    })
    assert response.status_code == 200
    assert response.json() == assess_history(payload, start="2024-07-01", end="2025-06-30")
    assert response.json()["synthetic"] is False
    assert response.json()["counts"]["missing"] == 11


@pytest.mark.parametrize("params,fragment", [
    ({"start": "20240101"}, "YYYY-MM-DD"),
    ({"end": "2024-02-30"}, "YYYY-MM-DD"),
    ({"start": "2025-01-01", "end": "2024-12-31"}, "on or after"),
    ({"start": "0001-01-01", "end": "9999-12-31"}, "1200 calendar months"),
])
def test_invalid_windows_return_friendly_422(client, params, fragment):
    response = client.get("/api/v1/regions/tapajos/coverage", params=params)
    assert response.status_code == 422
    assert fragment in response.json()["detail"]


def test_unknown_region_returns_404(client):
    response = client.get("/api/v1/regions/unknown/coverage")
    assert response.status_code == 404
    assert response.json()["detail"] == "Region not found"
