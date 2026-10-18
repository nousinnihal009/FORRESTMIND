"""Search provenance, local HTTP boundary, persistence and recovery contracts."""
import json
import sqlite3
from urllib.error import URLError

from fastapi.testclient import TestClient
import pytest

from forestmind.api.app import create_app
from forestmind.api.store import Store
from forestmind.data.adapters.network import validate_destination


@pytest.fixture
def app(tmp_path):
    return create_app(tmp_path / "study.db")


def test_place_search_disambiguation_cache_and_attribution(app, monkeypatch):
    calls = []
    def fetch(query):
        calls.append(query)
        return {"features": [
            {"geometry": {"coordinates": [-55.1, -3.5]}, "properties": {"name": "Tapajos", "country": "Brazil", "osm_type": "R", "osm_id": 123, "extent": [-56, -3, -54, -4]}},
            {"geometry": {"coordinates": [10, 50]}, "properties": {"name": "Tapajos", "country": "Germany"}},
            {"geometry": {"coordinates": [float("nan"), 0]}, "properties": {}},
        ]}
    monkeypatch.setattr(app.state.places, "_fetch", fetch)
    with TestClient(app) as client:
        response = client.get("/api/v1/places", params={"q": "Tapajos"})
        assert response.status_code == 200
        result = response.json()
        assert len(result["results"]) == 2
        assert result["results"][0]["bbox"] == [-56, -4, -54, -3]
        assert not result["results"][0]["verified_forest_boundary"]
        assert "OpenStreetMap" in result["source"]
        assert client.get("/api/v1/places?q=TAPAJOS").json()["cached"]
        assert client.get("/api/v1/places?q=Another").status_code == 429
    restarted = create_app(app.state.store.path)
    monkeypatch.setattr(restarted.state.places, "_fetch", lambda _: pytest.fail("Cache should survive restart"))
    with TestClient(restarted) as client:
        assert client.get("/api/v1/places?q=Tapajos").json()["cached"]
    assert calls == ["Tapajos"]


def test_place_failure_actionable_and_not_cached(app, monkeypatch):
    def fail(_):
        raise URLError("secret upstream message")
    monkeypatch.setattr(app.state.places, "_fetch", fail)
    with TestClient(app) as client:
        response = client.get("/api/v1/places?q=Missing")
        assert response.status_code == 503
        assert "coordinates" in response.json()["detail"]
        assert "secret" not in response.text


def test_local_security_boundary_and_body_limit(app):
    with TestClient(app) as client:
        assert client.get("/api/v1/status", headers={"Host": "evil.example"}).status_code == 403
        assert client.post("/api/v1/analysis", json={"region_id": "xingu"}, headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.post("/api/v1/analysis", json={"region_id": "xingu"}, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
        assert client.post("/api/v1/analysis", content=b"x" * (4 * 1024 * 1024 + 1)).status_code == 413
        assert client.post("/api/v1/analysis", json={"region_id": "xingu"}, headers={"Origin": "http://testserver"}).status_code == 202
    with TestClient(app, client=("192.0.2.1", 1234)) as remote:
        assert remote.get("/api/v1/status", headers={"Host": "localhost"}).status_code == 403


def test_backup_restores_measurements_alert_review_and_migrations(app, tmp_path):
    with TestClient(app) as client:
        alert = client.get("/api/v1/alerts").json()["alerts"][0]
        reviewed = client.patch("/api/v1/alerts/" + alert["id"], json={"status": "acknowledged", "note": "Review imagery"})
        assert reviewed.status_code == 200
        assert reviewed.json()["supporting_observations"]
        rid = alert["region_id"]
        client.post("/api/v1/analysis", json={"region_id": rid})
        assert next(a for a in client.get(f"/api/v1/regions/{rid}").json()["alerts"] if a["id"] == alert["id"])["status"] == "acknowledged"
        before = client.get(f"/api/v1/regions/{rid}").json()
        result = client.post("/api/v1/operations/backup")
        assert result.status_code == 201
        snapshot = app.state.store.path.parent / "backups" / result.json()["filename"]
        assert client.get("/api/v1/operations").json()["schema_migrations"] == [1, 2, 3]
    restored = create_app(snapshot, seed_demo=False)
    with TestClient(restored) as client:
        assert client.get(f"/api/v1/regions/{rid}").json() == before
        assert client.get(f"/api/v1/regions/{rid}/alert-history").json()["alerts"]
    with pytest.raises(FileExistsError):
        app.state.store.backup(snapshot)


def test_alert_thresholds_and_episode_survive_restart(app):
    with TestClient(app) as client:
        assert client.put("/api/v1/regions/xingu/alert-settings", json={"condition_drop_points": 0}).status_code == 422
        response = client.put("/api/v1/regions/xingu/alert-settings", json={"condition_drop_points": 100, "stale_days": 1})
        assert response.status_code == 200
        assert all(not a["id"].endswith("-decline") for a in client.get("/api/v1/regions/xingu").json()["alerts"])
        assert client.get("/api/v1/regions/xingu/operational-alerts").json()["alerts"][0]["kind"] == "monitoring_stale"
    with TestClient(create_app(app.state.store.path)) as client:
        assert client.get("/api/v1/regions/xingu/alert-settings").json()["condition_drop_points"] == 100


@pytest.mark.parametrize("url", ["http://127.0.0.1/x", "https://evil.example/x", "file:///etc/passwd", "https://sentinel-cogs.s3.amazonaws.com.evil/x", "https://user:pass@sentinel-cogs.s3.amazonaws.com/x", "https://sentinel-cogs.s3.amazonaws.com:8080/x"])
def test_outbound_destinations_are_restricted(url):
    with pytest.raises(ValueError):
        validate_destination(url)


def test_known_provider_allowed():
    assert validate_destination("https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/sentinel-2-c1-l2a/a.tif")
