"""Connected selected-area analysis: masks, cutoffs, provenance and abstention."""
from copy import deepcopy
import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from forestmind.api.store import Store
from forestmind.api.workflow import install_workflow
from forestmind.monitoring.analysis import analyze_region
from forestmind.monitoring.climate import VARIABLES, aggregate_climate
from forestmind.monitoring.spatial import fragmentation_metrics, recovery_diagnostics, spectral_change
from forestmind.monitoring.workflow import WorkflowService, digest, file_hash, model_eligibility
from forestmind.data.sequence import OPTICAL_BANDS, prepare_sequence
from test_prepare_sequence import cache


@pytest.fixture
def saved(tmp_path):
    store = Store(tmp_path / "workflow.db")
    region = {"id": "observed", "name": "Test evidence", "country": "Brazil", "center": [-55.09, -3.49], "area_km2": 4.9,
              "geometry": {"type": "Polygon", "coordinates": [[[-55.1, -3.5], [-55.08, -3.5], [-55.08, -3.48], [-55.1, -3.48], [-55.1, -3.5]]]},
              "provenance": {"source": "Controlled local fixture; not a scientific validation", "mode": "imported_observations", "synthetic": False}}
    observations = [{"date": "2024-01-02", "ndvi": .5, "ndmi": .3, "nbr": .4, "valid_fraction": 1.},
                    {"date": "2024-02-02", "ndvi": .3, "ndmi": .2, "nbr": .2, "valid_fraction": .75}]
    store.put_region(region, observations, analyze_region(region, observations))
    return store, region, observations


def test_accepted_only_sequence_excludes_rejected_and_future(tmp_path):
    first = cache(tmp_path, "first", "2024-01-02T10:00:00+00:00")
    second = cache(tmp_path, "second", "2024-02-02T10:00:00+00:00")
    cache(tmp_path, "rejected", "2024-01-02T10:00:00+00:00")  # duplicate would fail unfiltered
    future = cache(tmp_path, "future", "2024-03-02T10:00:00+00:00")
    prepared = prepare_sequence(tmp_path, through="2024-02-28", source_paths=[first, second, future])
    assert [r["name"] for r in prepared.provenance["sources"]] == [first.name, second.name]
    with pytest.raises(ValueError, match="unique"):
        prepare_sequence(tmp_path, through="2024-02-28", source_paths=[first, first])
    with pytest.raises(ValueError, match="inside"):
        prepare_sequence(tmp_path, through="2024-02-28", source_paths=[tmp_path.parent / "outside.npz"])


def test_spatial_change_uses_common_mask_and_physical_bands(tmp_path, saved):
    first = cache(tmp_path, "first", "2024-01-02T10:00:00+00:00")
    data = np.full((12, 2, 2), .3, dtype=np.float32)
    data[7] = .6
    second = cache(tmp_path, "second", "2024-02-02T10:00:00+00:00", mask=[[True, False], [True, True]], updates={"data": data})
    prepared = prepare_sequence(tmp_path, through="2024-02-28", source_paths=[first, second])
    result = spectral_change(prepared, saved[1]["geometry"])
    assert result["kind"] == "observed_spectral_change"
    assert result["layers"]["ndvi"]["values"][0][1] is None
    assert result["layers"]["ndvi"]["mean_delta"] == pytest.approx(1 / 3)
    assert result["layers"]["ndmi"]["valid_fraction"] == .75


def test_workflow_binds_manifest_and_detects_changed_sources(tmp_path, saved):
    store, region, observations = saved
    service = WorkflowService(store)
    first = cache(tmp_path, "first", "2024-01-02T10:00:00+00:00")
    second = cache(tmp_path, "second", "2024-02-02T10:00:00+00:00")
    result = service.run(region["id"], cache_dir=tmp_path, source_paths=[first, second])
    assert result["manifest"]["geometry"] == region["geometry"]
    assert result["manifest"]["observations_sha256"] == digest(observations)
    assert result["manifest"]["observed_through"] == "2024-02-02"
    assert result["spatial_change"]["available"]
    assert not result["model"]["available"]
    assert not next(c for c in result["capabilities"] if c["id"] == "risk_map")["available"]
    assert service.latest(region["id"])["manifest"] == result["manifest"]
    restarted = WorkflowService(Store(store.path))
    assert restarted.latest(region["id"])["manifest"] == result["manifest"]
    second.write_bytes(second.read_bytes() + b"changed")
    changed = service.run(region["id"])
    assert not changed["spatial_change"]["available"]
    assert "changed after import" in changed["sequence"]["reason"]
    service.executor.shutdown()
    restarted.executor.shutdown()


def test_cutoff_never_uses_future_observations(tmp_path, saved):
    store, region, observations = saved
    service = WorkflowService(store)
    result = service.run(region["id"], cutoff="2024-01-31")
    assert result["observations"] == observations[:1]
    assert result["manifest"]["observations_sha256"] == digest(observations[:1])
    assert result["manifest"]["observation_cutoff"] == "2024-01-31"
    assert service.latest(region["id"])["stale"]
    service.executor.shutdown()


def test_seasonal_recovery_and_missing_month_censoring():
    rows = [{"date": f"{2020 + i // 12}-{i % 12 + 1:02d}-15", "ndvi": .8, "valid_fraction": 1.} for i in range(30)]
    rows[24]["ndvi"], rows[25]["ndvi"] = .5, .6
    result = recovery_diagnostics(rows)
    assert result["available"]
    assert result["baseline_end"] == "2021-12-15"
    assert result["episodes"][0]["status"] == "observed_return"
    assert result["episodes"][0]["recovered_at"] == "2022-04-15"
    missing = recovery_diagnostics(rows[:25] + rows[26:])
    assert missing["episodes"][0]["status"] == "censored_missing_months"
    assert missing["episodes"][0]["recovery_days"] is None
    assert not recovery_diagnostics(rows[:23])["available"]


def test_fragmentation_uses_classified_cover_and_ignores_nodata_edges(saved):
    payload = {"forest": [[True, False, True], [True, False, True]], "valid_mask": [[True] * 3] * 2,
               "source": "Hand-labelled controlled fixture", "observed_at": "2024-01-02", "classification": "Explicit forest/nonforest"}
    result = fragmentation_metrics(payload, saved[1]["geometry"])
    assert result["patch_count"] == 2
    assert result["forest_fraction"] == pytest.approx(2 / 3)
    assert result["largest_patch_fraction"] == pytest.approx(.5)
    assert result["mean_nearest_centroid_m"] > 0
    payload["valid_mask"] = [[True, False, True]] * 2
    masked = fragmentation_metrics(payload, saved[1]["geometry"])
    assert masked["internal_edge_km"] == 0  # unknown pixels cannot be assumed nonforest


def test_climate_aggregation_retains_nulls_and_rejects_future_or_wrong_units():
    response = {"daily_units": VARIABLES.copy(), "daily": {"time": ["2024-01-01", "2024-01-02"]}}
    for key in VARIABLES:
        response["daily"][key] = [2., None]
    result = aggregate_climate(response, start="2024-01-01", end="2024-01-31")
    assert result[0]["temperature_2m_mean"] == 2.
    assert result[0]["precipitation_sum"] == 2.
    assert result[0]["quality"]["precipitation_sum"]["missing_days"] == 1
    with pytest.raises(ValueError, match="cutoff"):
        aggregate_climate(response, start="2024-01-01", end="2024-01-01")
    response["daily_units"]["precipitation_sum"] = "inch"
    with pytest.raises(ValueError, match="unit mismatch"):
        aggregate_climate(response, start="2024-01-01", end="2024-01-31")


def test_registry_rejects_synthetic_unscoped_missing_and_tampered_models(tmp_path, saved):
    model = tmp_path / "model.pt"
    model.write_bytes(b"Only fixture bytes; never deserialize this in a test")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({"kind": "forestmind_checkpoint_evaluation", "dataset": {"split": "test", "synthetic": True},
                                     "checkpoint": {"sha256": file_hash(model)}}))
    entry = {"id": "development", "path": str(model), "sha256": file_hash(model), "trained_on_synthetic": True,
             "evaluation": {"path": str(evaluation), "sha256": file_hash(evaluation)}}
    chosen, checks = model_eligibility([entry], saved[1], None)
    assert chosen is None
    reason = " ".join(checks[0]["reasons"])
    assert "nonsynthetic" in reason and "real-data test-split" in reason and "geographic scope" in reason
    model.write_bytes(b"tampered")
    assert "SHA256 differs" in " ".join(model_eligibility([entry], saved[1], None)[1][0]["reasons"])


def test_api_rejects_arbitrary_paths_and_exports_selected_manifest(tmp_path, saved):
    store, region, _ = saved
    app = FastAPI()
    service = install_workflow(app, store)
    with TestClient(app) as client:
        result = client.get(f"/api/v1/regions/{region['id']}/workflow").json()
        identifier = result["manifest"]["id"]
        assert client.get(f"/api/v1/regions/{region['id']}/workflow/export?manifest_id={identifier}").json()["manifest"] == result["manifest"]
        assert client.post(f"/api/v1/regions/{region['id']}/workflow", json={"checkpoint_path": "evil.pt"}).status_code == 422
        assert client.get("/api/v1/regions/missing/workflow").status_code == 404
        cover = {"geometry": region["geometry"], "observed_at": "2024-02-02", "source": "classified sample",
                 "classification": "Reference woodland class", "license": "CC0", "data_kind": "classified_forest_cover",
                 "forest": [[True, False]], "valid_mask": [[True, True]]}
        invalid = deepcopy(cover)
        invalid["forest"] = [[1, 0]]
        assert client.post(f"/api/v1/regions/{region['id']}/forest-cover", json=invalid).status_code == 422
        accepted = client.post(f"/api/v1/regions/{region['id']}/forest-cover", json=cover)
        assert accepted.status_code == 201
        service.executor.shutdown(wait=True)
        updated = client.get(f"/api/v1/regions/{region['id']}/workflow").json()
        assert updated["fragmentation"]["available"]
        assert updated["manifest"]["id"] != identifier
        assert client.get(f"/api/v1/regions/{region['id']}/workflow/export?manifest_id={identifier}").json()["manifest"]["id"] == identifier


def test_interrupted_analysis_is_recoverable_and_preserves_snapshots(saved):
    store, region, _ = saved
    service = WorkflowService(store)
    snapshot = service.run(region["id"])
    service._save_job({"id": "unfinished", "region_id": region["id"], "status": "running"})
    restarted = WorkflowService(Store(store.path))
    assert restarted.job("unfinished")["status"] == "interrupted"
    assert restarted.latest(region["id"])["manifest"]["id"] == snapshot["manifest"]["id"]
    service.executor.shutdown()
    restarted.executor.shutdown()
