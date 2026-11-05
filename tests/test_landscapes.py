"""Worldwide plans preserve exact footprints and never imply missing evidence."""
import json
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from forestmind.api.landscapes import install_landscapes
from forestmind.api.schemas import Geometry
from forestmind.api.store import Store
from forestmind.monitoring.acquisition import AcquisitionService, geometry_for_bbox, validate_request
from forestmind.monitoring.landscapes import LandscapeService, MAX_PLANS, TileGrid, rectangle_area
from test_history import fixture


def params(**changes):
    return {"name": "Worldwide forest study", "country": "Unspecified",
            "bbox": [-55.1, -3.5, -55.0, -3.4], **changes}


def acquisition(bbox, **changes):
    return {"name": "Selected monitoring area", "country": "Unspecified",
            "bbox": bbox, "provider": "earth-search", "start": "2024-01-01",
            "end": "2024-02-29", "patch_size": 32, "max_downloads": 1, **changes}


def service(tmp_path):
    return LandscapeService(Store(tmp_path / "study.db"))


def insert_region(store, identifier, bbox, *, synthetic=False, observations=None, geometry=None, mode="import"):
    region = {"id": identifier, "geometry": geometry or geometry_for_bbox(bbox),
              "provenance": {"synthetic": synthetic, "mode": mode}}
    rows = observations if observations is not None else [
        {"date": "2024-01-12", "valid_fraction": .8, "ndvi": .6}]
    with store.connection() as db:
        db.execute("INSERT INTO regions VALUES (?,?,?)", (identifier, store._json(region), store._json(rows)))


@pytest.mark.parametrize("bbox", [
    [-55.1, -3.5, -54.9, -3.3],       # Amazon / southern tropics
    [20.0, -1.0, 20.2, -.8],          # Congo
    [114.0, 1.0, 114.2, 1.2],         # Borneo
    [76.5, 11.5, 76.7, 11.7],         # India
    [10.0, 48.0, 10.2, 48.2],         # Europe
    [-120.0, 60.0, -119.8, 60.2],     # Canada
    [130.0, 65.0, 130.2, 65.2],       # Siberia
    [145.0, -42.0, 145.2, -41.8],     # Australia
    [-73.0, -42.0, -72.8, -41.8],     # Chile
    [179.9, -17.0, -179.9, -16.8],    # International date line
    [-180.0, 84.8, -179.8, 85.0],     # Northern supported limit
    [179.8, -85.0, 180.0, -84.8],      # Southern supported limit
])
def test_continents_dateline_and_latitude_limits_partition_exactly_into_acquirable_tiles(bbox):
    grid = TileGrid(bbox, 25)
    tiles = [grid.tile(index) for index in range(grid.tile_count)]
    assert len(tiles) > 1
    assert math.fsum(tile["area_km2"] for tile in tiles) == pytest.approx(grid.area_km2, rel=1e-12)
    for tile in tiles:
        w, s, e, n = tile["bbox"]
        assert .0001 <= tile["area_km2"] <= 25
        assert w < e and s < n and e - w <= 1 and n - s <= 1
        assert grid.index_for(tile["bbox"]) == tile["index"]
        assert TileGrid(bbox, 25).tile(tile["index"]) == tile
        validate_request(acquisition(tile["bbox"]))
        Geometry.model_validate(geometry_for_bbox(tile["bbox"]))
    # Every latitude row abuts its neighbours and covers each longitude segment.
    assert grid.rows[0][0] == bbox[1] and grid.rows[-1][1] == bbox[3]
    for previous, following in zip(grid.rows, grid.rows[1:]):
        assert previous[1] == following[0]
    expected_segments = [(bbox[0], bbox[2])] if bbox[0] < bbox[2] else [(bbox[0], 180), (-180, bbox[2])]
    for s, n, parts in grid.rows:
        assert [(part[0], part[1]) for part in parts] == expected_segments
        for west, east, count, start in parts:
            row_tiles = [grid.tile(i)["bbox"] for i in range(start, start + count)]
            assert row_tiles[0][0] == west and row_tiles[-1][2] == east
            assert all(left[2] == right[0] for left, right in zip(row_tiles, row_tiles[1:]))


def test_large_world_plan_is_lazy_paged_and_does_not_create_jobs(tmp_path, monkeypatch):
    planner = service(tmp_path)
    plan = planner.create(params(bbox=[-180, -85, 180, 85], tile_area_km2=1))
    assert plan["tile_count"] > 100_000_000
    assert plan["row_count"] < 20_000
    calls = []
    original = TileGrid.tile
    def tracked(grid, index):
        calls.append(index)
        return original(grid, index)
    monkeypatch.setattr(TileGrid, "tile", tracked)
    page = planner.page(plan["id"], plan["tile_count"] - 3, 50)
    assert len(page["tiles"]) == 3 and page["next_offset"] is None
    assert len(calls) == 3
    assert page["summary"]["tiles_without_observations"] == plan["tile_count"]
    assert page["summary"]["study_area_with_observations_km2"] == 0
    with planner.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM regions").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM landscape_plans").fetchone()[0] == 1
        assert len(db.execute("SELECT payload FROM landscape_plans").fetchone()[0]) < 2500


def test_plan_is_idempotent_persistent_and_survives_database_backup(tmp_path):
    planner = service(tmp_path)
    plan = planner.create(params())
    assert planner.create(params()) == plan
    original_page = planner.page(plan["id"], 1, 3)
    restored = LandscapeService(Store(planner.store.path))
    assert restored.list() == [plan]
    assert restored.page(plan["id"], 1, 3) == original_page
    backup = tmp_path / "snapshot.db"
    planner.store.backup(backup)
    assert LandscapeService(Store(backup)).page(plan["id"], 1, 3) == original_page
    assert original_page["next_offset"] == 4
    assert len(original_page["tiles"]) == 3


def test_plan_limit_retains_existing_and_idempotency(tmp_path):
    planner = service(tmp_path)
    first = None
    for index in range(MAX_PLANS):
        created = planner.create(params(name=f"Forest {index}"))
        first = first or created
    with pytest.raises(ValueError, match="at most 100"):
        planner.create(params(name="One plan too many"))
    assert planner.create(params(name="Forest 0")) == first
    assert len(planner.list()) == MAX_PLANS


@pytest.mark.parametrize("changes", [
    {"name": " "}, {"name": "a" * 121}, {"country": " "},
    {"bbox": [0, 0, 0, 1]}, {"bbox": [180, 0, -180, 1]},
    {"bbox": [181, 0, 1, 1]}, {"bbox": [0, -86, 1, 1]},
    {"bbox": [0, 1, 1, 1]}, {"bbox": [0, 2, 1, 1]},
    {"bbox": [0, 0, 1, 86]}, {"bbox": [False, 0, 1, 1]},
    {"bbox": ["0", 0, 1, 1]}, {"bbox": [0, 0, float("nan"), 1]},
    {"bbox": [0, 0, float("inf"), 1]}, {"bbox": [0, 0, 1]},
    {"bbox": [0, 0, 1e-9, 1e-9]},
    {"bbox": [179.999999999, 0, -179.9, .1]},
    {"tile_area_km2": 0}, {"tile_area_km2": 26}, {"tile_area_km2": True},
    {"tile_area_km2": float("inf")},
])
def test_invalid_or_degenerate_plans_are_rejected_without_persistence(tmp_path, changes):
    planner = service(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        planner.create(params(**changes))
    assert planner.list() == []


def test_coverage_requires_exact_observed_rectangles_and_counts_duplicates_once(tmp_path):
    planner = service(tmp_path)
    plan = planner.create(params())
    tiles = planner.page(plan["id"])["tiles"]
    first, second = tiles[0]["bbox"], tiles[1]["bbox"]
    insert_region(planner.store, "observed", first)
    insert_region(planner.store, "duplicate", first, observations=[
        {"date": "2024-01-12", "valid_fraction": .5, "ndvi": 0},
        {"date": "2024-02-12", "valid_fraction": .1, "ndvi": .2},
        {"date": "2024-02-12", "valid_fraction": .1, "ndvi": .2},
    ])
    insert_region(planner.store, "synthetic", second, synthetic=True)
    insert_region(planner.store, "demo", second, mode="demo")
    insert_region(planner.store, "unknown-source", second, synthetic=None)
    insert_region(planner.store, "empty", second, observations=[])
    insert_region(planner.store, "no-measurement", second, observations=[{"date": "2024-01-12", "valid_fraction": 1}])
    insert_region(planner.store, "masked", second, observations=[{"date": "2024-01-12", "valid_fraction": 0, "ndvi": .8}])
    overlap = list(second)
    overlap[0] += (overlap[2] - overlap[0]) / 2
    insert_region(planner.store, "overlap", overlap)
    triangle = geometry_for_bbox(second)
    triangle["coordinates"][0].pop(2)
    insert_region(planner.store, "triangle", second, geometry=triangle)
    page = planner.page(plan["id"])
    assert page["summary"]["tiles_with_observations"] == 1
    assert page["summary"]["tiles_without_observations"] == plan["tile_count"] - 1
    assert page["summary"]["study_area_with_observations_km2"] == rectangle_area(first)
    assert page["tiles"][0]["region_id"] == "duplicate"
    assert page["tiles"][0]["observation_count"] == 2
    assert page["tiles"][0]["latest_date"] == "2024-02-12"
    assert page["tiles"][1]["region_id"] is None
    assert "health" not in page["summary"]


def test_reversed_ring_and_far_page_match_without_enumerating_grid(tmp_path):
    planner = service(tmp_path)
    plan = planner.create(params(bbox=[179, -17, -179, -16]))
    grid = TileGrid(plan["bbox"], plan["tile_area_km2"])
    index = grid.tile_count - 1
    bbox = grid.tile(index)["bbox"]
    geometry = geometry_for_bbox(bbox)
    geometry["coordinates"][0].reverse()
    insert_region(planner.store, "west-of-dateline", bbox, geometry=geometry)
    assert planner.page(plan["id"], 0, 1)["summary"]["tiles_with_observations"] == 1
    assert planner.page(plan["id"], index, 1)["tiles"][0]["region_id"] == "west-of-dateline"


def test_integrity_and_bad_pagination_are_refused(tmp_path):
    planner = service(tmp_path)
    plan = planner.create(params())
    for offset, limit in [(-1, 50), (True, 50), (1.5, 50), (0, 101), (0, 0), (0, False)]:
        with pytest.raises(ValueError):
            planner.page(plan["id"], offset, limit)
    assert planner.page(plan["id"], plan["tile_count"], 1)["tiles"] == []
    with pytest.raises(KeyError, match="not found"):
        planner.page("missing")
    with planner.store.connection() as db:
        changed = dict(plan, name="Tampered")
        db.execute("UPDATE landscape_plans SET payload=? WHERE id=?", (planner.store._json(changed), plan["id"]))
    with pytest.raises(ValueError, match="integrity"):
        planner.page(plan["id"])


def test_api_validation_pages_and_no_background_work(tmp_path):
    app = FastAPI()
    planner = install_landscapes(app, Store(tmp_path / "api.db"))
    client = TestClient(app)
    response = client.post("/api/v1/landscapes", json=params(bbox=[179.9, -17, -179.9, -16.8]))
    assert response.status_code == 201
    plan = response.json()["landscape"]
    assert plan["crosses_antimeridian"]
    assert client.get("/api/v1/landscapes").json()["landscapes"] == [plan]
    page = client.get(f"/api/v1/landscapes/{plan['id']}?offset=1&limit=2")
    assert page.status_code == 200 and len(page.json()["tiles"]) == 2
    for suffix in ["?offset=-1", "?limit=101", "?limit=0", "?offset=true"]:
        assert client.get(f"/api/v1/landscapes/{plan['id']}" + suffix).status_code == 422
    assert client.get("/api/v1/landscapes/missing").status_code == 404
    for changes in [{"bbox": [True, 0, 1, 1]}, {"tile_area_km2": True}, {"unexpected": "field"}]:
        assert client.post("/api/v1/landscapes", json=params(**changes)).status_code == 422
    assert planner.store.list_regions() == []


def test_planned_tile_flows_through_existing_acquisition_and_retains_missing_neighbours(tmp_path):
    planner = service(tmp_path)
    plan = planner.create(params())
    first = planner.page(plan["id"])["tiles"][0]
    state, factory = fixture()
    worker = AcquisitionService(planner.store, adapter_factory=factory)
    submitted = worker.submit(acquisition(first["bbox"], name=first["name"]))
    assert state["searches"] == []
    assert planner.page(plan["id"])["summary"]["tiles_with_observations"] == 0
    assert worker.run_once()
    job = worker.job(submitted["id"])
    assert job["status"] == "partial" and job["region_id"]
    page = planner.page(plan["id"])
    assert page["tiles"][0]["region_id"] == job["region_id"]
    assert page["tiles"][0]["observation_count"] == 1
    assert page["summary"]["tiles_with_observations"] == 1
    assert all(tile["region_id"] is None for tile in page["tiles"][1:])
    worker.resume(job["id"])
    assert worker.run_once()
    assert planner.page(plan["id"])["tiles"][0]["observation_count"] == 2
    assert state["downloads"] == ["jan", "feb"]
