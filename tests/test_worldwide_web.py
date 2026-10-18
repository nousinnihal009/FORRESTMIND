"""Frontend geometry safety and served controls for the worldwide workflow."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from forestmind.api.app import create_app


ROOT = Path(__file__).resolve().parents[1]


def _geometry(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to execute the frontend coordinate contract")
    helper = json.dumps(str(ROOT / "forestmind/web/area-utils.js"))
    result = subprocess.run([node, "-e", f"const area=require({helper}); {script}"],
                            capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize("bounds,reason", [
    ([170, -2, -170, 2], "antimeridian"),
    ([-55.1, -3.5, -55.08, -3.5], "South"),
    ([-55, -3, -54, -2], "smaller"),
    ([0, 85.01, .01, 85.02], "Polar"),
    ([0, 0, 2, .00001], "1°"),
    ([None, 0, .01, .01], "finite"),
    ([0, 0, .00000001, .00000001], "at least"),
])
def test_area_rejects_unsupported_or_unsafe_bounds(bounds, reason):
    error = _geometry(f"process.stdout.write(JSON.stringify(area.validateBbox({json.dumps(bounds)})));")
    assert reason in error


def test_real_pilot_area_and_mercator_round_trip():
    result = _geometry("""
const bbox=[-55.1,-3.5,-55.08,-3.48];
const points=[[-55.09,-3.49],[20,80],[-150,-70],[179,0]];
process.stdout.write(JSON.stringify({error:area.validateBbox(bbox),size:area.areaKm2(bbox),
points:points.map(p=>area.unproject(...area.project(...p,12),12))}));
""")
    assert result["error"] is None
    assert 4.9 < result["size"] < 5
    for actual, expected in zip(result["points"], [[-55.09, -3.49], [20, 80], [-150, -70], [179, 0]]):
        assert actual == pytest.approx(expected, abs=1e-7)


def test_worldwide_assets_and_keyboard_alternatives_are_served(tmp_path):
    with TestClient(create_app(tmp_path / "workspace.db")) as client:
        page = client.get("/").text
        for control in ("place-search", "area-map", "apply-coordinates", "confirm-area",
                        "acquisition-form", "acquisition-jobs", "refresh-jobs",
                        "scenario-horizon", "local-operations", "refresh-operations"):
            assert f'id="{control}"' in page
        for tab in ("workflow", "monitor", "coverage", "models"):
            assert f'data-detail="{tab}"' in page
        assert 'id="area-map" viewBox="0 0 800 370" tabindex="0"' in page
        assert 'id="confirm-area" type="checkbox" disabled' in page
        assert 'id="start-acquisition" disabled' in page
        assert "not verified forest boundaries" in page or "not verify forest cover" in page
        for filename in ("area-utils.js", "acquisition.js", "workflow.js", "operations.js"):
            response = client.get("/static/" + filename)
            assert response.status_code == 200
            assert "javascript" in response.headers["content-type"]
            assert response.headers["cache-control"] == "no-cache"
