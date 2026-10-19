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


def _workflow(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for frontend rendering contracts")
    source = json.dumps(str(ROOT / "forestmind/web/workflow.js"))
    code = f"""
const vm=require('vm'),fs=require('fs');
const context={{esc:s=>String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;'),
numeric:v=>typeof v==='number'?String(v):'—',pct:v=>typeof v==='number'?String(v*100)+'%':'—',badge:s=>String(s)}};
vm.createContext(context);vm.runInContext(fs.readFileSync({source},'utf8'),context);
process.stdout.write(JSON.stringify(vm.runInContext({json.dumps(script)},context)));
"""
    result = subprocess.run([node, "-e", code], capture_output=True, text=True,
                            encoding="utf-8", check=True)
    return json.loads(result.stdout)


def test_climate_quality_distinguishes_partial_month_and_missing_fields():
    complete = _workflow("climateQuality({start:'2024-07-17',end:'2024-07-31'}, {month:'2024-07',quality:{temperature:{valid_days:15},rain:{valid_days:15}}})")
    assert complete == {"start": "2024-07-17", "end": "2024-07-31", "partial": True,
                        "summary": "15 / 15 requested days · all fields"}
    missing = _workflow("climateQuality({start:'2024-02-01',end:'2024-02-29'}, {month:'2024-02',quality:{temperature:{valid_days:29},rain:{valid_days:20}}})")
    assert missing["partial"] is False
    assert missing["summary"] == "20 / 29 requested days · 1 of 2 fields complete"


def test_readiness_details_and_failed_context_remain_explicit():
    rendered = _workflow("workflowCapabilities([{id:'radar',name:'Radar',available:false,reason:'No matched pair',requires:['Same orbit']}])")
    assert rendered.startswith('<details class="workflow-readiness')
    assert "0 with output · 1 need more evidence" in rendered
    assert "No matched pair" in rendered and "Same orbit" in rendered
    errors = _workflow("collectionErrors({collection_errors:{radar:'Provider <unavailable>',climate:null}})")
    assert "Provider &lt;unavailable>" in errors
    assert "earlier successful context remains dated" in errors


def test_radar_render_keeps_missing_values_and_context_interpretation():
    rendered = _workflow("renderRadarContext({available:true,dataset:'Sentinel-1 RTC',comparison:{vv_change_db:null,vh_change_db:1.25,common_valid_fraction:.8,vv_change_map_db:[[null,1]],vh_change_map_db:[[null,2]]},scenes:[],limitations:['Not a causal estimate']})")
    assert 'id="radar-change-map"' in rendered
    assert "not automatically fused" in rendered
    assert "<strong>—</strong>" in rendered
    assert "Not a causal estimate" in rendered


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
