"""Execute the selected-area presentation and forest reference file flows."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_js(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser presentation checks")
    source = f"""
const fs=require('fs'),vm=require('vm'),elements={{}};
function element(key) {{ return elements[key]??(elements[key]={{innerHTML:'',textContent:'',hidden:false,disabled:false,
  handlers:{{}},addEventListener(type,handler){{this.handlers[type]=handler;}}}}); }}
const context={{document:{{querySelector:element,querySelectorAll:()=>[]}},setTimeout,clearTimeout,TextEncoder,
  console,URLSearchParams,FormData,Intl,Date}};
vm.createContext(context);
const app=fs.readFileSync({json.dumps(str(ROOT / 'forestmind/web/app.js'))},'utf8').split('hydrateIcons();route();')[0];
vm.runInContext(app,context);
vm.runInContext(fs.readFileSync({json.dumps(str(ROOT / 'forestmind/web/workflow.js'))},'utf8'),context);
context.elements=elements;
Promise.resolve(vm.runInContext({json.dumps(script)},context)).then(value=>process.stdout.write(JSON.stringify(value))).catch(error=>{{console.error(error);process.exit(1);}});
"""
    result = subprocess.run([node, "-e", source], capture_output=True, text=True,
                            encoding="utf-8", check=True, timeout=30)
    return json.loads(result.stdout)


GEOMETRY = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
REFERENCE = {"observed_at": "2024-01-01", "source": "Field team", "classification": "Mapped forest cover",
             "license": "CC BY 4.0", "data_kind": "classified_forest_cover",
             "forest": [[True, False], [True, True]], "valid_mask": [[True, True], [False, False]]}


def test_reference_preview_respects_missing_pixels_and_escapes_source():
    data = {**REFERENCE, "source": "<script>bad</script>"}
    html = run_js(f"coverFilePreview({json.dumps(data)},{json.dumps(GEOMETRY)})")
    assert "50%" in html and "2 × 2 samples" in html
    assert "&lt;script&gt;bad&lt;/script&gt;" in html and "<script>" not in html
    assert "not been independently verified" in html
    assert "<pre" not in html and "valid_mask" not in html


@pytest.mark.parametrize("changes,expected", [
    ({"observed_at": "2024-02-30"}, "valid observation date"),
    ({"valid_mask": [[False, False], [False, False]]}, "no usable classified area"),
    ({"forest": [[1, 0], [1, 1]]}, "must share a valid grid"),
    ({"valid_mask": [[True]]}, "must share a valid grid"),
    ({"geometry": {"type": "Polygon", "coordinates": []}}, "different area"),
    ({"source": ""}, "needs its source"),
])
def test_invalid_reference_cannot_enable_save(changes, expected):
    data = {**REFERENCE, **changes}
    message = run_js(f"(()=>{{try{{coverFilePreview({json.dumps(data)},{json.dumps(GEOMETRY)});return 'accepted';}}catch(e){{return e.message;}}}})()")
    assert expected in message


def test_replacing_valid_file_with_invalid_file_clears_preview_and_disables_save():
    result = run_js(f"""(async()=>{{
      state.selected='area';state.tab='workflow';state.region={{geometry:{json.dumps(GEOMETRY)}}};wireCoverImport('area');
      const change=elements['#cover-import-file'].handlers.change;
      await change({{target:{{files:[{{size:100,text:async()=>{json.dumps(json.dumps(REFERENCE))}}}]}}}});
      const first={{disabled:elements['#save-cover-reference'].disabled,hidden:elements['#cover-import-preview'].hidden}};
      await change({{target:{{files:[{{size:100,text:async()=>'broken file'}}]}}}});
      return {{first,disabled:elements['#save-cover-reference'].disabled,hidden:elements['#cover-import-preview'].hidden,error:elements['#cover-import-status'].textContent}};
    }})()""")
    assert result["first"] == {"disabled": False, "hidden": False}
    assert result["disabled"] and result["hidden"]
    assert "could not be read" in result["error"]


def test_slow_previous_file_cannot_overwrite_new_file_selection():
    result = run_js(f"""(async()=>{{
      state.selected='area';state.tab='workflow';state.region={{geometry:{json.dumps(GEOMETRY)}}};wireCoverImport('area');
      const change=elements['#cover-import-file'].handlers.change;let finish;
      const previous=change({{target:{{files:[{{size:100,text:()=>new Promise(resolve=>{{finish=resolve;}})}}]}}}});
      await change({{target:{{files:[]}}}});finish({json.dumps(json.dumps(REFERENCE))});await previous;
      return {{disabled:elements['#save-cover-reference'].disabled,hidden:elements['#cover-import-preview'].hidden}};
    }})()""")
    assert result == {"disabled": True, "hidden": True}


def test_unavailable_analysis_sections_have_reasons_without_raw_records():
    report = {"observations": [{"date": "2024-01-01", "ndvi": None, "valid_fraction": 0}],
              "condition": {"score": 0, "status": "watch"},
              "resilience": {"status": "insufficient_data", "evidence": ["Need 36 months."]},
              "recovery": {"available": False, "reason": "Need 24 baseline months."},
              "fragmentation": {"available": False, "reason": "No classified cover."},
              "scenarios": {"available": False, "reason": "Need a one-year baseline."},
              "model": {"available": False, "reason": "No trusted local model is registered. Supply a real-trained compatible checkpoint and independent evaluation evidence in FORESTMIND_MODEL_REGISTRY."}}
    html = run_js(f"renderWorkflowSections({json.dumps(report)})")
    for message in ["Need 24 baseline months", "No classified cover", "Need a one-year baseline", "No compatible trained model", "Need 36 months", "0.00 / 100"]:
        assert message in html
    for token in ["<pre", "<textarea", "FORESTMIND_MODEL_REGISTRY", '"available":', "saved computation"]:
        assert token not in html


def test_manifest_and_alert_details_keep_evidence_without_internal_identifiers():
    manifest = {"created_at": "2024-02-01T12:00:00Z", "manifest": {"id": "SECRET_HASH", "synthetic": False,
        "geometry": GEOMETRY, "context_sha256": {"radar": "SECRET_HASH"}, "observation_count": 4}}
    html = run_js(f"renderWorkflowManifest({json.dumps(manifest)},{{name:'A <forest>'}})")
    assert "A &lt;forest&gt;" in html and "Radar" in html and "4" in html
    assert "SECRET_HASH" not in html and "<pre" not in html
    alert = {"episode_id": "SECRET_EPISODE", "note": "<review>", "thresholds": {"min_valid_fraction": .4},
             "supporting_observations": [{"date": "2024-01-01", "ndvi": 0, "ndmi": None, "valid_fraction": 0}]}
    html = run_js(f"renderAlertEvidence({json.dumps(alert)})")
    assert "&lt;review&gt;" in html and "40%" in html and "0%" in html and "Not available" in html
    assert "SECRET_EPISODE" not in html and "<pre" not in html


def test_spatial_provenance_shows_date_source_and_coverage_without_hashes():
    html = run_js("""renderSpatialEvidence({available:false,reason:'Insufficient common pixels',shape:[2,2],crs:'EPSG:4326',
      sources:[{acquired_at:'2024-01-01',provider:'element84-earth-search',valid_fraction:0,sha256:'SECRET_HASH'}]})""")
    assert "Insufficient common pixels" in html and "Sentinel-2 · Earth Search" in html and "0%" in html
    assert "SECRET_HASH" not in html and "<pre" not in html
