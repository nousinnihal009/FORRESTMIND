"""Presentation contracts for measured coverage and unsupported learned outputs."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def render(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to execute frontend rendering contracts")
    paths = [str(ROOT / "forestmind" / "web" / name) for name in ("workflow.js", "model-runs.js")]
    source = f"""
const vm=require('vm'),fs=require('fs'),elements={{}};
const context={{esc:s=>String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;'),
numeric:v=>Number.isFinite(v)?String(v):'—',pct:v=>Number.isFinite(v)?String(v*100)+'%':'—',
badge:s=>String(s),icon:()=>'',dateLabel:s=>String(s),
$:key=>elements[key]??(elements[key]={{innerHTML:'',addEventListener:()=>{{}}}})}};
vm.createContext(context);
for(const path of {json.dumps(paths)})vm.runInContext(fs.readFileSync(path,'utf8'),context);
process.stdout.write(JSON.stringify(vm.runInContext({json.dumps(script)},context)));
"""
    result = subprocess.run([node, "-e", source], capture_output=True, text=True,
                            encoding="utf-8", check=True, timeout=30)
    return json.loads(result.stdout)


def test_missing_driver_has_no_zero_bar_and_observed_zero_remains_visible():
    output = render("modelScoreRows([{name:'Missing <driver>',score:null},{name:'Measured zero',score:0}])")
    assert "Missing &lt;driver&gt;" in output
    assert "No trained output" in output and "<strong>Unavailable</strong>" in output
    assert output.count('class="tiny-track"') == 1
    assert output.count('width:0%') == 1
    assert "<strong>0</strong>" in output


def test_empty_class_scores_show_why_no_classification_exists():
    output = render("modelScoreRows([])")
    assert "Unavailable" in output and "no trained output" in output
    assert "tiny-track" not in output


def test_missing_trajectory_and_unestimated_spread_are_distinct_from_zero():
    output = render("modelTrajectoryRows([{horizon_days:30,score:null,stddev:null},{horizon_days:365,score:0,stddev:null},{horizon_days:1825,score:10,stddev:0}])")
    assert "Unavailable · no trained output" in output
    assert "<td>Not estimated</td>" in output
    assert "<td>365 days</td><td>0</td>" in output
    assert "<td>1825 days</td><td>10</td><td>0</td>" in output


def test_aligned_input_summary_retains_partial_feature_coverage_and_no_prediction():
    output = render("""renderAlignedInputs({sequence:{selected_acquisitions:4,alignment:{
      parameters:{climate_window_days:30,climate_lag_days:5},
      radar:{max_age_days:12,matches:[{scene_id:'one'},{scene_id:null},{scene_id:null},{scene_id:null}]},
      climate:{windows:[{mask:[false,false,false,false]},{mask:[true,true,true,true]},
        {mask:[true,false,true,true]},{mask:[false,false,false,false]}]}}},
      model:{available:false,reason:'No trusted local model is registered. Supply a real-trained compatible checkpoint and independent evaluation evidence in FORESTMIND_MODEL_REGISTRY.'}})""")
    assert "1 / 4 image dates" in output
    assert "1 / 4 image dates with all measurements" in output
    assert "1 image date has only some measurements" in output
    assert "30 days" in output and "5 days before" in output
    assert "No model prediction was produced" in output
    assert "No compatible trained model is registered for this area" in output
    assert "FORESTMIND_MODEL_REGISTRY" not in output
    assert "historical availability has not been verified" in output


@pytest.mark.parametrize("error_location", ["top_level", "manifest"])
def test_alignment_failures_are_escaped_and_not_claimed_as_available_inputs(error_location):
    report = {"sequence": {"selected_acquisitions": 2}, "radar": {"available": True}}
    errors = {"alignment_errors": {"radar": "Wrong <grid> or changed source"}}
    report.update(errors if error_location == "top_level" else {"manifest": errors})
    output = render(f"renderAlignedInputs({json.dumps(report)})")
    assert "Could not prepare" in output
    assert "Wrong &lt;grid&gt; or changed source" in output
    assert "Collect climate context" in output
    assert "Some evidence could not be used" in output
    assert "Collected radar evidence could not be used with these images" in output


def test_failed_image_preparation_shows_the_reason_without_implying_ready_inputs():
    output = render("""renderAlignedInputs({sequence:{available:false,reason:'No cached <imagery>'},
      model:{available:false,reason:'A prepared optical sequence is required.'}})""")
    assert "Images could not be prepared for a model" in output
    assert "No cached &lt;imagery&gt;" in output
    assert "No model prediction was produced" in output
    assert "A prepared optical sequence is required" in output
    assert "0 image dates" not in output and "<table>" not in output


def test_missing_health_head_does_not_show_a_zero_score_or_empty_class_panel():
    output = render("""(() => {showModelRun({id:'run',verification:'local_registered_execution',report:{
      region_id:'region',observed_through:'2024-01-01',created_at:'2024-01-02',
      checkpoint:{name:'model',backbone:'custom_vit',trained_on_synthetic:false,sha256:'hash',training_source:'independent'},
      input:{name:'sequence',timesteps:3,synthetic:false,sha256:'hash'},
      predictions:{health:{state:null,score:null,class_scores:[]},drivers:[],
        trajectory:[{horizon_days:30,score:null,stddev:null}]},caveats:['No trained health output.']}});
      return $('#model-run-detail').innerHTML;})()""")
    assert "Condition class unavailable" in output
    assert "No trained condition-score output" in output
    assert "/ 100 model score" not in output
    assert "<strong>Unavailable</strong>" in output


def test_saved_radar_caveat_is_clarified_without_changing_evidence_or_real_zero():
    legacy = "This separately dated context is not inserted into optical model inputs or monthly observations. A paired trained multimodal model remains a separate requirement."
    record = {"available": True, "sha256": "saved-source-hash", "scenes": [],
              "comparison": {"vv_change_db": 0, "vh_change_db": None},
              "limitations": [legacy, "Unverified <terrain> coverage"]}
    result = render(f"""(() => {{const data={json.dumps(record)},before=JSON.stringify(data);
      const html=renderRadarContext(data);return {{html,unchanged:before===JSON.stringify(data)}};}})()""")
    assert result["unchanged"] is True
    assert legacy not in result["html"]
    assert "Individual radar scenes can be prepared for a compatible trained model" in result["html"]
    assert "pair comparison remains descriptive evidence" in result["html"]
    assert "Unverified &lt;terrain&gt; coverage" in result["html"]
    assert "<strong>0</strong>" in result["html"]
    assert "<strong>—</strong>" in result["html"]
