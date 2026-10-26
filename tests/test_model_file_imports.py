"""Exercise model file previews and async import state without browser-only stubs."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def execute(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the model import interaction checks")
    source = r"""
const vm=require('vm'),fs=require('fs'),elements={};
function element(key){return elements[key]??(elements[key]={innerHTML:'',textContent:'',hidden:false,disabled:false,value:'',
  handlers:{},addEventListener(name,handler){this.handlers[name]=handler;},reset(){},showModal(){this.open=true;},close(){this.open=false;this.handlers.close?.();}});}
const context={document:{querySelector:element,querySelectorAll:()=>[]},console,TextEncoder,URL,Blob,setTimeout,clearTimeout};
vm.createContext(context);
const app=fs.readFileSync(APP_PATH,'utf8').split('hydrateIcons();route();')[0];
vm.runInContext(app,context);
vm.runInContext(fs.readFileSync(MODEL_PATH,'utf8'),context);
vm.runInContext("state.region={name:'Test forest'};state.selected='forest';state.tab='models';",context);
Promise.resolve(vm.runInContext(SCRIPT,context)).then(value=>process.stdout.write(JSON.stringify(value))).catch(error=>{console.error(error);process.exitCode=1;});
"""
    for key, value in {
        "APP_PATH": str(ROOT / "forestmind/web/app.js"),
        "MODEL_PATH": str(ROOT / "forestmind/web/model-runs.js"),
        "SCRIPT": script,
    }.items():
        source = source.replace(key, json.dumps(value))
    result = subprocess.run([node, "-e", source], capture_output=True, text=True,
                            encoding="utf-8", check=True, timeout=30)
    return json.loads(result.stdout)


def report():
    return {"kind": "forestmind_inference", "schema_version": 1, "region_id": "forest",
            "observed_through": "2024-02-29", "created_at": "2024-03-01T00:00:00Z",
            "checkpoint": {"name": "Partial model", "backbone": "custom_vit",
                           "trained_on_synthetic": None, "training_source": "unknown",
                           "sha256": "checkpoint-private-fingerprint"},
            "input": {"name": "sequence", "synthetic": False, "timesteps": 2,
                      "sha256": "input-private-fingerprint", "observed_start": "2024-01-01"},
            "predictions": {"health": {"state": None, "score": 0, "class_scores": []},
                            "drivers": [{"name": "Wildfire", "score": None}],
                            "trajectory": [{"horizon_days": 30, "score": None, "stddev": None},
                                           {"horizon_days": 365, "score": 0, "stddev": 0}]},
            "caveats": ["No trained class output."]}


def test_preview_distinguishes_available_zero_and_unknown_training_origin():
    result = execute(f"modelFilePreview({json.dumps(report())},'forest')")
    assert result["targets"] == "Condition score · 1 forecast horizon"
    assert result["source"] == "Declared observed inputs · training data origin unknown"
    assert result["period"] == "Jan 1, 2024 – Feb 29, 2024"


@pytest.mark.parametrize("change,expected", [
    ({"region_id": "another-forest"}, "different monitoring region"),
    ({"observed_through": "2023-02-29"}, "invalid observation period"),
    ({"kind": "some_other_record"}, "different format"),
    ({"predictions": {}}, "missing its model"),
])
def test_wrong_or_incomplete_report_gets_actionable_message(change, expected):
    payload = {**report(), **change}
    result = execute(f"(() => {{try{{modelFilePreview({json.dumps(payload)},'forest');return 'accepted';}}catch(error){{return error.message;}}}})()")
    assert expected in result
    assert "region_id" not in result


def test_invalid_or_removed_file_clears_previous_preview_and_submit_permission():
    result = execute(f"""(async () => {{
      openModelImport('forest');
      await loadModelFile({{target:{{files:[{{name:'valid.json',size:10,text:async()=>JSON.stringify({json.dumps(report())})}}]}}}});
      const before={{disabled:$('#submit-model-import').disabled,hidden:$('#model-import-preview').hidden}};
      await loadModelFile({{target:{{files:[{{name:'bad.json',size:10,text:async()=>'<html>not a report</html>'}}]}}}});
      const invalid={{disabled:$('#submit-model-import').disabled,hidden:$('#model-import-preview').hidden,report:modelImportReport,error:$('#model-import-error').textContent}};
      await loadModelFile({{target:{{files:[]}}}});
      return {{before,invalid,removed:{{disabled:$('#submit-model-import').disabled,hidden:$('#model-import-preview').hidden,report:modelImportReport}}}};
    }})()""")
    assert result["before"] == {"disabled": False, "hidden": False}
    assert result["invalid"]["disabled"] and result["invalid"]["hidden"]
    assert result["invalid"]["report"] is None
    assert "could not be read as a model report" in result["invalid"]["error"]
    assert "<html>" not in result["invalid"]["error"]
    assert result["removed"] == {"disabled": True, "hidden": True, "report": None}


def test_oversize_does_not_read_file_or_leave_old_report_selected():
    result = execute("""(async () => {
      openModelImport('forest');let reads=0;
      await loadModelFile({target:{files:[{name:'large.json',size:2*1024*1024+1,text:async()=>{reads++;return '{}';}}]}});
      return {reads,disabled:$('#submit-model-import').disabled,report:modelImportReport,error:$('#model-import-error').textContent};
    })()""")
    assert result["reads"] == 0 and result["disabled"]
    assert result["report"] is None and "smaller than 2 MB" in result["error"]


def test_slow_old_file_and_closed_dialog_cannot_restore_stale_report():
    result = execute(f"""(async () => {{
      openModelImport('forest');let resolveOld;
      const old=loadModelFile({{target:{{files:[{{name:'old.json',size:10,text:()=>new Promise(resolve=>resolveOld=resolve)}}]}}}});
      await loadModelFile({{target:{{files:[{{name:'new.json',size:10,text:async()=>JSON.stringify({json.dumps(report())})}}]}}}});
      resolveOld('invalid');await old;
      const selected={{name:$('#model-file-name').textContent,errorHidden:$('#model-import-error').hidden,disabled:$('#submit-model-import').disabled}};
      let resolveClosed;
      const closed=loadModelFile({{target:{{files:[{{name:'closing.json',size:10,text:()=>new Promise(resolve=>resolveClosed=resolve)}}]}}}});
      $('#model-import-dialog').close();resolveClosed(JSON.stringify({json.dumps(report())}));await closed;
      return {{selected,closed:{{report:modelImportReport,disabled:$('#submit-model-import').disabled,hidden:$('#model-import-preview').hidden}}}};
    }})()""")
    assert result["selected"] == {"name": "new.json", "errorHidden": True, "disabled": False}
    assert result["closed"] == {"report": None, "disabled": True, "hidden": True}


def test_region_change_blocks_submission_without_exposing_internal_id():
    result = execute(f"""(async () => {{
      openModelImport('forest');modelImportReport={json.dumps(report())};state.selected='other';
      let calls=0;api=async()=>{{calls++;}};
      await importModelRun({{preventDefault(){{}}}});
      return {{calls,error:$('#model-import-error').textContent}};
    }})()""")
    assert result["calls"] == 0
    assert "monitoring region changed" in result["error"]
    assert "region_id" not in result["error"]


def test_visible_provenance_has_no_raw_fingerprint_and_preserves_export():
    result = execute(f"""(() => {{
      const report={json.dumps(report())},before=JSON.stringify(report);let exported;
      downloadJSON=data=>{{exported=data;}};
      showModelRun({{id:'internal-run-id',report}});
      $('#export-model-run').handlers.click();
      return {{html:$('#model-run-detail').innerHTML,unchanged:before===JSON.stringify(report),exported}};
    }})()""")
    assert result["unchanged"] and result["exported"] == report()
    assert "checkpoint-private-fingerprint" not in result["html"]
    assert "input-private-fingerprint" not in result["html"]
    assert "<code>" not in result["html"]
    assert "Training data origin unknown" in result["html"]
    assert "No trained condition-class output" in result["html"] or "no trained condition-class output" in result["html"]


def test_browser_dialog_ids_match_import_handlers_without_json_editor():
    html = (ROOT / "forestmind/web/index.html").read_text(encoding="utf-8")
    code = (ROOT / "forestmind/web/model-runs.js").read_text(encoding="utf-8")
    for name in ("model-preview-name", "model-preview-period", "model-preview-targets", "model-preview-source"):
        assert f'id="{name}"' in html
    assert "model-import-json" not in code
    assert 'id="model-import-json"' not in html
