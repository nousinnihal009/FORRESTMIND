"""Exercise user-facing error handling and observation import state in JavaScript."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_js(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for interactive presentation checks")
    source = f"""
const vm=require('vm'),fs=require('fs'),elements={{}};
const element=key=>elements[key]??(elements[key]={{innerHTML:'',textContent:'',hidden:false,
 disabled:false,value:'',reset(){{}},showModal(){{}},close(){{}},addEventListener(){{}}}});
const context={{document:{{querySelector:element,querySelectorAll:()=>[]}},console,TextEncoder,
 FormData:class{{constructor(){{return new Map([['region_id','area'],['fire_reduction','0'],['pressure_reduction','0'],['restoration','0'],['climate_stress','0']]);}}}},
 location:{{hash:''}},setTimeout,clearTimeout}};
vm.createContext(context);
vm.runInContext(fs.readFileSync({json.dumps(str(ROOT/'forestmind/web/app.js'))},'utf8').split('hydrateIcons();route();')[0],context);
Promise.resolve(vm.runInContext({json.dumps(script)},context)).then(value=>process.stdout.write(JSON.stringify(value))).catch(error=>{{console.error(error);process.exit(1);}});
"""
    result = subprocess.run([node, "-e", source], text=True, encoding="utf-8", capture_output=True, check=True, timeout=30)
    return json.loads(result.stdout)


def test_errors_are_actionable_without_exposing_internal_paths_or_traces():
    result = run_js("""[
      userMessage(new Error('Failed to fetch')),
      userMessage('ValueError: failed C:/private/data/file.py','Could not load the observations. Try again.'),
      userMessage('The observation date must be before today.'),
      validationMessage([{loc:['body','observations',2,'valid_fraction'],type:'missing',msg:'Field required'}])
    ]""")
    assert "local workspace could not be reached" in result[0]
    assert result[1] == "Could not load the observations. Try again."
    assert result[2] == "The observation date must be before today."
    assert result[3] == "Observations · record 3 · Usable pixels: This information is required."


def test_api_server_error_does_not_render_internal_exception_details():
    result = run_js("""(async()=>{fetch=async()=>({ok:false,status:500,json:async()=>({detail:'private service traceback'})});
      try{await api('/regions');}catch(error){return error.message;}})()""")
    assert "could not complete this request" in result
    assert "traceback" not in result


def test_import_preview_preserves_demo_label_and_missing_measurements():
    result = run_js("""regionFilePreview({name:'Example',geometry:{type:'Polygon',coordinates:[]},
      observations:[{date:'2024-02-01',ndvi:null},{date:'2024-01-01',ndvi:0}],synthetic:true})""")
    assert result["count"] == 2
    assert result["period"] == "Jan 1, 2024 – Feb 1, 2024"
    assert result["source"] == "Synthetic example · not real observations"


@pytest.mark.parametrize("value", ["null", "[]", "{name:'Empty'}", "{name:'Bad date',geometry:{type:'Polygon',coordinates:[]},observations:[{date:'2024-02-30'}]}"])
def test_invalid_observation_files_explain_what_needs_fixing(value):
    result = run_js(f"""(()=>{{try{{regionFilePreview({value});return null;}}catch(error){{return error.message;}}}})()""")
    assert result and "TypeError" not in result and "JSON.parse" not in result


def test_invalid_replacement_clears_successful_preview_and_disables_import():
    result = run_js("""(async()=>{
      const good={name:'Example',geometry:{type:'Polygon',coordinates:[]},observations:[{date:'2024-01-01'}]};
      await loadRegionFile({target:{files:[{name:'valid.json',size:100,text:async()=>JSON.stringify(good)}]}});
      const ready=!$('#submit-import').disabled&&!$('#import-preview').hidden;
      await loadRegionFile({target:{files:[{name:'broken.json',size:10,text:async()=>'{broken'}]}});
      return {ready,disabled:$('#submit-import').disabled,hidden:$('#import-preview').hidden,
        retained:regionImportFile,error:$('#import-error').textContent};})()""")
    assert result["ready"] and result["disabled"] and result["hidden"]
    assert result["retained"] is None
    assert "could not be read" in result["error"]


def test_older_slow_file_cannot_replace_newer_selection():
    result = run_js("""(async()=>{
      let release;
      const pending=loadRegionFile({target:{files:[{name:'old.json',size:10,text:()=>new Promise(resolve=>release=resolve)}]}});
      await loadRegionFile({target:{files:[]}});
      release(JSON.stringify({name:'Old',geometry:{type:'Polygon',coordinates:[]},observations:[{date:'2024-01-01'}]}));await pending;
      return {disabled:$('#submit-import').disabled,hidden:$('#import-preview').hidden,payload:regionImportFile};})()""")
    assert result == {"disabled": True, "hidden": True, "payload": None}


def test_five_year_scenario_with_no_forecast_shows_missing_history_instead_of_crashing():
    result = run_js("""(async()=>{
      $('#scenario-horizon').value='five-year';
      ensureWorkflowSnapshot=async()=>({manifest:{id:'record'},forecast:{historical:[]}});
      api=async path=>path.endsWith('/workflow/scenarios')?{baseline:[],comparisons:[],limitations:[]}:{};
      await runScenario({preventDefault(){}});
      return {html:$('#scenario-results').innerHTML,enabled:!$('#compare-scenario').disabled};})()""")
    assert result["enabled"]
    assert "More observations are needed" in result["html"]


def test_static_markup_has_no_code_editors_or_json_output_blocks():
    for path in (ROOT / "forestmind/web").glob("*"):
        if path.suffix not in {".js", ".html"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "<pre>" not in text, path.name
        assert 'id="import-json"' not in text, path.name
        assert 'id="model-import-json"' not in text, path.name
        assert 'id="cover-import-json"' not in text, path.name
