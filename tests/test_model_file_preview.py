"""File import state and missing model outputs in the human-readable preview."""
from test_plain_language_ui import run_js


FIXTURE = """const report={schema_version:1,kind:'forestmind_inference',region_id:'area',
 observed_through:'2024-01-01',checkpoint:{name:'Research model',trained_on_synthetic:true},
 input:{synthetic:false,timesteps:2},predictions:{health:{score:null,class_scores:[]},
 drivers:[{name:'No result',score:null},{name:'Zero score',score:0}],
 trajectory:[{horizon_days:30,score:null}]}};"""


def model_js(script):
    return run_js(script, extra_scripts=("model-runs.js",))


def test_partial_model_preview_keeps_zero_score_and_excludes_missing_heads():
    result = model_js("(()=>{" + FIXTURE + "return modelFilePreview(report,'area');})()")
    assert result["targets"] == "1 candidate driver score"
    assert result["source"].startswith("Synthetic research data")


def test_wrong_region_file_is_rejected_before_import():
    result = model_js("(()=>{" + FIXTURE + "try{modelFilePreview(report,'different');}catch(error){return error.message;}})()")
    assert "different monitoring region" in result


def test_invalid_file_replacement_clears_preview_and_old_payload():
    result = model_js("(async()=>{" + FIXTURE + """
      modelImportRegion='area';
      await loadModelFile({target:{files:[{name:'good.json',size:100,text:async()=>JSON.stringify(report)}]}});
      const ready=!$('#submit-model-import').disabled;
      await loadModelFile({target:{files:[{name:'bad.json',size:2,text:async()=>'{'}]}});
      return {ready,disabled:$('#submit-model-import').disabled,hidden:$('#model-import-preview').hidden,
       report:modelImportReport,error:$('#model-import-error').textContent};})()""")
    assert result["ready"] and result["disabled"] and result["hidden"]
    assert result["report"] is None and "could not be read" in result["error"]


def test_late_file_read_is_ignored_after_clearing_selection():
    result = model_js("(async()=>{" + FIXTURE + """
      modelImportRegion='area';let finish;
      const pending=loadModelFile({target:{files:[{name:'old.json',size:100,text:()=>new Promise(resolve=>finish=resolve)}]}});
      await loadModelFile({target:{files:[]}});finish(JSON.stringify(report));await pending;
      return {disabled:$('#submit-model-import').disabled,report:modelImportReport};})()""")
    assert result == {"disabled": True, "report": None}


def test_changed_selection_cannot_save_model_to_the_wrong_region():
    result = model_js("(async()=>{" + FIXTURE + """
      modelImportRegion='area';modelImportReport=report;state.selected='other';
      let calls=0;api=async()=>{++calls;};await importModelRun({preventDefault(){}});
      return {calls,error:$('#model-import-error').textContent};})()""")
    assert result["calls"] == 0
    assert "selected monitoring region changed" in result["error"]
