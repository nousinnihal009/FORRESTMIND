"""Opening/reloading a workspace must preserve an intentional region selection."""

import pytest

from test_plain_language_ui import run_js


REGIONS = """[{id:'demo',provenance:{synthetic:true}},
  {id:'real',provenance:{synthetic:false,mode:'satellite_acquisition'}},
  {id:'other',provenance:{synthetic:false}}]"""


def test_new_workspace_prefers_real_observations_but_keeps_demo_only_workspaces_usable():
    result = run_js(f"""[preferredRegionId({REGIONS}),
      preferredRegionId([{{id:'demo',provenance:{{synthetic:true}}}}]),preferredRegionId([])]""")
    assert result == ["real", "demo", None]


@pytest.mark.parametrize(
    "requested,current,remembered,expected",
    [
        ("other", "real", "demo", "other"),
        (None, "real", "demo", "real"),
        (None, None, "demo", "demo"),
        (None, "deleted", "other", "other"),
        ("deleted", "missing", "removed", "real"),
    ],
)
def test_selection_precedence_and_deleted_ids(requested, current, remembered, expected):
    import json

    result = run_js(f"""(()=>{{
      state.selected={json.dumps(current)};
      localStorage={{getItem:()=>{json.dumps(remembered)}}};
      return preferredRegionId({REGIONS},{json.dumps(requested)});
    }})()""")
    assert result == expected


@pytest.mark.parametrize("blocked", [False, True])
def test_successful_selection_still_opens_when_storage_is_absent_or_blocked(blocked):
    setup = "localStorage={getItem(){throw new Error('Denied');},setItem(){throw new Error('Quota');}};" if blocked else ""
    result = run_js(f"""(async()=>{{
      {setup}
      state.overview={{regions:{REGIONS}}};
      renderPriorities=renderMap=renderDetail=invalidateScenario=()=>{{}};
      api=async()=>({{id:'real',name:'Actual forest',latest_date:'2025-01-01'}});
      await selectRegion('real');
      return {{remembered:rememberedRegionId(),selected:state.selected,
        title:$('#detail-title').textContent,enabled:!$('#export-region').disabled}};
    }})()""")
    assert result == {"remembered": None, "selected": "real", "title": "Actual forest", "enabled": True}


def test_only_latest_successful_selection_is_remembered():
    result = run_js("""(async()=>{
      const saved=[];let release;
      localStorage={setItem:(key,value)=>saved.push([key,value])};
      renderPriorities=renderMap=renderDetail=invalidateScenario=()=>{};
      api=async path=>path.endsWith('/slow')?new Promise(resolve=>release=resolve):
        {id:'real',name:'Latest forest',latest_date:'2025-01-01'};
      const stale=selectRegion('slow');
      await selectRegion('real');
      release({id:'slow',name:'Old forest'});await stale;
      api=async()=>{throw new Error('Unavailable');};reportError=()=>{};
      await selectRegion('failed');
      return saved;
    })()""")
    assert result == [["forestmind.selectedRegion", "real"]]


def test_workspace_reload_uses_memory_and_empty_workspace_clears_previous_controls():
    result = run_js(f"""(async()=>{{
      localStorage={{getItem:()=> 'other'}};
      renderMetrics=renderPriorities=renderTable=renderProvenance=renderMap=()=>{{}};
      let chosen=null;selectRegion=async id=>chosen=id;
      api=async()=>({{regions:{REGIONS}}});
      await loadWorkspace();
      state.selected='other';state.region={{id:'other'}};
      api=async()=>({{regions:[]}});
      await loadWorkspace();
      return {{chosen,selected:state.selected,region:state.region,
        exportDisabled:$('#export-region').disabled,analysisDisabled:$('#run-analysis').disabled}};
    }})()""")
    assert result == {"chosen": "other", "selected": None, "region": None,
                      "exportDisabled": True, "analysisDisabled": True}
