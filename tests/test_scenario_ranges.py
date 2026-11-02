"""Assumption-envelope arithmetic and persistence; no causal uncertainty claim."""
from copy import deepcopy
from itertools import product
from fastapi.testclient import TestClient
import pytest

from forestmind.api.app import create_app
from forestmind.monitoring.projections import scenario_projection
from test_plain_language_ui import run_js


def analysis():
    return {"latest_date": "2025-01-01", "health_score": 50.,
            "forecast": {"points": [{"horizon_days": 365, "score": 50.}], "caveat": "Test fixture"}}


def test_signed_ranges_contain_central_and_preserve_baseline():
    original = analysis()
    central = {"restoration": 20., "climate_stress": 10.}
    ranges = {"restoration": {"lower": 10., "upper": 30.}, "climate_stress": {"lower": 5., "upper": 20.}}
    point = scenario_projection(original, central, parameter_ranges=ranges)["comparisons"][0]
    plain = scenario_projection(original, central)["comparisons"][0]
    assert [p["score"] for p in point["points"]] == [p["score"] for p in plain["points"]]
    assert point["points"][-1]["sensitivity_lower"] == 43.5
    assert point["points"][-1]["sensitivity_upper"] == 52.5
    assert all(p["sensitivity_lower"] <= p["score"] <= p["sensitivity_upper"] for p in point["points"])
    assert point["points"][0]["sensitivity_lower"] == point["points"][0]["sensitivity_upper"] == 50.
    assert point["uncertainty"]["calibrated"] is False
    assert point["uncertainty"]["confidence_level"] is None
    assert original == analysis()


def test_degenerate_ranges_and_clipped_bounds_are_exact():
    result = scenario_projection(analysis(), {"climate_stress": 100}, parameter_ranges={"climate_stress": {"lower": 100, "upper": 100}})
    assert all(p["score"] == p["sensitivity_lower"] == p["sensitivity_upper"] for p in result["comparisons"][0]["points"])
    assert result["comparisons"][0]["points"][-1]["score"] == 10
    low = analysis()
    low["health_score"] = low["forecast"]["points"][0]["score"] = 5
    result = scenario_projection(low, {"climate_stress": 100}, parameter_ranges={"climate_stress": {"lower": 0, "upper": 100}})
    assert result["comparisons"][0]["points"][-1]["sensitivity_lower"] == 0
    assert result["comparisons"][0]["points"][-1]["sensitivity_upper"] == 5


def test_envelope_matches_all_corners_of_supplied_assumptions():
    central = {"pressure_reduction": 50, "fire_reduction": 30, "restoration": 40, "climate_stress": 60}
    ranges = {"pressure_reduction": {"lower": 0, "upper": 100}, "fire_reduction": {"lower": 10, "upper": 80},
              "restoration": {"lower": 20, "upper": 70}, "climate_stress": {"lower": 5, "upper": 100}}
    bounded = scenario_projection(analysis(), central, parameter_ranges=ranges)["comparisons"][0]["points"]
    corners = [scenario_projection(analysis(), dict(zip(central, corner)))["comparisons"][0]["points"]
               for corner in product(*[(value["lower"], value["upper"]) for value in ranges.values()])]
    for index, point in enumerate(bounded):
        outcomes = [corner[index]["score"] for corner in corners]
        assert point["sensitivity_lower"] == min(outcomes)
        assert point["sensitivity_upper"] == max(outcomes)


@pytest.mark.parametrize("ranges", [{}, {"unknown": {"lower": 0, "upper": 10}},
    {"restoration": {"lower": 21, "upper": 30}}, {"restoration": {"lower": 0, "upper": 19}},
    {"restoration": {"lower": True, "upper": 30}}, {"restoration": {"lower": 0, "upper": float("nan")}},
    {"restoration": {"lower": 0}}, {"restoration": None}])
def test_invalid_ranges_are_rejected_before_calculation(ranges):
    with pytest.raises(ValueError):
        scenario_projection(analysis(), {"restoration": 20}, parameter_ranges=ranges)


@pytest.mark.parametrize("central", [{"restoration": True}, {"restoration": "20"}, {"unknown": 10}])
def test_direct_python_call_cannot_bypass_strict_input_validation(central):
    with pytest.raises(ValueError):
        scenario_projection(analysis(), central)


def test_ranges_do_not_create_a_forecast_for_insufficient_history():
    result = scenario_projection({"health_score": 50, "forecast": {"points": [], "caveat": "Missing history"}},
                                 {"restoration": 20}, parameter_ranges={"restoration": {"lower": 10, "upper": 30}})
    assert not result["available"] and result["comparisons"] == []


def test_api_ranges_retain_exact_snapshot_and_survive_reload(tmp_path):
    app = create_app(tmp_path / "ranges.db")
    with TestClient(app) as client:
        snapshot = app.state.workflow.run("xingu", execute_model=False)
        payload = {"manifest_id": snapshot["manifest"]["id"], "restoration": 20,
                   "parameter_ranges": {"restoration": {"lower": 10, "upper": 30}}}
        response = client.post("/api/v1/regions/xingu/workflow/scenarios", json=payload)
        assert response.status_code == 201, response.text
        result = response.json()
        assert result["manifest"] == snapshot["manifest"]
        assert result["comparisons"][0]["uncertainty"]["kind"] == "explicit_assumption_range"
        before = client.get("/api/v1/scenarios/" + result["id"]).json()
        invalid = deepcopy(payload)
        invalid["parameter_ranges"]["restoration"]["lower"] = 25
        assert client.post("/api/v1/regions/xingu/workflow/scenarios", json=invalid).status_code == 422
        invalid["parameter_ranges"]["restoration"]["lower"] = True
        assert client.post("/api/v1/regions/xingu/workflow/scenarios", json=invalid).status_code == 422
        assert client.post("/api/v1/regions/madeira/workflow/scenarios", json=payload).status_code == 404
    with TestClient(create_app(tmp_path / "ranges.db")) as client:
        assert client.get("/api/v1/scenarios/" + result["id"]).json() == before


def test_browser_range_parser_preserves_explicit_values_and_rejects_empty_or_inverted():
    result = run_js("""(()=>{
      $('#scenario-ranges-enabled').checked=true;
      const values={pressure_reduction:20,fire_reduction:20,restoration:20,climate_stress:20};
      for(const key of Object.keys(values)){$('#range-'+key+'-lower').value='10';$('#range-'+key+'-upper').value='30';}
      const parsed=scenarioAssumptionRanges(values);
      $('#range-restoration-lower').value='';let message;
      try{scenarioAssumptionRanges(values);}catch(error){message=error.message;}
      $('#scenario-ranges-enabled').checked=false;
      return {parsed,message,disabled:scenarioAssumptionRanges(values)};
    })()""")
    assert result["parsed"]["restoration"] == {"lower": 10, "upper": 30}
    assert "must contain" in result["message"]
    assert result["disabled"] is None


@pytest.mark.parametrize("value", ["' '", "'NaN'", "'101'", "'-1'", "'21'"])
def test_browser_rejects_invalid_lower_bound(value):
    result = run_js(f"""(()=>{{
      $('#scenario-ranges-enabled').checked=true;
      const values={{pressure_reduction:20,fire_reduction:20,restoration:20,climate_stress:20}};
      for(const key of Object.keys(values)){{$('#range-'+key+'-lower').value='10';$('#range-'+key+'-upper').value='30';}}
      $('#range-restoration-lower').value={value};
      try{{scenarioAssumptionRanges(values);return null;}}catch(error){{return error.message;}}
    }})()""")
    assert "must contain" in result


def test_range_controls_follow_horizon_and_preserve_cleared_input():
    result = run_js("""(()=>{
      $('#scenario-horizon').value='five-year';$('#scenario-ranges-enabled').checked=true;
      $('#range-restoration-lower').value='';$('#range-restoration-upper').value='30';
      syncScenarioRanges();
      const visible=!$('#scenario-range-controls').hidden,enabled=!$('#range-restoration-lower').disabled;
      $('#scenario-horizon').value='baseline';syncScenarioRanges();
      const hidden=$('#scenario-range-controls').hidden,disabled=$('#range-restoration-lower').disabled;
      $('#scenario-horizon').value='five-year';syncScenarioRanges();
      return {visible,enabled,hidden,disabled,cleared:$('#range-restoration-lower').value,upper:$('#range-restoration-upper').value};
    })()""")
    assert result == {"visible": True, "enabled": True, "hidden": True, "disabled": True, "cleared": "", "upper": "30"}


def test_workflow_shortcut_reveals_range_controls():
    result = run_js("""(()=>{
      let click;$('#adjust-five-year').addEventListener=(name,handler)=>{click=handler;};
      $('#scenario-horizon').value='baseline';$('#scenario-ranges-enabled').checked=false;
      $('#five-year-scenario').value='0';syncScenarioRanges();
      wireFiveYearScenarios({scenarios:{baseline:[{date:'2025-01-01',score:50}],comparisons:[{
        name:'Fixture',points:[{date:'2030-01-01',score:55}],five_year_difference:5,parameters:{restoration:10}}]},forecast:{historical:[]}});
      click();
      return {horizon:$('#scenario-horizon').value,hidden:$('#scenario-range-controls').hidden,disabled:$('#range-restoration-lower').disabled};
    })()""", extra_scripts=("workflow.js",))
    assert result == {"horizon": "five-year", "hidden": False, "disabled": True}


def test_chart_draws_assumption_band_without_probability_label():
    result = run_js("""chart([],[{date:'2025-01-01',score:50},{date:'2030-01-01',score:50}],{
      alternative:[{date:'2025-01-01',score:50,sensitivity_lower:50,sensitivity_upper:50},
                   {date:'2030-01-01',score:51,sensitivity_lower:45,sensitivity_upper:57}],
      label:'Explicit scenario sensitivity'})""")
    assert 'fill="#b3935c" fill-opacity=".18"' in result
    assert "supplied assumptions; not a confidence interval" in result
    assert "assumption sensitivity 45.0–57.0" in result
    assert "95%" not in result


def test_changed_inputs_discard_in_flight_scenario_without_saving_mixed_assumptions():
    result = run_js("""(async()=>{
      $('#scenario-horizon').value='five-year';$('#scenario-ranges-enabled').checked=false;
      let release,posts=0;
      ensureWorkflowSnapshot=()=>new Promise(resolve=>release=resolve);
      api=async()=>{posts++;return {};};
      const pending=runScenario({preventDefault(){}});
      invalidateScenario();
      release({manifest:{id:'record'},forecast:{historical:[]}});await pending;
      return {posts,retained:state.scenario,pending:state.scenarioPending,enabled:!$('#compare-scenario').disabled,html:$('#scenario-results').innerHTML};
    })()""")
    assert result["posts"] == 0
    assert result["retained"] is None and not result["pending"] and result["enabled"]
    assert "Assumptions changed" in result["html"]


def test_late_saved_scenario_cannot_replace_newer_input_state():
    result = run_js("""(async()=>{
      let release;api=()=>new Promise(resolve=>release=resolve);
      const pending=runScenario({preventDefault(){}});invalidateScenario();
      release({id:'outdated',baseline:[],scenario:[]});await pending;
      return {retained:state.scenario,enabled:!$('#compare-scenario').disabled,html:$('#scenario-results').innerHTML};
    })()""")
    assert result["retained"] is None and result["enabled"]
    assert "Assumptions changed" in result["html"]


def test_browser_submits_and_displays_explicit_ranges_with_exportable_evidence():
    result = run_js("""(async()=>{
      $('#scenario-horizon').value='five-year';$('#scenario-ranges-enabled').checked=true;
      for(const key of ['pressure_reduction','fire_reduction','restoration','climate_stress']){
        $('#range-'+key+'-lower').value='0';$('#range-'+key+'-upper').value='10';}
      $('#scenario-results .scenario-assumptions').insertAdjacentHTML=(where,html)=>{
        $('#scenario-results .scenario-assumptions').innerHTML+=html;};
      $('#scenario-results .chart-legend').insertAdjacentHTML=(where,html)=>{
        $('#scenario-results .chart-legend').innerHTML+=html;};
      ensureWorkflowSnapshot=async()=>({manifest:{id:'frozen'},forecast:{historical:[]}});toast=()=>{};
      let submitted;
      api=async(path,options)=>{
        if(!path.endsWith('/workflow/scenarios'))return {name:'Controlled UI fixture'};
        submitted=JSON.parse(options.body);
        return {id:'fixture',baseline:[{date:'2030-01-01',horizon_days:1825,score:50}],limitations:['Unvalidated assumptions.'],assumptions:['Fixed coefficients.'],
          comparisons:[{parameter_ranges:submitted.parameter_ranges,uncertainty:{kind:'explicit_assumption_range',calibrated:false,confidence_level:null,definition:'Not a confidence interval.'},
            points:[{date:'2030-01-01',horizon_days:1825,score:50,sensitivity_lower:46,sensitivity_upper:56}]}]};
      };
      await runScenario({preventDefault(){}});
      return {submitted,evidence:state.scenario.comparisons[0],html:$('#scenario-results').innerHTML,
        range:$('#scenario-results .scenario-assumptions').innerHTML,legend:$('#scenario-results .chart-legend').innerHTML,enabled:!$('#compare-scenario').disabled};
    })()""")
    assert result["submitted"]["manifest_id"] == "frozen"
    assert "region_id" not in result["submitted"]
    assert result["submitted"]["parameter_ranges"]["restoration"] == {"lower": 0, "upper": 10}
    assert result["evidence"]["uncertainty"]["calibrated"] is False
    assert result["evidence"]["uncertainty"]["confidence_level"] is None
    assert "Export comparison" in result["html"] and "assumption sensitivity 46.0–56.0" in result["html"]
    assert "Not a confidence interval" in result["range"] and "56.00" in result["range"]
    assert "Supplied assumption range" in result["legend"] and result["enabled"]
