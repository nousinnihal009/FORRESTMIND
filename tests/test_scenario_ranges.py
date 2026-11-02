"""Assumption-envelope arithmetic and persistence; no causal uncertainty claim."""
from copy import deepcopy
import json

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
