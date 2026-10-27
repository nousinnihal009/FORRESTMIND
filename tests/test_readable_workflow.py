"""Readable result presentation without weakening data or area constraints."""
from test_plain_language_ui import run_js


def workflow_js(script):
    return run_js(script, extra_scripts=("workflow.js",))


COVER = """const geometry={type:'Polygon',coordinates:[[[0,0],[1,0],[1,1],[0,1],[0,0]]]};
const payload={data_kind:'classified_forest_cover',observed_at:'2024-01-01',source:'Example survey',
 classification:'Verified tree classes',license:'CC BY 4.0',forest:[[true,false],[true,true]],
 valid_mask:[[true,true],[false,true]],geometry:{coordinates:geometry.coordinates,type:'Polygon'}};"""


def test_equivalent_geometry_key_order_is_accepted_and_record_is_unchanged():
    result = workflow_js("(()=>{" + COVER + "const before=JSON.stringify(payload);return {html:coverFilePreview(payload,geometry),unchanged:before===JSON.stringify(payload)};})()")
    assert result["unchanged"]
    assert "67%" in result["html"] and "75%" in result["html"]
    assert "Example survey" in result["html"] and "Jan 1, 2024" in result["html"]
    assert "<pre" not in result["html"]


def test_changed_coordinates_are_rejected_even_when_shape_and_counts_match():
    result = workflow_js("(()=>{" + COVER + "payload.geometry={type:'Polygon',coordinates:[[[2,0],[3,0],[3,1],[2,1],[2,0]]]};try{coverFilePreview(payload,geometry);}catch(error){return error.message;}})()")
    assert "different area" in result


def test_unknown_coverage_is_not_displayed_as_zero_forest():
    result = workflow_js("(()=>{" + COVER + "payload.valid_mask=[[false,false],[false,false]];try{coverFilePreview(payload,geometry);}catch(error){return error.message;}})()")
    assert "no usable classified area" in result


def test_integer_counts_and_absent_values_are_distinct():
    result = workflow_js("workflowFacts([['Recorded observations',4],['Missing months',0],['Unavailable measurement',null]])")
    assert '<dd>4</dd>' in result and '<dd>0</dd>' in result
    assert '<dd>Not available</dd>' in result
    assert '4.000' not in result


def test_unavailable_sections_keep_reasons_and_supported_measurements_remain_visible():
    result = workflow_js("""renderWorkflowSections({
      observations:[{date:'2024-01-01',ndvi:0,ndmi:null,nbr:.1,valid_fraction:.75}],
      condition:{score:0,status:'watch',change:null},
      forecast:{points:[],caveat:'At least 12 consecutive usable months are required.'},
      recovery:{available:false,reason:'Need 24 baseline months and 3 follow-up months.'},
      fragmentation:{available:false,reason:'No dated classification is available.'},
      radar:{available:false,reason:'No compatible observations.'},
      model:{available:false,reason:'No trusted local model is registered. Supply a real-trained compatible checkpoint and independent evaluation evidence in FORESTMIND_MODEL_REGISTRY.'}})""")
    assert '0.00 / 100' in result
    assert 'Need 24 baseline months' in result and 'No dated classification' in result
    assert 'No compatible trained model' in result
    assert 'FORESTMIND_MODEL_REGISTRY' not in result
    assert '<pre' not in result and '<textarea' not in result


def test_manifest_has_readable_dates_sources_and_no_identifiers_or_fingerprints():
    result = workflow_js("""renderWorkflowManifest({created_at:'2024-02-02T12:00:00Z',manifest:{
      id:'private-record-id',observed_through:'2024-01-01',observation_count:4,synthetic:false,
      context_sha256:{radar:'private-hash'},model:{id:'technical-model-id'}}},{name:'Test area'})""")
    assert 'Jan 1, 2024' in result and 'Feb 2, 2024' in result
    assert 'Radar' in result
    assert 'private-record-id' not in result and 'private-hash' not in result
    assert 'technical-model-id' not in result
