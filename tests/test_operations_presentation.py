"""Readable operational screens keep underlying data and actions intact."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def render(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for interface behavior checks")
    web = json.dumps(str(ROOT / "forestmind" / "web"))
    source = f"""
const vm=require('vm'),fs=require('fs'),path=require('path'),elements={{}},downloads=[];
function element(selector){{return elements[selector]??(elements[selector]={{innerHTML:'',textContent:'',value:'',hidden:false,disabled:false,listeners:{{}},addEventListener(name,fn){{this.listeners[name]=fn;}},setAttribute(){{}},querySelector(s){{return element(s);}},querySelectorAll(){{return [];}}}});}}
const context={{document:{{querySelector:element,querySelectorAll:()=>[]}},location:{{hash:''}},console,Intl,Date,JSON,Number,Promise,setTimeout,clearTimeout,downloads,elements}};
vm.createContext(context);
const web={web};
vm.runInContext(fs.readFileSync(path.join(web,'app.js'),'utf8').split('hydrateIcons();route();')[0],context);
vm.runInContext(fs.readFileSync(path.join(web,'acquisition.js'),'utf8').replace(/\\ninitializeAreaMap\\(\\);\\s*$/,''),context);
vm.runInContext(fs.readFileSync(path.join(web,'coverage.js'),'utf8'),context);
vm.runInContext(fs.readFileSync(path.join(web,'operations.js'),'utf8'),context);
vm.runInContext("downloadJSON=(data,name)=>downloads.push({{data,name}})",context);
(async()=>process.stdout.write(JSON.stringify(await vm.runInContext({json.dumps(script)},context))))().catch(error=>{{console.error(error);process.exit(1);}});
"""
    completed = subprocess.run([node, "-e", source], capture_output=True, text=True,
                               encoding="utf-8", check=True, timeout=30)
    return json.loads(completed.stdout)


def visible(markup):
    return re.sub(r"<[^>]*>", " ", markup)


def test_saved_collection_keeps_source_boundaries_missing_values_and_zero_threshold():
    result = render("""savedAcquisitionSummary({configuration:{request:{provider:'earth-search',
      start:'2024-07-01',end:'2024-08-31',bbox:[-55.1,-3.5,-55.08,-3.48],patch_size:64,
      max_cloud_cover:0,min_valid_fraction:.6,max_downloads:2,max_attempts:6,candidates_per_month:3},
      latest_observation:'2024-08-26'},snapshots:[{},{}]})""")
    assert "Earth Search · Sentinel-2" in result
    assert "Jul 1, 2024 to Aug 31, 2024" in result
    assert "Aug 26, 2024" in result and "West -55.100000°" in result
    assert "0%" in result and "60%" in result
    assert "64 × 64 samples" in result and "2.000" not in result
    assert "<pre" not in result and '"request"' not in result
    missing = render("savedAcquisitionSummary({configuration:{request:{}}})")
    assert "Unavailable" in missing and "No observation date recorded" in missing
    absent = render("savedAcquisitionSummary({configuration:null})")
    assert "No automatic collection settings" in absent


def test_partial_collection_has_readable_failure_resume_and_download_controls():
    result = render("""(() => {acquisition.jobs=[{id:'job-opaque-123',status:'partial',phase:'saved',
      request:{name:'Pilot <area>',provider:'earth-search',start:'2024-07-01',end:'2024-08-31'},
      updated_at:'2024-09-01T12:30:00Z',error:'Traceback at C:/private/provider.py',
      progress:{requested_months:2,accepted_months:1,missing_months:['2024-08'],downloads:2,
      months:[{month:'2024-08',status:'missing',reason:'No eligible catalog candidates'}]}}];
      renderAcquisitionJobs();return elements['#job-list'].innerHTML;})()""")
    text = visible(result)
    assert "Some months still missing" in text and "Observations saved" in text
    assert "1 new months saved / 2 requested" in text
    assert "Resume collection" in text and "Download full collection record" in text
    assert "No suitable satellite images were found" in text
    assert "Pilot &lt;area&gt;" in text
    assert "job-opaque-123" not in text and "Traceback" not in text and "private" not in text


def test_unconfigured_notifications_are_disabled_with_no_machine_terms():
    result = render("notificationSummary({configuration:{enabled:false,authorized:false,endpoint_configured:false,signing_configured:false},outbox:[]})")
    assert "Notifications are not set up" in result and "No delivery attempts recorded" in result
    assert result.count(' disabled') == 3
    assert ' checked' not in result
    for raw in ("webhook", "event_id", "FORESTMIND_", "signing secret", "deduplicate"):
        assert raw not in result


def test_delivery_failures_show_retry_uncertainty_and_preserve_zero_attempts():
    result = render("""notificationSummary({configuration:{endpoint_configured:true,signing_configured:true,
      destination_host:'alerts.example',enabled:false,authorized:false,reauthorization_required:true,
      capacity_reached:true,outbox_events:10000,outbox_limit:10000},outbox:[
      {id:'notice-opaque',status:'pending',attempts:0,error:null},
      {id:'notice-2',status:'retry',attempts:1,error:'Delivery failed (ConnectionError); inspect receiver availability.'},
      {id:'notice-3',status:'failed',attempts:3,error:'Delivery attempt limit reached; an interrupted final attempt may have reached the receiver. Check the receiver using the event_id.'}]})""")
    text = visible(result)
    assert "Destination changed" in text and "Notification storage is full" in text
    assert "Waiting to send" in text and "Waiting to retry" in text and "Delivery stopped" in text
    assert '<td>0</td>' in result
    assert "may have arrived" in text and "alerts.example" in text
    assert 'notice-opaque' not in text and 'ConnectionError' not in text and 'event_id' not in text


def test_database_failure_is_visible_without_printing_internal_diagnostics():
    result = render("""operationsSummary({status:'degraded',access:'local-single-user-only',
      database_check:'*** Page 100: btreeInitPage() returns error code 11',database_bytes:1048576,disk_free_bytes:1073741824})""")
    assert "Needs attention" in result and "This computer only" in result
    assert "1.00 MB" in result and "1.00 GB" in result
    assert "btreeInitPage" not in result and "local-single-user-only" not in result
    assert "Download service record" in result


def test_service_and_notification_failure_do_not_hide_each_other():
    result = render("""(async()=>{api=async path=>{if(path==='/operations')throw new Error('TypeError: provider unavailable');
      return {configuration:{endpoint_configured:false,signing_configured:false},outbox:[]};};
      await renderOperations();return elements['#local-operations'].innerHTML;})()""")
    assert "local service status could not be loaded" in result
    assert "Optional alert notifications" in result and "TypeError" not in result
    reverse = render("""(async()=>{api=async path=>{if(path==='/notifications')throw new Error('TypeError: malformed response');
      return {status:'ok',database_check:'ok',access:'local-single-user-only',database_bytes:1024,disk_free_bytes:1024};};
      await renderOperations();return elements['#local-operations'].innerHTML;})()""")
    assert "Running normally" in reverse and "Notification settings could not be loaded" in reverse
    assert "TypeError" not in reverse


def test_monitoring_export_keeps_original_record_without_displaying_internal_fields():
    result = render("""(async()=>{state.selected='area';state.tab='monitor';
      const record={configuration:{request:{provider:'earth-search',max_downloads:2}},snapshots:[],internal_hash:'original-hash',jobs:[]};
      api=async path=>path.endsWith('/schedule')?{enabled:false,interval_days:37}:path.endsWith('/acquisition')?record:{alerts:[]};
      await renderMonitoringControls('area');elements['#download-acquisition-settings'].listeners.click();
      return {html:elements['#detail-body'].innerHTML,interval:elements['#monitor-interval'].value,downloads};})()""")
    assert result["interval"] == "37"
    assert "original-hash" not in result["html"]
    assert "<pre" not in result["html"]
    assert result["downloads"][0]["data"]["internal_hash"] == "original-hash"
    assert result["downloads"][0]["data"]["configuration"]["request"]["max_downloads"] == 2


def test_opening_operations_only_reads_configuration_and_export_keeps_original_record():
    result = render("""(async()=>{const calls=[];api=async (path,options)=>{calls.push([path,options?.method || 'GET']);
      return path==='/operations'?{status:'ok',database_check:'ok',access:'local-single-user-only'}:
      {configuration:{endpoint_configured:false,signing_configured:false},outbox:[{id:'record-kept',status:'sent',attempts:1,error:null}]};};
      await renderOperations();elements['#download-deliveries'].listeners.click();return {calls,downloads};})()""")
    assert result["calls"] == [["/operations", "GET"], ["/notifications", "GET"]]
    assert result["downloads"][0]["data"]["outbox"][0]["id"] == "record-kept"


def test_history_refresh_defaults_keep_saved_start_and_limit_rolling_window():
    result = render("""[
      refreshDefaults({start:'2025-07-01',max_downloads:2},'2026-10-29'),
      refreshDefaults({start:'2015-06-23',max_downloads:6},'2026-10-29')
    ]""")
    assert result[0] == {"start": "2025-07-01", "end": "2026-10-29", "max_downloads": 2}
    assert result[1] == {"start": "2021-11-01", "end": "2026-10-29", "max_downloads": 6}


@pytest.mark.parametrize("values,fragment", [
    ({"start": "2024-02-30", "end": "2024-03-31", "max_downloads": 1}, "valid"),
    ({"start": "2014-01-01", "end": "2014-12-31", "max_downloads": 1}, "June 23"),
    ({"start": "2024-02-01", "end": "2024-01-31", "max_downloads": 1}, "on or after"),
    ({"start": "2024-01-01", "end": "2027-01-01", "max_downloads": 1}, "today or earlier"),
    ({"start": "2020-01-31", "end": "2025-01-01", "max_downloads": 1}, "60 calendar months"),
    ({"start": "2024-01-01", "end": "2024-12-31", "max_downloads": 0}, "1 to 24"),
    ({"start": "2024-01-01", "end": "2024-12-31", "max_downloads": 1.5}, "whole attempts"),
    ({"start": "2024-01-01", "end": "2024-12-31", "max_downloads": 25}, "1 to 24"),
])
def test_history_refresh_validation_rejects_invalid_dates_and_budgets(values, fragment):
    result = render(f"refreshInputError({json.dumps(values)},'2026-10-29')")
    assert fragment in result


def test_history_refresh_validates_before_submit_and_preserves_selected_area():
    result = render("""(async()=>{state.selected='area';state.tab='monitor';location.hash='overview';
      const calls=[];api=async(path,options)=>{calls.push([path,options]);
        if(path.endsWith('/acquisition'))return {configuration:{request:{start:'2026-09-01',max_downloads:2},study_period:{start:'2025-07-01',end:'2026-10-29'}}};
        return {};};
      refreshAcquisitionJobs=async()=>{};toast=()=>{};
      await renderMonitoringControls('area');
      const initial=elements['#refresh-start'].value;
      elements['#refresh-start'].value='2024-03-01';elements['#refresh-end'].value='2024-02-29';
      await elements['#refresh-form'].listeners.submit({preventDefault(){}});
      const invalid={posts:calls.filter(row=>row[1]?.method==='POST').length,error:elements['#monitor-error'].textContent,disabled:elements['#refresh-region'].disabled};
      elements['#refresh-end'].value='2024-12-31';elements['#refresh-budget'].value='4';
      await elements['#refresh-form'].listeners.submit({preventDefault(){}});
      return {initial,invalid,posts:calls.filter(row=>row[1]?.method==='POST'),selected:state.selected,hash:location.hash,status:elements['#refresh-status'].textContent};})()""")
    assert result["initial"] == "2025-07-01"
    assert result["invalid"]["posts"] == 0 and not result["invalid"]["disabled"]
    assert "on or after" in result["invalid"]["error"]
    assert len(result["posts"]) == 1
    assert result["posts"][0][0] == "/regions/area/refresh"
    assert json.loads(result["posts"][0][1]["body"]) == {
        "start": "2024-03-01", "end": "2024-12-31", "max_downloads": 4,
    }
    assert result["selected"] == "area" and result["hash"] == "overview"
    assert "Collection queued" in result["status"]


def test_coverage_default_includes_requested_missing_months_without_starting_collection():
    result = render("""(async()=>{state.selected='area';state.tab='coverage';
      state.region={observations:[{date:'2025-10-12'},{date:'2026-08-30'}]};
      const calls=[],periods=[];api=async(path,options)=>{calls.push([path,options?.method || 'GET']);return {configuration:{request:{start:'2026-09-01',end:'2026-10-29'},study_period:{start:'2025-07-01',end:'2026-10-29'}}};};
      loadCoverage=async(...args)=>periods.push(args);
      await renderCoverage('area');return {calls,periods,html:elements['#detail-body'].innerHTML};})()""")
    assert result["calls"] == [["/regions/area/acquisition", "GET"]]
    assert result["periods"] == [["area", "2025-07-01", "2026-10-29"]]
    assert "including months without an observation" in result["html"]
    assert "does not start a download" in result["html"]


def test_coverage_preserves_selected_period_and_falls_back_when_settings_unavailable():
    result = render("""(async()=>{state.selected='area';state.tab='coverage';
      state.region={observations:[{date:'2025-10-12'},{date:'2026-08-30'}]};
      const periods=[];let calls=0;api=async()=>{calls++;throw new Error('unavailable');};
      loadCoverage=async(...args)=>periods.push(args);await renderCoverage('area');
      const fallback=elements['#detail-body'].innerHTML;
      coveragePeriods.set('area',{start:'2024-01-01',end:'2024-12-31'});await renderCoverage('area');
      return {calls,periods,fallback};})()""")
    assert result["calls"] == 1
    assert result["periods"] == [["area", "2025-10-12", "2026-08-30"], ["area", "2024-01-01", "2024-12-31"]]
    assert "could not be loaded" in result["fallback"]


def test_stale_coverage_period_request_cannot_replace_a_different_area():
    result = render("""(async()=>{state.selected='area';state.tab='coverage';
      state.region={observations:[{date:'2025-10-12'}]};let release;const periods=[];
      api=()=>new Promise(resolve=>release=resolve);loadCoverage=async(...args)=>periods.push(args);
      const pending=renderCoverage('area');state.selected='other';elements['#detail-body'].innerHTML='Other area';
      release({configuration:{study_period:{start:'2025-07-01',end:'2026-10-29'}}});await pending;
      return {html:elements['#detail-body'].innerHTML,periods};})()""")
    assert result == {"html": "Other area", "periods": []}
