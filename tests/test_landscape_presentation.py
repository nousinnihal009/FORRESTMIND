"""Execute worldwide-plan interactions, including asynchronous selection races."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_ui(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for interface behavior checks")
    source = r"""
const vm=require('vm'),fs=require('fs'),path=require('path'),elements={},stored={},downloads=[],events={};
function element(selector){return elements[selector]??(elements[selector]={innerHTML:'',textContent:'',value:'',
 hidden:false,disabled:false,checked:false,style:{},listeners:{},addEventListener(name,fn){this.listeners[name]=fn;},
 setAttribute(){},querySelector(s){return element(s);},querySelectorAll(){return [];},
 scrollIntoView(){this.scrolled=true;},focus(){this.focused=true;}});}
const context={document:{querySelector:element,querySelectorAll:()=>[]},location:{hash:''},console,Intl,Date,JSON,
 Number,Promise,setTimeout,clearTimeout,elements,stored,downloads,events,
 localStorage:{getItem:key=>stored[key]??null,setItem:(key,value)=>stored[key]=value,removeItem:key=>delete stored[key]}};
vm.createContext(context);
vm.runInContext('window=globalThis;window.addEventListener=(name,fn)=>events[name]=fn;',context);
const web=WEBPATH;
vm.runInContext(fs.readFileSync(path.join(web,'app.js'),'utf8').split('hydrateIcons();route();')[0],context);
vm.runInContext(fs.readFileSync(path.join(web,'area-utils.js'),'utf8'),context);
vm.runInContext(fs.readFileSync(path.join(web,'acquisition.js'),'utf8').replace(/\ninitializeAreaMap\(\);\s*$/,''),context);
vm.runInContext(fs.readFileSync(path.join(web,'landscapes.js'),'utf8'),context);
vm.runInContext("downloadJSON=(data,name)=>downloads.push({data,name});",context);
vm.runInContext(`
 function plan(id='plan-a',name='Forest study'){return {id,name,country:'Multiple countries',bbox:[20,-2,20.2,-1.8],
   area_km2:494.2,tile_area_km2:25,tile_count:30,method:'spherical-rectangle-rows-v1'};}
 function page(id='plan-a',offset=0){return {landscape:plan(id),summary:{tiles_with_observations:0,total_tiles:30},
   tiles:[{index:offset,name:'Forest study · area '+(offset+1),bbox:[20,-2,20.04002301234567,-1.96],area_km2:19.78,
     region_id:null,observation_count:0,latest_date:null}],next_offset:offset===0?20:null};}
 function fillForm(){
   $('#landscape-name').value='Worldwide study';$('#landscape-country').value='';
   ['west','south','east','north'].forEach((key,i)=>$('#landscape-'+key).value=String([179.9,-17,-179.9,-16.9][i]));
   $('#landscape-tile-area').value='25';
 }
`,context);
Promise.resolve(vm.runInContext(SCRIPT,context)).then(value=>process.stdout.write(JSON.stringify(value))).catch(error=>{console.error(error);process.exit(1);});
""".replace("WEBPATH", json.dumps(str(ROOT / "forestmind/web"))).replace("SCRIPT", json.dumps(script))
    result = subprocess.run([node, "-e", source], capture_output=True, text=True,
                            encoding="utf-8", check=True, timeout=30)
    return json.loads(result.stdout)


def test_worldwide_and_date_line_rectangles_are_accepted_without_country_restriction():
    result = run_ui("""[
      [-180,-85,180,85],[179.9,-17,-179.9,-16.9],[-60,-4,-59,-3],[20,-2,21,-1],
      [110,2,111,3],[10,49,11,50],[-110,60,-109,61],[145,-42,146,-41]
    ].map(bbox=>landscapeInputError({name:'Forest study',country:'',bbox,tile_area_km2:25}))""")
    assert result == [None] * 8


@pytest.mark.parametrize("bbox", ["[180,-1,-180,1]", "[0,0,0,1]", "[1,2,3,1]",
                                      "[0,-86,1,-84]", "[181,0,182,1]", "[0,0,NaN,1]"])
def test_invalid_geography_explained_before_saving(bbox):
    result = run_ui(f"landscapeInputError({{name:'Forest',bbox:{bbox},tile_area_km2:25}})")
    assert result


def test_planned_area_reaches_acquisition_exactly_and_fit_apply_preserve_it():
    result = run_ui("""(()=>{
      let calls=0;api=async()=>{calls++;throw new Error('Unexpected network request');};
      const record=page(),tile=record.tiles[0],before=JSON.stringify(tile.bbox);
      acquisition.confirmed=true;acquisition.requestKey='previous';acquisition.place={label:'Old'};
      $('#confirm-area').checked=true;$('#start-acquisition').disabled=false;
      landscapeState.page=record;selectLandscapeArea(tile.index);
      const initial={bbox:acquisition.bbox,fields:areaFields()};
      updateArea(acquisition.bbox,true);updateArea(areaFields(),true);updateArea(areaFields(),true);
      return {initial,final:acquisition.bbox,source:tile.bbox,unchanged:JSON.stringify(tile.bbox)===before,
        calls,confirmed:acquisition.confirmed,checked:$('#confirm-area').checked,disabled:$('#start-acquisition').disabled,
        confirmationAvailable:!$('#confirm-area').disabled,requestKey:acquisition.requestKey,place:acquisition.place,
        name:$('#area-name').value,country:$('#area-country').value};})()""")
    assert result["initial"]["bbox"] == result["initial"]["fields"] == result["source"] == result["final"]
    assert result["unchanged"] and result["calls"] == 0
    assert not result["confirmed"] and not result["checked"] and result["disabled"]
    assert result["confirmationAvailable"] and result["requestKey"] is None and result["place"] is None
    assert result["name"] == "Forest study · area 1" and result["country"] == "Multiple countries"


def test_create_uses_wrapped_api_result_and_never_starts_collection():
    result = run_ui("""(async()=>{
      const calls=[];fillForm();
      api=async(url,options)=>{calls.push({url,method:options?.method || 'GET',body:options?.body && JSON.parse(options.body)});
        if(options?.method==='POST')return {landscape:plan('new')};
        if(url==='/landscapes')return {landscapes:[plan('new'),plan('older')]};
        return page('new');};
      await createLandscape({preventDefault(){}});
      return {calls,selected:landscapeState.selected,saved:stored['forestmind-landscape'],busy:landscapeState.creating,
        disabled:$('#save-landscape').disabled,count:landscapeState.page.summary.tiles_with_observations};})()""")
    assert [(row["method"], row["url"]) for row in result["calls"]] == [
        ("POST", "/landscapes"), ("GET", "/landscapes"), ("GET", "/landscapes/new?offset=0&limit=20")]
    assert result["calls"][0]["body"]["bbox"] == [179.9, -17, -179.9, -16.9]
    assert result["calls"][0]["body"]["country"] == "Unspecified"
    assert result["selected"] == result["saved"] == "new"
    assert not result["busy"] and not result["disabled"] and result["count"] == 0


def test_saved_selection_is_restored_and_paged_controls_request_only_one_page():
    result = run_ui("""(async()=>{
      const calls=[];stored['forestmind-landscape']='remembered';
      api=async(url)=>{calls.push(url);return url==='/landscapes'?{landscapes:[plan('other'),plan('remembered')]}:
        page('remembered',url.includes('offset=20')?20:0);};
      await refreshLandscapes();const first=landscapeState.selected;
      await $('#landscape-next').listeners.click();await Promise.resolve();
      const last={offset:landscapeState.offset,previous:$('#landscape-previous').disabled,next:$('#landscape-next').disabled};
      await $('#landscape-previous').listeners.click();await Promise.resolve();
      return {calls,first,last,offset:landscapeState.offset,previous:$('#landscape-previous').disabled};})()""")
    assert result["first"] == "remembered"
    assert result["last"] == {"offset": 20, "previous": False, "next": True}
    assert result["offset"] == 0 and result["previous"]
    assert result["calls"] == ["/landscapes", "/landscapes/remembered?offset=0&limit=20",
                               "/landscapes/remembered?offset=20&limit=20",
                               "/landscapes/remembered?offset=0&limit=20"]


def test_slow_page_cannot_replace_newer_selection():
    result = run_ui("""(async()=>{
      let release;api=url=>url.includes('/old?')?new Promise(resolve=>release=resolve):Promise.resolve(page('new'));
      const old=loadLandscapePage('old');await loadLandscapePage('new');release(page('old'));await old;
      return {selected:landscapeState.selected,page:landscapeState.page.landscape.id,saved:stored['forestmind-landscape']};})()""")
    assert result == {"selected": "new", "page": "new", "saved": "new"}


def test_slow_page_cannot_restore_an_explicitly_cleared_selection():
    result = run_ui("""(async()=>{
      let release;api=()=>new Promise(resolve=>release=resolve);
      const pending=loadLandscapePage('old');await loadLandscapePage('');release(page('old'));await pending;
      return {selected:landscapeState.selected,page:landscapeState.page,saved:savedLandscapeId(),
        disabled:$('#export-landscape').disabled,hidden:$('#landscape-pagination').hidden};})()""")
    assert result == {"selected": None, "page": None, "saved": None, "disabled": True, "hidden": True}


def test_slow_list_refresh_cannot_replace_a_later_cleared_selection():
    result = run_ui("""(async()=>{
      let release;const calls=[];api=url=>{calls.push(url);return new Promise(resolve=>release=resolve);};
      const pending=refreshLandscapes();await loadLandscapePage('');release({landscapes:[plan('old')]});await pending;
      return {calls,selected:landscapeState.selected,page:landscapeState.page,selection:$('#landscape-select').value};})()""")
    assert result == {"calls": ["/landscapes"], "selected": None, "page": None, "selection": ""}


def test_older_list_response_cannot_overwrite_a_more_recent_refresh():
    result = run_ui("""(async()=>{
      let release;api=url=>url==='/landscapes'?new Promise(resolve=>release=resolve):Promise.resolve(page('new'));
      const old=refreshLandscapes();api=async url=>url==='/landscapes'?{landscapes:[plan('new')]}:page('new');
      await refreshLandscapes();release({landscapes:[plan('old')]});await old;
      return {selected:landscapeState.selected,plans:landscapeState.plans.map(p=>p.id),page:landscapeState.page.landscape.id,
        disabled:$('#refresh-landscapes').disabled};})()""")
    assert result == {"selected": "new", "plans": ["new"], "page": "new", "disabled": False}


def test_save_completion_does_not_steal_a_later_manual_plan_selection():
    result = run_ui("""(async()=>{
      let release;fillForm();api=async(url,options)=>options?.method==='POST'?new Promise(resolve=>release=resolve):
        url==='/landscapes'?{landscapes:[plan('created'),plan('chosen')]}:page('chosen');
      const saving=createLandscape({preventDefault(){}});await loadLandscapePage('chosen');
      release({landscape:plan('created')});await saving;
      return {selected:landscapeState.selected,page:landscapeState.page.landscape.id};})()""")
    assert result == {"selected": "chosen", "page": "chosen"}


def test_failed_page_clears_previous_tiles_and_disables_export_until_retry():
    result = run_ui("""(async()=>{
      api=async()=>page();await loadLandscapePage('plan-a');
      api=async()=>{throw new Error('Failed to fetch');};await loadLandscapePage('plan-a',20);
      const failed={page:landscapeState.page,tiles:$('#landscape-tiles').innerHTML,hidden:$('#landscape-pagination').hidden,
        disabled:$('#export-landscape').disabled,error:$('#landscape-error').textContent};
      api=async()=>page();await loadLandscapePage('plan-a');return {failed,recovered:landscapeState.page.landscape.id};})()""")
    assert result["failed"]["page"] is None and result["failed"]["tiles"] == ""
    assert result["failed"]["hidden"] and result["failed"]["disabled"]
    assert "could not be reached" in result["failed"]["error"] and result["recovered"] == "plan-a"


def test_rendering_escapes_saved_text_and_preserves_unknown_coverage():
    result = run_ui("""(()=>{
      const record=page();record.landscape.name='<img src=x onerror=attack()>';
      record.tiles[0].name='<script>attack()</script>';record.tiles[0].region_id='x" onclick="attack()';
      renderLandscapePage(record);return {summary:$('#landscape-summary').innerHTML,tiles:$('#landscape-tiles').innerHTML};})()""")
    assert "<img" not in result["summary"] and "&lt;img" in result["summary"]
    assert "<script>" not in result["tiles"] and "&lt;script&gt;" in result["tiles"]
    assert 'onclick="attack()' not in result["tiles"]
    assert "0 of 30 areas have saved observations" in result["summary"]
    assert "No observations yet" in result["tiles"] and "no inferred condition" in result["summary"]


def test_export_is_explicitly_a_plan_definition_and_not_full_observation_evidence():
    result = run_ui("""(()=>{landscapeState.page=page();$('#export-landscape').listeners.click();return downloads;})()""")
    assert len(result) == 1 and result[0]["name"] == "forestmind-forest-plan.json"
    payload = result[0]["data"]
    assert payload["landscape"]["method"] == "spherical-rectangle-rows-v1"
    assert "No observations or full tile list" in payload["note"]
    assert "tiles" not in payload and "observations" not in payload


def test_storage_restrictions_do_not_prevent_opening_a_plan():
    result = run_ui("""(async()=>{
      localStorage={getItem(){throw new Error('Blocked');},setItem(){throw new Error('Blocked');},removeItem(){throw new Error('Blocked');}};
      api=async url=>url==='/landscapes'?{landscapes:[plan()]}:page();await refreshLandscapes();
      return {selected:landscapeState.selected,saved:savedLandscapeId(),page:landscapeState.page.landscape.id};})()""")
    assert result == {"selected": "plan-a", "saved": None, "page": "plan-a"}
