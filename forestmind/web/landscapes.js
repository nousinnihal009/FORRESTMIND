/* A worldwide forest can contain many bounded monitoring areas. Planning is local. */
'use strict';
const landscapeState = {plans:[],selected:null,page:null,offset:0,pageSize:20,listSequence:0,pageSequence:0,creating:false};

function savedLandscapeId() {
  try {return localStorage.getItem('forestmind-landscape') || null;} catch {return null;}
}
function rememberLandscape(id) {
  try {if(id)localStorage.setItem('forestmind-landscape',id);else localStorage.removeItem('forestmind-landscape');} catch {}
}
function landscapeError(message) {
  $('#landscape-error').textContent=message || '';
  $('#landscape-error').hidden=!message;
}
function landscapeFormValues() {
  return {name:$('#landscape-name').value.trim(),country:$('#landscape-country').value.trim() || 'Unspecified',
    bbox:['west','south','east','north'].map(key=>{const value=$('#landscape-'+key).value.trim();return value===''?NaN:Number(value);}),
    tile_area_km2:Number($('#landscape-tile-area').value)};
}
function landscapeInputError(values) {
  if(values.name.length<2)return 'Give this forest plan a name with at least two characters.';
  const b=values.bbox;
  if(!Array.isArray(b)||b.length!==4||b.some(v=>!Number.isFinite(v)))return 'Enter all four coordinates for the forest study area.';
  if(b[0]<-180||b[0]>180||b[2]<-180||b[2]>180||b[1]<-85||b[3]>85)return 'Use longitudes from −180° to 180° and latitudes from −85° to 85°.';
  if(b[1]>=b[3]||b[0]===b[2]||(b[0]===180&&b[2]===-180))return 'North must be above south, and the west and east boundaries must enclose a nonzero span. Use −180 to 180 for all longitudes.';
  if(!Number.isFinite(values.tile_area_km2)||values.tile_area_km2<1||values.tile_area_km2>25)return 'Choose a maximum monitoring area from 1 to 25 km².';
  return null;
}
function clearLandscapePage(message) {
  landscapeState.page=null;
  $('#landscape-summary').textContent=message;
  $('#landscape-tiles').innerHTML='';
  $('#landscape-pagination').hidden=true;
  $('#export-landscape').disabled=true;
}
async function refreshLandscapes(preferredId=null) {
  const sequence=++landscapeState.listSequence,pageSequence=landscapeState.pageSequence;
  $('#refresh-landscapes').disabled=true;
  landscapeError(null);
  try {
    const result=await api('/landscapes');
    if(sequence!==landscapeState.listSequence)return;
    landscapeState.plans=result.landscapes || [];
    const selectionChanged=pageSequence!==landscapeState.pageSequence;
    const selected=selectionChanged?landscapeState.selected:([preferredId,landscapeState.selected,savedLandscapeId()].find(id=>landscapeState.plans.some(plan=>plan.id===id)) || landscapeState.plans[0]?.id || null);
    $('#landscape-select').innerHTML='<option value="">Choose a saved forest plan</option>'+landscapeState.plans.map(plan=>`<option value="${esc(plan.id)}">${esc(plan.name)} · ${numeric(plan.tile_count)} areas</option>`).join('');
    $('#landscape-select').value=selected || '';
    $('#landscape-select').disabled=!landscapeState.plans.length;
    // A refresh must not replace a later page choice, including explicitly
    // clearing the selection, or make that choice issue a duplicate request.
    if(selectionChanged)return;
    if(selected)await loadLandscapePage(selected,selected===landscapeState.selected?landscapeState.offset:0);
    else {++landscapeState.pageSequence;landscapeState.selected=null;rememberLandscape(null);clearLandscapePage('No saved forest plans yet. Define a larger study area above to begin.');}
  } catch(error) {
    if(sequence===landscapeState.listSequence)landscapeError(userMessage(error,'Saved forest plans could not be loaded. Use Refresh plans to retry.'));
  } finally {if(sequence===landscapeState.listSequence)$('#refresh-landscapes').disabled=false;}
}
async function createLandscape(event) {
  event.preventDefault();if(landscapeState.creating)return;
  const values=landscapeFormValues(),error=landscapeInputError(values);landscapeError(error);if(error)return;
  landscapeState.creating=true;$('#save-landscape').disabled=true;
  const pageSequence=landscapeState.pageSequence;
  try {
    const result=await api('/landscapes',{method:'POST',body:JSON.stringify(values)});
    await refreshLandscapes(pageSequence===landscapeState.pageSequence?(result.landscape || result).id:null);
  } catch(error) {landscapeError(userMessage(error,'The forest plan could not be saved. Check its boundaries and try again.'));}
  finally {landscapeState.creating=false;$('#save-landscape').disabled=false;}
}
async function loadLandscapePage(id,offset=0) {
  const sequence=++landscapeState.pageSequence;
  landscapeState.selected=id || null;landscapeState.offset=offset;rememberLandscape(id);
  $('#landscape-select').value=id || '';
  clearLandscapePage(id?'Loading monitoring areas…':'Choose a saved forest plan.');
  landscapeError(null);if(!id)return;
  try {
    const page=await api('/landscapes/'+encodeURIComponent(id)+'?offset='+offset+'&limit='+landscapeState.pageSize);
    if(sequence!==landscapeState.pageSequence||id!==landscapeState.selected)return;
    landscapeState.page=page;renderLandscapePage(page);
  } catch(error) {if(sequence===landscapeState.pageSequence){clearLandscapePage('This plan could not be loaded. Refresh plans to retry.');landscapeError(userMessage(error,'Monitoring areas could not be loaded. Please retry.'));}}
}
function renderLandscapePage(page) {
  const plan=page.landscape,summary=page.summary,tiles=page.tiles || [];
  const count=summary.tiles_with_observations || 0;
  $('#landscape-summary').innerHTML=`<strong>${esc(plan.name)}</strong><span>${numeric(plan.area_km2,2)} km² study rectangle · ${numeric(plan.tile_count)} monitoring areas</span><span>${numeric(count)} of ${numeric(summary.total_tiles ?? plan.tile_count)} areas have saved observations. The remaining areas have no matched observations.</span><small>Observation presence is not complete seasonal coverage or verified forest extent. Each area needs its own historical baseline and evidence; a missing area has no inferred condition.</small>`;
  $('#landscape-tiles').innerHTML=tiles.length?`<div class="table-scroll"><table class="landscape-tile-table"><thead><tr><th>Monitoring area</th><th>Study area</th><th>Saved observations</th><th>Next step</th></tr></thead><tbody>${tiles.map(tile=>`<tr><td><strong>${esc(tile.name)}</strong><small>${tile.bbox.map(v=>numeric(v,5)+'°').join(', ')}</small></td><td>${numeric(tile.area_km2,3)} km²</td><td>${tile.observation_count?`${numeric(tile.observation_count)} records<small>Latest: ${esc(tile.latest_date || 'Date unavailable')}</small>`:'No observations yet'}</td><td><button type="button" class="button secondary" data-landscape-tile="${tile.index}">Select area</button>${tile.region_id?` <button type="button" class="text-button" data-landscape-region="${esc(tile.region_id)}">View evidence ↗</button>`:''}</td></tr>`).join('')}</tbody></table></div>`:'<p class="observations-empty">No monitoring areas on this page.</p>';
  $('#landscape-pagination').hidden=false;
  $('#landscape-page-label').textContent=tiles.length?`Areas ${numeric(landscapeState.offset+1)}–${numeric(landscapeState.offset+tiles.length)} of ${numeric(plan.tile_count)}`:'No areas on this page';
  $('#landscape-previous').disabled=landscapeState.offset===0;
  $('#landscape-next').disabled=page.next_offset===null || page.next_offset===undefined;
  $('#export-landscape').disabled=false;
  $$('[data-landscape-tile]',$('#landscape-tiles')).forEach(button=>button.addEventListener('click',()=>selectLandscapeArea(Number(button.dataset.landscapeTile))));
  $$('[data-landscape-region]',$('#landscape-tiles')).forEach(button=>button.addEventListener('click',async()=>{location.hash='overview';await loadWorkspace(button.dataset.landscapeRegion);}));
}
function selectLandscapeArea(index) {
  const page=landscapeState.page,tile=page?.tiles.find(value=>value.index===index);
  if(!tile)return;
  selectPlannedAcquisitionArea(tile,page.landscape);
}
function useCurrentLandscapeBounds() {
  if(typeof acquisition==='undefined'||!acquisition.bbox?.every(Number.isFinite)){landscapeError('Choose or draw a study rectangle on the map first, or enter its coordinates here.');return;}
  ['west','south','east','north'].forEach((key,i)=>{$('#landscape-'+key).value=String(acquisition.bbox[i]);});
  $('#landscape-name').value=$('#area-name').value;
  $('#landscape-country').value=$('#area-country').value;
  landscapeError(null);
}
function initializeLandscapes() {
  $('#landscape-form').addEventListener('submit',createLandscape);
  $('#refresh-landscapes').addEventListener('click',()=>refreshLandscapes());
  $('#landscape-select').addEventListener('change',event=>loadLandscapePage(event.target.value));
  $('#landscape-use-map').addEventListener('click',useCurrentLandscapeBounds);
  $('#landscape-previous').addEventListener('click',()=>{if(landscapeState.page)loadLandscapePage(landscapeState.selected,Math.max(0,landscapeState.offset-landscapeState.pageSize));});
  $('#landscape-next').addEventListener('click',()=>{const offset=landscapeState.page?.next_offset;if(Number.isInteger(offset))loadLandscapePage(landscapeState.selected,offset);});
  $('#export-landscape').addEventListener('click',()=>{const page=landscapeState.page;if(page)downloadJSON({landscape:page.landscape,summary:page.summary,note:'Plan definition and observation-presence summary only. No observations or full tile list are included; the versioned plan defines all monitoring areas.'},'forestmind-forest-plan.json');});
  window.addEventListener('hashchange',()=>{if(location.hash==='#explore')refreshLandscapes();});
  if(location.hash==='#explore')refreshLandscapes();
}
if(typeof window!=='undefined')initializeLandscapes();
