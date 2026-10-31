/* Explicit worldwide search, bounded real acquisition, and durable local monitoring. */
'use strict';
const acquisition = {bbox:null,confirmed:false,place:null,limits:{max_area_km2:25},jobs:[],timer:null,busy:false,requestKey:null,map:{center:[0,15],zoom:2,mode:'pan',drag:null},tileErrors:0};

function acquisitionLabel(value) {
  const labels={'earth-search':'Earth Search · Sentinel-2','element84-earth-search':'Earth Search · Sentinel-2','planetary-computer':'Microsoft Planetary Computer',queued:'Waiting to start',running:'Collecting observations',retry_wait:'Waiting to retry',retrying_provider:'Retrying the satellite provider',planning:'Finding suitable images',acquiring:'Downloading and checking images',recovered:'Resuming after restart',interrupted:'Interrupted',saved:'Observations saved',completed:'Collection complete',partial:'Some months still missing',failed:'Needs attention',cancelled:'Cancelled',accepted:'Observation saved',pending:'Not yet checked',missing:'No usable observation'};
  return labels[value] || humanLabel(value || 'Unavailable');
}
function acquisitionWhen(value, fallback='Unavailable') {
  if(typeof value!=='string'||!/^\d{4}-\d{2}-\d{2}(?:T|$)/.test(value))return fallback;
  const time=new Date(value);if(!Number.isFinite(+time))return fallback;
  return new Intl.DateTimeFormat('en',{dateStyle:'medium',...(value.includes('T')?{timeStyle:'short'}:{}),timeZone:'UTC'}).format(time)+(value.includes('T')?' UTC':'');
}
function acquisitionGapReason(month) {
  const reasons={'Catalog search failed':'The satellite image search failed.','No eligible catalog candidates':'No suitable satellite images were found.','No quality-passing acquisition within the available candidates and budgets':'No image met the quality requirements within this collection budget.'};
  return month.reason?(reasons[month.reason] || userMessage(month.reason,'No usable observation was saved for this month.')):acquisitionLabel(month.status);
}
function collectionDate(value) {
  if(typeof value!=='string'||!/^\d{4}-\d{2}-\d{2}$/.test(value))return null;
  const parsed=new Date(value+'T00:00:00Z');
  return Number.isFinite(+parsed)&&parsed.toISOString().slice(0,10)===value?parsed:null;
}
function refreshDefaults(request, today=new Date().toISOString().slice(0,10)) {
  const end=collectionDate(today),earliest=new Date(+end);
  earliest.setUTCMonth(earliest.getUTCMonth()-59,1);
  const savedStart=collectionDate(request.start)?request.start:today;
  return {start:savedStart<earliest.toISOString().slice(0,10)?earliest.toISOString().slice(0,10):savedStart,
    end:today,max_downloads:Number.isInteger(request.max_downloads)?request.max_downloads:6};
}
function refreshInputError(values, today=new Date().toISOString().slice(0,10)) {
  const start=collectionDate(values.start),end=collectionDate(values.end);
  if(!start||!end)return 'Choose a valid starting and ending date.';
  if(values.start<'2015-06-23')return 'Sentinel-2 history starts on June 23, 2015.';
  if(values.end<values.start)return 'The ending date must be on or after the starting date.';
  if(values.end>today)return 'The ending date must be today or earlier.';
  const months=(end.getUTCFullYear()-start.getUTCFullYear())*12+end.getUTCMonth()-start.getUTCMonth()+1;
  if(months>60)return 'Choose at most 60 calendar months per collection. Collect longer histories in separate batches.';
  if(!Number.isInteger(values.max_downloads)||values.max_downloads<1||values.max_downloads>24)return 'Choose a download budget from 1 to 24 whole attempts.';
  return null;
}
function savedAcquisitionSummary(record) {
  const history=record.configuration,request=history?.request || {};
  if(!history)return '<p class="form-help">No automatic collection settings are saved for this area. Create a real collection in Find a forest to enable updates.</p>';
  const bounds=Array.isArray(request.bbox)&&request.bbox.length===4&&request.bbox.every(Number.isFinite)?['West','South','East','North'].map((name,i)=>`${name} ${numeric(request.bbox[i],6)}°`).join(' · '):'Unavailable';
  const rows=[['Satellite provider',acquisitionLabel(request.provider)],...(history.study_period? [['Requested study period',`${acquisitionWhen(history.study_period.start)} to ${acquisitionWhen(history.study_period.end)}`]]:[]),['Last successful collection period',`${acquisitionWhen(request.start)} to ${acquisitionWhen(request.end)}`],['Latest saved observation',acquisitionWhen(history.latest_observation,'No observation date recorded')],['Selected boundary',bounds],['Image grid',Number.isFinite(request.patch_size)?`${numeric(request.patch_size)} × ${numeric(request.patch_size)} samples`:'Unavailable'],['Maximum scene cloud cover',Number.isFinite(request.max_cloud_cover)?`${numeric(request.max_cloud_cover)}%`:'Unavailable'],['Minimum usable area',Number.isFinite(request.min_valid_fraction)?pct(request.min_valid_fraction):'Unavailable'],['Downloads per collection',numeric(request.max_downloads)],['Image checks per collection',numeric(request.max_attempts)],['Candidate images per month',numeric(request.candidates_per_month)],['Saved collection records',Array.isArray(record.snapshots)?numeric(record.snapshots.length):'Unavailable']];
  return `<div class="monitor-status">${rows.map(([label,value])=>`<div><small>${esc(label)}</small><strong>${esc(value)}</strong></div>`).join('')}</div><p class="form-help">Updates extend the saved period through today and look back at most 60 months. Existing months are preserved. The grid controls sampling; it does not increase the satellite's original detail.</p>`;
}

function areaFields() {return ['west','south','east','north'].map(key=>{const value=$('#area-'+key).value.trim();return value===''?NaN:Number(value);});}
function areaError(message) {$('#area-error').textContent=message || '';$('#area-error').hidden=!message;}
function currentAreaLimit() {return acquisition.limits.max_area_km2 || 25;}
function updateArea(bbox,fit=false) {
  acquisition.bbox=bbox;acquisition.confirmed=false;acquisition.requestKey=null;
  ['west','south','east','north'].forEach((key,i)=>{$('#area-'+key).value=Number.isFinite(bbox[i])?bbox[i].toFixed(6):'';});
  $('#confirm-area').checked=false;
  const error=ForestArea.validateBbox(bbox,currentAreaLimit());areaError(error);
  $('#area-summary').textContent=bbox.every(Number.isFinite)?`${bbox.map(v=>v.toFixed(6)).join(', ')} · ${numeric(ForestArea.areaKm2(bbox),3)} km²`:'Choose a rectangle or enter its coordinates.';
  $('#confirm-area').disabled=Boolean(error);$('#start-acquisition').disabled=true;
  if(fit&&bbox.every(Number.isFinite)&&bbox[0]<bbox[2]&&bbox[1]<bbox[3]){
    acquisition.map.center=[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2];
    let zoom=16;while(zoom>1){const a=ForestArea.project(bbox[0],bbox[1],zoom),b=ForestArea.project(bbox[2],bbox[3],zoom);if(Math.abs(b[0]-a[0])<570&&Math.abs(b[1]-a[1])<240)break;zoom--;}
    acquisition.map.zoom=zoom;
  }
  renderAreaMap();
}
function mapPoint(event) {
  const point=new DOMPoint(event.clientX,event.clientY).matrixTransform($('#area-map').getScreenCTM().inverse());
  return [point.x,point.y];
}
function mapCoordinate(point) {
  const center=ForestArea.project(...acquisition.map.center,acquisition.map.zoom);
  const coord=ForestArea.unproject(center[0]+point[0]-400,center[1]+point[1]-185,acquisition.map.zoom);
  return [Math.max(-180,Math.min(180,coord[0])),Math.max(-85.051129,Math.min(85.051129,coord[1]))];
}
function renderAreaMap() {
  const svg=$('#area-map');if(!svg)return;
  const {center,zoom}=acquisition.map, origin=ForestArea.project(...center,zoom),left=origin[0]-400,top=origin[1]-185;
  let tiles='';const count=2**zoom;
  for(let x=Math.floor(left/256);x<=Math.floor((left+800)/256);x++)for(let y=Math.floor(top/256);y<=Math.floor((top+370)/256);y++){
    if(y<0||y>=count||x<0||x>=count)continue;
    tiles+=`<image href="https://tile.openstreetmap.org/${zoom}/${x}/${y}.png" x="${x*256-left}" y="${y*256-top}" width="256" height="256" aria-hidden="true"/>`;
  }
  let selection='';const bbox=acquisition.bbox;
  if(bbox&&bbox.every(Number.isFinite)&&bbox[0]<bbox[2]&&bbox[1]<bbox[3]){
    const a=ForestArea.project(bbox[0],bbox[3],zoom),b=ForestArea.project(bbox[2],bbox[1],zoom);
    selection=`<rect x="${a[0]-left}" y="${a[1]-top}" width="${b[0]-a[0]}" height="${b[1]-a[1]}" fill="#335b4280" stroke="#173e32" stroke-width="3" stroke-dasharray="6 3"/>`;
  }
  svg.innerHTML=`<rect width="800" height="370" fill="#e6eee7"/>${tiles}${selection}<path d="M390 185H410M400 175V195" stroke="#173e32" stroke-width="1.5"/><text x="14" y="351" font-size="12" fill="#173e32" stroke="#fff" stroke-width="3" paint-order="stroke">${center[1].toFixed(4)}°, ${center[0].toFixed(4)}° · zoom ${zoom}</text>`;
  $$('image',svg).forEach(node=>node.addEventListener('error',()=>{$('#map-tile-status').hidden=false;},{once:true}));
  $('#area-map-mode').textContent=acquisition.map.mode==='draw'?'Draw mode: drag a rectangle. Use coordinates below for keyboard selection.':'Pan mode: drag to move. Use + / − or the arrow keys to explore.';
  $('#draw-area').setAttribute('aria-pressed',String(acquisition.map.mode==='draw'));
  svg.style.cursor=acquisition.map.mode==='draw'?'crosshair':'grab';
}
function centerAreaOnPoint(point,label,country='') {
  const lat=Math.max(-84.9,Math.min(84.9,point[1])),lon=Math.max(-179.98,Math.min(179.98,point[0]));
  const dx=Math.min(.2,.01/Math.max(.1,Math.cos(lat*Math.PI/180)));
  $('#area-name').value=(label || 'My monitoring area').slice(0,120);$('#area-country').value=country.slice(0,100);
  updateArea([lon-dx,lat-.01,lon+dx,lat+.01],true);
}
async function searchPlaces(event) {
  event.preventDefault();const query=$('#place-query').value.trim();if(query.length<2)return;
  const button=$('#place-search-button');button.disabled=true;$('#place-results').innerHTML='<p class="inline-status" role="status">Searching worldwide places…</p>';
  try{
    const result=await api('/places?q='+encodeURIComponent(query)),places=result.results || [];
    $('#place-results').innerHTML=places.length?`<ul class="place-results">${places.map((p,i)=>`<li><button type="button" data-place="${i}"><strong>${esc(p.label)}</strong><span>${esc(p.country || 'Country not supplied')} · ${esc(typeof p.source==='string'?p.source:'OpenStreetMap / Photon')}</span></button></li>`).join('')}</ul><p class="form-help">Named places are search references, not verified forest boundaries. A small rectangle will be suggested around your selection.</p>`:'<p class="inline-status" role="status">No matching places. Add a country or nearby city, or enter coordinates below.</p>';
    $$('[data-place]').forEach(button=>button.addEventListener('click',()=>{
      const place=places[Number(button.dataset.place)];acquisition.place=place;
      const center=place.center || (place.bbox?[(place.bbox[0]+place.bbox[2])/2,(place.bbox[1]+place.bbox[3])/2]:null);
      if(!center){areaError('This result has no usable coordinates. Choose another result.');return;}
      centerAreaOnPoint(center,place.label,place.country);$('#place-selection').textContent=`Search reference: ${place.label}. This rectangle is your monitoring boundary; forest cover inside it has not been verified.`;
      $('#place-results').replaceChildren();$('#area-name').focus();
    }));
  }catch(error){$('#place-results').innerHTML=`<p class="form-error" role="alert">${esc(userMessage(error,'Place search is unavailable. Please try again.'))} Coordinate selection remains available.</p>`;}
  finally{button.disabled=false;}
}
function acquisitionPayload() {
  return {name:$('#area-name').value.trim(),country:$('#area-country').value.trim() || 'Not specified',bbox:acquisition.bbox,start:$('#history-start').value,end:$('#history-end').value,provider:$('#history-provider').value,patch_size:Number($('#history-patch').value),max_cloud_cover:Number($('#history-cloud').value),min_valid_fraction:Number($('#history-quality').value),max_downloads:Number($('#history-budget').value),max_attempts:Number($('#history-budget').value)*3,candidates_per_month:3,search_limit:30,timeout_seconds:30};
}
async function submitAcquisition(event) {
  event.preventDefault();if(acquisition.busy)return;
  const error=ForestArea.validateBbox(acquisition.bbox,currentAreaLimit());if(error||!acquisition.confirmed){areaError(error || 'Confirm the exact monitoring rectangle first.');return;}
  if($('#history-end').value<$('#history-start').value){areaError('The ending date must be on or after the starting date.');return;}
  acquisition.busy=true;$('#start-acquisition').disabled=true;areaError(null);
  try{
    acquisition.requestKey=acquisition.requestKey || (crypto.randomUUID?crypto.randomUUID():String(Date.now()));
    const job=await api('/acquisition/jobs',{method:'POST',headers:{'Content-Type':'application/json','Idempotency-Key':acquisition.requestKey},body:JSON.stringify(acquisitionPayload())});
    toast('Real-data collection queued. You can leave this page while it runs.');
    await refreshAcquisitionJobs();$('#acquisition-jobs').scrollIntoView({behavior:'smooth',block:'start'});
    if(job.region_id)await loadWorkspace(job.region_id);
  }catch(error){areaError(userMessage(error,'Collection could not start. Check the area and dates, then try again.'));}
  finally{acquisition.busy=false;$('#start-acquisition').disabled=!acquisition.confirmed;}
}
function jobProgress(job) {
  const p=job.progress || {}, requested=Array.isArray(p.requested_months)?p.requested_months.length:Number(p.requested_months || 0),accepted=Array.isArray(p.accepted_months)?p.accepted_months.length:Number(p.accepted_months || 0);
  const saved=Array.isArray(p.already_saved_months)?p.already_saved_months.length:0;
  return {requested,accepted,saved,ratio:requested?Math.min(100,(accepted+saved)/requested*100):0};
}
function renderAcquisitionJobs() {
  const jobs=acquisition.jobs;
  $('#job-list').innerHTML=jobs.length?jobs.map(job=>{
    const p=job.progress || {},progress=jobProgress(job),active=['queued','running','retry_wait'].includes(job.status),resume=['partial','failed','cancelled','interrupted'].includes(job.status);
    const gaps=p.remaining_area_months || p.missing_months,missing=Array.isArray(gaps)?gaps.join(', '):gaps;
    const title=job.request?.name || job.name || 'Monitoring update';
    const phase=job.cancel_requested&&active?'Stopping after the current image check':acquisitionLabel(job.phase || job.status);
    return `<article class="acquisition-job"><div class="job-heading"><div><h3>${esc(title)}</h3><p>${esc(acquisitionWhen(job.request?.start))} → ${esc(acquisitionWhen(job.request?.end))} · ${esc(acquisitionLabel(job.request?.provider))}</p></div>${badge(acquisitionLabel(job.status))}</div><p class="job-phase">${esc(phase)} · ${progress.accepted} new months saved${progress.saved?` · ${progress.saved} already saved`:''} / ${progress.requested || '—'} requested${p.searched_months!==undefined?` · ${numeric(p.searched_months)} months searched`:''} · ${numeric(p.downloads ?? job.download_attempts)} downloads attempted</p><div class="job-progress" role="progressbar" aria-label="Saved months for ${esc(title)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${progress.ratio.toFixed(0)}"><i style="width:${progress.ratio}%"></i></div>${job.error?`<p class="form-error">${esc(userMessage(job.error,'Collection needs attention. Saved observations are preserved; refresh the collection list or try resuming.'))}</p>`:''}${job.analysis_error?`<p class="evidence-disclaimer">Observations were saved, but their analysis could not finish. ${esc(userMessage(job.analysis_error,'Open the saved area and run its analysis again.'))}</p>`:''}${missing?`<details class="job-gaps"><summary>Months without a saved observation</summary><p>${esc(missing)}</p>${Array.isArray(p.months)?`<ul>${p.months.filter(month=>month.status!=='accepted').map(month=>`<li>${esc(month.month)}: ${esc(acquisitionGapReason(month))}</li>`).join('')}</ul>`:''}</details>`:''}<div class="job-actions">${active?`<button class="button secondary" data-job-cancel="${esc(job.id)}" ${job.cancel_requested?'disabled':''}>${job.cancel_requested?'Stopping…':'Cancel'}</button>`:''}${resume?`<button class="button secondary" data-job-resume="${esc(job.id)}">Resume collection</button>`:''}${job.region_id?`<button class="button primary" data-job-region="${esc(job.region_id)}">Inspect saved observations ↗</button>`:''}<details><summary>Collection details</summary><p>Updated ${esc(acquisitionWhen(job.updated_at))}</p>${job.next_retry_at?`<p>Next retry: ${esc(acquisitionWhen(job.next_retry_at))}</p>`:''}<p>Cancel stops after the current image check; saved observations are preserved. Resume keeps the saved plan and starts another collection budget.</p><button type="button" class="button secondary" data-job-export="${esc(job.id)}">Download full collection record</button></details></div></article>`;
  }).join(''):'<p class="observations-empty">No collection jobs yet. Confirm a small area and start collecting observations above.</p>';
  $$('[data-job-cancel]').forEach(button=>button.addEventListener('click',()=>changeJob(button,'cancel')));
  $$('[data-job-resume]').forEach(button=>button.addEventListener('click',()=>changeJob(button,'resume')));
  $$('[data-job-region]').forEach(button=>button.addEventListener('click',async()=>{state.tab='coverage';location.hash='overview';await loadWorkspace(button.dataset.jobRegion);$('#region-detail').scrollIntoView({behavior:'smooth'});}));
  $$('[data-job-export]').forEach(button=>button.addEventListener('click',()=>downloadJSON(jobs.find(job=>job.id===button.dataset.jobExport),'forestmind-job-'+button.dataset.jobExport+'.json')));
}
async function changeJob(button,action) {
  button.disabled=true;
  try{await api('/acquisition/jobs/'+encodeURIComponent(button.dataset[action==='cancel'?'jobCancel':'jobResume'])+'/'+action,{method:'POST'});await refreshAcquisitionJobs();toast(action==='cancel'?'Cancellation requested. Accepted observations are retained.':'Collection resumed with its saved plan and download limit.');}
  catch(error){toast(userMessage(error,action==='cancel'?'Cancellation could not be requested. Refresh the list and try again.':'Collection could not resume. Refresh the list and try again.'));button.disabled=false;}
}
async function refreshAcquisitionJobs({preserveSelection=false}={}) {
  clearTimeout(acquisition.timer);
  try{
    const result=await api('/acquisition/jobs'),jobs=result.jobs || [],old=new Map(acquisition.jobs.map(j=>[j.id,j.status]));acquisition.jobs=jobs;
    renderAcquisitionJobs();$('#job-status').textContent=`${jobs.length} saved jobs · updated ${new Date().toLocaleTimeString()}`;
    const complete=jobs.find(j=>j.region_id&&['completed','partial'].includes(j.status)&&old.has(j.id)&&old.get(j.id)!==j.status);
    if(complete&&!preserveSelection){await loadWorkspace(complete.region_id);toast('Collection saved. Inspect real observations and remaining gaps.');}
  }catch(error){$('#job-status').textContent=userMessage(error,'The collection list could not be loaded.')+' Use Refresh jobs to retry.';}
  finally{if(acquisition.jobs.some(j=>['queued','running','retry_wait'].includes(j.status)))acquisition.timer=setTimeout(refreshAcquisitionJobs,3500);}
}
async function renderMonitoringControls(id) {
  $('#detail-body').innerHTML='<p class="inline-status" role="status">Loading saved monitoring settings…</p>';
  try{
    const [schedule,record,operations]=await Promise.all([api('/regions/'+encodeURIComponent(id)+'/schedule'),api('/regions/'+encodeURIComponent(id)+'/acquisition'),api('/regions/'+encodeURIComponent(id)+'/operational-alerts')]);
    if(state.selected!==id||state.tab!=='monitor')return;
    const config=schedule.schedule || schedule,history=record.configuration || {},linked=Boolean(record.configuration);
    const lastError=config.last_error || record.jobs?.[0]?.error;
    const today=new Date().toISOString().slice(0,10),defaults=refreshDefaults({...history.request,start:history.study_period?.start || history.request?.start},today);
    $('#detail-body').innerHTML=`<div class="observations-caption"><div><strong>Continuing real-data monitoring</strong><p class="form-help">Refresh uses this saved area's exact boundary and collection settings. Existing observations are preserved and duplicate months are skipped.</p></div><button class="button primary" type="submit" form="refresh-form" id="refresh-region" ${linked?'':'disabled'}>Collect missing observations</button></div><form id="refresh-form" class="coverage-form"><label for="refresh-start">History start<input id="refresh-start" type="date" min="2015-06-23" max="${today}" required ${linked?'':'disabled'}></label><label for="refresh-end">History end<input id="refresh-end" type="date" min="2015-06-23" max="${today}" required ${linked?'':'disabled'}></label><label for="refresh-budget">Download attempts per collection<input id="refresh-budget" type="number" min="1" max="24" step="1" required ${linked?'':'disabled'}></label></form><p class="form-help">Choose an earlier start to extend this area's history, or a smaller period to revisit gaps. Up to 60 calendar months per collection; failed downloads also use the budget. After a successful collection, this budget becomes the default for future updates, including local schedules. Existing observations stay unchanged. Defaults cover the saved start through today, limited to the latest 60 months.</p><p id="refresh-status" class="inline-status" role="status"></p>${(operations.alerts || []).map(alert=>`<p class="evidence-disclaimer"><strong>Monitoring freshness:</strong> ${esc(userMessage(alert.message,'Review the latest observation date and collect missing observations if needed.'))}</p>`).join('')}<div class="monitor-status"><div><small>Last successful update</small><strong>${esc(acquisitionWhen(config.last_successful_update || config.last_success_at || history.last_successful_update,'No successful update yet'))}</strong></div><div><small>Next local update</small><strong>${esc(config.enabled?acquisitionWhen(config.next_run_at,'Scheduled by service'):'Scheduling is off')}</strong></div><div><small>Last update failure</small><strong>${esc(lastError?userMessage(lastError,'The last update could not finish. Check collection jobs and retry.'):'None recorded')}</strong></div></div><form id="monitor-form" class="monitor-form"><label class="check-label"><input type="checkbox" id="monitor-enabled" ${linked?'':'disabled'} ${config.enabled?'checked':''}> Keep this saved area updated</label><label>Update interval (days)<input type="number" id="monitor-interval" min="1" max="365" step="1" required ${linked?'':'disabled'}></label><button class="button secondary" type="submit" ${linked?'':'disabled'}>Save schedule</button><p class="form-help">Local scheduling runs only while ForestMind is running. Provider availability and the saved download budget still apply.</p></form><div id="monitor-error" class="form-error" role="alert"></div><details class="model-hashes"><summary>Saved collection settings</summary>${savedAcquisitionSummary(record)}${linked?'<button type="button" class="button secondary" id="download-acquisition-settings">Download full collection details</button>':''}</details><p class="form-help">View all running, failed and resumable updates in <a class="text-link" href="#explore">Collection jobs in Find a forest</a>.</p>`;
    $('#monitor-interval').value=String(config.interval_days || 30);
    $('#refresh-start').value=defaults.start;$('#refresh-end').value=defaults.end;$('#refresh-budget').value=String(defaults.max_downloads);
    $('#download-acquisition-settings')?.addEventListener('click',()=>downloadJSON(record,'forestmind-collection-settings.json'));
    const refreshForm=$('#refresh-form'),refreshButton=$('#refresh-region'),refreshStatus=$('#refresh-status'),refreshError=$('#monitor-error');
    const currentRefresh=()=>state.selected===id&&state.tab==='monitor'&&$('#refresh-form')===refreshForm;
    refreshForm.addEventListener('submit',async event=>{
      event.preventDefault();const button=refreshButton;if(!currentRefresh()||button.disabled)return;
      const values={start:$('#refresh-start').value,end:$('#refresh-end').value,max_downloads:Number($('#refresh-budget').value)};
      const error=refreshInputError(values);refreshError.textContent=error || '';if(error)return;
      button.disabled=true;refreshStatus.textContent='Queuing collection…';
      try{
        const submitted=await api('/regions/'+encodeURIComponent(id)+'/refresh',{method:'POST',body:JSON.stringify(values)});
        if(currentRefresh())refreshStatus.textContent='Collection queued. Follow its progress using the Collection jobs link below.';
        toast('Area update queued. Saved observations are preserved.');await refreshAcquisitionJobs({preserveSelection:true});
        if(!currentRefresh())return;
        button.disabled=false;
        // A small cached collection can finish before the first jobs refresh;
        // it has no previous status for the normal polling transition check.
        const latest=acquisition.jobs.find(job=>job.id===submitted.id) || submitted;
        if(['completed','partial','failed','cancelled'].includes(latest.status)){
          refreshStatus.textContent='Collection finished. Review the result in Collection jobs.';
          if(['completed','partial'].includes(latest.status)&&latest.region_id===id)await loadWorkspace(id);
        }
      }catch(error){if(currentRefresh()){refreshStatus.textContent='';refreshError.textContent=userMessage(error,'The update could not start. Check collection jobs, then try again.');button.disabled=false;}}
    });
    $('#monitor-form').addEventListener('submit',async event=>{event.preventDefault();const button=$('button',event.target);button.disabled=true;try{await api('/regions/'+encodeURIComponent(id)+'/schedule',{method:'PUT',body:JSON.stringify({enabled:$('#monitor-enabled').checked,interval_days:Number($('#monitor-interval').value)})});toast('Local monitoring schedule saved.');await renderMonitoringControls(id);}catch(error){$('#monitor-error').textContent=userMessage(error,'The schedule could not be saved. Check the interval and try again.');button.disabled=false;}});
  }catch(error){if(state.selected===id&&state.tab==='monitor')$('#detail-body').innerHTML=`<p class="observations-empty">${esc(userMessage(error,'Monitoring settings could not be loaded.'))} Select Monitoring updates again to retry, or create a real collection through <a href="#explore" class="text-link">Find a forest</a>.</p>`;}
}
function initializeAreaMap() {
  const map=$('#area-map');
  map.addEventListener('pointerdown',event=>{if(event.button!==0)return;event.preventDefault();map.setPointerCapture(event.pointerId);acquisition.map.drag={start:mapPoint(event),center:[...acquisition.map.center]};});
  map.addEventListener('pointermove',event=>{
    const drag=acquisition.map.drag;if(!drag||acquisition.map.mode!=='draw')return;
    const end=mapPoint(event);let preview=$('#area-draw-preview');
    if(!preview){preview=document.createElementNS('http://www.w3.org/2000/svg','rect');preview.id='area-draw-preview';preview.setAttribute('fill','#285d4533');preview.setAttribute('stroke','#173e32');preview.setAttribute('stroke-width','2');preview.setAttribute('stroke-dasharray','5 3');map.append(preview);}
    preview.setAttribute('x',Math.min(drag.start[0],end[0]));preview.setAttribute('y',Math.min(drag.start[1],end[1]));preview.setAttribute('width',Math.abs(end[0]-drag.start[0]));preview.setAttribute('height',Math.abs(end[1]-drag.start[1]));
  });
  map.addEventListener('pointerup',event=>{
    const drag=acquisition.map.drag;if(!drag)return;acquisition.map.drag=null;$('#area-draw-preview')?.remove();const end=mapPoint(event);
    if(acquisition.map.mode==='draw'){
      const a=mapCoordinate(drag.start),b=mapCoordinate(end);if(Math.abs(end[0]-drag.start[0])<3||Math.abs(end[1]-drag.start[1])<3){areaError('Drag a rectangle, or enter exact coordinates below.');return;}
      acquisition.place=null;$('#place-selection').textContent='Drawn WGS84 rectangle. Forest cover within this boundary has not been verified.';
      updateArea([Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])]);
    }else{
      const center=ForestArea.project(...drag.center,acquisition.map.zoom);acquisition.map.center=ForestArea.unproject(center[0]-(end[0]-drag.start[0]),center[1]-(end[1]-drag.start[1]),acquisition.map.zoom);acquisition.map.center[0]=Math.max(-180,Math.min(180,acquisition.map.center[0]));acquisition.map.center[1]=Math.max(-84,Math.min(84,acquisition.map.center[1]));renderAreaMap();
    }
  });
  map.addEventListener('pointercancel',()=>{acquisition.map.drag=null;$('#area-draw-preview')?.remove();});
  map.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','='].includes(event.key))return;event.preventDefault();if(['+','-','='].includes(event.key)){acquisition.map.zoom=Math.max(1,Math.min(18,acquisition.map.zoom+(event.key==='-'?-1:1)));}else{const center=ForestArea.project(...acquisition.map.center,acquisition.map.zoom);center[0]+=event.key==='ArrowLeft'?-100:event.key==='ArrowRight'?100:0;center[1]+=event.key==='ArrowUp'?-80:event.key==='ArrowDown'?80:0;acquisition.map.center=ForestArea.unproject(...center,acquisition.map.zoom);acquisition.map.center[0]=Math.max(-180,Math.min(180,acquisition.map.center[0]));acquisition.map.center[1]=Math.max(-84,Math.min(84,acquisition.map.center[1]));}renderAreaMap();});
  $('#draw-area').addEventListener('click',()=>{acquisition.map.mode=acquisition.map.mode==='draw'?'pan':'draw';renderAreaMap();});
  $('#area-zoom-in').addEventListener('click',()=>{acquisition.map.zoom=Math.min(18,acquisition.map.zoom+1);renderAreaMap();});
  $('#area-zoom-out').addEventListener('click',()=>{acquisition.map.zoom=Math.max(1,acquisition.map.zoom-1);renderAreaMap();});
  $('#area-fit').addEventListener('click',()=>{if(acquisition.bbox)updateArea(acquisition.bbox,true);});
  $('#apply-coordinates').addEventListener('click',()=>{acquisition.place=null;$('#place-selection').textContent='User-selected WGS84 rectangle. Forest cover inside it has not been verified.';updateArea(areaFields(),true);});
  ['west','south','east','north'].forEach(key=>$('#area-'+key).addEventListener('input',()=>{acquisition.confirmed=false;$('#confirm-area').checked=false;$('#confirm-area').disabled=true;$('#start-acquisition').disabled=true;$('#area-summary').textContent='Coordinates changed. Select “Apply coordinates” to preview and validate.';}));
  $('#area-example').addEventListener('click',()=>{acquisition.place=null;$('#area-name').value='Tapajos small real-data study';$('#area-country').value='Brazil';$('#place-selection').textContent='Small coordinate example in Brazil. The outline is a study area, not a verified forest boundary. No sample observations are inserted.';updateArea([-55.10,-3.50,-55.08,-3.48],true);});
  $('#confirm-area').addEventListener('change',event=>{acquisition.confirmed=event.target.checked;$('#start-acquisition').disabled=!acquisition.confirmed;});
  $('#place-search').addEventListener('submit',searchPlaces);$('#acquisition-form').addEventListener('submit',submitAcquisition);$('#refresh-jobs').addEventListener('click',refreshAcquisitionJobs);
  const today=new Date().toISOString().slice(0,10);$('#history-end').max=today;$('#history-start').max=today;$('#history-end').value=today;const start=new Date();start.setUTCMonth(start.getUTCMonth()-11,1);$('#history-start').value=start.toISOString().slice(0,10);
  renderAreaMap();
  api('/acquisition/capabilities').then(capabilities=>{acquisition.limits=capabilities.limits || capabilities;$('#acquisition-limits').textContent=`Up to ${currentAreaLimit()} km² · maximum 60 months per collection · one local acquisition at a time. Satellite detail varies; sampling is set by your rectangle and grid size.`;if(capabilities.available===false)$('#acquisition-availability').textContent=userMessage(capabilities.reason,'Satellite collection is unavailable. The local installation needs attention.');}).catch(error=>{$('#acquisition-availability').textContent=userMessage(error,'Collection availability could not be checked. Refresh this page to retry.');});
  refreshAcquisitionJobs();
}
initializeAreaMap();
