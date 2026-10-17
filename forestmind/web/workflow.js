/* The selected-area analysis record and alert lifecycle. All output comes from the API. */
'use strict';
let workflowRequest=0;
function workflowText(value) {return typeof value==='string'?value:typeof value==='number'?numeric(value,3):Array.isArray(value)?value.map(workflowText).join(' · '):value?JSON.stringify(value):'Not available';}
function workflowCapabilities(items) {
  return `<div class="workflow-capabilities">${items.map(item=>`<article class="capability-card"><div><h3>${esc(item.name || item.id)}</h3>${badge(item.status || (item.available?'Available':'Unavailable'))}</div><p>${esc(item.reason || item.description || '')}</p>${item.requires?.length?`<details><summary>Required evidence</summary><ul>${item.requires.map(value=>`<li>${esc(workflowText(value))}</li>`).join('')}</ul></details>`:''}${item.evidence?`<details><summary>Evidence and computation</summary><p>${esc(workflowText(item.evidence))}</p></details>`:''}</article>`).join('')}</div>`;
}
function renderSpatialEvidence(data) {
  if(!data)return '';
  const values=data.layers?.ndvi?.values;
  const metadata={...data,layers:data.layers?Object.fromEntries(Object.entries(data.layers).map(([key,value])=>[key,{mean_delta:value.mean_delta,valid_fraction:value.valid_fraction}])):undefined};
  return `<section class="spatial-evidence"><h3>Spatial observation evidence</h3><p class="form-help">${esc(data.reason || 'Two accepted optical acquisitions are required.')}</p>${values?`<div class="spatial-toolbar"><label>Observed index change <select id="spatial-index"><option value="ndvi">NDVI · vegetation</option><option value="ndmi">NDMI · moisture</option><option value="nbr">NBR · burn-sensitive index</option></select></label><span>${esc(data.start)} → ${esc(data.end)}</span></div><p class="form-help">Latest minus first · north at top · geographic extent ${esc((data.bbox || []).map(value=>Number(value).toFixed(5)).join(', '))}</p><canvas id="spatial-change-map" width="${values[0].length}" height="${values.length}" tabindex="0" role="img" aria-label="Observed spatial spectral change on the selected area. Use arrow keys to inspect sampled pixels."></canvas><p id="spatial-pixel-value" class="form-help" aria-live="polite"></p><div class="spatial-grid-legend"><span style="--swatch:#a34f32">Decrease &lt; −0.2</span><span style="--swatch:#d2aa67">Decrease −0.2 to −0.05</span><span style="--swatch:#e7eadc">Change −0.05 to +0.05</span><span style="--swatch:#87a86c">Increase +0.05 to +0.2</span><span style="--swatch:#296947">Increase &gt; +0.2</span><span style="--swatch:#dce2d8">Missing / masked</span></div>`:''}<p class="form-help">${esc((data.limitations || []).join(' '))}</p><details class="model-hashes"><summary>Spatial method, dates, geometry and source details</summary><pre>${esc(JSON.stringify(metadata,null,2))}</pre><p class="form-help">The reproducible export includes every grid value, including null for masked pixels.</p></details></section>`;
}
function drawSpatialEvidence(data) {
  const canvas=$('#spatial-change-map');if(!canvas||!data.layers)return;
  const context=canvas.getContext('2d'),pixel=[0,0];
  const inspect=()=>{const layer=data.layers[$('#spatial-index').value],value=layer.values[pixel[1]][pixel[0]],bbox=data.bbox;$('#spatial-pixel-value').textContent=`Pixel ${pixel[0]+1}, ${pixel[1]+1}: ${typeof value==='number'?value.toFixed(5)+' index difference':'missing or masked'} · ${bbox?(bbox[0]+(pixel[0]+.5)*(bbox[2]-bbox[0])/canvas.width).toFixed(6)+'°, '+(bbox[3]-(pixel[1]+.5)*(bbox[3]-bbox[1])/canvas.height).toFixed(6)+'°':''} · area mean ${numeric(layer.mean_delta,3)} · usable pixels ${pct(layer.valid_fraction)}`;};
  const paint=()=>{const values=data.layers[$('#spatial-index').value].values,img=context.createImageData(canvas.width,canvas.height);values.forEach((row,y)=>row.forEach((v,x)=>{const color=typeof v!=='number'?[220,226,216]:v<-.2?[163,79,50]:v<-.05?[210,170,103]:v>.2?[41,105,71]:v>.05?[135,168,108]:[231,234,220];const index=(y*canvas.width+x)*4;img.data.set([...color,255],index);}));context.putImageData(img,0,0);inspect();};
  $('#spatial-index').addEventListener('change',paint);
  canvas.addEventListener('click',event=>{const rect=canvas.getBoundingClientRect();pixel[0]=Math.max(0,Math.min(canvas.width-1,Math.floor((event.clientX-rect.left)/rect.width*canvas.width)));pixel[1]=Math.max(0,Math.min(canvas.height-1,Math.floor((event.clientY-rect.top)/rect.height*canvas.height)));inspect();});
  canvas.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;event.preventDefault();pixel[0]=Math.max(0,Math.min(canvas.width-1,pixel[0]+(event.key==='ArrowLeft'?-1:event.key==='ArrowRight'?1:0)));pixel[1]=Math.max(0,Math.min(canvas.height-1,pixel[1]+(event.key==='ArrowUp'?-1:event.key==='ArrowDown'?1:0)));inspect();});paint();
}
function renderClimateContext(data) {
  if(!data?.available)return '';
  return `<section class="spatial-evidence"><h3>Climate context · ${esc(data.dataset || data.provider)}</h3><p class="form-help">Reanalysis at the selected area's centroid. Context only: these fields are not automatically treated as trained multimodal model inputs.</p><div class="table-scroll"><table><thead><tr><th>Month</th><th>Temperature (°C)</th><th>Rainfall (mm)</th><th>Reference ET₀ (mm)</th><th>Surface soil moisture (m³/m³)</th><th>Quality</th></tr></thead><tbody>${(data.monthly || []).map(row=>`<tr><td>${esc(row.month)}</td><td>${numeric(row.temperature_2m_mean,2)}</td><td>${numeric(row.precipitation_sum,2)}</td><td>${numeric(row.et0_fao_evapotranspiration,2)}</td><td>${numeric(row.soil_moisture_0_to_7cm_mean,3)}</td><td>${esc(workflowText(row.quality))}</td></tr>`).join('')}</tbody></table></div><p class="form-help">${esc((data.limitations || []).join(' '))}</p></section>`;
}
function renderRecoveryEvidence(data) {
  if(!data?.available)return '';
  return `<section class="spatial-evidence"><h3>Observed recovery episodes</h3><p class="form-help">${esc(data.reason || '')} Baseline ${esc(data.baseline_start || '')} → ${esc(data.baseline_end || '')}.</p>${data.episodes?.length?`<div class="table-scroll"><table><thead><tr><th>Episode starts</th><th>Status</th><th>Observed recovery date</th><th>Recovery duration (days)</th><th>Minimum anomaly</th></tr></thead><tbody>${data.episodes.map(row=>`<tr><td>${esc(row.start)}</td><td>${esc(row.status)}</td><td>${esc(row.recovered_at || 'Not observed')}</td><td>${numeric(row.recovery_days)}</td><td>${numeric(row.minimum_anomaly,3)}</td></tr>`).join('')}</tbody></table></div>`:''}<p class="form-help">Observed recovery is a retrospective index diagnostic. Missing or unresolved episodes do not have an invented recovery date.</p></section>`;
}
function renderFragmentation(data) {
  if(!data?.available)return '';
  return `<section class="spatial-evidence"><h3>Classified forest topology</h3><p class="form-help">${esc(data.source)} · ${esc(data.observed_at)} · ${esc(data.classification)}. This classification is declared by its provider and remains unverified.</p><div class="monitor-status"><div><small>Forest cover inside valid area</small><strong>${pct(data.forest_fraction)} · ${numeric(data.forest_area_km2,3)} km²</strong></div><div><small>Connected patches · four neighbours</small><strong>${numeric(data.patch_count)}</strong></div><div><small>Internal edge density</small><strong>${numeric(data.edge_density_m_per_ha,2)} m/ha</strong></div><div><small>Largest patch / all forest</small><strong>${pct(data.largest_patch_fraction)}</strong></div><div><small>Mean nearest patch centroid</small><strong>${numeric(data.mean_nearest_centroid_m,1)} m</strong></div><div><small>Classification coverage</small><strong>${pct(data.valid_fraction)}</strong></div></div><p class="form-help">${esc((data.limitations || []).join(' '))}</p></section>`;
}
function coverImportForm() {
  return `<details class="workflow-section model-hashes"><summary>Attach a classified forest-cover reference</summary><p class="form-help">A dated, licensed classification enables patch, edge and structural connectivity measurements. Provide actual boolean forest and valid-mask grids aligned to this exact area's WGS84 rectangle; maximum 128 × 128. Vegetation indices alone are not a forest classification.</p><form id="cover-import-form"><label class="field-label" for="cover-import-file">Classification JSON file (maximum 2 MB)</label><input type="file" id="cover-import-file" accept=".json,application/json"><label class="field-label" for="cover-import-json">Classification record</label><textarea id="cover-import-json" rows="8" required spellcheck="false" placeholder='{"observed_at":"YYYY-MM-DD","source":"Reference","classification":"Forest class definition","license":"License","data_kind":"classified_forest_cover","forest":[],"valid_mask":[]}'></textarea><p class="form-help">The selected area's exact geometry is attached when you submit. Values must describe that same grid, with north at the top. This upload does not infer a class or invent missing labels.</p><button type="submit" class="button secondary">Save classification & analyze topology</button><p id="cover-import-status" class="form-error" role="status" aria-live="polite"></p></form></details>`;
}
function wireCoverImport(id) {
  const status=$('#cover-import-status');
  $('#cover-import-file').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;if(file.size>2*1024*1024){status.textContent='Choose a JSON file smaller than 2 MB.';return;}$('#cover-import-json').value=await file.text();status.textContent='';});
  $('#cover-import-form').addEventListener('submit',async event=>{
    event.preventDefault();const button=$('button',event.target);button.disabled=true;
    try{
      const raw=$('#cover-import-json').value;if(new TextEncoder().encode(raw).length>2*1024*1024)throw new Error('Keep the JSON record below 2 MB.');
      let payload;try{payload=JSON.parse(raw);}catch{throw new Error('This is not valid JSON. Check the classification record.');}
      payload.geometry=payload.geometry || state.region.geometry;
      const result=await api('/regions/'+encodeURIComponent(id)+'/forest-cover',{method:'POST',body:JSON.stringify(payload)});let job=result.job;
      status.textContent='Classification saved. Computing topology for the selected area…';
      for(let count=0;count<120&&job?.id&&!['completed','failed'].includes(job.status);count++){await new Promise(resolve=>setTimeout(resolve,1000));job=await api('/workflow-jobs/'+encodeURIComponent(job.id));}
      if(job?.status==='failed')throw new Error(job.error || 'Topology analysis failed; its durable job retains diagnostics.');
      if(job?.status!=='completed')throw new Error('Analysis is still running. Job '+job.id+' is saved; reopen Connected analysis to inspect its result.');
      if(state.selected===id&&state.tab==='workflow')await renderWorkflow(id);toast('Classified-cover reference and topology analysis saved.');
    }catch(error){status.textContent=error.message;button.disabled=false;}
  });
}
function renderWorkflowSections(report) {
  const sections=report.sections || report.results || report;
  const names={observations:'Multimodal observation record',condition:'Observed condition and disturbance',drivers:'Candidate drivers and competing explanations',forecast:'Temporal forecasts',resilience:'Resilience diagnostics',recovery:'Observed recovery',fragmentation:'Fragmentation and connectivity',model:'Checkpoint inference',scenarios:'Five-year hypothetical scenarios',climate:'Climate reanalysis context',alerts:'Prioritized alerts',evaluation:'Historical baseline comparison'};
  return renderSpatialEvidence(sections.spatial_change || report.spatial_change)+renderClimateContext(sections.climate)+renderRecoveryEvidence(sections.recovery)+renderFragmentation(sections.fragmentation)+coverImportForm()+Object.entries(names).filter(([key])=>sections[key]).map(([key,title])=>`<details class="workflow-section model-hashes"><summary>${esc(title)} · saved computation</summary><pre>${esc(JSON.stringify(sections[key],null,2))}</pre></details>`).join('');
}
async function renderWorkflow(id) {
  const request=++workflowRequest;
  $('#detail-body').innerHTML='<p class="inline-status" role="status">Loading the selected area’s connected analysis record…</p>';
  try{
    const result=await api('/regions/'+encodeURIComponent(id)+'/workflow');
    if(request!==workflowRequest||state.selected!==id||state.tab!=='workflow')return;
    const report=result.report || result,manifest=report.manifest || report.context || {},capabilities=report.capabilities || [],list=Array.isArray(capabilities)?capabilities:Object.entries(capabilities).map(([id,value])=>({id,...value}));
    $('#detail-body').innerHTML=`<div class="workflow-intro"><div><div class="eyebrow">ONE AREA · ONE EVIDENCE RECORD</div><h3>Connected analysis</h3><p class="form-help">Run the available computations for this area. Missing data and unsupported model scope remain explicit. Observation indicators, model predictions and hypothetical scenarios retain their own interpretation.</p></div>${badge(isDemo(state.region)?'Synthetic demonstration':'Real-data research')}</div><div class="workflow-actions"><button class="button primary" id="run-workflow">Run selected-area analysis</button><button class="button secondary" id="collect-climate">Collect climate context</button><button class="button secondary" id="export-workflow">Export reproducible record</button></div><div id="workflow-status" class="inline-status" role="status" aria-live="polite"></div><div class="workflow-manifest"><strong>Selected area: ${esc(state.region.name)}</strong><br>Observation cutoff: ${esc(manifest.observed_through || manifest.observation_cutoff || report.observed_through || state.region.latest_date || 'No observations')}<br>Geometry and data/model versions are retained in this report.<details><summary>Inspect shared analysis context</summary><pre>${esc(JSON.stringify(manifest,null,2))}</pre></details></div>${list.length?workflowCapabilities(list):'<p class="form-help">No persisted analysis record yet. Run selected-area analysis to evaluate the available evidence.</p>'}${renderWorkflowSections(report)}`;
    drawSpatialEvidence(report.spatial_change || report.sections?.spatial_change);
    wireCoverImport(id);
    $('#run-workflow').addEventListener('click',()=>runWorkflow(id,false));
    $('#collect-climate').addEventListener('click',()=>runWorkflow(id,true));
    $('#export-workflow').addEventListener('click',async()=>{try{downloadJSON(await api('/regions/'+encodeURIComponent(id)+'/workflow/export'),'forestmind-'+id+'-connected-analysis.json');}catch(error){toast(error.message);}});
  }catch(error){if(request===workflowRequest&&state.selected===id&&state.tab==='workflow')$('#detail-body').innerHTML=`<p class="observations-empty">${esc(error.message)} Select Connected analysis again to retry.</p>`;}
}
async function runWorkflow(id,includeClimate) {
  const status=$('#workflow-status');$('#run-workflow').disabled=true;$('#collect-climate').disabled=true;status.textContent=includeClimate?'Collecting bounded climate reanalysis context and running eligible analyses…':'Preparing compatible inputs and running eligible analyses…';
  try{
    let job=await api('/regions/'+encodeURIComponent(id)+'/workflow',{method:'POST',body:JSON.stringify({include_climate:includeClimate})});
    for(let count=0;count<240&&job.id&&!['completed','failed','cancelled'].includes(job.status);count++){
      await new Promise(resolve=>setTimeout(resolve,1500));job=await api('/workflow-jobs/'+encodeURIComponent(job.id));
      if(state.selected===id&&state.tab==='workflow'&&$('#workflow-status'))$('#workflow-status').textContent=`${job.phase || job.status || 'Running'} · job ${job.id}`;
    }
    if(job.status==='failed')throw new Error(job.error || 'The analysis job failed. Its diagnostics have been saved.');
    if(job.id&&!['completed','cancelled'].includes(job.status))throw new Error('Analysis is still running. Its durable job ID is '+job.id+'. Reopen Connected analysis to inspect the saved result.');
    if(state.selected===id&&state.tab==='workflow')await renderWorkflow(id);
    toast('Selected-area analysis saved with its evidence and limitations.');
  }catch(error){if(state.selected===id&&state.tab==='workflow'&&$('#workflow-status')){$('#workflow-status').textContent=error.message;$('#run-workflow').disabled=false;$('#collect-climate').disabled=false;}else toast(error.message);}
}
async function renderAlertLifecycle(region) {
  const target=$('#alert-lifecycle');if(!target)return;
  try{
    const settings=await api('/regions/'+encodeURIComponent(region.id)+'/alert-settings');
    if(state.selected!==region.id||state.tab!=='observations'||!$('#alert-lifecycle'))return;
    const config=settings.settings || settings;
    $('#alert-lifecycle').innerHTML=`<section class="alert-management"><h3>Alert review and thresholds</h3><p class="form-help">Review dated supporting observations above. Acknowledgement records a review; resolution records your decision. Collection failures are shown separately in Collection jobs.</p>${(region.alerts || []).map(alert=>`<div class="alert-management-row"><strong>${esc(alert.title)}</strong>${badge(alert.status || 'open')}<button class="button secondary" data-alert-id="${esc(alert.id)}" data-alert-state="acknowledged">Acknowledge</button><button class="button secondary" data-alert-id="${esc(alert.id)}" data-alert-state="resolved">Resolve</button></div>`).join('')}<form id="alert-settings-form" class="alert-settings-form"><label>Condition drop (index points)<input name="condition_drop_points" type="number" min="1" max="100" step="0.1" value="${Number(config.condition_drop_points ?? 12)}" required></label><label>Moisture drop (NDMI units)<input name="moisture_drop" type="number" min="0.01" max="2" step="0.01" value="${Number(config.moisture_drop ?? .12)}" required></label><label>Minimum valid pixel fraction<input name="min_valid_fraction" type="number" min="0" max="1" step="0.01" value="${Number(config.min_valid_fraction ?? .4)}" required></label><label>Stale observation age (days)<input name="stale_days" type="number" min="1" max="3650" step="1" value="${Number(config.stale_days ?? 90)}" required></label><button type="submit" class="button secondary">Save thresholds & refresh assessment</button><p class="form-help">Thresholds are screening choices, not validated ecological standards. No external notifications are sent.</p></form><p id="alert-settings-error" class="form-error" role="alert"></p></section>`;
    $$('[data-alert-id]').forEach(button=>button.addEventListener('click',async()=>{button.disabled=true;try{await api('/alerts/'+encodeURIComponent(button.dataset.alertId),{method:'PATCH',body:JSON.stringify({status:button.dataset.alertState,note:''})});await selectRegion(region.id);toast('Alert review saved.');}catch(error){$('#alert-settings-error').textContent=error.message;button.disabled=false;}}));
    $('#alert-settings-form').addEventListener('submit',async event=>{event.preventDefault();const button=$('button',event.target);button.disabled=true;try{const values=Object.fromEntries([...new FormData(event.target)].map(([key,value])=>[key,Number(value)]));await api('/regions/'+encodeURIComponent(region.id)+'/alert-settings',{method:'PUT',body:JSON.stringify(values)});await selectRegion(region.id);toast('Alert thresholds saved.');}catch(error){$('#alert-settings-error').textContent=error.message;button.disabled=false;}});
  }catch(error){if(state.selected===region.id&&state.tab==='observations'&&$('#alert-lifecycle'))$('#alert-lifecycle').innerHTML=`<p class="form-error">Alert controls: ${esc(error.message)}</p>`;}
}
