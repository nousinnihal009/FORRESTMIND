/* The selected-area analysis record and alert lifecycle. All output comes from the API. */
'use strict';
let workflowRequest=0;
function workflowText(value) {return typeof value==='string'?value:typeof value==='number'?numeric(value,3):Array.isArray(value)?value.map(workflowText).join(' · '):value?JSON.stringify(value):'Not available';}
function workflowCapabilities(items) {
  return `<div class="workflow-capabilities">${items.map(item=>`<article class="capability-card"><div><h3>${esc(item.name || item.id)}</h3>${badge(item.status || (item.available?'Available':'Unavailable'))}</div><p>${esc(item.reason || item.description || '')}</p>${item.requires?.length?`<details><summary>Required evidence</summary><ul>${item.requires.map(value=>`<li>${esc(workflowText(value))}</li>`).join('')}</ul></details>`:''}${item.evidence?`<details><summary>Evidence and computation</summary><p>${esc(workflowText(item.evidence))}</p></details>`:''}</article>`).join('')}</div>`;
}
function renderSpatialEvidence(data) {
  if(!data)return '';
  const grid=data.layers?.ndvi?.values || data.grid || data.change_grid || data.ndvi_change_grid || data.cells,values=Array.isArray(grid)&&Array.isArray(grid[0])?grid:null;
  let image='';
  if(values){
    const h=values.length,w=values[0].length;
    const color=value=>typeof value!=='number'||!Number.isFinite(value)?'#dce2d8':value<-.2?'#a34f32':value<-.05?'#d2aa67':value>.2?'#296947':value>.05?'#87a86c':'#e7eadc';
    image=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Observed spatial index change, north at top. Missing pixels are gray.">${values.flatMap((row,y)=>row.map((value,x)=>`<rect x="${x}" y="${y}" width="1" height="1" fill="${color(value)}"><title>${typeof value==='number'?value.toFixed(3):'Missing or masked'}</title></rect>`)).join('')}</svg><div class="spatial-grid-legend"><span style="--swatch:#a34f32">Larger index decrease</span><span style="--swatch:#e7eadc">Small observed change</span><span style="--swatch:#296947">Larger index increase</span><span style="--swatch:#dce2d8">Missing / masked</span></div>`;
  }
  return `<section class="spatial-evidence"><h3>Spatial observation evidence${values?' · NDVI change':''}</h3><p class="form-help">${esc(data.reason || data.caveat || 'Observed index changes describe sampled pixels. They do not establish land-cover class, forest extent, or a calibrated future risk.')}</p>${values?`<p class="form-help">${esc(data.start)} → ${esc(data.end)} · latest minus first · north at top · valid fraction ${pct(data.layers?.ndvi?.valid_fraction)} · mean change ${numeric(data.layers?.ndvi?.mean_delta,3)}</p>`:''}${image}<p class="form-help">${esc((data.limitations || []).join(' '))}</p><details class="model-hashes"><summary>Spatial method, dates, geometry and source details</summary><pre>${esc(JSON.stringify(data,null,2))}</pre></details></section>`;
}
function renderClimateContext(data) {
  if(!data?.available)return '';
  return `<section class="spatial-evidence"><h3>Climate context · ${esc(data.dataset || data.provider)}</h3><p class="form-help">Reanalysis at the selected area's centroid. Context only: these fields are not automatically treated as trained multimodal model inputs.</p><div class="table-scroll"><table><thead><tr><th>Month</th><th>Temperature (°C)</th><th>Rainfall (mm)</th><th>Reference ET₀ (mm)</th><th>Surface soil moisture (m³/m³)</th><th>Quality</th></tr></thead><tbody>${(data.monthly || []).map(row=>`<tr><td>${esc(row.month)}</td><td>${numeric(row.temperature_2m_mean,2)}</td><td>${numeric(row.precipitation_sum,2)}</td><td>${numeric(row.et0_fao_evapotranspiration,2)}</td><td>${numeric(row.soil_moisture_0_to_7cm_mean,3)}</td><td>${esc(workflowText(row.quality))}</td></tr>`).join('')}</tbody></table></div><p class="form-help">${esc((data.limitations || []).join(' '))}</p></section>`;
}
function renderRecoveryEvidence(data) {
  if(!data?.available)return '';
  return `<section class="spatial-evidence"><h3>Observed recovery episodes</h3><p class="form-help">${esc(data.reason || '')} Baseline ${esc(data.baseline_start || '')} → ${esc(data.baseline_end || '')}.</p>${data.episodes?.length?`<div class="table-scroll"><table><thead><tr><th>Episode starts</th><th>Status</th><th>Observed recovery date</th><th>Recovery duration (days)</th><th>Minimum anomaly</th></tr></thead><tbody>${data.episodes.map(row=>`<tr><td>${esc(row.start)}</td><td>${esc(row.status)}</td><td>${esc(row.recovered_at || 'Not observed')}</td><td>${numeric(row.recovery_days)}</td><td>${numeric(row.minimum_anomaly,3)}</td></tr>`).join('')}</tbody></table></div>`:''}<p class="form-help">Observed recovery is a retrospective index diagnostic. Missing or unresolved episodes do not have an invented recovery date.</p></section>`;
}
function renderWorkflowSections(report) {
  const sections=report.sections || report.results || report;
  const names={observations:'Multimodal observation record',condition:'Observed condition and disturbance',drivers:'Candidate drivers and competing explanations',forecast:'Temporal forecasts',resilience:'Resilience diagnostics',recovery:'Observed recovery',fragmentation:'Fragmentation and connectivity',model:'Checkpoint inference',scenarios:'Five-year hypothetical scenarios',climate:'Climate reanalysis context',alerts:'Prioritized alerts',evaluation:'Historical baseline comparison'};
  return renderSpatialEvidence(sections.spatial_change || report.spatial_change)+renderClimateContext(sections.climate)+renderRecoveryEvidence(sections.recovery)+Object.entries(names).filter(([key])=>sections[key]).map(([key,title])=>`<details class="workflow-section model-hashes"><summary>${esc(title)} · saved computation</summary><pre>${esc(JSON.stringify(sections[key],null,2))}</pre></details>`).join('');
}
async function renderWorkflow(id) {
  const request=++workflowRequest;
  $('#detail-body').innerHTML='<p class="inline-status" role="status">Loading the selected area’s connected analysis record…</p>';
  try{
    const result=await api('/regions/'+encodeURIComponent(id)+'/workflow');
    if(request!==workflowRequest||state.selected!==id||state.tab!=='workflow')return;
    const report=result.report || result,manifest=report.manifest || report.context || {},capabilities=report.capabilities || [],list=Array.isArray(capabilities)?capabilities:Object.entries(capabilities).map(([id,value])=>({id,...value}));
    $('#detail-body').innerHTML=`<div class="workflow-intro"><div><div class="eyebrow">ONE AREA · ONE EVIDENCE RECORD</div><h3>Connected analysis</h3><p class="form-help">Run the available computations for this area. Missing data and unsupported model scope remain explicit. Observation indicators, model predictions and hypothetical scenarios retain their own interpretation.</p></div>${badge(isDemo(state.region)?'Synthetic demonstration':'Real-data research')}</div><div class="workflow-actions"><button class="button primary" id="run-workflow">Run selected-area analysis</button><button class="button secondary" id="collect-climate">Collect climate context</button><button class="button secondary" id="export-workflow">Export reproducible record</button></div><div id="workflow-status" class="inline-status" role="status" aria-live="polite"></div><div class="workflow-manifest"><strong>Selected area: ${esc(state.region.name)}</strong><br>Observation cutoff: ${esc(manifest.observed_through || manifest.observation_cutoff || report.observed_through || state.region.latest_date || 'No observations')}<br>Geometry and data/model versions are retained in this report.<details><summary>Inspect shared analysis context</summary><pre>${esc(JSON.stringify(manifest,null,2))}</pre></details></div>${list.length?workflowCapabilities(list):'<p class="form-help">No persisted analysis record yet. Run selected-area analysis to evaluate the available evidence.</p>'}${renderWorkflowSections(report)}`;
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
