/* The selected-area analysis record and alert lifecycle. All output comes from the API. */
'use strict';
let workflowRequest=0;
function workflowText(value) {return typeof value==='string'?value:typeof value==='number'?numeric(value,3):typeof value==='boolean'?(value?'Yes':'No'):Array.isArray(value)?value.filter(item=>item===null||typeof item!=='object').map(workflowText).join(' · '):'Not available';}
function workflowLabel(value) {
  const labels={sar:'Radar',optical:'Optical imagery',climate:'Climate',activity:'Human activity',insufficient_data:'Insufficient observations',review_signal:'Review suggested',no_joint_increase:'No joint increase detected',candidate_pressure:'Candidate pressure',unresolved:'Unresolved',right_censored:'Still unresolved',censored_missing_months:'Follow-up interrupted by missing months',observed_return:'Return observed',persistence:'Last observed value',seasonal_naive:'Same season last year',damped_trend:'Damped historical trend',active:'Open',low_coverage:'Low image coverage',missing_indices:'Missing spectral measurements'};
  return labels[value] || (typeof humanLabel==='function'?humanLabel(value):String(value || 'Not available').replaceAll('_',' '));
}
function workflowReason(value) {
  const message=workflowText(value).replace('No trusted local model is registered. Supply a real-trained compatible checkpoint and independent evaluation evidence in FORESTMIND_MODEL_REGISTRY.','No compatible trained model is registered for this area. A model trained on real observations and independently evaluated for this use is required.');
  return typeof userMessage==='function'?userMessage(message,'This evidence could not be prepared. Retry the collection or analysis action.'):message;
}
function workflowNumber(value,digits=2,unit='') {return Number.isFinite(value)?numeric(value,digits)+unit:'Not available';}
function workflowPercent(value) {return Number.isFinite(value)?pct(value):'Not available';}
function workflowFacts(items) {return `<dl class="workflow-facts">${items.map(([label,value])=>`<dt>${esc(label)}</dt><dd>${esc(workflowText(value))}</dd>`).join('')}</dl>`;}
function workflowDate(value) {return typeof value==='string'?dateLabel(value):'Not recorded';}
function workflowSource(value) {return value==='element84-earth-search'?'Sentinel-2 · Earth Search':String(value || '').startsWith('Cached sentinel2 surface reflectance')?'Sentinel-2 satellite measurements':value || 'Source not supplied';}
function workflowList(items) {return items?.length?`<ul class="workflow-evidence-list">${items.map(value=>`<li>${esc(workflowReason(value))}</li>`).join('')}</ul>`:'';}
function workflowTable(headings,rows) {return `<div class="table-scroll"><table class="workflow-results-table"><thead><tr>${headings.map(label=>`<th>${esc(label)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(value=>`<td>${esc(workflowText(value))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
function workflowUnavailable(title,data,fallback) {return `<section class="spatial-evidence"><h3>${esc(title)}</h3><p class="observations-empty"><strong>Not available.</strong> ${esc(workflowReason(data?.reason || data?.caveat || fallback))}</p></section>`;}
function workflowCapabilityEvidence(item) {
  const evidence=item.evidence;if(!evidence)return '';
  if(item.id==='optical'&&Number.isFinite(evidence.count))return `<p>${numeric(evidence.count)} dated observations retained.</p>`;
  if(item.id==='aligned_inputs')return `<p>${esc(evidence.modalities?.length?evidence.modalities.map(workflowLabel).join(' · '):'No model-ready measurements yet.')}</p>${workflowList(Object.entries(evidence.errors || {}).filter(([,reason])=>reason).map(([name,reason])=>workflowLabel(name)+': '+workflowReason(reason)))}`;
  return typeof evidence==='string'?`<p>${esc(evidence)}</p>`:'';
}
function workflowCapabilities(items) {
  const available=items.filter(item=>item.available).length;
  return `<details class="workflow-readiness"><summary>Analysis readiness and missing evidence · ${available} with output · ${items.length-available} need more evidence</summary><div class="workflow-capabilities">${items.map(item=>`<article class="capability-card"><div><h3>${esc(item.name || workflowLabel(item.id))}</h3>${badge(workflowLabel(item.status || (item.available?'Available':'Unavailable')))}</div><p>${esc(workflowReason(item.reason || item.description || 'No assessment recorded.'))}</p>${workflowCapabilityEvidence(item)}${item.requires?.length?`<details><summary>Required evidence</summary>${workflowList(item.requires)}</details>`:''}</article>`).join('')}</div></details>`;
}
function collectionErrors(report) {
  const names={climate:'Climate context',radar:'Radar context',forest_cover:'Forest-cover reference'};
  const errors=Object.entries(report.collection_errors || {}).filter(([,value])=>value);
  return errors.length?`<div class="collection-errors" role="status"><strong>Some requested evidence could not be updated</strong>${errors.map(([key,value])=>`<p><strong>${esc(names[key] || workflowLabel(key))}:</strong> ${esc(workflowReason(value))}</p>`).join('')}<p>Any earlier successful context remains dated in this record. Use the corresponding collection action to retry; unavailable data has not been filled in.</p></div>`:'';
}
function renderAlignedInputs(report) {
  const sequence=report.sequence || report.sections?.sequence;
  if(!sequence)return '';
  const model=report.model || report.sections?.model;
  const modelReason=(model?.reason || '').replace('No trusted local model is registered. Supply a real-trained compatible checkpoint and independent evaluation evidence in FORESTMIND_MODEL_REGISTRY.','No compatible trained model is registered for this area.');
  const modelStatus=model?`<p class="form-help"><strong>${model.available?'A model result was saved.':'No model prediction was produced.'}</strong> ${esc(workflowReason(modelReason))}</p>`:'';
  if(sequence.available===false)return `<section class="spatial-evidence"><h3>Measured inputs available to a model</h3><p class="observations-empty"><strong>Images could not be prepared for a model.</strong> ${esc(workflowReason(sequence.reason || 'No usable image sequence is available for this area.'))}</p>${modelStatus}</section>`;
  const alignment=sequence.alignment || {},parameters=alignment.parameters || {};
  const dates=Number.isInteger(sequence.selected_acquisitions)?sequence.selected_acquisitions:(sequence.sources || []).length;
  const radar=alignment.radar,climate=alignment.climate;
  const radarDates=radar?(radar.matches || []).filter(row=>row.scene_id).length:0;
  const windows=climate?.windows || [];
  const climateDates=windows.filter(row=>Array.isArray(row.mask)&&row.mask.length>0&&row.mask.every(Boolean)).length;
  const partialDates=windows.filter(row=>Array.isArray(row.mask)&&row.mask.some(Boolean)&&!row.mask.every(Boolean)).length;
  const errors={...(report.manifest?.alignment_errors || {}),...(report.alignment_errors || {})};
  const errorDetails=Object.entries(errors).filter(([,reason])=>reason).map(([name,reason])=>`<p><strong>${esc(name==='radar'?'Radar':name==='climate'?'Climate':name)}:</strong> ${esc(workflowReason(reason))}</p>`).join('');
  const radarStatus=errors.radar?'Could not prepare':radar?`${radarDates} / ${dates} image dates`:'No matched inputs in this saved record';
  const climateStatus=errors.climate?'Could not prepare':climate?`${climateDates} / ${dates} image dates with all measurements`:'No matched inputs in this saved record';
  const radarReason=errors.radar?'Collected radar evidence could not be used with these images. The reason is shown below.':radar?`Radar must be from the same time or earlier, within ${numeric(radar.max_age_days)} days. ${dates-radarDates} image dates have no usable match.`:report.radar?.available?'Radar evidence has been collected. Run selected-area analysis to check matching dates and coverage.':'Collect radar context to look for compatible observations.';
  const climateReason=errors.climate?'Collected climate evidence could not be used with these images. The reason is shown below.':climate?`${partialDates?partialDates+' image '+(partialDates===1?'date has':'dates have')+' only some measurements. ':''}Temperature, rainfall, reference evaporation and surface soil moisture retain separate coverage checks.`:report.climate?.available?'Climate evidence has been collected. Run selected-area analysis to check matching dates and coverage.':'Collect climate context to prepare dated measurements.';
  const timing=climate?`Climate windows cover ${numeric(parameters.climate_window_days)} days and end ${numeric(parameters.climate_lag_days)} days before each image. Reanalysis can be revised later; historical availability has not been verified.`:'';
  return `<section class="spatial-evidence"><h3>Measured inputs available to a model</h3><p class="form-help">This summary shows which measurements can accompany each optical image. Missing dates remain missing. A compatible trained model and sufficient coverage are required before prediction.</p><div class="table-scroll"><table class="aligned-input-table"><thead><tr><th>Input</th><th>Coverage</th><th>What is available</th></tr></thead><tbody><tr><td>Optical imagery</td><td>${numeric(dates)} image dates</td><td>Dated satellite images with valid-pixel masks.</td></tr><tr><td>Radar</td><td>${esc(radarStatus)}</td><td>${esc(radarReason)}</td></tr><tr><td>Climate</td><td>${esc(climateStatus)}</td><td>${esc(climateReason)}</td></tr></tbody></table></div>${timing?`<p class="form-help">${esc(timing)}</p>`:''}${errorDetails?`<div class="collection-errors" role="status"><strong>Some evidence could not be used as model input</strong>${errorDetails}</div>`:''}${modelStatus}</section>`;
}
function renderSpatialEvidence(data) {
  if(!data)return '';
  const values=data.layers?.ndvi?.values;
  const metadata=workflowFacts([['Measurement','Later image minus earlier image'],['Earlier image',workflowDate(data.start)],['Later image',workflowDate(data.end)],['Grid',Array.isArray(data.shape)?data.shape.join(' × ')+' samples':'Not supplied'],['Map reference',data.crs==='EPSG:4326'?'Latitude and longitude (WGS84)':data.crs || 'Not supplied']])+workflowTable(['Index','Mean observed difference','Usable area'],Object.entries(data.layers || {}).map(([name,layer])=>[workflowLabel(name),workflowNumber(layer.mean_delta,3),workflowPercent(layer.valid_fraction)]))+workflowTable(['Image date','Source','Usable image pixels'],(data.sources || []).map(source=>[workflowDate(source.acquired_at),workflowSource(source.provider),workflowPercent(source.valid_fraction)]));
  return `<section class="spatial-evidence"><h3>Spatial observation evidence</h3><p class="form-help">${esc(data.reason || 'Two accepted optical acquisitions are required.')}</p>${values?`<div class="spatial-toolbar"><label>Observed index change <select id="spatial-index"><option value="ndvi">NDVI · vegetation</option><option value="ndmi">NDMI · moisture</option><option value="nbr">NBR · burn-sensitive index</option></select></label><span>${esc(data.start)} → ${esc(data.end)}</span></div><p class="form-help">Latest minus first · north at top · geographic extent ${esc((data.bbox || []).map(value=>Number(value).toFixed(5)).join(', '))}</p><canvas id="spatial-change-map" width="${values[0].length}" height="${values.length}" tabindex="0" role="img" aria-label="Observed spatial spectral change on the selected area. Use arrow keys to inspect sampled pixels."></canvas><p id="spatial-pixel-value" class="form-help" aria-live="polite"></p><div class="spatial-grid-legend"><span style="--swatch:#a34f32">Decrease &lt; −0.2</span><span style="--swatch:#d2aa67">Decrease −0.2 to −0.05</span><span style="--swatch:#e7eadc">Change −0.05 to +0.05</span><span style="--swatch:#87a86c">Increase +0.05 to +0.2</span><span style="--swatch:#296947">Increase &gt; +0.2</span><span style="--swatch:#dce2d8">Missing / masked</span></div>`:''}<p class="form-help">${esc((data.limitations || []).join(' '))}</p><details class="model-hashes"><summary>Image dates, coverage and source</summary>${metadata}<p class="form-help">The reproducible export includes every grid value, with missing pixels kept separate.</p></details></section>`;
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
function climateQuality(data,row) {
  const first=row.month+'-01',year=Number(row.month.slice(0,4)),month=Number(row.month.slice(5,7)),last=new Date(Date.UTC(year,month,0)).toISOString().slice(0,10);
  const start=data.start&&data.start>first?data.start:first,end=data.end&&data.end<last?data.end:last,expected=Math.round((Date.parse(end+'T00:00:00Z')-Date.parse(start+'T00:00:00Z'))/86400000)+1;
  const counts=Object.values(row.quality || {}).map(value=>value.valid_days).filter(Number.isFinite);
  const complete=counts.filter(value=>value===expected).length,min=counts.length?Math.min(...counts):null;
  return {start,end,partial:start!==first||end!==last,summary:counts.length?`${min} / ${expected} requested days · ${complete===counts.length?'all fields':complete+' of '+counts.length+' fields complete'}`:'Quality not supplied'};
}
function climateBalanceMarkup(row) {
  const balance=row.water_balance;
  if(!balance)return 'Not computed in this saved record';
  const coverage=`${balance.paired_days} / ${balance.expected_days} paired days`,period=balance.complete_calendar_month?'complete month':balance.complete_requested_period?'partial month':'partial sum; missing days';
  return `${numeric(balance.sum_mm,2)}<small>${esc(coverage)} · ${esc(period)}</small>`;
}
function renderClimateContext(data) {
  if(!data?.available)return workflowUnavailable('Climate context',data,'Collect climate context to see dated rainfall and temperature measurements.');
  return `<section class="spatial-evidence"><h3>Climate context · ${esc(data.dataset || data.provider)}</h3><p class="form-help">Reanalysis at the selected area's centroid · ${esc(data.start || '')} → ${esc(data.end || '')}. Monthly summaries describe the collected climate evidence. Daily measurements can be prepared for a compatible trained model after date and coverage checks. Requested ranges shorter than a calendar month are labeled partial.</p><div class="table-scroll"><table><thead><tr><th>Month / requested range</th><th>Temperature (°C)</th><th>Rainfall (mm)</th><th>Reference ET₀ (mm)</th><th>Paired P − ET₀ (mm)</th><th>Surface soil moisture (m³/m³)</th><th>Data coverage</th></tr></thead><tbody>${(data.monthly || []).map(row=>{const quality=climateQuality(data,row);return `<tr><td>${esc(row.month)}${quality.partial?`<small>Partial: ${esc(quality.start)} → ${esc(quality.end)}</small>`:''}</td><td>${numeric(row.temperature_2m_mean,2)}</td><td>${numeric(row.precipitation_sum,2)}</td><td>${numeric(row.et0_fao_evapotranspiration,2)}</td><td>${climateBalanceMarkup(row)}</td><td>${numeric(row.soil_moisture_0_to_7cm_mean,3)}</td><td>${esc(quality.summary)}</td></tr>`;}).join('')}</tbody></table></div><p class="form-help">${esc(data.diagnostics?.water_balance?.definition || 'Paired P − ET₀ is a climatic water-balance diagnostic. It is not measured forest water use or a validated drought index.')} Partial sums are not comparable with complete-month totals; days lacking either input are excluded from the paired diagnostic.</p><p class="form-help">${esc((data.limitations || []).join(' '))} Per-variable and paired valid/missing day counts, dates, source references and units remain available in the downloaded report.</p></section>`;
}
function renderRecoveryEvidence(data) {
  if(!data?.available)return workflowUnavailable('Observed recovery',data,'Recovery needs 24 consecutive baseline months and at least 3 subsequent months.');
  return `<section class="spatial-evidence"><h3>Observed recovery episodes</h3><p class="form-help">${esc(data.reason || '')} Baseline ${esc(data.baseline_start || '')} → ${esc(data.baseline_end || '')}.</p>${data.episodes?.length?`<div class="table-scroll"><table><thead><tr><th>Episode starts</th><th>Status</th><th>Observed recovery date</th><th>Recovery duration (days)</th><th>Minimum anomaly</th></tr></thead><tbody>${data.episodes.map(row=>`<tr><td>${esc(row.start)}</td><td>${esc(workflowLabel(row.status))}</td><td>${esc(row.recovered_at || 'Not observed')}</td><td>${numeric(row.recovery_days)}</td><td>${numeric(row.minimum_anomaly,3)}</td></tr>`).join('')}</tbody></table></div>`:''}<p class="form-help">Observed recovery is a retrospective index diagnostic. Missing or unresolved episodes do not have an invented recovery date.</p></section>`;
}
function renderRadarContext(data) {
  if(!data?.available)return workflowUnavailable('Radar context',data,'Collect radar context to look for compatible dated observations.');
  const comparison=data.comparison || {};
  const legacyCaveat='This separately dated context is not inserted into optical model inputs or monthly observations. A paired trained multimodal model remains a separate requirement.';
  const inputCaveat='Individual radar scenes can be prepared for a compatible trained model after date, area and coverage checks. The pair comparison remains descriptive evidence, and monthly optical observations remain unchanged.';
  // Clarify only the old software caveat in the display; saved records and their hashes remain intact.
  const limitations=(data.limitations || []).map(value=>value===legacyCaveat?inputCaveat:value);
  return `<section class="spatial-evidence"><h3>Measured radar context · ${esc(data.dataset || 'Sentinel-1 RTC')}</h3><p class="form-help">${esc(data.provider || '')} · ${esc(comparison.start || data.start)} → ${esc(comparison.end || data.end)}. Changes use common valid pixels from matched satellite, orbit and pass observations. This pair comparison describes radar change; it does not identify a structural cause or supply a model prediction.</p><div class="monitor-status"><div><small>VV change · dB gamma0</small><strong>${numeric(comparison.vv_change_db,3)}</strong></div><div><small>VH change · dB gamma0</small><strong>${numeric(comparison.vh_change_db,3)}</strong></div><div><small>Common valid pixels</small><strong>${pct(comparison.common_valid_fraction)}</strong></div></div>${radarGridMarkup(data)}<div class="table-scroll"><table><thead><tr><th>Acquired (UTC)</th><th>Satellite</th><th>Relative orbit</th><th>Pass</th><th>Valid pixels</th></tr></thead><tbody>${(data.scenes || []).map(row=>`<tr><td>${esc(workflowDate(row.datetime))}</td><td>${esc(row.platform)}</td><td>${esc(row.relative_orbit)}</td><td>${esc(workflowLabel(row.orbit_state))}</td><td>${pct(row.valid_fraction)}</td></tr>`).join('')}</tbody></table></div><p class="form-help">${esc(limitations.join(' '))}</p></section>`;
}
function radarGridMarkup(data) {
  const values=data.comparison?.vv_change_map_db;if(!Array.isArray(values)||!values[0]?.length)return '';
  return `<div class="spatial-toolbar"><label>Measured radar change <select id="radar-polarization"><option value="vv">VV</option><option value="vh">VH</option></select></label><span>Latest minus first · dB · north at top</span></div><canvas id="radar-change-map" width="${values[0].length}" height="${values.length}" tabindex="0" role="img" aria-label="Measured radar backscatter change. Arrow keys inspect pixels; gray means missing or masked."></canvas><p id="radar-pixel-value" class="form-help" aria-live="polite"></p><div class="spatial-grid-legend"><span style="--swatch:#73538e">Decrease &lt; −3 dB</span><span style="--swatch:#b5a0c3">Decrease −3 to −1 dB</span><span style="--swatch:#e7eadc">Change −1 to +1 dB</span><span style="--swatch:#9abccc">Increase +1 to +3 dB</span><span style="--swatch:#376e8c">Increase &gt; +3 dB</span><span style="--swatch:#dce2d8">Missing / masked</span></div><p class="form-help">These display ranges are visual aids, not disturbance or ecological alert thresholds. Raster spacing and native radar resolving power are different quantities; the source record retains both.</p>`;
}
function drawRadarEvidence(data) {
  const canvas=$('#radar-change-map');if(!canvas)return;
  const context=canvas.getContext('2d'),pixel=[0,0],grid=()=>data.comparison[$('#radar-polarization').value+'_change_map_db'];
  const inspect=()=>{const value=grid()[pixel[1]][pixel[0]];$('#radar-pixel-value').textContent=`Pixel ${pixel[0]+1}, ${pixel[1]+1}: ${typeof value==='number'?value.toFixed(4)+' dB change':'missing or masked'}. Backscatter change does not identify disturbance or its cause.`;};
  const paint=()=>{const img=context.createImageData(canvas.width,canvas.height);grid().forEach((row,y)=>row.forEach((v,x)=>{const rgb=typeof v!=='number'?[220,226,216]:v< -3?[115,83,142]:v< -1?[181,160,195]:v>3?[55,110,140]:v>1?[154,188,204]:[231,234,220];img.data.set([...rgb,255],(y*canvas.width+x)*4);}));context.putImageData(img,0,0);inspect();};
  $('#radar-polarization').addEventListener('change',paint);
  canvas.addEventListener('click',event=>{const rect=canvas.getBoundingClientRect();pixel[0]=Math.max(0,Math.min(canvas.width-1,Math.floor((event.clientX-rect.left)/rect.width*canvas.width)));pixel[1]=Math.max(0,Math.min(canvas.height-1,Math.floor((event.clientY-rect.top)/rect.height*canvas.height)));inspect();});
  canvas.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;event.preventDefault();pixel[0]=Math.max(0,Math.min(canvas.width-1,pixel[0]+(event.key==='ArrowLeft'?-1:event.key==='ArrowRight'?1:0)));pixel[1]=Math.max(0,Math.min(canvas.height-1,pixel[1]+(event.key==='ArrowUp'?-1:event.key==='ArrowDown'?1:0)));inspect();});paint();
}
function renderFragmentation(data) {
  if(!data?.available)return workflowUnavailable('Forest patches and connectivity',data,'Collect a forest-cover reference or attach a classified forest-cover file to measure patches.');
  const shape=data.shape;
  const shapeSummary=shape?`<div><small>Mean compactness · complete boundaries</small><strong>${numeric(shape.mean_compactness,3)}</strong></div><div><small>Patch boundaries</small><strong>${numeric(shape.complete_patch_count)} complete · ${numeric(shape.censored_patch_count)} clipped or unknown</strong></div>`:'';
  const shapeDetails=shape?`<details class="workflow-section"><summary>Patch shape measurements · ${numeric(data.patch_count)} patches</summary><p class="form-help">${esc(shape.interpretation)} Compactness = 4π × area / perimeter². Patches touching the area boundary or unknown cells have no complete perimeter and no compactness value. Showing up to 20 patches; the reproducible export includes every patch.</p><div class="table-scroll"><table class="observations-table"><thead><tr><th>Patch</th><th>Area km²</th><th>Boundary</th><th>Perimeter m</th><th>Compactness</th></tr></thead><tbody>${(shape.patches||[]).slice(0,20).map(p=>`<tr><td>${numeric(p.patch_id)}</td><td>${numeric(p.area_km2,4)}</td><td>${p.boundary_complete?'Complete':'Clipped or unknown'}</td><td>${numeric(p.perimeter_m,1)}</td><td>${numeric(p.compactness,3)}</td></tr>`).join('')||'<tr><td colspan="5">No classified forest patches in the valid area.</td></tr>'}</tbody></table></div></details>`:'';
  return `<section class="spatial-evidence"><h3>Forest patches and connectivity</h3><p class="form-help">${esc(data.source)} · ${esc(data.observed_at)} · ${esc(data.classification)}. This classification is declared by its provider and remains unverified.</p><div class="monitor-status"><div><small>Forest cover inside valid area</small><strong>${pct(data.forest_fraction)} · ${numeric(data.forest_area_km2,3)} km²</strong></div><div><small>Connected patches · four neighbours</small><strong>${numeric(data.patch_count)}</strong></div><div><small>Internal edge density</small><strong>${numeric(data.edge_density_m_per_ha,2)} m/ha</strong></div><div><small>Largest patch / all forest</small><strong>${pct(data.largest_patch_fraction)}</strong></div><div><small>Mean nearest patch centroid</small><strong>${numeric(data.mean_nearest_centroid_m,1)} m</strong></div><div><small>Classification coverage</small><strong>${pct(data.valid_fraction)}</strong></div>${shapeSummary}</div>${shapeDetails}<p class="form-help">${esc((data.limitations || []).join(' '))}</p></section>`;
}
function coverImportForm() {
  return `<details class="workflow-section"><summary>Attach your own forest-cover reference</summary><p class="form-help">Already have a classified forest-cover map for this area? Choose a prepared ForestMind reference file to review its date, source and coverage before saving. You can also use Collect forest-cover reference above to request a public reference.</p><form id="cover-import-form"><label class="file-drop" for="cover-import-file"><strong>Choose forest-cover file</strong><span>ForestMind reference (.json) · maximum 2 MB</span><input type="file" id="cover-import-file" accept=".json,application/json" required aria-describedby="cover-file-help"></label><p id="cover-file-help" class="form-help">The map must describe this exact monitoring boundary, with a known observation date, forest definition and usage license. Its grid must distinguish forest, nonforest and missing measurements; up to 128 × 128 samples, with north at the top.</p><section id="cover-import-preview" class="workflow-import-preview" aria-label="Selected forest-cover file summary" aria-live="polite" hidden></section><button type="submit" class="button secondary" id="save-cover-reference" disabled>Save reference & measure patches</button><p id="cover-import-status" class="form-error" role="status" aria-live="polite"></p></form></details>`;
}
function coverFilePreview(payload,geometry) {
  if(!payload||typeof payload!=='object'||Array.isArray(payload))throw new Error('Choose a ForestMind forest-cover reference file.');
  const date=payload.observed_at;
  if(typeof date!=='string'||!/^\d{4}-\d{2}-\d{2}$/.test(date)||!Number.isFinite(Date.parse(date))||new Date(date).toISOString().slice(0,10)!==date)throw new Error('The reference needs a valid observation date. Ask the data provider to correct this file.');
  for(const [key,label,min] of [['source','source',3],['classification','forest definition',5],['license','usage license',2]])if(typeof payload[key]!=='string'||payload[key].trim().length<min)throw new Error('The reference needs its '+label+' before it can be saved.');
  if(payload.data_kind!=='classified_forest_cover')throw new Error('This file is not a classified forest-cover reference.');
  const forest=payload.forest,mask=payload.valid_mask;
  const validGrid=grid=>Array.isArray(grid)&&grid.length>0&&grid.length<=128&&Array.isArray(grid[0])&&grid[0].length>0&&grid[0].length<=128&&grid.every(row=>Array.isArray(row)&&row.length===grid[0].length&&row.every(value=>typeof value==='boolean'));
  if(!validGrid(forest)||!validGrid(mask)||forest.length!==mask.length||forest[0].length!==mask[0].length)throw new Error('The forest map and its coverage mask must share a valid grid, up to 128 × 128 samples. Ask the data provider to correct this file.');
  if(payload.geometry&&JSON.stringify(payload.geometry)!==JSON.stringify(geometry))throw new Error('This reference describes a different area. Choose a file prepared for the selected monitoring boundary.');
  const total=forest.length*forest[0].length;let valid=0,tree=0;
  mask.forEach((row,y)=>row.forEach((value,x)=>{if(value){valid++;if(forest[y][x])tree++;}}));
  if(!valid)throw new Error('This reference has no usable classified area. Choose a file with valid measurements.');
  return workflowFacts([['Observed',workflowDate(date)],['Source',payload.source],['Forest definition',payload.classification],['Usage license',payload.license],['Grid',`${forest[0].length} × ${forest.length} samples`],['Usable coverage',workflowPercent(valid/total)],['Forest within usable coverage',workflowPercent(tree/valid)]])+`<p class="form-help">The exact selected boundary is checked again when saving. This supplied classification has not been independently verified.</p>`;
}
function wireCoverImport(id) {
  const status=$('#cover-import-status'),form=$('#cover-import-form'),input=$('#cover-import-file'),preview=$('#cover-import-preview'),button=$('#save-cover-reference'),geometry=state.region.geometry;
  let selected=null,read=0;
  input.addEventListener('change',async event=>{
    const request=++read,file=event.target.files[0];selected=null;button.disabled=true;preview.hidden=true;preview.innerHTML='';status.textContent='';if(!file)return;
    try {
      if(file.size>2*1024*1024)throw new Error('Choose a forest-cover file smaller than 2 MB.');
      let payload;try{payload=JSON.parse(await file.text());}catch{throw new Error('This file could not be read as a forest-cover reference. Choose a valid JSON file from your data provider.');}
      const summary=coverFilePreview(payload,geometry);if(request!==read||state.selected!==id||state.tab!=='workflow')return;
      selected={...payload,geometry:payload.geometry || geometry};preview.innerHTML=summary;preview.hidden=false;button.disabled=false;
    }catch(error){if(request!==read)return;status.textContent=userMessage(error,'This file could not be read. Please choose it again.');}
  });
  form.addEventListener('submit',async event=>{
    event.preventDefault();button.disabled=true;input.disabled=true;
    try{
      if(!selected||state.selected!==id||state.tab!=='workflow')throw new Error('Choose a reference file for the current area and review its summary before saving.');
      const result=await api('/regions/'+encodeURIComponent(id)+'/forest-cover',{method:'POST',body:JSON.stringify(selected)});let job=result.job;
      status.textContent='Reference saved. Measuring forest patches for the selected area…';
      for(let count=0;count<120&&job?.id&&!['completed','failed'].includes(job.status);count++){await new Promise(resolve=>setTimeout(resolve,1000));job=await api('/workflow-jobs/'+encodeURIComponent(job.id));}
      if(job?.status==='failed')throw new Error(job.error || 'The reference was saved, but patch measurements could not be completed. Run selected-area analysis to retry.');
      if(job?.status!=='completed')throw new Error('Analysis is still running. Reopen Connected analysis shortly to inspect its result.');
      if(state.selected===id&&state.tab==='workflow')await renderWorkflow(id);toast('Forest-cover reference and patch measurements saved.');
    }catch(error){status.textContent=userMessage(error);button.disabled=!selected;}
    finally{input.disabled=false;}
  });
}
function renderFiveYearScenarios(data) {
  if(!data?.available)return workflowUnavailable('Five-year hypothetical sensitivity',data,'A usable one-year indicator baseline is required before comparing assumptions.');
  return `<section class="spatial-evidence"><h3>Five-year hypothetical sensitivity</h3><p class="form-help">${esc((data.limitations || []).join(' '))}</p><div class="spatial-toolbar"><label>Explicit assumptions <select id="five-year-scenario">${data.comparisons.map((item,index)=>`<option value="${index}">${esc(item.name)}</option>`).join('')}</select></label><a class="button secondary" href="#scenarios" id="adjust-five-year">Adjust assumptions ↗</a></div><div id="five-year-comparison"></div><details class="model-hashes"><summary>Coefficients and assumptions</summary><ul>${(data.assumptions || []).map(item=>`<li>${esc(item)}</li>`).join('')}</ul></details></section>`;
}
function wireFiveYearScenarios(report) {
  const selector=$('#five-year-scenario');if(!selector)return;
  const data=report.scenarios || report.sections?.scenarios;
  const draw=()=>{const item=data.comparisons[Number(selector.value)];$('#five-year-comparison').innerHTML=`<div class="monitor-status"><div><small>Baseline index at five years</small><strong>${numeric(data.baseline.at(-1).score,2)}</strong></div><div><small>Under these assumptions</small><strong>${numeric(item.points.at(-1).score,2)}</strong></div><div><small>Hypothetical difference</small><strong>${numeric(item.five_year_difference,2)} points</strong></div></div><div class="chart-legend"><span><i class="legend-line dashed"></i>No additional intervention</span><span><i class="legend-line" style="border-color:#b3935c"></i>${esc(item.name)}</span></div>${chart((report.forecast?.historical || []).slice(-12),data.baseline,{alternative:item.points,label:'Five-year hypothetical indicator sensitivity; not an ecological forecast'})}<p class="form-help">${esc(Object.entries(item.parameters).map(([key,value])=>workflowLabel(key)+': '+value+'%').join(' · '))}</p>`;};
  selector.addEventListener('change',draw);$('#adjust-five-year').addEventListener('click',()=>{$('#scenario-horizon').value='five-year';invalidateScenario();});draw();
}
async function ensureWorkflowSnapshot(id) {
  let report=await api('/regions/'+encodeURIComponent(id)+'/workflow');
  if(report.status!=='not_run'&&!report.stale)return report;
  let job=await api('/regions/'+encodeURIComponent(id)+'/workflow',{method:'POST',body:'{}'});
  for(let count=0;count<240&&!['completed','failed'].includes(job.status);count++){await new Promise(resolve=>setTimeout(resolve,1000));job=await api('/workflow-jobs/'+encodeURIComponent(job.id));}
  if(job.status!=='completed')throw new Error(job.error || 'Analysis is still running. Reopen Connected analysis before comparing assumptions.');
  return api('/regions/'+encodeURIComponent(id)+'/workflow');
}
function renderWorkflowManifest(report,region={}) {
  const manifest=report.manifest || report.context || {},points=manifest.geometry?.coordinates?.[0] || [];
  const extent=points.length?`West ${workflowNumber(Math.min(...points.map(point=>point[0])),5)}° · East ${workflowNumber(Math.max(...points.map(point=>point[0])),5)}° · South ${workflowNumber(Math.min(...points.map(point=>point[1])),5)}° · North ${workflowNumber(Math.max(...points.map(point=>point[1])),5)}°`:'Area boundary not supplied';
  const sourceNames=Object.entries(manifest.context_sha256 || {}).filter(([,value])=>value).map(([key])=>key==='forest_cover'?'Forest-cover reference':workflowLabel(key));
  return `<section class="workflow-manifest"><h3>${esc(region.name || 'Selected area')} · analysis record</h3>${workflowFacts([
    ['Latest observation',workflowDate(manifest.observed_through || report.observed_through || region.latest_date)],
    ['Analysis cutoff',workflowDate(manifest.observation_cutoff)],['Dated observations',manifest.observation_count ?? 'Not recorded'],
    ['Saved',report.created_at?workflowDate(report.created_at):'Analysis has not run yet'],['Evidence type',manifest.synthetic===true?'Synthetic demonstration':manifest.synthetic===false?'Real observations; scientific validity remains unverified':'Not declared'],
    ['Additional evidence',sourceNames.length?sourceNames.join(' · '):'None recorded'],['Model used',manifest.model?.name || (manifest.model?'Registered model; see Model runs for details':'No model selected')]
  ])}<p class="form-help">${report.stale?'The inputs, methods or model configuration have changed. Run selected-area analysis to refresh these results.':'Results apply to the dates and area shown here.'}</p><details><summary>Area and record history</summary><p>${esc(extent)}</p><p class="form-help">${manifest.software_version?'Analysis software version '+esc(manifest.software_version)+'. ':''}The downloadable record retains the exact boundary, source identifiers and file fingerprints needed to trace this analysis. These records do not establish ecological accuracy.</p></details></section>`;
}
function renderObservationEvidence(rows,coverage) {
  if(!Array.isArray(rows))return '';
  if(!rows.length)return workflowUnavailable('Observation history',null,'Collect dated satellite observations to begin.');
  const counts=coverage?.counts,shown=rows.slice(-36);
  const measures=['ndvi','ndmi','nbr'];
  return `<section class="spatial-evidence"><h3>Observation history and coverage</h3>${workflowFacts([
    ['Observed dates',`${workflowDate(rows[0].date)} → ${workflowDate(rows.at(-1).date)}`],['Recorded observations',rows.length],
    ['Usable months',counts?.usable ?? 'Not assessed'],['Missing months',counts?.missing ?? 'Not assessed'],
    ['Months with low coverage',counts?.low_coverage ?? 'Not assessed'],['Months missing measurements',counts?.missing_indices ?? 'Not assessed'],
    ['Consecutive usable months through the ending month',coverage?.trailing_usable_run ?? 'Not assessed']
  ])}<p class="form-help">Vegetation (NDVI), moisture (NDMI) and burn-sensitive (NBR) measurements are spectral indices. Missing measurements remain unavailable. ${rows.length>shown.length?'Showing the latest 36 observations; the export includes the full history.':''}</p><details><summary>Review dated measurements</summary>${workflowTable(['Observation date','Vegetation index','Moisture index','Burn-sensitive index','Usable image pixels'],shown.map(row=>[workflowDate(row.date),...measures.map(key=>workflowNumber(row[key],3)),workflowPercent(row.valid_fraction)]))}</details>${workflowList(coverage?.reasons || [])}</section>`;
}
function renderConditionEvidence(data) {
  if(!data)return '';
  return `<section class="spatial-evidence"><h3>Observed condition indicator</h3>${workflowFacts([
    ['Condition index',workflowNumber(data.score,2,' / 100')],['Screening category',workflowLabel(data.status)],
    ['Recent change from baseline',workflowNumber(data.change,2,' index points')],['Recent annual trend',workflowNumber(data.trend_per_year,2,' index points per year')],['Recent image coverage',workflowPercent(data.coverage)]
  ])}<p class="form-help">${esc(data.interpretation || 'This spectral screening index is not a field-validated forest-health diagnosis.')}</p>${data.components?.length?workflowTable(['Measurement','Latest index'],data.components.map(item=>[item.name,workflowNumber(item.value,3)])):''}</section>`;
}
function renderDriverEvidence(rows) {
  if(!Array.isArray(rows))return '';
  return `<section class="spatial-evidence"><h3>Possible drivers and competing explanations</h3><p class="form-help">Evidence scores compare screening signals. They are not probabilities, causal contributions or proof of a disturbance.</p><div class="workflow-summary-grid">${rows.map(row=>`<article class="workflow-summary-card"><h4>${esc(row.name || workflowLabel(row.id))}</h4><strong>${Number.isFinite(row.evidence_score)?workflowNumber(row.evidence_score,2,' / 100 evidence score'):'Evidence unavailable'}</strong><p>${esc(workflowReason(row.evidence || 'No supporting evidence supplied.'))}</p><p class="form-help">${esc(workflowLabel(row.direction))} · confidence has not been calibrated.</p></article>`).join('') || '<p>No driver assessment is saved.</p>'}</div></section>`;
}
function renderIndicatorForecast(data) {
  if(!data)return '';
  if(!data.points?.length)return workflowUnavailable('Illustrative indicator forecast',data,'At least 12 consecutive usable months are required.');
  return `<section class="spatial-evidence"><h3>Illustrative indicator forecast</h3><p class="form-help">${esc(data.caveat || 'Historical trend extrapolation; no ecological event prediction or calibrated confidence interval.')}</p>${workflowTable(['Date','Horizon','Indicator / 100','Lower envelope','Upper envelope'],data.points.map(row=>[row.date,workflowNumber(row.horizon_days,0,' days'),workflowNumber(row.score),workflowNumber(row.lower),workflowNumber(row.upper)]))}<p class="form-help">Lower and upper values are heuristic residual envelopes. Disturbances and interventions are not predicted.</p></section>`;
}
function renderResilienceEvidence(data) {
  if(!data)return '';
  return `<section class="spatial-evidence"><h3>Resilience research diagnostics</h3>${workflowFacts([['Assessment',workflowLabel(data.status)],['Lag-one correlation',workflowNumber(data.ar1,3)],['Residual variance',workflowNumber(data.variance,3)],['Change across comparison windows',data.trend || 'Not available']])}${workflowList(data.evidence || [])}<p class="form-help">${esc(data.caveat || 'Descriptive diagnostics do not establish ecological stability or a tipping point.')} No validated tipping probability or transition date is available.</p></section>`;
}
function renderWorkflowEvaluation(data) {
  if(!data)return '';
  const rows=[];
  for(const [method,result] of Object.entries(data.methods || {}))for(const [horizon,metrics] of Object.entries(result.by_horizon || {all:result.aggregate || {}}))rows.push([
    workflowLabel(method),horizon==='all'?'All horizons':`${horizon} month${horizon==='1'?'':'s'}`,metrics.n ?? 0,workflowNumber(metrics.mae),workflowNumber(metrics.rmse),workflowPercent(metrics.coverage),
    metrics.skipped_missing_target ?? 0,metrics.skipped_unusable_target ?? 0,metrics.outside_observation_period ?? 0
  ]);
  const scored=Object.values(data.methods || {}).some(result=>Number.isFinite(result.aggregate?.n)&&result.aggregate.n>0);
  return `<section class="spatial-evidence"><h3>Historical baseline comparison</h3><p class="form-help">${scored?'Predictions were compared with later spectral observations, using only earlier observations at each cutoff.':'No predictions could be scored against a usable later observation in this record.'} These results do not validate forest health, canopy loss or a trained model.</p>${data.synthetic_warning?`<p class="collection-errors">${esc(data.synthetic_warning)}</p>`:''}${rows.length?workflowTable(['Baseline','Horizon','Scored cases','Mean absolute error','Root mean squared error','Envelope coverage','Missing targets','Unusable targets','Beyond observed period'],rows):''}<p class="form-help">Errors use indicator points on a 0–100 scale. Envelope coverage is empirical and uncalibrated. Missing targets remain skipped, and an unavailable metric is not a zero error.</p>${workflowList(data.limitations || [])}</section>`;
}
function renderWorkflowModel(data) {
  if(!data)return '';
  if(!data.available)return workflowUnavailable('Trained model result',{reason:workflowReason(data.reason)},'No compatible trained model result is available.');
  const report=data.report || {},prediction=report.predictions || {},health=prediction.health || {};
  return `<section class="spatial-evidence"><h3>Trained model result</h3><p class="form-help">${esc(workflowReason(data.reason || 'A compatible registered model produced a saved research result.'))}</p>${workflowFacts([
    ['Model',report.checkpoint?.name || 'Name not supplied'],['Observed through',workflowDate(report.observed_through)],['Result saved',workflowDate(report.created_at)],
    ['Condition class',health.state?workflowLabel(health.state):'No trained output'],['Condition score',Number.isFinite(health.score)?workflowNumber(health.score,2,' / 100'):'No trained output']
  ])}${prediction.drivers?.length?workflowTable(['Candidate driver','Uncalibrated model score'],prediction.drivers.map(row=>[row.name,Number.isFinite(row.score)?workflowNumber(row.score,3):'No trained output'])):''}${prediction.trajectory?.length?workflowTable(['Horizon','Model score / 100','Model spread, score points'],prediction.trajectory.map(row=>[workflowNumber(row.horizon_days,0,' days'),Number.isFinite(row.score)?workflowNumber(row.score):'No trained output',Number.isFinite(row.stddev)?workflowNumber(row.stddev):'Not estimated'])):''}${workflowList(report.caveats || [])}<p class="form-help">Open Model runs to review the complete saved prediction and its training and input provenance.</p></section>`;
}
function renderWorkflowAlerts(alerts) {
  if(!Array.isArray(alerts))return '';
  return `<section class="spatial-evidence"><h3>Evidence alerts</h3>${alerts.length?alerts.map(alert=>`<article class="workflow-summary-card"><h4>${esc(alert.title)}</h4><p>${esc(alert.message || 'Review the dated source observations.')}</p>${workflowFacts([['Observation date',workflowDate(alert.date)],['Priority',workflowLabel(alert.severity)],['Review status',workflowLabel(alert.status || 'open')]])}</article>`).join(''):'<p class="form-help">No screening alerts are recorded for this assessment. This does not establish ecological stability.</p>'}<p class="form-help">Use Observations to review supporting measurements and acknowledge or resolve an alert.</p></section>`;
}
function renderWorkflowSections(report) {
  const sections=report.sections || report.results || report;
  return renderObservationEvidence(sections.observations,sections.coverage)+renderConditionEvidence(sections.condition)+renderAlignedInputs(report)+renderSpatialEvidence(sections.spatial_change || report.spatial_change)+renderClimateContext(sections.climate)+renderRadarContext(sections.radar)+renderDriverEvidence(sections.drivers)+renderIndicatorForecast(sections.forecast)+renderResilienceEvidence(sections.resilience)+renderRecoveryEvidence(sections.recovery)+renderFragmentation(sections.fragmentation)+renderWorkflowModel(sections.model)+renderFiveYearScenarios(sections.scenarios)+renderWorkflowAlerts(sections.alerts)+renderWorkflowEvaluation(sections.evaluation)+coverImportForm();
}
async function renderWorkflow(id) {
  const request=++workflowRequest;
  $('#detail-body').innerHTML='<p class="inline-status" role="status">Loading the selected area’s connected analysis record…</p>';
  try{
    const result=await api('/regions/'+encodeURIComponent(id)+'/workflow');
    if(request!==workflowRequest||state.selected!==id||state.tab!=='workflow')return;
    const report=result.report || result,manifest=report.manifest || report.context || {},capabilities=report.capabilities || [],list=Array.isArray(capabilities)?capabilities:Object.entries(capabilities).map(([id,value])=>({id,...value}));
    $('#detail-body').innerHTML=`<div class="workflow-intro"><div><div class="eyebrow">ONE AREA · ONE EVIDENCE RECORD</div><h3>Connected analysis</h3><p class="form-help">Run the available computations for this area. Missing data and unsupported model scope remain explicit. Observation indicators, model predictions and hypothetical scenarios retain their own interpretation.</p></div>${badge(isDemo(state.region)?'Synthetic demonstration':'Real-data research')}</div><div class="workflow-actions"><button class="button primary" id="run-workflow">Run selected-area analysis</button><button class="button secondary" id="collect-climate">Collect climate context</button><button class="button secondary" id="collect-landcover">Collect forest-cover reference</button><button class="button secondary" id="collect-radar">Collect radar context</button><button class="button secondary" id="export-workflow">Export reproducible record</button></div><div id="workflow-status" class="inline-status" role="status" aria-live="polite"></div>${collectionErrors(report)}${renderWorkflowManifest(report,state.region)}${list.length?workflowCapabilities(list):'<p class="form-help">No persisted analysis record yet. Run selected-area analysis to evaluate the available evidence.</p>'}${renderWorkflowSections(report)}`;
    drawSpatialEvidence(report.spatial_change || report.sections?.spatial_change);
    drawRadarEvidence(report.radar || report.sections?.radar);
    wireCoverImport(id);
    wireFiveYearScenarios(report);
    $('#run-workflow').addEventListener('click',()=>runWorkflow(id,false));
    $('#collect-climate').addEventListener('click',()=>runWorkflow(id,true));
    $('#collect-landcover').addEventListener('click',()=>runWorkflow(id,false,true));
    $('#collect-radar').addEventListener('click',()=>runWorkflow(id,false,false,true));
    $('#export-workflow').addEventListener('click',async()=>{try{downloadJSON(await api('/regions/'+encodeURIComponent(id)+'/workflow/export'+(manifest.id?'?manifest_id='+encodeURIComponent(manifest.id):'')),'forestmind-'+id+'-connected-analysis.json');}catch(error){toast(userMessage(error));}});
  }catch(error){if(request===workflowRequest&&state.selected===id&&state.tab==='workflow')$('#detail-body').innerHTML=`<p class="observations-empty">${esc(userMessage(error))} Select Connected analysis again to retry.</p>`;}
}
function workflowProgress(phase) {
  const phases={queued:'Waiting to start…',running:'Analyzing the selected area…',starting:'Preparing the selected area…',manifest:'Checking the selected area and observation record…',climate:'Collecting climate measurements…',radar:'Collecting radar observations…',forest_cover:'Collecting a forest-cover reference…',sequence:'Preparing dated model inputs…',model:'Checking and running an eligible trained model…',analysis:'Computing the available evidence…',saving:'Saving the analysis record…',completed:'Analysis complete.',cancelled:'Analysis was cancelled.'};
  return phases[phase] || 'Analyzing the selected area…';
}
async function runWorkflow(id,includeClimate,includeLandcover=false,includeRadar=false) {
  const status=$('#workflow-status');$('#run-workflow').disabled=true;$('#collect-climate').disabled=true;$('#collect-landcover').disabled=true;$('#collect-radar').disabled=true;status.textContent=includeRadar?'Collecting compatible radar observations for this area…':includeLandcover?'Collecting a dated ESA WorldCover tree-cover reference for this exact area…':includeClimate?'Collecting climate context and running available analyses…':'Preparing compatible inputs and running eligible analyses…';
  try{
    let job=await api('/regions/'+encodeURIComponent(id)+'/workflow',{method:'POST',body:JSON.stringify({include_climate:includeClimate,include_landcover:includeLandcover,...(includeRadar?{include_radar:true}:{})})});
    for(let count=0;count<240&&job.id&&!['completed','failed','cancelled'].includes(job.status);count++){
      await new Promise(resolve=>setTimeout(resolve,1500));job=await api('/workflow-jobs/'+encodeURIComponent(job.id));
      if(state.selected===id&&state.tab==='workflow'&&$('#workflow-status'))$('#workflow-status').textContent=`${workflowProgress(job.phase || job.status)}`;
    }
    if(job.status==='failed')throw new Error(job.error || 'The analysis job failed. Its diagnostics have been saved.');
    if(job.id&&!['completed','cancelled'].includes(job.status))throw new Error('Analysis is still running. Reopen Connected analysis shortly to inspect the saved result.');
    if(state.selected===id&&state.tab==='workflow')await renderWorkflow(id);
    toast(job.status==='cancelled'?'Analysis was cancelled. Earlier saved results remain available.':'Selected-area analysis saved with its evidence and limitations.');
  }catch(error){if(state.selected===id&&state.tab==='workflow'&&$('#workflow-status')){$('#workflow-status').textContent=userMessage(error);$('#run-workflow').disabled=false;$('#collect-climate').disabled=false;$('#collect-landcover').disabled=false;$('#collect-radar').disabled=false;}else toast(userMessage(error));}
}
function renderAlertEvidence(alert) {
  const thresholds=alert.thresholds || {},rows=alert.supporting_observations || [];
  return workflowFacts([['Source',workflowSource(alert.source)],['Last reviewed',alert.reviewed_at?workflowDate(alert.reviewed_at):'Not reviewed yet'],['Review note',alert.note || 'No note recorded']])+`<h4>Screening thresholds used</h4>`+workflowFacts([
    ['Condition drop',workflowNumber(thresholds.condition_drop_points,1,' index points')],['Moisture-index drop',workflowNumber(thresholds.moisture_drop,3)],['Minimum usable pixels',workflowPercent(thresholds.min_valid_fraction)],['Observation considered old after',workflowNumber(thresholds.stale_days,0,' days')]
  ])+(rows.length?workflowTable(['Observation date','Vegetation index','Moisture index','Burn-sensitive index','Usable pixels','Condition / 100'],rows.map(row=>[workflowDate(row.date),workflowNumber(row.ndvi,3),workflowNumber(row.ndmi,3),workflowNumber(row.nbr,3),workflowPercent(row.valid_fraction),workflowNumber(row.health_score,1)])):'<p class="form-help">No supporting measurements were retained with this older alert. Review the observation record above.</p>');
}
async function renderAlertLifecycle(region) {
  const target=$('#alert-lifecycle');if(!target)return;
  try{
    const settings=await api('/regions/'+encodeURIComponent(region.id)+'/alert-settings');
    if(state.selected!==region.id||state.tab!=='observations'||!$('#alert-lifecycle'))return;
    const config=settings.settings || settings;
    $('#alert-lifecycle').innerHTML=`<section class="alert-management"><h3>Alert review and thresholds</h3><p class="form-help">Review dated supporting observations above. Acknowledgement records a review; resolution records your decision. Collection failures are shown separately in Collection jobs.</p>${(region.alerts || []).map(alert=>`<div class="alert-management-row"><strong>${esc(alert.title)}</strong>${badge(alert.status || 'open')}<button class="button secondary" data-alert-id="${esc(alert.id)}" data-alert-state="acknowledged">Acknowledge</button><button class="button secondary" data-alert-id="${esc(alert.id)}" data-alert-state="resolved">Resolve</button>${alert.status==='resolved'||alert.status==='acknowledged'?`<button class="button secondary" data-alert-id="${esc(alert.id)}" data-alert-state="open">Reopen</button>`:''}<details class="alert-support"><summary>Dated supporting evidence & quality</summary>${renderAlertEvidence(alert)}</details></div>`).join('')}<label class="field-label" for="alert-review-note">Note for the next review action (optional)</label><input id="alert-review-note" class="alert-review-note" maxlength="2000" placeholder="Reason for acknowledgement, resolution or reopening"><form id="alert-settings-form" class="alert-settings-form"><label>Condition drop (index points)<input name="condition_drop_points" type="number" min="0.1" max="100" step="0.1" value="${Number(config.condition_drop_points ?? 5)}" required></label><label>Moisture drop (NDMI units)<input name="moisture_drop" type="number" min="0.01" max="2" step="0.01" value="${Number(config.moisture_drop ?? .12)}" required></label><label>Minimum valid pixel fraction<input name="min_valid_fraction" type="number" min="0" max="1" step="0.01" value="${Number(config.min_valid_fraction ?? .4)}" required></label><label>Stale observation age (days)<input name="stale_days" type="number" min="1" max="1095" step="1" value="${Number(config.stale_days ?? 90)}" required></label><button type="submit" class="button secondary">Save thresholds & refresh assessment</button><p class="form-help">Thresholds are screening choices, not validated ecological standards. Review actions do not themselves send notifications; optional delivery requires authorization in Data & methods.</p></form><p id="alert-settings-error" class="form-error" role="alert"></p></section>`;
    $$('[data-alert-id]').forEach(button=>button.addEventListener('click',async()=>{button.disabled=true;try{await api('/alerts/'+encodeURIComponent(button.dataset.alertId),{method:'PATCH',body:JSON.stringify({status:button.dataset.alertState,note:$('#alert-review-note').value})});await selectRegion(region.id);toast('Alert review saved.');}catch(error){$('#alert-settings-error').textContent=userMessage(error);button.disabled=false;}}));
    $('#alert-settings-form').addEventListener('submit',async event=>{event.preventDefault();const button=$('button',event.target);button.disabled=true;try{const values=Object.fromEntries([...new FormData(event.target)].map(([key,value])=>[key,Number(value)]));await api('/regions/'+encodeURIComponent(region.id)+'/alert-settings',{method:'PUT',body:JSON.stringify(values)});await selectRegion(region.id);toast('Alert thresholds saved.');}catch(error){$('#alert-settings-error').textContent=userMessage(error);button.disabled=false;}});
  }catch(error){if(state.selected===region.id&&state.tab==='observations'&&$('#alert-lifecycle'))$('#alert-lifecycle').innerHTML=`<p class="form-error">Alert controls: ${esc(userMessage(error))}</p>`;}
}
