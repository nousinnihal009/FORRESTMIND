/* Saved neural inference reports, kept distinct from observation baselines. */
'use strict';
let modelImportRegion = null;
let modelRequest = 0;
let modelImportReport = null;
let modelImportRead = 0;

function modelRunHeader(id) {
  return `<div class="observations-caption"><div><strong>Saved model predictions</strong><p class="chart-caveat">Results for this monitoring region, with their model, observation dates, and evidence source.</p></div><button class="button secondary" id="open-model-import">${icon('upload',14)}Import model run</button></div><p class="chart-caveat">Connected analysis can run a compatible installed model when sufficient measurements are available. Imported reports retain their declared observation dates.</p>`;
}

function modelProvenance(report, run) {
  const checkpoint = report.checkpoint, input = report.input;
  const backbone={custom_vit:'ForestMind image model',prithvi_proxy:'Prithvi-style research model',terramind_proxy:'TerraMind-style research model',prithvi:'Prithvi',terramind:'TerraMind'}[checkpoint.backbone] || humanLabel(checkpoint.backbone);
  const training=checkpoint.trained_on_synthetic===true?'Synthetic training data':checkpoint.trained_on_synthetic===false?'Training source declared by the report':'Training data origin unknown';
  return `<dl class="model-provenance"><div><dt>Model</dt><dd>${esc(checkpoint.name)}<small>${esc(backbone)} · ${esc(training)}</small></dd></div><div><dt>Input measurements</dt><dd>${input.synthetic ? 'Synthetic input' : 'Declared observed input'}<small>${numeric(input.timesteps)} prepared time steps${Number.isInteger(input.observed_timesteps)?' · '+numeric(input.observed_timesteps)+' with observations':''}</small></dd></div><div><dt>Observed through</dt><dd>${esc(dateLabel(report.observed_through))}<small>Report created ${esc(dateLabel(report.created_at))}</small></dd></div><div><dt>How this result was added</dt><dd>${run?.verification==='local_registered_execution'?'Run locally with a registered model':'User-provided · unverified'}<small>Neither saving a report nor running a model verifies its ecological accuracy.</small></dd></div></dl><details class="model-hashes"><summary>Training source and reproducibility</summary><p>Training source: ${esc(checkpoint.training_source==='unknown'?'Not supplied':checkpoint.training_source==='synthetic'?'Synthetic demonstration data':checkpoint.training_source)}</p><p class="form-help">The exported report includes the original model and input file names, their fingerprints, and all saved limitations so the result can be traced.</p></details>`;
}

function modelScoreRows(scores, unit = 'score') {
  if(!scores.length)return '<p class="observations-empty">Unavailable · this checkpoint has no trained output for these classes.</p>';
  return scores.map(item => Number.isFinite(item.score)?`<div class="model-score-row"><span>${esc(item.name)}</span><div class="tiny-track"><i style="width:${Math.max(0,Math.min(100,item.score * 100))}%"></i></div><strong>${numeric(item.score,3)}</strong><span class="sr-only">${esc(unit)}, uncalibrated</span></div>`:`<div class="model-score-row model-output-missing"><span>${esc(item.name)}<small>No trained output</small></span><strong>Unavailable</strong></div>`).join('');
}

function modelTrajectoryRows(points) {
  return points.map(point=>`<tr><td>${numeric(point.horizon_days)} days</td><td>${Number.isFinite(point.score)?numeric(point.score,2):'Unavailable · no trained output'}</td><td>${Number.isFinite(point.stddev)?numeric(point.stddev,2):Number.isFinite(point.score)?'Not estimated':'Unavailable'}</td></tr>`).join('');
}

function showModelRun(run) {
  const report = run.report, prediction = report.predictions;
  const synthetic = report.input.synthetic || report.checkpoint.trained_on_synthetic;
  $('#model-run-detail').innerHTML = `<div class="model-run-heading"><div>${badge(synthetic ? 'Synthetic research run' : 'Unvalidated research run')}<h3>${prediction.health.state?'Top-scoring class: '+esc(prediction.health.state):'Condition class unavailable'}</h3><p>${prediction.health.state?'Model class assignment · uncalibrated':'This checkpoint has no trained condition-class output.'}</p></div><div class="model-score"><strong>${Number.isFinite(prediction.health.score)?numeric(prediction.health.score,1):'Unavailable'}</strong><span>${Number.isFinite(prediction.health.score)?'/ 100 model score':'No trained condition-score output'}</span></div><button class="button secondary" id="export-model-run">${icon('download',14)}Export report</button></div><div class="evidence-disclaimer">${synthetic ? 'This run uses synthetic inputs or a model trained on synthetic data. It cannot establish real forest condition. ' : ''}Class and driver scores are uncalibrated. Forecast spread is a model output, not a validated confidence interval. These predictions do not establish a tipping point.</div>${modelProvenance(report,run)}<div class="model-output-grid"><section><h3>Condition class scores</h3><p class="chart-caveat">Available class scores sum to one · missing supervision is shown as unavailable</p>${modelScoreRows(prediction.health.class_scores)}</section><section><h3>Candidate driver scores</h3><p class="chart-caveat">Multiple drivers can score highly · no causal claim</p>${modelScoreRows(prediction.drivers)}</section></div><h3 class="model-forecast-title">Configured forecast horizons</h3><div class="table-scroll"><table><thead><tr><th>Horizon from cutoff</th><th>Model score / 100</th><th>Standard deviation (score points)</th></tr></thead><tbody>${modelTrajectoryRows(prediction.trajectory)}</tbody></table></div><div class="model-limitations"><strong>Limitations saved with the run</strong><ul>${report.caveats.map(item=>`<li>${esc(item)}</li>`).join('')}</ul></div>`;
  $('#export-model-run').addEventListener('click',()=>downloadJSON(report,'forestmind-'+report.region_id+'-'+run.id+'.json'));
}

async function renderModelRuns(id, preferredRun) {
  const request = ++modelRequest;
  $('#detail-body').innerHTML = '<div class="chart-skeleton skeleton" aria-label="Loading model runs"></div>';
  try {
    const result = await api('/regions/'+encodeURIComponent(id)+'/model-runs');
    if(request !== modelRequest || state.selected !== id || state.tab !== 'models')return;
    const runs = result.runs || [];
    $('#detail-body').innerHTML = modelRunHeader(id) + (runs.length ? `<div class="model-picker"><label class="field-label" for="model-run-select">Saved run</label><select id="model-run-select">${runs.map((run,index)=>`<option value="${esc(run.id)}">Run ${index+1} · ${esc(run.report.checkpoint.name)} · observed through ${esc(dateLabel(run.report.observed_through))}</option>`).join('')}</select></div><div id="model-run-detail"></div>` : '<div class="empty-state model-empty"><span class="large-symbol">↗</span><h3>No saved model prediction yet</h3><p>Open Connected analysis to check whether this area has enough measurements for a compatible installed model. Missing measurements or a missing trained model will be explained there.</p><p class="chart-caveat">The observation baseline remains available without a trained model. Saved automatic and imported model runs appear here.</p></div>');
    $('#open-model-import').addEventListener('click',()=>openModelImport(id));
    if(runs.length){
      const picker = $('#model-run-select');
      if(runs.some(run=>run.id===preferredRun))picker.value=preferredRun;
      const render = ()=>showModelRun(runs.find(run=>run.id===picker.value));
      picker.addEventListener('change',render);render();
    }
  } catch(error) {
    if(request === modelRequest && state.selected === id && state.tab === 'models'){
      $('#detail-body').innerHTML = `<p class="observations-empty">${esc(userMessage(error))} Try selecting Model runs again.</p>`;
    }
  }
}

function openModelImport(id) {
  ++modelImportRead;
  modelImportReport = null;
  modelImportRegion = id;
  $('#model-import-form').reset();
  $('#model-import-region').textContent = state.region?.name || 'the selected monitoring region';
  $('#model-import-error').hidden = true;
  $('#model-import-preview').hidden = true;
  $('#model-file-name').textContent = 'ForestMind model report (.json) · maximum 2 MB';
  $('#submit-model-import').disabled = true;
  $('#model-import-dialog').showModal();
}

function modelFilePreview(report,regionId) {
  if(!report||typeof report!=='object'||Array.isArray(report)||report.kind!=='forestmind_inference'||report.schema_version!==1)throw new Error('Choose a ForestMind model report. This file uses a different format.');
  if(report.region_id!==regionId)throw new Error('This report belongs to a different monitoring region. Select that region before importing its report.');
  const {checkpoint,input,predictions}=report;
  if(!checkpoint||typeof checkpoint.name!=='string'||!checkpoint.name.trim()||!input||typeof input.synthetic!=='boolean'||!Number.isInteger(input.timesteps)||input.timesteps<1||!predictions?.health||!Array.isArray(predictions.health.class_scores)||!Array.isArray(predictions.drivers)||!Array.isArray(predictions.trajectory))throw new Error('This report is missing its model, input measurements, or prediction results. Choose a complete exported report.');
  const validDate=date=>typeof date==='string'&&/^\d{4}-\d{2}-\d{2}$/.test(date)&&Number.isFinite(Date.parse(date))&&new Date(date).toISOString().slice(0,10)===date;
  if(!validDate(report.observed_through)||(input.observed_start!==undefined&&input.observed_start!==null&&(!validDate(input.observed_start)||input.observed_start>report.observed_through)))throw new Error('This report has an invalid observation period. Ask the report provider to correct it.');
  const targets=[];
  if(Number.isFinite(predictions.health.score))targets.push('Condition score');
  if(predictions.health.class_scores.some(item=>Number.isFinite(item?.score)))targets.push('Condition classes');
  const drivers=predictions.drivers.filter(item=>Number.isFinite(item?.score)).length,forecasts=predictions.trajectory.filter(item=>Number.isFinite(item?.score)).length;
  if(drivers)targets.push(drivers+' candidate driver '+(drivers===1?'score':'scores'));
  if(forecasts)targets.push(forecasts+' forecast '+(forecasts===1?'horizon':'horizons'));
  const synthetic=input.synthetic||checkpoint.trained_on_synthetic===true;
  return {name:checkpoint.name,period:(input.observed_start?dateLabel(input.observed_start)+' – ':'Through ')+dateLabel(report.observed_through),targets:targets.length?targets.join(' · '):'No trained outputs in this report',source:synthetic?'Synthetic research data · not evidence of real forest condition':checkpoint.trained_on_synthetic===false?'Declared observed inputs · training source unverified':'Declared observed inputs · training data origin unknown'};
}

async function loadModelFile(event) {
  const request=++modelImportRead,regionId=modelImportRegion,file=event.target.files[0];
  modelImportReport=null;$('#submit-model-import').disabled=true;$('#model-import-preview').hidden=true;$('#model-import-error').hidden=true;
  $('#model-file-name').textContent=file?.name || 'ForestMind model report (.json) · maximum 2 MB';
  if(!file)return;
  try{
    if(file.size>2*1024*1024)throw new Error('Choose a model report smaller than 2 MB.');
    let report;try{report=JSON.parse(await file.text());}catch{throw new Error('This file could not be read as a model report. Choose a valid JSON report exported by ForestMind.');}
    const preview=modelFilePreview(report,regionId);
    if(request!==modelImportRead||regionId!==modelImportRegion)return;
    modelImportReport=report;
    for(const [field,value] of Object.entries(preview))$('#model-preview-'+field).textContent=value;
    $('#model-import-preview').hidden=false;$('#submit-model-import').disabled=false;
  }catch(error){if(request!==modelImportRead||regionId!==modelImportRegion)return;$('#model-import-error').textContent=userMessage(error,'This file could not be read. Please choose it again.');$('#model-import-error').hidden=false;}
}

async function importModelRun(event) {
  event.preventDefault();const button=$('#submit-model-import');button.disabled=true;
  const id=modelImportRegion,request=modelImportRead;$('#model-import-error').hidden=true;
  try{
    const report=modelImportReport;
    if(!report)throw new Error('Choose a model report and review its summary before saving.');
    if(state.selected!==id||report.region_id!==id)throw new Error('The selected monitoring region changed. Close this dialog and open the import again for the intended region.');
    const result=await api('/regions/'+encodeURIComponent(id)+'/model-runs',{method:'POST',body:JSON.stringify(report)});
    if(request===modelImportRead)$('#model-import-dialog').close();
    if(state.selected===id&&state.tab==='models')await renderModelRuns(id,result.id);
    toast('Model run saved. Its predictions and source remain unverified.');
  }catch(error){if(request===modelImportRead){$('#model-import-error').textContent=userMessage(error);$('#model-import-error').hidden=false;}}
  finally{if(request===modelImportRead)button.disabled=!modelImportReport;}
}

$('#close-model-import').addEventListener('click',()=>$('#model-import-dialog').close());
$('#model-import-dialog').addEventListener('close',()=>{++modelImportRead;modelImportReport=null;$('#submit-model-import').disabled=true;});
$('#model-import-file').addEventListener('change',loadModelFile);
$('#model-import-form').addEventListener('submit',importModelRun);
