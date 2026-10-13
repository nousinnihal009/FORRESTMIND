/* Saved neural inference reports, kept distinct from observation baselines. */
'use strict';
let modelImportRegion = null;
let modelRequest = 0;

function modelRunHeader(id) {
  return `<div class="observations-caption"><div><strong>Saved neural predictions</strong><p class="chart-caveat">Checkpoint-specific results, with their input record and observation cutoff.</p></div><button class="button secondary" id="open-model-import">${icon('upload',14)}Import model run</button></div><p class="chart-caveat">Region ID: <code>${esc(id)}</code> · Use this ID when preparing an inference report.</p>`;
}

function modelProvenance(report) {
  const checkpoint = report.checkpoint, input = report.input;
  return `<dl class="model-provenance"><div><dt>Checkpoint</dt><dd>${esc(checkpoint.name)}<small>${esc(checkpoint.backbone)} · ${checkpoint.trained_on_synthetic ? 'Synthetic training data' : 'Training provenance declared by report'}</small></dd></div><div><dt>Input sequence</dt><dd>${esc(input.name)}<small>${input.synthetic ? 'Synthetic input' : 'Declared observed input'} · ${numeric(input.timesteps)} timesteps</small></dd></div><div><dt>Observed through</dt><dd>${esc(dateLabel(report.observed_through))}<small>Report created ${esc(dateLabel(report.created_at))}</small></dd></div><div><dt>Provenance status</dt><dd>User-provided · unverified<small>Hashes identify artifacts; they do not verify ecological accuracy.</small></dd></div></dl><details class="model-hashes"><summary>Artifact fingerprints and training source</summary><p>Checkpoint SHA-256 <code>${esc(checkpoint.sha256)}</code></p><p>Input SHA-256 <code>${esc(input.sha256)}</code></p><p>Training source: ${esc(checkpoint.training_source)}</p></details>`;
}

function modelScoreRows(scores, unit = 'score') {
  return scores.map(item => `<div class="model-score-row"><span>${esc(item.name)}</span><div class="tiny-track"><i style="width:${Math.max(0,Math.min(100,item.score * 100))}%"></i></div><strong>${numeric(item.score,3)}</strong><span class="sr-only">${esc(unit)}, uncalibrated</span></div>`).join('');
}

function showModelRun(run) {
  const report = run.report, prediction = report.predictions;
  const synthetic = report.input.synthetic || report.checkpoint.trained_on_synthetic;
  $('#model-run-detail').innerHTML = `<div class="model-run-heading"><div>${badge(synthetic ? 'Synthetic research run' : 'Unvalidated research run')}<h3>${esc(prediction.health.state)}</h3><p>Model class assignment · uncalibrated</p></div><div class="model-score"><strong>${numeric(prediction.health.score,1)}</strong><span>/ 100 model score</span></div><button class="button secondary" id="export-model-run">${icon('download',14)}Export report</button></div><div class="evidence-disclaimer">${synthetic ? 'This run uses synthetic inputs or a model trained on synthetic data. It cannot establish real forest condition. ' : ''}Class and driver scores are uncalibrated. Forecast spread is a model output, not a validated confidence interval. These predictions do not establish a tipping point.</div>${modelProvenance(report)}<div class="model-output-grid"><section><h3>Condition class scores</h3><p class="chart-caveat">Mutually exclusive model classes · scores sum to one</p>${modelScoreRows(prediction.health.class_scores)}</section><section><h3>Candidate driver scores</h3><p class="chart-caveat">Multiple drivers can score highly · no causal claim</p>${modelScoreRows(prediction.drivers)}</section></div><h3 class="model-forecast-title">Configured forecast horizons</h3><div class="table-scroll"><table><thead><tr><th>Horizon from cutoff</th><th>Model score / 100</th><th>Standard deviation (score points)</th></tr></thead><tbody>${prediction.trajectory.map(point=>`<tr><td>${numeric(point.horizon_days)} days</td><td>${numeric(point.score,2)}</td><td>${numeric(point.stddev,2)}</td></tr>`).join('')}</tbody></table></div><div class="model-limitations"><strong>Limitations saved with the run</strong><ul>${report.caveats.map(item=>`<li>${esc(item)}</li>`).join('')}</ul></div>`;
  $('#export-model-run').addEventListener('click',()=>downloadJSON(report,'forestmind-'+report.region_id+'-'+run.id+'.json'));
}

async function renderModelRuns(id, preferredRun) {
  const request = ++modelRequest;
  $('#detail-body').innerHTML = '<div class="chart-skeleton skeleton" aria-label="Loading model runs"></div>';
  try {
    const result = await api('/regions/'+encodeURIComponent(id)+'/model-runs');
    if(request !== modelRequest || state.selected !== id || state.tab !== 'models')return;
    const runs = result.runs || [];
    $('#detail-body').innerHTML = modelRunHeader(id) + (runs.length ? `<div class="model-picker"><label class="field-label" for="model-run-select">Saved run</label><select id="model-run-select">${runs.map(run=>`<option value="${esc(run.id)}">${esc(run.report.checkpoint.name)} · ${esc(dateLabel(run.report.observed_through))} · ${esc(run.id)}</option>`).join('')}</select></div><div id="model-run-detail"></div>` : '<div class="empty-state model-empty"><span class="large-symbol">↗</span><h3>Connect your trained model</h3><p>Run the prediction command on a prepared satellite sequence, then import its JSON report here. Every saved run retains its model and input fingerprints.</p><p class="chart-caveat">No trained neural model is used by the observation baseline. Model runs appear here once you import them.</p></div>');
    $('#open-model-import').addEventListener('click',()=>openModelImport(id));
    if(runs.length){
      const picker = $('#model-run-select');
      if(runs.some(run=>run.id===preferredRun))picker.value=preferredRun;
      const render = ()=>showModelRun(runs.find(run=>run.id===picker.value));
      picker.addEventListener('change',render);render();
    }
  } catch(error) {
    if(request === modelRequest && state.selected === id && state.tab === 'models'){
      $('#detail-body').innerHTML = `<p class="observations-empty">${esc(error.message)} Try selecting Model runs again.</p>`;
    }
  }
}

function openModelImport(id) {
  modelImportRegion = id;
  $('#model-import-region').textContent = state.region?.name || id;
  $('#model-import-error').hidden = true;
  $('#model-import-dialog').showModal();
}

$('#close-model-import').addEventListener('click',()=>$('#model-import-dialog').close());
$('#model-import-json').addEventListener('input',()=>{$('#model-import-error').hidden=true;});
$('#model-import-file').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file)return;
  try{
    if(file.size>2*1024*1024)throw new Error('Choose a JSON report smaller than 2 MB.');
    $('#model-import-json').value=await file.text();$('#model-file-name').textContent=file.name;
    $('#model-import-error').hidden=true;
  }catch(error){$('#model-import-error').textContent=error.message;$('#model-import-error').hidden=false;}
});
$('#model-import-form').addEventListener('submit',async event=>{
  event.preventDefault();const button=$('#submit-model-import');button.disabled=true;
  const id=modelImportRegion;$('#model-import-error').hidden=true;
  try{
    const raw=$('#model-import-json').value;
    if(new TextEncoder().encode(raw).length>2*1024*1024)throw new Error('Keep the JSON report below 2 MB.');
    let report;try{report=JSON.parse(raw);}catch{throw new Error('This is not valid JSON. Paste a report generated by the prediction command.');}
    if(report.region_id!==id)throw new Error('The report region_id must match the selected region: '+id);
    const result=await api('/regions/'+encodeURIComponent(id)+'/model-runs',{method:'POST',body:JSON.stringify(report)});
    $('#model-import-dialog').close();
    if(state.selected===id&&state.tab==='models')await renderModelRuns(id,result.id);
    toast('Model run saved with unverified provenance.');
  }catch(error){$('#model-import-error').textContent=error.message;$('#model-import-error').hidden=false;}
  finally{button.disabled=false;}
});
