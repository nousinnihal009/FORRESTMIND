/* Observed history and explicit gaps; no interpolation or ecological claims. */
'use strict';
let coverageRequest = 0;
const coveragePeriods = new Map();
const coverageLabels = {usable:'Usable', low_coverage:'Low coverage', missing_indices:'Missing indices', missing:'No observation'};

function coverageCalendar(report) {
  const years = new Map();
  for (const month of report.months) {
    const year = month.month.slice(0,4);
    if (!years.has(year)) years.set(year,[]);
    years.get(year).push(month);
  }
  return [...years].map(([year, months])=>`<section class="coverage-year"><h4>${esc(year)}</h4><div class="coverage-months">${months.map(month=>{
    const label = coverageLabels[month.state] || 'Unavailable';
    const detail = `${month.month}: ${label}${month.date ? ` · Observed ${dateLabel(month.date)} · ${pct(month.valid_fraction)} valid pixels` : ''}`;
    return `<div class="coverage-month ${Object.hasOwn(coverageLabels,month.state) ? month.state : 'missing'}" title="${esc(detail)}" aria-label="${esc(detail)}"><span>${esc(dateLabel(month.month+'-01',true).split(' ')[0])}</span><strong>${month.date ? pct(month.valid_fraction) : '—'}</strong><small>${esc(label)}</small></div>`;
  }).join('')}</div></section>`).join('');
}

function displayCoverage(report, id) {
  const counts = report.counts;
  const forecast = report.baseline_forecast_ready_at_requested_end;
  const resilience = report.resilience_ready_at_requested_end;
  $('#coverage-results').innerHTML = `<div class="coverage-summary"><div><strong>${numeric(counts.usable)}<small> / ${numeric(counts.expected_months)}</small></strong><span>Usable months</span></div><div><strong>${numeric(counts.missing)}</strong><span>Missing months</span></div><div><strong>${numeric(report.longest_usable_run)}</strong><span>Longest usable run</span></div><div><strong>${numeric(report.trailing_usable_run)}</strong><span>Usable run at period end</span></div></div><div class="coverage-readiness"><p><strong>Baseline forecast</strong><span>${forecast ? 'Minimum history met' : 'More usable history needed'}</span></p><p><strong>Resilience diagnostics</strong><span>${resilience ? 'Minimum history met' : 'More usable history needed'}</span></p></div><p class="chart-caveat">Readiness is checked at the end of the selected period. Forecasts need at least 12 consecutive usable monthly observations; resilience diagnostics need at least 36 and a regular record. Meeting these software requirements does not establish predictive accuracy.</p><div class="coverage-legend">${Object.entries(coverageLabels).map(([key,label])=>`<span><i class="coverage-dot ${key}"></i>${label}</span>`).join('')}</div>${coverageCalendar(report)}<div class="model-limitations"><strong>What this record supports</strong><ul>${report.reasons.map(reason=>`<li>${esc(reason)}</li>`).join('')}</ul></div><div class="detail-foot"><span>${report.synthetic ? 'Synthetic demonstration' : 'User-provided observations · unverified'} · ${esc(report.source)}</span><span>${numeric(counts.excluded_observations)} records outside the selected period</span></div>`;
  const button = $('#export-coverage');
  button.disabled = false;
  button.onclick = ()=>downloadJSON(report, 'forestmind-'+id+'-coverage.json');
}

async function loadCoverage(id, start, end) {
  const request = ++coverageRequest;
  $('#coverage-results').innerHTML = '<p class="observations-empty" role="status">Checking the observation history…</p>';
  $('#coverage-error').hidden = true;
  $('#export-coverage').disabled = true;
  const query = new URLSearchParams({start,end});
  try {
    const report = await api('/regions/'+encodeURIComponent(id)+'/coverage?'+query);
    if(request !== coverageRequest || state.selected !== id || state.tab !== 'coverage')return;
    displayCoverage(report,id);
  }catch(error){
    if(request !== coverageRequest || state.selected !== id || state.tab !== 'coverage')return;
    $('#coverage-results').replaceChildren();
    $('#coverage-error').textContent = userMessage(error);
    $('#coverage-error').hidden = false;
  }
}

function renderCoverage(id) {
  const rows = state.region.observations;
  const period = coveragePeriods.get(id) || {start:rows[0].date, end:rows[rows.length-1].date};
  $('#detail-body').innerHTML = `<div class="observations-caption"><div><strong>Every month accounted for</strong><p class="chart-caveat">Inspect gaps and quality before interpreting a trend.</p></div><button class="button secondary" id="export-coverage" disabled>${icon('download',14)}Export coverage</button></div><form class="coverage-form" id="coverage-form"><label for="coverage-start">Period start<input id="coverage-start" type="date" required value="${esc(period.start)}"></label><label for="coverage-end">Period end<input id="coverage-end" type="date" required value="${esc(period.end)}"></label><button class="button secondary" type="submit">Check period</button></form><p class="coverage-error" id="coverage-error" role="alert" hidden></p><div id="coverage-results" aria-live="polite"></div>`;
  $('#coverage-form').addEventListener('submit',event=>{
    event.preventDefault();
    const start = $('#coverage-start').value, end = $('#coverage-end').value;
    if(end < start){$('#coverage-error').textContent='Period end must be on or after period start.';$('#coverage-error').hidden=false;return;}
    coveragePeriods.set(id,{start,end});
    loadCoverage(id,start,end);
  });
  // Do not let old results appear to describe newly edited dates.
  for (const input of $$('#coverage-form input')) input.addEventListener('input',()=>{
    ++coverageRequest;
    $('#coverage-results').innerHTML='<p class="observations-empty">Select Check period to assess these dates.</p>';
    $('#export-coverage').disabled=true;
    $('#coverage-error').hidden=true;
  });
  loadCoverage(id,period.start,period.end);
}
