/* ForestMind local research workspace. No client-side model or fabricated API fallback. */
'use strict';
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const state = {overview: null, region: null, selected: null, tab: 'trajectory', horizon: 365, zoom: 1, request: 0, scenario: null};
const paths = {
  grid: '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
  map: '<path d="m3 5 6-2 6 2 6-2v16l-6 2-6-2-6 2V5Zm6-2v16m6-14v16"/>',
  branch: '<path d="M6 6v12m0-6h7a5 5 0 0 0 5-5"/><circle cx="6" cy="4" r="2"/><circle cx="6" cy="20" r="2"/><circle cx="18" cy="4" r="2"/>',
  layers: '<path d="m12 3 10 5-10 5L2 8l10-5Zm-10 9 10 5 10-5M2 16l10 5 10-5"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v.1"/>',
  scan: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M3 12h18"/>',
  refresh: '<path d="M20 8a8 8 0 0 0-14-3L3 8m0-5v5h5m-4 8a8 8 0 0 0 14 3l3-3m0 5v-5h-5"/>',
  download: '<path d="M12 3v12m-4-4 4 4 4-4M4 16v5h16v-5"/>',
  upload: '<path d="M12 16V3m-4 4 4-4 4 4M4 16v5h16v-5"/>',
  search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
  leaf: '<path d="M20 3C7 2 1 9 5 16c5 7 16 0 15-13ZM4 21 16 8"/>',
  alert: '<path d="m12 3 10 18H2L12 3Zm0 6v5m0 3v.1"/>',
  trend: '<path d="m3 16 6-6 4 4 8-9m-6 0h6v6"/>',
  globe: '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18"/>'
};
function icon(name, size = 18) {return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.leaf}</svg>`;}
function hydrateIcons(root = document) {$$('[data-icon]', root).forEach(el => {el.innerHTML = icon(el.dataset.icon);});}
function esc(value) {return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function numeric(value, digits = 0) {return typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('en-US', {maximumFractionDigits: digits, minimumFractionDigits: digits}) : '—';}
function dateLabel(value, short = false) {if (!value) return 'Unavailable'; const d = new Date(value.slice(0, 10) + 'T00:00:00Z'); return Number.isNaN(+d) ? value : new Intl.DateTimeFormat('en', {month: 'short', ...(short ? {} : {day:'numeric'}), year:'numeric', timeZone:'UTC'}).format(d);}
function isDemo(region) {return Boolean(region.provenance?.synthetic || String(region.provenance?.mode || region.provenance?.data_mode || region.provenance?.source || '').match(/synthetic|demo/i));}
function isAcquired(region) {return region.provenance?.mode === 'satellite_acquisition';}
function observationSource(region) {return isDemo(region)?'Synthetic observations':isAcquired(region)?'Satellite observations':'User-provided observations';}
function sourceSummary(region) {const source=region.provenance?.source || '';return isDemo(region)?'Synthetic example observations':source.startsWith('Cached sentinel2 surface reflectance')?'Sentinel-2 satellite surface measurements':source || 'Source recorded with observations';}
function methodLabel(value) {return value==='transparent-indicator-baseline-v1'?'Vegetation condition baseline':humanLabel(value || 'Vegetation condition baseline');}
function statusClass(value) {const s = String(value).toLowerCase(); return /severe|critical|degrad|high/.test(s) ? 'critical' : /stress|watch|warning|elevat/.test(s) ? 'warning' : /unknown|insuff|unavail/.test(s) ? 'neutral' : '';}
function badge(value) {return `<span class="status ${statusClass(value)}">${esc(String(value || 'Unavailable').replaceAll('_', ' '))}</span>`;}
function pct(value) {return typeof value === 'number' && Number.isFinite(value) ? numeric(value * 100) + '%' : '—';}
function humanLabel(value) {
  const names={ndvi:'Vegetation index (NDVI)',ndmi:'Moisture index (NDMI)',nbr:'Burn-sensitive index (NBR)',valid_fraction:'Usable pixels',geometry:'Area boundary',observed_at:'Observation date',observed_through:'Last observation',region_id:'Monitoring region',pressure_reduction:'Human pressure reduction',fire_reduction:'Fire pressure reduction',climate_stress:'Additional climate stress',restoration:'Restoration intensity',uncalibrated:'Accuracy not calibrated',not_available:'Unavailable',local_registered_execution:'Run locally',imported_unverified:'Imported · unverified',damped_trend:'Damped trend baseline',seasonal_naive:'Same month in the previous year',persistence:'Last observed value'};
  const text=String(value ?? '').trim();
  return names[text] || text.replace(/([a-z])([A-Z])/g,'$1 $2').replace(/[_-]+/g,' ').replace(/^./,c=>c.toUpperCase()) || 'Not provided';
}
function readableValue(value) {
  if(value===null||value===undefined)return 'Not available';
  if(typeof value==='boolean')return value?'Yes':'No';
  if(typeof value==='number')return numeric(value,3);
  if(typeof value==='string')return value;
  if(Array.isArray(value))return value.length?value.map(readableValue).join(' · '):'None recorded';
  return value.description || value.reason || value.label || value.name || 'Full details are included in the downloaded report.';
}
function userMessage(error, fallback='This action could not be completed. Please try again.') {
  const message=typeof error==='string'?error:error?.message;
  if(!message)return fallback;
  if(/failed to fetch|networkerror|network request failed|load failed/i.test(message))return 'The local workspace could not be reached. Check that ForestMind is running, then try again.';
  if(/FORESTMIND_MODEL_REGISTRY/.test(message))return 'No compatible trained model is registered for this area.';
  if(/traceback|\bat \S+ \(.+:\d+:\d+\)|[A-Za-z]:[\\/]|\b(?:TypeError|ReferenceError|SyntaxError|KeyError|ValueError|RuntimeError|HTTPError|HTTPSConnectionPool)\b|\bFORESTMIND_[A-Z_]+\b|<!doctype|<html|\[object Object\]/i.test(message)||/^\s*[\[{]/.test(message))return fallback;
  return message;
}
function validationMessage(detail) {
  return detail.slice(0,5).map(item=>{
    const parts=(item.loc || []).filter(p=>!['body','query','path'].includes(p));
    const label=parts.map(part=>typeof part==='number'?'record '+(part+1):humanLabel(part)).join(' · ');
    let message=String(item.msg || 'Check this value.').replace(/^Value error, /,'');
    if(item.type==='missing')message='This information is required.';
    else if(item.type==='extra_forbidden')message='This field is not supported by this import format.';
    else message=userMessage(message,'Check this value and try again.');
    return (label?label+': ':'')+message;
  }).join('\n')+(detail.length>5?'\nMore fields need attention. Review the file against the import guide.':'');
}
function toast(message) {const node = $('#toast'); node.textContent = message; node.hidden = false; clearTimeout(toast.timer); toast.timer = setTimeout(() => {node.hidden = true;}, 5000);}
function reportError(error) {const node = $('#error-banner'); node.replaceChildren(document.createTextNode(userMessage(error))); const retry = document.createElement('button'); retry.textContent = 'Refresh workspace'; retry.addEventListener('click', loadWorkspace); node.append(retry); node.hidden = false;}
async function api(path, options = {}) {
  let response;
  try {response = await fetch('/api/v1' + path, {headers: {'Content-Type': 'application/json'}, ...options});}
  catch(error){throw new Error(userMessage(error,'The local workspace could not be reached. Please try again.'));}
  if (!response.ok) {
    let body; try {body = await response.json();} catch {body = {};}
    const detail = body.detail;
    const fallback=response.status>=500?'ForestMind could not complete this request. Please retry; if it continues, the local service needs attention.':response.status===404?'This saved item is no longer available. Refresh the workspace.':response.status===413?'This file is too large. Choose a file smaller than the displayed limit.':'This request could not be accepted. Check the supplied information and try again.';
    const message = response.status>=500?fallback:Array.isArray(detail) ? validationMessage(detail) : userMessage(detail,fallback);
    throw new Error(message);
  }
  return response.json();
}
function route() {
  let name = location.hash.slice(1) || 'overview';
  if (!['overview','explore','monitoring','scenarios','methods'].includes(name)) name = 'overview';
  $$('.view').forEach(el => {el.hidden = el.id !== 'view-' + name;});
  $$('[data-route]').forEach(el => {el.classList.toggle('active', el.dataset.route === name); if (el.dataset.route === name) el.setAttribute('aria-current','page'); else el.removeAttribute('aria-current');});
  $('#page-name').textContent = {overview:'Overview',explore:'Find a forest',monitoring:'Monitoring regions',scenarios:'Scenario lab',methods:'Data & methods'}[name];
  if (name === 'monitoring') renderTable();
  if (name === 'methods' && typeof renderOperations === 'function') renderOperations();
  if (name === 'explore' && typeof renderAreaMap === 'function') {renderAreaMap();refreshAcquisitionJobs();}
}
async function loadWorkspace(selectId) {
  $('#error-banner').hidden = true;
  try {
    state.overview = await api('/overview');
    const regions = state.overview.regions || [];
    $('#region-count').textContent = regions.length;
    const demoCount = regions.filter(isDemo).length;
    $('#mode-badge').innerHTML = `<span class="live-dot"></span>${!regions.length ? 'Empty workspace' : demoCount === regions.length ? 'Demo workspace' : demoCount ? 'Mixed data sources' : regions.every(isAcquired)?'Satellite observations':'Observed records'}`;
    $('#demo-notice').innerHTML = `${icon('info',16)}<span><strong>A research view, with transparent evidence.</strong> ${demoCount ? `${demoCount} sample regions use synthetic observations.` : 'Satellite and imported records retain their source and processing provenance.'} Indicators are exploratory, not validated ecological predictions.</span><a href="#methods">How it works ↗</a>`;
    renderMetrics(); renderPriorities(); renderTable(); renderProvenance();
    const previous = $('#scenario-region').value;
    $('#scenario-region').innerHTML = regions.map(r => `<option value="${esc(r.id)}">${esc(r.name)}</option>`).join('');
    if (regions.some(r => r.id === previous)) $('#scenario-region').value = previous;
    const id = typeof selectId === 'string' ? selectId : state.selected || regions[0]?.id;
    if (id) await selectRegion(id);
    else {$('#detail-title').textContent = 'No regions yet'; $('#detail-body').innerHTML = '<p class="observations-empty"><a class="text-link" href="#explore">Find a forest and collect real observations</a>, or import an existing record to start monitoring.</p>'; renderMap();}
  } catch (error) {reportError(error);}
}
function renderMetrics() {
  const m = state.overview.metrics;
  const regions = state.overview.regions;
    const entries = [
    ['Monitored landscape', numeric(m.monitored_area_km2, 0), 'km²', 'globe', `${regions.length} region footprints · WGS84`],
    ['Mean condition index', numeric(m.mean_health, 1), '/ 100', 'leaf', 'Vegetation proxy · not a probability'],
    ['Active alerts', numeric(m.active_alerts), 'to investigate', 'alert', 'Evidence that merits a closer look'],
    ['Monitoring regions', numeric(m.regions ?? regions.length), 'regions', 'layers', `${regions.filter(isDemo).length} synthetic · ${regions.filter(r => !isDemo(r)).length} observed records`]
  ];
  $('#metrics').innerHTML = entries.map(([label,value,unit,i,foot]) => `<div class="metric"><div class="metric-label">${label}${icon(i,16)}</div><div class="metric-value">${value}<small>${unit}</small></div><div class="metric-foot">${esc(foot)}</div></div>`).join('');
}
function renderPriorities() {
  if (!state.overview) return;
  const regions = [...state.overview.regions].sort((a,b) => (b.alert_count || 0) - (a.alert_count || 0) || (a.health_score ?? 100) - (b.health_score ?? 100)).slice(0,3);
  $('#investigations').innerHTML = regions.map((r,i) => `<button class="priority-card ${r.id === state.selected ? 'selected' : ''}" data-region="${esc(r.id)}"><div class="priority-card-top"><h3>${esc(r.name)}</h3>${badge(r.status)}</div><p>${r.alert_count ? `${r.alert_count} signals flagged for review. Inspect the observed changes and competing explanations.` : 'No active alert. Review condition and observation coverage.'}</p><div class="priority-card-bottom"><span>${isDemo(r) ? 'SYNTHETIC OBSERVATIONS' : 'OBSERVED RECORD'}</span><span>Investigate ↗</span></div></button>`).join('') || '<p class="no-results">No monitored regions.</p>';
  $$('[data-region]', $('#investigations')).forEach(el => el.addEventListener('click', () => selectRegion(el.dataset.region)));
}
function renderMap() {
  if (!state.overview) return;
  const regions = state.overview.regions;
  const points = regions.flatMap(r => r.geometry?.coordinates?.[0] || [r.center]).filter(p => p && p.length === 2);
  if (!points.length) {$('#region-map').innerHTML = ''; return;}
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  let west = Math.min(...xs), east = Math.max(...xs), south = Math.min(...ys), north = Math.max(...ys);
  let dx = Math.max(east-west,2)*1.55, dy = Math.max(north-south,2)*1.8;
  if (dx/dy < 840/410) dx=dy*840/410; else dy=dx*410/840;
  let cx=(west+east)/2,cy=(south+north)/2;
  if (state.zoom > 1 && state.region?.center) {cx=state.region.center[0];cy=state.region.center[1];}
  dx/=state.zoom;dy/=state.zoom; west=cx-dx/2;north=cy+dy/2;
  const project = p => [(p[0]-west)/dx*840,(north-p[1])/dy*410];
  const path = coords => coords.map((p,i) => `${i?'L':'M'}${project(p).map(n=>n.toFixed(1)).join(',')}`).join(' ') + ' Z';
  const layer=$('#map-layer').value;
  const color = r => layer==='alerts' ? (r.alert_count>1?'#ddb27b':r.alert_count?'#c5cd84':'#8abb8d') : layer==='coverage' ? ((r.coverage ?? r.data_coverage ?? .7) < .65 ? '#ddb27b' : '#9bbc82') : (r.health_score === null ? '#9da89c' : r.health_score<40?'#d9a57b':r.health_score<58?'#d6bd7b':r.health_score<75?'#ced088':'#a3c58c');
  let grid='';
  const step = dx > 70 ? 20 : dx > 30 ? 10 : dx > 15 ? 5 : 2;
  for(let x=Math.ceil(west/step)*step;x<west+dx;x+=step){const p=project([x,cy])[0];grid+=`<path d="M${p},0V410"/><text x="${p+5}" y="397" stroke="none" fill="#78917b" font-size="8">${Math.abs(x)}°${x<0?'W':'E'}</text>`;}
  for(let y=Math.ceil((north-dy)/step)*step;y<north;y+=step){const p=project([cx,y])[1];grid+=`<path d="M0,${p}H840"/><text x="7" y="${p-6}" stroke="none" fill="#78917b" font-size="8">${Math.abs(y)}°${y<0?'S':'N'}</text>`;}
  // Deliberately coarse geographic context. This is a schematic, never a satellite raster.
  const basin=[[-77,0],[-72,3],[-65,4],[-58,3],[-52,0],[-49,-3],[-51,-7],[-55,-11],[-63,-13],[-69,-11],[-74,-7],[-77,0]];
  const river=[[-75,-4.5],[-72,-4],[-69,-3.2],[-66,-3.7],[-63,-3.2],[-60,-3.1],[-57,-2.3],[-54,-2.1],[-51,-1.2],[-49,-.2]];
  const riverPath=river.map((p,i)=>`${i?'L':'M'}${project(p).join(',')}`).join(' ');
  const basinOnMap=cx>-80&&cx<-45&&cy>-20&&cy<10;
  const shapes=regions.map(r=>{const c=project(r.center);const coords=r.geometry?.coordinates?.[0] || [];const selected=r.id===state.selected;const w=Math.min(220,Math.max(100,r.name.length*6+20));return `<g class="map-region ${selected?'selected':''}" tabindex="0" role="button" aria-label="Inspect ${esc(r.name)}" data-region="${esc(r.id)}"><title>${esc(r.name)} · ${esc(r.status)} · Condition ${numeric(r.health_score,1)}</title><path class="region-shape" d="${path(coords)}" fill="${color(r)}" fill-opacity="${selected ? 0.4 : 0.2}" stroke="${color(r)}" stroke-width="${selected?2.5:1.5}" stroke-dasharray="${selected?'0':'4 3'}"/><circle cx="${c[0]}" cy="${c[1]}" r="4" fill="${color(r)}" stroke="#1c4434" stroke-width="2"/><rect x="${c[0]-w/2}" y="${c[1]+13}" width="${w}" height="25" rx="4" fill="${selected?'#dce8c5':'#224737'}" fill-opacity=".95" stroke="${selected?'#c3d99d':'#6e88624d'}"/><text x="${c[0]}" y="${c[1]+29}" text-anchor="middle" font-size="10" font-weight="500" fill="${selected?'#355138':'#cbdab9'}">${esc(r.name)}${selected?' ↗':''}</text></g>`;}).join('');
  $('#region-map').innerHTML=`<defs><radialGradient id="mapGlow"><stop stop-color="#31533c"/><stop offset="1" stop-color="#143c31"/></radialGradient><pattern id="mapDots" width="18" height="18" patternUnits="userSpaceOnUse"><circle cx="3" cy="3" r=".7" fill="#799164" opacity=".1"/></pattern></defs><rect width="840" height="410" fill="url(#mapGlow)"/><rect width="840" height="410" fill="url(#mapDots)"/><g stroke="#7d9678" stroke-opacity=".12" stroke-width=".7">${grid}</g>${basinOnMap?`<path d="${path(basin)}" fill="#6e8c4c" fill-opacity=".075" stroke="#86a167" stroke-opacity=".18" stroke-width="1"/><path d="${riverPath}" fill="none" stroke="#789e8b" stroke-opacity=".28" stroke-width="3" stroke-linejoin="round"/><text x="${project([-61,-.5])[0]}" y="${project([-61,-.5])[1]}" fill="#8aa07f" opacity=".55" font-size="9" letter-spacing="4" text-anchor="middle">AMAZON BASIN</text>`:''}${shapes}`;
  $$('.map-region').forEach(el=>{const select=()=>selectRegion(el.dataset.region);el.addEventListener('click',select);el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});});
  $('#map-legend').innerHTML=layer==='condition'?'<span><i class="small-dot" style="background:#a3c58c"></i>Higher condition</span><span><i class="small-dot" style="background:#ced088"></i>Stressed</span><span><i class="small-dot" style="background:#d9a57b"></i>Lower condition</span>':layer==='alerts'?'<span><i class="small-dot" style="background:#ddb27b"></i>Multiple alerts</span><span><i class="small-dot" style="background:#8abb8d"></i>No alerts</span>':'<span><i class="small-dot" style="background:#9bbc82"></i>Higher coverage</span><span><i class="small-dot" style="background:#ddb27b"></i>Partial coverage</span>';
  $('#map-date').textContent='Latest observation · '+dateLabel(state.overview.as_of,true);
}
async function selectRegion(id) {
  const request=++state.request;
  state.selected=id;
  if ($('#scenario-region').value !== id) {$('#scenario-region').value=id;invalidateScenario();}
  $('#detail-title').textContent=state.overview?.regions.find(r=>r.id===id)?.name || 'Loading region…';
  $('#run-analysis').disabled=true;$('#export-region').disabled=true;
  $('#detail-body').innerHTML='<div class="chart-skeleton skeleton" aria-label="Loading analysis"></div>';
  renderPriorities();
  try {
    const region=await api('/regions/'+encodeURIComponent(id));
    if(request!==state.request)return;
    state.region=region;
    $('#detail-title').textContent=region.name;
    $('#detail-meta').textContent=`${region.country || 'Unspecified'} · ${numeric(region.area_km2)} km² · ${observationSource(region)} · Updated ${dateLabel(region.latest_date)}`;
    if($('#scenario-region').value!==id)invalidateScenario();
    $('#scenario-region').value=id;
    renderMap();renderDetail();
  }catch(error){if(request===state.request){state.region=null;$('#detail-body').innerHTML='<p class="observations-empty">Could not load this region. Retry the connection above.</p>';reportError(error);}}
  finally{if(request===state.request){$('#run-analysis').disabled=!state.region;$('#export-region').disabled=!state.region;}}
}
function chart(history, forecast, options={}) {
  const past=(history||[]).map(p=>({...p,stamp:Date.parse(p.date)}));
  const future=(forecast||[]).map(p=>({...p,stamp:Date.parse(p.date)}));
  const alternative=(options.alternative||[]).map(p=>({...p,stamp:Date.parse(p.date)}));
  const all=[...past,...future,...alternative].filter(p=>Number.isFinite(p.stamp));
  if(!all.some(p=>typeof p.score==='number'))return '<div class="observations-empty">Not enough valid observations to draw a trajectory.</div>';
  const width=760,height=245,left=34,right=16,top=12,bottom=30;
  let min=Math.min(...all.map(p=>p.stamp)),max=Math.max(...all.map(p=>p.stamp));if(min===max){min-=86400000*30;max+=86400000*30;}
  const x=p=>left+(p.stamp-min)/(max-min)*(width-left-right),y=score=>top+(100-score)/100*(height-top-bottom);
  let axes='';for(const value of [0,25,50,75,100]) axes+=`<path d="M${left},${y(value)}H${width-right}" stroke="#e9ede0" stroke-width=".7" stroke-dasharray="3 4"/><text x="${left-10}" y="${y(value)+3}" fill="#a4ad96" font-size="9" text-anchor="end">${value}</text>`;
  for(let i=0;i<5;i++){const stamp=min+(max-min)*i/4;axes+=`<text x="${x({stamp})}" y="${height-6}" fill="#a1aa94" font-size="9" text-anchor="${i===0?'start':i===4?'end':'middle'}">${esc(dateLabel(new Date(stamp).toISOString(),true))}</text>`;}
  const line=points=>{let active=false;return points.map(p=>{if(typeof p.score!=='number'||!Number.isFinite(p.score)){active=false;return '';}const s=`${active?'L':'M'}${x(p).toFixed(2)},${y(p.score).toFixed(2)}`;active=true;return s;}).join(' ');};
  const last=[...past].reverse().find(p=>typeof p.score==='number');
  const f=last?[last,...future]:future;
  const a=last?[last,...alternative]:alternative;
  const band=f.filter(p=>typeof p.lower==='number'&&typeof p.upper==='number');
  let interval='';if(band.length){const b=last?[{...last,lower:last.score,upper:last.score},...band]:band;interval=`<path d="${b.map((p,i)=>`${i?'L':'M'}${x(p)},${y(p.upper)}`).join(' ')} ${[...b].reverse().map(p=>`L${x(p)},${y(p.lower)}`).join(' ')} Z" fill="#e1eacb" fill-opacity=".7"/>`;}
  const split=last&&future.length?`<path d="M${x(last)},${top}V${height-bottom}" stroke="#cbd6bb" stroke-dasharray="3 4"/><text x="${Math.min(x(last)+8,width-105)}" y="${top+11}" fill="#8e9e79" font-size="8">PROJECTION →</text>`:'';
  const circles=[...past,...future,...alternative].filter(p=>typeof p.score==='number').map(p=>`<circle cx="${x(p)}" cy="${y(p.score)}" r="2.3" fill="${alternative.includes(p)?'#b3935c':'#7e9a62'}"><title>${esc(dateLabel(p.date))}: ${numeric(p.score,1)}${typeof p.lower==='number'?` (range ${numeric(p.lower,1)}–${numeric(p.upper,1)})`:''}</title></circle>`).join('');
  return `<svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(options.label||'Observed vegetation condition index and uncalibrated baseline forecast interval')}"><title>${esc(options.label||'Vegetation condition trajectory')}</title>${axes}${interval}${split}<path d="${line(past)}" fill="none" stroke="#416b4c" stroke-width="2.2" stroke-linecap="round"/><path d="${line(f)}" fill="none" stroke="#8fa763" stroke-width="2" stroke-dasharray="5 4"/>${alternative.length?`<path d="${line(a)}" fill="none" stroke="#b3935c" stroke-width="2.3"/>`:''}${circles}</svg>`;
}
function renderDetail() {
  if(!state.region)return;
  const r=state.region,h=r.health;
  $$('[data-detail]').forEach(el=>{const selected=el.dataset.detail===state.tab;el.setAttribute('aria-selected',String(selected));el.tabIndex=selected?0:-1;});
  $('#detail-body').setAttribute('aria-labelledby','detail-tab-'+state.tab);
  if(state.tab==='trajectory'){
    const history=(r.forecast.historical||[]).slice(-24);
    const points=(r.forecast.points||[]).filter(p=>p.horizon_days<=state.horizon);
    $('#detail-body').innerHTML=`<div class="trajectory-layout"><div class="condition-summary"><div class="eyebrow">CURRENT CONDITION</div><div class="condition-big">${numeric(h.score,1)}<small>/100</small></div>${badge(h.status)}<div class="condition-change">${typeof h.change==='number'?(h.change>0?'↑ +':h.change<0?'↓ ':'')+numeric(h.change,1)+' points · recent 3 vs prior 12 records':'Change unavailable'}</div><p>A vegetation condition proxy, interpreted with the available observation record.</p></div><div><div class="chart-topline"><div class="chart-legend"><span><i class="legend-line"></i>Observed</span><span><i class="legend-line dashed"></i>Baseline</span><span><i class="legend-band"></i>Model range</span></div><div class="segmented" aria-label="Forecast horizon">${[[30,'30D'],[180,'6M'],[365,'1Y']].map(([d,label])=>`<button data-horizon="${d}" aria-pressed="${state.horizon===d}" class="${state.horizon===d?'active':''}">${label}</button>`).join('')}</div></div>${chart(history,points)}<p class="chart-caveat">${esc(r.forecast.caveat || 'Illustrative historical baseline. Intervals are not calibrated.')}</p></div></div><div class="detail-foot"><span>${numeric(r.observations?.length)} monthly records · Coverage ${pct(h.coverage)} · Last observed ${dateLabel(r.latest_date)}</span><span>${esc(methodLabel(h.method))}</span></div>`;
    $$('[data-horizon]').forEach(el=>el.addEventListener('click',()=>{state.horizon=Number(el.dataset.horizon);renderDetail();}));
  }else if(state.tab==='evidence'){
    $('#detail-body').innerHTML=`<div class="evidence-grid">${(r.drivers||[]).map(d=>`<article class="evidence-card"><h3>${esc(d.name)}</h3><div class="evidence-meter"><div class="tiny-track"><i style="width:${Math.max(0,Math.min(100,(d.evidence_score || 0)))}%"></i></div><strong>${numeric(d.evidence_score,1)}</strong></div><div class="evidence-caption">Evidence strength / 100 · not causal probability</div><p>${esc(Array.isArray(d.evidence)?d.evidence.join(' '):d.evidence)}</p>${badge(d.direction || 'Candidate driver')}</article>`).join('') || '<p class="observations-empty">Cause uncertain. There is not enough evidence to identify a candidate driver.</p>'}</div><div class="evidence-disclaimer">Multiple drivers can act together. These associations do not establish causation. Check event timing, independent reference observations, and competing explanations before acting.</div>`;
  }else if(state.tab==='resilience'){
    const s=r.resilience;
    $('#detail-body').innerHTML=`<div class="resilience-grid"><div class="signal-card"><small>Lag-1 autocorrelation</small><strong>${numeric(s.ar1,3)}</strong><p>Persistence in the prepared vegetation series. A statistical indicator, not a tipping-point probability.</p></div><div class="signal-card"><small>Residual variance</small><strong>${numeric(s.variance,4)}</strong><p>Variability after preprocessing. Interpretation depends on record quality and length.</p></div><div class="signal-note"><h3>${esc(String(s.status || 'Insufficient evidence').replaceAll('_', ' '))}</h3><p>${esc(Array.isArray(s.evidence)?s.evidence.join(' '):s.evidence || '')}</p><p style="margin-top:10px">${esc(s.trend || '')}</p></div></div><div class="evidence-disclaimer">${esc(s.caveat || 'These exploratory indicators cannot establish an irreversible ecological tipping point.')}</div>`;
  }else if(state.tab==='observations'){
    $('#detail-body').innerHTML=`<div class="observed-alerts">${(r.alerts||[]).map(a=>`<article class="observation-alert"><div>${badge(a.severity)}<strong>${esc(a.title)}</strong><small>${esc(dateLabel(a.date))}</small></div><p>${esc(a.message)}</p></article>`).join('') || '<p class="chart-caveat">No active alert in the current assessment.</p>'}</div><div class="observations-caption"><strong>The observation record</strong><span>${observationSource(r)} · — denotes missing data</span></div><div class="observation-table table-scroll"><table><thead><tr><th>Date</th><th>NDVI</th><th>NDMI</th><th>NBR</th><th>Rainfall anomaly (σ)</th><th>Heat anomaly (°C)</th><th>Valid pixels</th><th>Index / 100</th></tr></thead><tbody>${[...r.observations].reverse().map(o=>`<tr><td>${esc(dateLabel(o.date))}</td><td>${numeric(o.ndvi,3)}</td><td>${numeric(o.ndmi,3)}</td><td>${numeric(o.nbr,3)}</td><td>${numeric(o.precipitation_anomaly,2)}</td><td>${numeric(o.temperature_anomaly,2)}</td><td>${pct(o.valid_fraction)}</td><td>${numeric(o.health_score,1)}</td></tr>`).join('')}</tbody></table></div><div class="detail-foot"><span>${esc(sourceSummary(r))}</span><span>${esc(r.provenance?.geometry_note || '')}</span></div><div id="alert-lifecycle"></div>`;
    if(typeof renderAlertLifecycle==='function')renderAlertLifecycle(r);
  }else if(state.tab==='coverage'){
    renderCoverage(r.id);
  }else if(state.tab==='models'){
    renderModelRuns(r.id);
  }else if(state.tab==='monitor'){
    renderMonitoringControls(r.id);
  }else if(state.tab==='workflow'){
    renderWorkflow(r.id);
  }else{
    renderEvaluation(r.id);
  }
}
async function renderEvaluation(id) {
  $('#detail-body').innerHTML='<div class="chart-skeleton skeleton" aria-label="Running historical evaluation"></div>';
  try{
    const result=await api('/regions/'+encodeURIComponent(id)+'/evaluation');
    if(state.selected!==id||state.tab!=='evaluation')return;
    const names={persistence:'Persistence',seasonal_naive:'Seasonal naive',damped_trend:'Damped trend'};
    const rows=Object.entries(result.methods).flatMap(([method,scores])=>Object.entries(scores.by_horizon).map(([h,m])=>`<tr><td><strong>${names[method]}</strong></td><td>${h} month${h==='1'?'':'s'}</td><td>${m.n}</td><td>${numeric(m.mae,2)}</td><td>${numeric(m.rmse,2)}</td><td>${pct(m.coverage)}</td></tr>`));
    $('#detail-body').innerHTML=`<div class="observations-caption"><strong>Historical baseline comparison</strong><button class="button secondary" id="export-evaluation">${icon('download',14)}Export evaluation</button></div><p class="chart-caveat" style="margin-bottom:16px">Each prediction uses only the observations available at its historical cutoff. Errors measure spectral index points; this is not validation against field-measured ecosystem health.</p><div class="table-scroll"><table><thead><tr><th>Baseline</th><th>Horizon</th><th>Scored targets</th><th>MAE ↓</th><th>RMSE ↓</th><th>Empirical range coverage</th></tr></thead><tbody>${rows.join('')}</tbody></table></div><div class="evidence-disclaimer">${esc(result.synthetic_warning || 'Remote measurements and their ecological interpretation have not been independently validated.')} Intervals remain uncalibrated. Overlapping cutoffs are dependent. Use an untouched period and region before claiming generalization.</div><p class="chart-caveat" style="margin-top:12px">Calendar-month targets are matched to the actual dated observation; nominal forecasts use 30, 180, and 365 days. Missing targets are skipped. A dash means unavailable.</p>`;
    $('#export-evaluation').addEventListener('click',()=>downloadJSON(result,'forestmind-'+id+'-backtest.json'));
  }catch(error){if(state.selected===id&&state.tab==='evaluation')$('#detail-body').innerHTML=`<p class="observations-empty">${esc(userMessage(error))} Try selecting Backtesting again.</p>`;}
}
function renderTable() {
  if(!state.overview)return;
  const query=$('#region-search').value.toLowerCase(),filter=$('#region-filter').value;
  const rows=state.overview.regions.filter(r=>`${r.name} ${r.country}`.toLowerCase().includes(query)&&(filter==='all'||filter==='alerts'&&r.alert_count>0||filter==='imported'&&!isDemo(r)||filter==='demo'&&isDemo(r)));
  $('#region-table').innerHTML=rows.map(r=>`<tr><td><strong>${esc(r.name)}</strong><small>${esc(r.country)} · ${numeric(r.area_km2)} km²</small></td><td><div class="index-cell"><strong>${numeric(r.health_score,1)}</strong><div class="tiny-track"><i style="width:${Math.max(0,Math.min(100,r.health_score || 0))}%"></i></div></div></td><td>${badge(r.status)}</td><td>${esc(dateLabel(r.latest_date))}</td><td>${isDemo(r)?'Synthetic':'Observed record'}<small>${esc(r.quality || 'Coverage recorded')}</small></td><td>${r.alert_count || '—'}</td><td><button class="button secondary" data-inspect="${esc(r.id)}">Inspect ↗</button></td></tr>`).join('') || '<tr><td colspan="7" class="no-results">No regions match these filters.</td></tr>';
  $('#table-footer').textContent=`Showing ${rows.length} of ${state.overview.regions.length} regions. Condition values are proxy indices, not calibrated ecological probabilities.`;
  $$('[data-inspect]').forEach(el=>el.addEventListener('click',async()=>{location.hash='overview';await selectRegion(el.dataset.inspect);$('#region-detail').scrollIntoView({behavior:'smooth',block:'start'});}));
}
function renderProvenance() {
  if(!state.overview)return;
  $('#provenance-list').innerHTML=state.overview.regions.map(r=>`<div class="provenance-row"><div><strong>${esc(r.name)}</strong><p>${esc(sourceSummary(r))} · ${esc(methodLabel(r.provenance?.method))}</p></div><div>${badge(observationSource(r))}<p>Last observed ${esc(dateLabel(r.latest_date))}</p></div></div>`).join('');
}
async function runAnalysis() {
  if(!state.selected)return;
  const button=$('#run-analysis'),id=state.selected;
  button.disabled=true;button.innerHTML=icon('refresh',14)+' Analyzing…';
  try{
    let job=await api('/analysis',{method:'POST',body:JSON.stringify({region_id:id})});
    for(let n=0;n<120&&!['completed','failed'].includes(job.status);n++){await new Promise(resolve=>setTimeout(resolve,500));job=await api('/analysis/'+encodeURIComponent(job.id));}
    if(job.status==='failed')throw new Error(job.error || 'Analysis failed. Check the observation record.');
    if(job.status!=='completed')throw new Error('Analysis is still running. Refresh the workspace shortly to view its result.');
    await loadWorkspace(id);toast('Analysis complete. Evidence and forecasts refreshed.');
  }catch(error){reportError(error);}
  finally{button.disabled=false;button.innerHTML=icon('refresh',14)+' Run analysis';}
}
async function runScenario(event) {
  event.preventDefault();
  const button=$('#compare-scenario');button.disabled=true;
  try{
    const values=Object.fromEntries(new FormData($('#scenario-form')));
    for(const k of ['fire_reduction','pressure_reduction','restoration','climate_stress'])values[k]=Number(values[k]);
    let result;
    if($('#scenario-horizon').value==='five-year'){
      const snapshot=await ensureWorkflowSnapshot(values.region_id);
      const parameters={...values,manifest_id:snapshot.manifest.id};delete parameters.region_id;
      result=await api('/regions/'+encodeURIComponent(values.region_id)+'/workflow/scenarios',{method:'POST',body:JSON.stringify(parameters)});
      result={...result,scenario:result.comparisons[0]?.points || [],caveat:result.limitations.join(' '),historical:snapshot.forecast.historical};
    }else result=await api('/scenarios',{method:'POST',body:JSON.stringify(values)});
    const region=await api('/regions/'+encodeURIComponent(values.region_id));
    state.scenario=result;
    if(!result.baseline.length){$('#scenario-results').innerHTML='<div class="empty-state"><h2>More observations are needed</h2><p>This region has no eligible forecast. No scenario projection was produced. Supply twelve consecutive usable monthly records, including the latest month.</p></div>';return;}
    const b=result.baseline.at(-1),s=result.scenario.at(-1),delta=b&&s?s.score-b.score:null;
    const assumptions=Array.isArray(result.assumptions)?result.assumptions.map(readableValue):Object.entries(result.assumptions||{}).map(([k,v])=>`${humanLabel(k)}: ${readableValue(v)}`);
    $('#scenario-results').innerHTML=`<div class="panel-heading"><div><div class="eyebrow">EXPLICIT ASSUMPTIONS · ILLUSTRATIVE OUTPUT</div><h2>${esc(region.name)}</h2></div>${badge('Uncalibrated')}</div><div class="scenario-result-body"><div class="scenario-metrics"><div class="scenario-metric"><small>Baseline · ${b?.horizon_days || '—'} days</small><strong>${numeric(b?.score,1)}</strong></div><div class="scenario-metric"><small>Scenario · same horizon</small><strong>${numeric(s?.score,1)}</strong></div><div class="scenario-metric"><small>Condition-point difference</small><strong>${typeof delta==='number'&&delta>0?'+':''}${numeric(delta,1)}</strong></div></div><div class="chart-topline"><div class="chart-legend"><span><i class="legend-line"></i>Observed</span><span><i class="legend-line dashed"></i>Baseline</span><span><i class="legend-line" style="border-color:#b3935c"></i>Scenario</span></div></div>${chart((result.historical || region.forecast.historical).slice(-12),result.baseline,{alternative:result.scenario,label:'Baseline and hypothetical scenario condition trajectories'})}<div class="evidence-disclaimer">${esc(result.caveat)}</div><div class="scenario-assumptions"><strong>Assumptions saved with this comparison</strong><ul>${assumptions.map(a=>`<li>${esc(a)}</li>`).join('')}</ul><button id="export-scenario" class="button secondary">${icon('download',14)}Export comparison</button></div></div>`;
    $('#export-scenario').addEventListener('click',()=>downloadJSON(state.scenario,'forestmind-scenario-'+state.scenario.id+'.json'));
    toast('Scenario comparison saved.');
  }catch(error){reportError(error);toast(userMessage(error));}
  finally{button.disabled=false;}
}
function downloadJSON(data, filename) {const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=filename;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
let regionImportFile=null,regionImportRead=0;
function openImport() {
  ++regionImportRead;regionImportFile=null;$('#import-form').reset();$('#import-error').hidden=true;$('#import-preview').hidden=true;
  $('#file-name').textContent='Prepared observations · maximum 2 MB';$('#submit-import').disabled=true;$('#import-dialog').showModal();
}
function regionFilePreview(payload) {
  if(!payload||typeof payload!=='object'||Array.isArray(payload))throw new Error('Choose a ForestMind observation file containing a region and its dated observations.');
  if(typeof payload.name!=='string'||!payload.name.trim())throw new Error('This file is missing the monitoring region name.');
  if(payload.geometry?.type!=='Polygon'||!Array.isArray(payload.geometry.coordinates))throw new Error('This file is missing the monitoring area boundary.');
  if(!Array.isArray(payload.observations)||!payload.observations.length)throw new Error('This file has no dated observations to import.');
  const dates=payload.observations.map(row=>row?.date);
  if(dates.some(date=>typeof date!=='string'||!/^\d{4}-\d{2}-\d{2}$/.test(date)||!Number.isFinite(Date.parse(date))||new Date(date).toISOString().slice(0,10)!==date))throw new Error('Every observation needs a valid calendar date. Ask the data provider to correct the file.');
  dates.sort();
  return {name:payload.name,count:payload.observations.length,period:dateLabel(dates[0])+' – '+dateLabel(dates.at(-1)),source:isDemo(payload)?'Synthetic example · not real observations':'User-provided observations · not independently verified'};
}
async function loadRegionFile(event) {
  const request=++regionImportRead,file=event.target.files[0];regionImportFile=null;
  $('#submit-import').disabled=true;$('#import-preview').hidden=true;$('#import-error').hidden=true;$('#file-name').textContent=file?.name || 'Prepared observations · maximum 2 MB';
  if(!file)return;
  try {
    if(file.size>2*1024*1024)throw new Error('Choose an observation file smaller than 2 MB.');
    let payload;try{payload=JSON.parse(await file.text());}catch{throw new Error('This file could not be read as a ForestMind observation file. Choose a valid JSON file from your data provider.');}
    const preview=regionFilePreview(payload);if(request!==regionImportRead)return;
    regionImportFile=payload;
    $('#import-preview-region').textContent=preview.name;$('#import-preview-period').textContent=preview.period;
    $('#import-preview-count').textContent=numeric(preview.count);$('#import-preview-source').textContent=preview.source;
    $('#import-preview').hidden=false;$('#submit-import').disabled=false;
  }catch(error){if(request!==regionImportRead)return;$('#import-error').textContent=userMessage(error,'This file could not be read. Please choose it again.');$('#import-error').hidden=false;}
}
async function downloadObservationTemplate(){try{downloadJSON(await api('/import-example'),'forestmind-observation-example.json');toast('Example downloaded. Its measurements are synthetic; replace them with your own dated evidence before using it as real data.');}catch(error){toast(userMessage(error));}}
async function importRegion(event) {
  event.preventDefault();const button=$('#submit-import');button.disabled=true;$('#import-error').hidden=true;
  try{
    if(!regionImportFile)throw new Error('Choose an observation file and review its summary before importing.');
    const body=regionImportFile;
    const result=await api('/regions/import',{method:'POST',body:JSON.stringify(body)});
    $('#import-dialog').close();location.hash='overview';await loadWorkspace(result.id || result.region_id);toast(isDemo(body)?'Synthetic example imported. It remains labeled as demonstration data.':'Region imported and analyzed. Its supplied source is recorded.');
  }catch(error){$('#import-error').textContent=userMessage(error);$('#import-error').hidden=false;}
  finally{button.disabled=!regionImportFile;}
}
hydrateIcons();route();
window.addEventListener('hashchange',route);
$('#map-layer').addEventListener('change',renderMap);
$('#zoom-in').addEventListener('click',()=>{state.zoom=Math.min(8,state.zoom*1.5);renderMap();});
$('#zoom-out').addEventListener('click',()=>{state.zoom=Math.max(.5,state.zoom/1.5);renderMap();});
$('#zoom-reset').addEventListener('click',()=>{state.zoom=1;renderMap();});
$$('[data-detail]').forEach(el=>{el.id='detail-tab-'+el.dataset.detail;el.setAttribute('aria-controls','detail-body');el.tabIndex=el.dataset.detail===state.tab?0:-1;el.addEventListener('click',()=>{state.tab=el.dataset.detail;renderDetail();});});
$('#detail-body').setAttribute('role','tabpanel');$('#detail-body').setAttribute('tabindex','0');$('#detail-body').setAttribute('aria-labelledby','detail-tab-'+state.tab);
$('.detail-tabs').addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const tabs=$$('[data-detail]'),i=tabs.indexOf(document.activeElement),next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(i+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;tabs[next].focus();tabs[next].click();});
$('#region-search').addEventListener('input',renderTable);$('#region-filter').addEventListener('change',renderTable);
$('#run-analysis').addEventListener('click',runAnalysis);
$('#export-region').addEventListener('click',async()=>{if(!state.selected)return;try{downloadJSON(await api('/regions/'+encodeURIComponent(state.selected)+'/export'),'forestmind-'+state.selected+'.json');}catch(error){toast(userMessage(error));}});
$('#scenario-form').addEventListener('submit',runScenario);
$$('input[type="range"]').forEach(el=>el.addEventListener('input',()=>{$('#'+el.id+'-output').textContent=el.value+'%';}));
function invalidateScenario(){if(state.scenario){state.scenario=null;$('#scenario-results').innerHTML='<div class="empty-state"><h2>Assumptions changed</h2><p>Compare trajectories again to evaluate the selected region with these inputs.</p></div>';}}
$('#scenario-region').addEventListener('change',()=>{invalidateScenario();selectRegion($('#scenario-region').value);});
$('#scenario-horizon').addEventListener('change',invalidateScenario);
$$('input[type="range"]').forEach(el=>el.addEventListener('input',invalidateScenario));
$$('.import-trigger').forEach(el=>el.addEventListener('click',openImport));
$('#close-import').addEventListener('click',()=>$('#import-dialog').close());
$('#import-form').addEventListener('submit',importRegion);
$('#download-template').addEventListener('click',downloadObservationTemplate);
$('#download-import-template').addEventListener('click',downloadObservationTemplate);
$('#import-file').addEventListener('change',loadRegionFile);
$('#refresh-data').addEventListener('click',async()=>{await loadWorkspace();toast('Workspace refreshed.');});
loadWorkspace();
