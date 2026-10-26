/* Local service status and optional, explicitly authorized alert notifications. */
'use strict';
let operationsRequest=0;
function operationsSummary(data) {
  const status=data.status==='ok'?'Running normally':data.status==='degraded'?'Needs attention':'Status unavailable';
  const integrity=data.database_check==='ok'?'Check passed':data.database_check?'Needs attention':'Not checked';
  const access=data.access==='local-single-user-only'?'This computer only':humanLabel(data.access || 'Access unavailable');
  return `<div class="monitor-status"><div><small>Local service</small><strong>${esc(status)}</strong><p class="form-help">${esc(access)}</p></div><div><small>Saved database</small><strong>${esc(integrity)} · ${numeric(data.database_bytes/1048576,2)} MB</strong></div><div><small>Disk space available</small><strong>${numeric(data.disk_free_bytes/1073741824,2)} GB</strong></div></div>${data.database_check&&data.database_check!=='ok'?'<p class="evidence-disclaimer">The saved database needs attention. Download the service record and ask the person maintaining this installation to review it before relying on new saves.</p>':''}<div class="workflow-actions"><button class="button secondary" id="backup-workspace">Create database backup</button><button class="button secondary" id="download-operations">Download service record</button></div><p id="backup-status" class="form-help" role="status" aria-live="polite"></p><p class="form-help">A database backup preserves measurements, analyses, schedules and collection records. Downloaded satellite images need a separate backup. This workspace is for one person on this computer; public access and shared accounts are not available.</p>`;
}
function deliveryLabel(value) {
  return {pending:'Waiting to send',retry:'Waiting to retry',sending:'Sending',sent:'Delivered',failed:'Delivery stopped'}[value] || humanLabel(value || 'Unavailable');
}
function deliveryReason(row) {
  if(!row.error)return 'None recorded';
  if(/attempt limit|interrupted final attempt/i.test(row.error))return 'The retry limit was reached. An earlier attempt may have arrived; check the receiving service before sending again.';
  if(/Delivery failed|receiver availability/i.test(row.error))return 'The receiving service could not be reached or did not accept the alert.';
  return userMessage(row.error,'Delivery could not finish. Check the receiving service; the downloaded record includes troubleshooting details.');
}
function notificationSummary(data) {
  const config=data.configuration || data,configured=Boolean(config.endpoint_configured&&config.signing_configured),rows=data.outbox || [];
  return `<div class="notification-settings"><h3>Optional alert notifications</h3><p class="form-help">${configured?`Destination: <strong>${esc(config.destination_host)}</strong>. Enabling delivery shares dated alert summaries and their supporting observations with this service.`:'Notifications are not set up. Ask the person maintaining this installation to connect a receiving service before enabling delivery.'}</p>${config.reauthorization_required?'<p class="evidence-disclaimer"><strong>Destination changed.</strong> Delivery is off. Review the displayed destination, then authorize it and enable delivery to resume.</p>':''}${config.capacity_reached?`<p class="evidence-disclaimer"><strong>Notification storage is full: ${numeric(config.outbox_events)} / ${numeric(config.outbox_limit)} records.</strong> New alerts cannot be added to the delivery queue. Existing records are preserved. Ask the person maintaining this installation to review storage.</p>`:''}<form id="notification-form"><label class="check-label"><input id="notifications-enabled" type="checkbox" ${config.enabled?'checked':''} ${configured?'':'disabled'}> Enable alert delivery to the configured destination</label><label class="check-label"><input id="notifications-authorized" type="checkbox" ${config.authorized?'checked':''} ${configured?'':'disabled'}> I authorize sending these alert summaries to ${esc(config.destination_host || 'the configured destination')} while the service is running.</label><button type="submit" class="button secondary" ${configured?'':'disabled'}>Save notification preferences</button></form><p id="notification-status" class="form-help" role="status" aria-live="polite">${config.enabled&&config.authorized&&configured?'Delivery is enabled while ForestMind is running.':'Delivery is off until a destination is connected and you authorize it.'} An interrupted delivery may be retried, so the same alert can arrive more than once.</p><details class="model-hashes"><summary>Notification delivery record · ${rows.length} entries</summary>${rows.length?`<div class="table-scroll"><table><thead><tr><th>Entry</th><th>Status</th><th>Attempts</th><th>What happened</th></tr></thead><tbody>${rows.map((row,index)=>`<tr><td>${index+1}</td><td>${esc(deliveryLabel(row.status))}</td><td>${numeric(row.attempts)}</td><td>${esc(deliveryReason(row))}</td></tr>`).join('')}</tbody></table></div>`:'<p class="form-help">No delivery attempts recorded.</p>'}<button type="button" class="button secondary" id="download-deliveries">Download full delivery record</button></details></div>`;
}
async function renderOperations() {
  const target=$('#local-operations');if(!target)return;
  const request=++operationsRequest;
  target.innerHTML='<p class="inline-status" role="status">Checking the local service and notification settings…</p>';
  const [operations,notifications]=await Promise.allSettled([api('/operations'),api('/notifications')]);
  if(request!==operationsRequest)return;
  let markup=operations.status==='fulfilled'?operationsSummary(operations.value):`<p class="form-error">${esc(userMessage(operations.reason,'The local service status could not be loaded.'))} Use Refresh settings to try again.</p>`;
  markup+=notifications.status==='fulfilled'?notificationSummary(notifications.value):`<p class="form-error">${esc(userMessage(notifications.reason,'Notification settings could not be loaded.'))} Use Refresh settings to try again.</p>`;
  target.innerHTML=markup;
  $('#download-operations')?.addEventListener('click',()=>downloadJSON(operations.value,'forestmind-service-record.json'));
  $('#download-deliveries')?.addEventListener('click',()=>downloadJSON(notifications.value,'forestmind-notification-record.json'));
  $('#backup-workspace')?.addEventListener('click',async event=>{
    event.target.disabled=true;
    try{const result=await api('/operations/backup',{method:'POST'});$('#backup-status').textContent=`${result.verified?'Backup saved and checked: ':'Backup saved; verification was not confirmed: '}${result.filename}. Downloaded satellite images still need a separate backup.`;}
    catch(error){$('#backup-status').textContent=userMessage(error,'The backup could not be created. Please try again.');}
    finally{event.target.disabled=false;}
  });
  $('#notification-form')?.addEventListener('submit',async event=>{
    event.preventDefault();const button=$('button',event.target);button.disabled=true;
    try{const enabled=$('#notifications-enabled').checked,authorized=$('#notifications-authorized').checked;if(enabled&&!authorized)throw new Error('Authorize delivery to the displayed destination before enabling notifications.');await api('/notifications',{method:'PUT',body:JSON.stringify({enabled,authorize_delivery:authorized})});await renderOperations();toast('Notification preferences saved.');}
    catch(error){$('#notification-status').textContent=userMessage(error,'Notification preferences could not be saved. Please try again.');button.disabled=false;}
  });
}
$('#refresh-operations').addEventListener('click',renderOperations);
if(location.hash==='#methods')renderOperations();
