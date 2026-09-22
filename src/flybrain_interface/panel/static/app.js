import {createBrainView} from './brain-view.js';
import {activityIsStale,shouldClearActivity} from './brain-state.mjs';
import {parseIndices} from './indices.mjs';

const $=id=>document.getElementById(id);
const colors=['#c9f55f','#66d7ce','#ffac69','#c49cff'];
let history=[],socket,selectedPreset=null,lastTelemetryAt=0;
const brain=createBrainView($('brainViewport'),$('atlasState'));

function statusName(s){return s?String(s).replaceAll('_',' '):'idle'}
function update(t){
  const s=t.status||'idle';
  $('status').className=`status ${s}`;$('status').querySelector('b').textContent=statusName(s);
  $('simTime').textContent=(t.simulated_time_s||0).toFixed(3);
  $('progress').style.width=`${Math.min(100,100*(t.simulated_time_s||0)/(t.duration_s||1))}%`;
  $('speed').textContent=t.speed_ratio==null?'—':`${t.speed_ratio.toFixed(2)}×`;
  $('wallTime').textContent=t.wall_elapsed_s==null?'sim / active wall':`${t.wall_elapsed_s.toFixed(2)} s wall elapsed`;
  $('spikes').textContent=(t.total_spikes||0).toLocaleString();
  $('memory').textContent=t.process_rss_bytes?`${(t.process_rss_bytes/1048576).toFixed(0)} MiB`:'—';
  $('events').textContent=(t.pending_delayed_events||0).toLocaleString();
  $('experiment').textContent=t.experiment_id||'—';
  $('loop').textContent=t.worker_loop_seconds==null?'—':`${t.worker_loop_seconds.toFixed(3)} s`;
  $('dropped').textContent=(t.browser_frames_dropped||0).toLocaleString();
  $('ipcDropped').textContent=(t.ipc_telemetry_dropped_total||0).toLocaleString();
  $('edges').textContent=(t.visited_edges||0).toLocaleString();
  const active=['running','paused'].includes(s);
  $('start').disabled=active;$('pause').disabled=!active;
  $('pause').textContent=s==='paused'?'Resume':'Pause';
  $('reset').disabled=!['running','paused','completed','failed'].includes(s);
  if(t.error)$('error').textContent=t.error;
  if(t.kind==='telemetry'){
    lastTelemetryAt=performance.now();
    $('activityState').classList.remove('stale');
    history.push(t);if(history.length>120)history.shift();
    drawRate();drawVoltage();renderWatch(t);
    brain.updateActivity(t.brain_activity||null);
    renderActivityMeta(t);
  }else if(shouldClearActivity(s)){
    brain.updateActivity(null);
    $('activityState').textContent=s==='idle'?'idle · no activity supplied':`${s} · activity cleared`;
  }
}

function canvas(id){const c=$(id),d=devicePixelRatio||1,r=c.getBoundingClientRect();c.width=r.width*d;c.height=r.height*d;const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);return [x,r.width,r.height]}
function axes(x,w,h){x.strokeStyle='#263632';x.lineWidth=1;x.beginPath();for(let i=1;i<4;i++){let y=i*h/4;x.moveTo(0,y);x.lineTo(w,y)}x.stroke()}
function plot(id,series){const [x,w,h]=canvas(id);x.clearRect(0,0,w,h);axes(x,w,h);const vals=series.flatMap(s=>s.values).filter(Number.isFinite),lo=Math.min(...vals,0),hi=Math.max(...vals,1);series.forEach((s,j)=>{x.strokeStyle=colors[j%colors.length];x.lineWidth=2;x.beginPath();let connected=false;s.values.forEach((v,i)=>{if(!Number.isFinite(v)){connected=false;return}let px=i*w/Math.max(1,s.values.length-1),py=h-(v-lo)*h/(hi-lo);connected?x.lineTo(px,py):x.moveTo(px,py);connected=true});x.stroke()})}
function drawRate(){const names=[...new Set(history.flatMap(t=>Object.keys(t.population_rates_hz||{})))];plot('rateChart',names.map(n=>({values:history.map(t=>t.population_rates_hz?.[n]??NaN)})));$('legend').innerHTML=names.map((n,i)=>`<span><i style="background:${colors[i%colors.length]}"></i>${escapeHtml(n)}</span>`).join('')||'No population samples yet.'}
function drawVoltage(){const count=Math.max(0,...history.map(t=>(t.voltage_mv||[]).length));plot('voltageChart',Array.from({length:count},(_,i)=>({values:history.map(t=>t.voltage_mv?.[i]??NaN)})))}
function renderWatch(t){$('watchValues').innerHTML=(t.watch_indices||[]).map((n,i)=>{const v=t.voltage_mv?.[i],d=t.synaptic_drive_mv?.[i];return `<div><b>Neuron ${n}</b><span>${Number.isFinite(v)?v.toFixed(3)+' mV':'voltage unavailable'} · ${Number.isFinite(d)?'drive '+d.toExponential(2)+' mV':'drive unavailable'}</span></div>`}).join('')||'<p>Watchlist is empty.</p>'}
function renderActivityMeta(t){const a=t.brain_activity;if(!a)return;$('activityState').textContent=`live · ${(a.bin_duration_s*1000).toFixed(1)} ms simulation bin`;$('activitySignal').textContent=`${a.active_selected_count}/${a.selected_neuron_count} selected neurons spiked; ${a.active_without_visible_soma_count} active lacked a displayed soma. Brightness = emitted spike rate / ${a.normalization.reference_rate_hz} Hz, clamped [0,1].`}
function escapeHtml(value){const node=document.createElement('span');node.textContent=String(value);return node.innerHTML}

async function action(path,body){$('error').textContent='';try{const r=await fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:body?JSON.stringify(body):undefined});const d=await r.json();if(!r.ok)throw Error(d.detail||'Request failed')}catch(e){$('error').textContent=e.message}}
function selectedValues(){const targets=parseIndices($('targets').value,{required:true,label:'Target neurons'});const watch=parseIndices($('watch').value,{label:'Watchlist'});const outputs=selectedPreset?selectedPreset.outputs.flatMap(x=>x.neuron_indices):targets;const displayOutputs=selectedPreset?.display_output_indices||outputs;const visualization=selectedPreset?.visualization_indices||[...new Set([...targets,...watch,...outputs])].sort((a,b)=>a-b);return {targets,watch,outputs,displayOutputs,visualization}}

$('start').onclick=()=>{let values;try{values=selectedValues();if(values.visualization.length>4096)throw Error('Brain View selection exceeds 4,096 neurons.')}catch(e){$('error').textContent=e.message;return}const duration=Number($('duration').value),start=Number($('stimStart').value),stop=Number($('stimStop').value),populations=selectedPreset?selectedPreset.outputs.map(x=>({name:x.label.slice(0,40),neuron_indices:x.neuron_indices})):[{name:'stimulus targets',neuron_indices:values.targets}];history=[];void refreshSelection(values.targets,values.displayOutputs,values.watch);action('/api/experiments',{duration_s:duration,seed:Number($('seed').value),backend:'numba',subnormal_drive_policy:$('policy').value,chunk_duration_s:0.02,telemetry_hz:10,watch_indices:values.watch,visualization_indices:values.visualization,populations,stimulus:{neuron_indices:values.targets,start_s:start,stop_s:stop,interval_ms:Number($('interval').value),amplitude_mv:Number($('amplitude').value)}})};
$('pause').onclick=()=>action(`/api/experiments/${$('pause').textContent==='Resume'?'resume':'pause'}`);
$('reset').onclick=()=>action('/api/experiments/reset');
$('policy').onchange=()=>{$('policyNote').textContent=$('policy').value==='preserve'?'Exact IEEE-754 subnormal decay is preserved. This is the scientific default and can become slow in long sparse runs.':'Tiny drive values below the normal float64 range are set to zero. This explicit numerical approximation can improve long sparse-run performance.'};

async function postJson(path,body){const response=await fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});const payload=await response.json();if(!response.ok)throw Error(payload.detail||'Request failed');return payload}
async function refreshSelection(inputs,outputs,watch){try{const unique=[...new Set([...inputs,...outputs,...watch])];const mapped=await postJson('/api/anatomy/map',{neuron_indices:unique});const byIndex=new Map(mapped.entries.map(x=>[x.neuron_index,x]));const inputRows=inputs.map(x=>byIndex.get(x)),outputRows=outputs.map(x=>byIndex.get(x)),watchRows=watch.map(x=>byIndex.get(x));brain.setSelection(inputRows,outputRows,watchRows);$('selectionCoverage').textContent=`${mapped.visible_soma_count}/${mapped.requested_count} selected stable indices have a displayed measured soma; ${mapped.without_visible_soma_count} are omitted with reasons below.`;const rows=mapped.entries.filter(x=>!x.visible_soma).slice(0,12);$('selectionDetails').innerHTML=rows.map(x=>`<div><b>${x.neuron_index}</b><span>body ${escapeHtml(x.body_id)} · ${escapeHtml(x.omission_reason)}</span></div>`).join('')||'<p>All selected neurons map to displayed measured somata.</p>'}catch(e){$('selectionCoverage').textContent=e.message}}

async function loadPresets(){try{const [sensoryResponse,visualResponse]=await Promise.all([fetch('/api/presets/sensory-descending'),fetch('/api/presets/visual-pathway')]);if(!sensoryResponse.ok||!visualResponse.ok)throw Error('Preset metadata unavailable.');const sensory=await sensoryResponse.json(),visual=await visualResponse.json();const presets=sensory.inputs.map(input=>({input,outputs:sensory.outputs,caution:`${input.label}: ${input.selected_count} of ${input.available_count}, selected by the documented stable-body-ID rule. Output charts use annotated descending groups; no behavioral function is implied.`}));presets.push({input:visual.input,outputs:visual.outputs,watch_indices:visual.watch_indices,display_output_indices:visual.display_output_indices,visualization_indices:visual.visualization_indices,protocol:visual.interactive_protocol,caution:visual.caution});presets.forEach((x,i)=>{$('preset').add(new Option(`${x.input.label} (${x.input.selected_count}/${x.input.available_count})`,String(i)))});$('preset').onchange=()=>{if($('preset').value==='manual'){selectedPreset=null;$('presetNote').textContent='Manual engineering stimulation and observation assignments.';return}selectedPreset=presets[Number($('preset').value)];const input=selectedPreset.input;$('targets').value=input.neuron_indices.join(',');const watch=selectedPreset.watch_indices||selectedPreset.outputs.filter(x=>x.key!=='all_descending').flatMap(x=>x.neuron_indices);$('watch').value=watch.join(',');if(selectedPreset.protocol){$('duration').value=selectedPreset.protocol.duration_s;$('stimStart').value=selectedPreset.protocol.start_s;$('stimStop').value=selectedPreset.protocol.stop_s;$('interval').value=selectedPreset.protocol.interval_ms;$('amplitude').value=selectedPreset.protocol.amplitude_mv}$('presetNote').textContent=selectedPreset.caution;void refreshSelection(input.neuron_indices,selectedPreset.display_output_indices||selectedPreset.outputs.flatMap(x=>x.neuron_indices),watch)}}catch(e){$('presetNote').textContent=e.message}}

$('xyView').onclick=()=>{brain.resetView();$('orbit').textContent='Orbit off'};
$('orbit').onclick=()=>{const enabled=$('orbit').textContent.endsWith('off');brain.setOrbit(enabled);$('orbit').textContent=`Orbit ${enabled?'on':'off'}`};
$('render').onclick=()=>{const enabled=$('render').textContent==='Resume rendering';brain.setRendering(enabled);$('render').textContent=enabled?'Pause rendering':'Resume rendering'};
$('overlay').onclick=async()=>{try{if($('overlay').dataset.active==='true'){brain.setNeighborhood(null);$('overlay').dataset.active='false';$('overlay').textContent='Show connections';$('overlayState').textContent='off';return}const seeds=parseIndices($('targets').value,{required:true,label:'Target neurons'}).slice(0,64);const value=await postJson('/api/anatomy/neighborhood',{seed_indices:seeds,max_nodes:Number($('maxNodes').value),max_edges:Number($('maxEdges').value),min_synapse_count:Number($('minWeight').value)});brain.setNeighborhood(value);$('overlay').dataset.active='true';$('overlay').textContent='Hide connections';$('overlayState').textContent=`${value.edges.length} directed relationships · ${value.nodes.length} somata${value.truncated?' · capped':''}; ${value.relationships_without_visible_soma_count} candidates omitted for missing/view-excluded coordinates.`}catch(e){$('overlayState').textContent=e.message}};

async function loadAnatomy(){try{const r=await fetch('/api/anatomy');const a=await r.json();if(!r.ok)throw Error(a.detail||'Anatomy metadata unavailable');$('atlasCoverage').textContent=`${a.graph_visible_soma_count.toLocaleString()} displayed / ${a.graph_neuron_count.toLocaleString()} graph neurons (${(a.graph_visible_coverage_fraction*100).toFixed(2)}%). ${a.graph_measured_but_view_omitted_count.toLocaleString()} measured VNC-associated/unclassified and ${a.graph_without_measured_soma_count.toLocaleString()} without atlas somata are not drawn.`}catch(e){$('atlasCoverage').textContent=e.message}}

function connect(){socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/telemetry`);socket.onopen=()=>{$('connection').textContent='live';$('activityState').classList.remove('stale')};socket.onmessage=e=>update(JSON.parse(e.data));socket.onclose=()=>{$('connection').textContent='reconnecting';$('activityState').textContent='disconnected · activity cleared';$('activityState').classList.add('stale');brain.updateActivity(null);setTimeout(connect,1000)}}
setInterval(()=>{if(activityIsStale(lastTelemetryAt,performance.now())&&$('connection').textContent==='live'){$('activityState').textContent='stale · latest activity cleared';$('activityState').classList.add('stale');brain.updateActivity(null)}const p=brain.getPerformance();$('renderPerf').textContent=`WebGL CPU submit: ${p.mean_cpu_submit_ms.toFixed(2)} ms mean, ${p.max_cpu_submit_ms.toFixed(2)} ms max over ${p.render_count.toLocaleString()} draws (GPU completion not measured).`},500);

void loadPresets();void loadAnatomy();void refreshSelection([0],[0],[0]);connect();
addEventListener('resize',()=>{drawRate();drawVoltage()});
