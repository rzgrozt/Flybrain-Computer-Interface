import {clampNormalized, replayDirection, scoreRows, selectRecordedStep} from './replay-state.mjs';
import {appendNeuralSample} from './live-timeline.mjs';
import {initPathwayInspector} from './pathway-ui.mjs';

const $ = id => document.getElementById(id);
const state = {recording: null, recordingId: null, episode: 0, step: 0, mode: 'guest', playing: false, timer: null, liveSamples: []};
const fmt = value => Number.isFinite(value) ? value.toFixed(4) : '—';
const escapeHtml = value => {const node = document.createElement('span'); node.textContent = String(value); return node.innerHTML;};

function stopPlayback() {
  if (state.timer != null) clearInterval(state.timer);
  state.timer = null;
  state.playing = false;
  $('replayPlay').textContent = 'Play';
  $('replayPlay').setAttribute('aria-label', 'Play recorded steps');
}
function renderLiveTimeline() {
  const samples = state.liveSamples;
  const mostRecent = samples.at(-1);
  $('timelineSource').textContent = samples.length ? 'LIVE NEURAL SAMPLES' : 'NO LIVE EPISODE FEED';
  $('timelineSource').className = 'source-pill ' + (samples.length ? 'measured' : 'unavailable');
  $('timelineEpisode').textContent = mostRecent
    ? 'Simulation ' + mostRecent.session_id
    : 'No live neural samples yet';
  $('timelineDistance').textContent = 'Motor / reward unavailable';
  $('replayStepText').textContent = 'live only';
  $('stepScrub').disabled = true;
  $('stepScrub').value = '0';
  if (!samples.length) {
    const empty = document.createElement('li');
    empty.className = 'timeline-empty';
    empty.textContent = 'Ordinary stimulation does not generate guest actions or rewards.';
    $('episodeTimeline').replaceChildren(empty);
    return;
  }
  const items = samples.slice(-40).reverse().map(sample => {
    const row = document.createElement('li');
    const item = document.createElement('div');
    item.className = 'neural-event';
    const label = document.createElement('span');
    label.textContent = 'CHUNK ' + sample.step_id + ' · ' +
      sample.simulated_time_s.toFixed(3) + ' simulated s';
    const spikes = document.createElement('small');
    spikes.textContent = sample.chunk_spikes == null
      ? 'spikes unavailable'
      : sample.chunk_spikes.toLocaleString() + ' simulated spikes';
    item.append(label, spikes);
    row.append(item);
    return row;
  });
  $('episodeTimeline').replaceChildren(...items);
}
function clearRecordedPanels() {
  $('motorSource').textContent = 'NO LIVE MOTOR FEED';
  $('motorSource').className = 'source-pill unavailable';
  $('motorAction').textContent = 'No live motor command available';
  $('motorCommand').textContent = 'dx — / dy —';
  $('motorArrow').style.opacity = '.25';
  $('motorArrow').style.transform = 'rotate(0deg)';
  renderProbe('horizontalScores', 'horizontalModality', null);
  renderProbe('verticalScores', 'verticalModality', null);
  $('motorObserved').textContent = '—';
  $('motorClick').textContent = 'Not implemented';
  $('motorAttribution').textContent = 'Unavailable live';
  $('actionHistory').replaceChildren();
  const item = document.createElement('li');
  item.textContent = 'No live motor session is attached.';
  $('actionHistory').append(item);
  renderLiveTimeline();
}
function setMode(mode) {
  state.mode = mode;
  const replay = mode === 'replay';
  $('modeLive').classList.toggle('selected', !replay);
  $('modeReplay').classList.toggle('selected', replay);
  $('modeLive').setAttribute('aria-pressed', String(!replay));
  $('modeReplay').setAttribute('aria-pressed', String(replay));
  $('sandboxLiveEmpty').hidden = replay;
  $('virtualStage').hidden = !replay;
  $('replayToolbar').hidden = !replay;
  $('sandboxSource').textContent = replay ? 'RECORDED CURSOR · NOT A VM' : 'VM NOT CONNECTED';
  $('sandboxSource').className = 'source-pill ' + (replay ? 'recorded' : 'disconnected');
  $('observationSource').textContent = replay ? 'Live brain / recorded motor' : 'Live simulation';
  $('sandboxMessage').textContent = replay
    ? 'Separate horizontal and vertical neural probes; recorded environment motion, not guest OS output.'
    : 'The QEMU display adapter is not connected. No host desktop is being controlled.';
  if (!replay) {
    stopPlayback();
    clearRecordedPanels();
  }
  render();
}
function setPoint(id, point) {
  $(id).style.left = (clampNormalized(point?.x) * 100) + '%';
  $(id).style.top = (clampNormalized(point?.y) * 100) + '%';
}
function renderStage(episode, step, index) {
  setPoint('virtualCursor', step.cursor_after);
  setPoint('virtualTarget', step.target);
  $('stageEpisode').textContent = `EPISODE ${episode.episode_index + 1} / STEP ${index + 1}`;
  $('stageDistance').textContent = `DISTANCE ${fmt(step.distance_after)}`;
  const dx = step.motor_command.dx;
  const dy = step.motor_command.dy;
  const distance = Math.hypot(dx, dy);
  const vector = $('virtualVector');
  vector.style.left = (clampNormalized(step.cursor_before.x) * 100) + '%';
  vector.style.top = (clampNormalized(step.cursor_before.y) * 100) + '%';
  const stage = $('virtualStage');
  const bounds = stage.getBoundingClientRect();
  vector.style.width = Math.hypot(dx * bounds.width, dy * bounds.height) + 'px';
  vector.style.transform = `rotate(${Math.atan2(dy * bounds.height, dx * bounds.width)}rad)`;
  vector.hidden = distance === 0;
  const prior = episode.steps.slice(0, index + 1);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 100 100');
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('aria-hidden', 'true');
  const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
  const path = [episode.steps[0].cursor_before, ...prior.map(item => item.cursor_after)];
  polyline.setAttribute('points', path.map(point => `${clampNormalized(point.x) * 100},${clampNormalized(point.y) * 100}`).join(' '));
  polyline.setAttribute('fill', 'none');
  polyline.setAttribute('stroke', '#6edfe1');
  polyline.setAttribute('stroke-opacity', '.7');
  polyline.setAttribute('stroke-width', '.35');
  polyline.setAttribute('vector-effect', 'non-scaling-stroke');
  svg.append(polyline);
  $('virtualPath').replaceChildren(svg);
}
function renderProbe(id, modalityId, probe) {
  const group = $(id);
  if (!probe) {
    $(modalityId).textContent = 'Not available';
    group.replaceChildren();
    const blank = document.createElement('span');
    blank.className = 'empty-score';
    blank.textContent = 'No recorded probe';
    group.append(blank);
    return;
  }
  $(modalityId).textContent = probe.feature_modality.replaceAll('_', ' ');
  group.replaceChildren();
  scoreRows(probe).forEach(row => {
    const item = document.createElement('div');
    item.className = 'score-item';
    const label = document.createElement('span');
    label.className = 'label';
    label.textContent = row.label;
    const figure = document.createElement('strong');
    figure.textContent = Number.isFinite(row.value) ? row.value.toFixed(4) : '—';
    const bar = document.createElement('div');
    bar.className = 'score-bar';
    bar.title = 'Relative score magnitude; not a probability';
    const fill = document.createElement('span');
    fill.style.width = (row.magnitude * 100) + '%';
    bar.append(fill);
    item.append(label, figure, bar);
    group.append(item);
  });
}
function renderMotor(episode, step, index) {
  $('motorSource').textContent = 'RECORDED READOUT';
  $('motorSource').className = 'source-pill recorded';
  const horizontal = step.probes.find(probe => probe.axis === 'horizontal');
  const vertical = step.probes.find(probe => probe.axis === 'vertical');
  renderProbe('horizontalScores', 'horizontalModality', horizontal);
  renderProbe('verticalScores', 'verticalModality', vertical);
  const actions = step.probes.map(probe => probe.selected_action).join(' + ');
  $('motorAction').textContent = actions || 'No recorded action';
  $('motorCommand').textContent = `Recorded Δx ${fmt(step.motor_command.dx)} / Δy ${fmt(step.motor_command.dy)}`;
  $('motorObserved').textContent = `(${fmt(step.cursor_before.x)}, ${fmt(step.cursor_before.y)}) → (${fmt(step.cursor_after.x)}, ${fmt(step.cursor_after.y)})`;
  $('motorClick').textContent = 'Unavailable';
  $('motorAttribution').textContent = 'Recorded · separate axis probes';
  const angle = replayDirection(step);
  $('motorArrow').style.opacity = angle == null ? '.25' : '1';
  $('motorArrow').style.transform = `rotate(${angle ?? 0}deg)`;
  $('actionHistory').replaceChildren(...episode.steps.slice(Math.max(0, index - 5), index + 1).map(item => {
    const node = document.createElement('li');
    node.textContent = item.probes.map(probe => probe.selected_action).join(' + ');
    return node;
  }));
}
function renderTimeline(episode, selected) {
  $('timelineSource').textContent = 'RECORDED STEPS';
  $('timelineSource').className = 'source-pill recorded';
  $('timelineEpisode').textContent = `Episode ${episode.episode_index + 1} · ${episode.success ? 'validated success' : 'not successful'}`;
  $('timelineDistance').textContent = `Δ distance ${fmt(episode.distance_reduction)}`;
  $('replayStepText').textContent = `${selected + 1} / ${episode.steps.length}`;
  $('stepScrub').disabled = episode.steps.length <= 1;
  $('stepScrub').max = Math.max(0, episode.steps.length - 1);
  $('stepScrub').value = String(selected);
  const entries = episode.steps.map((step, index) => {
    const row = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = selected === index ? 'active' : '';
    button.setAttribute('aria-current', String(selected === index));
    const label = document.createElement('span');
    label.textContent = `STEP ${String(index + 1).padStart(2, '0')} · ${step.probes.map(probe => probe.selected_action).join(' / ')}`;
    const distance = document.createElement('small');
    distance.textContent = `d ${fmt(step.distance_after)}`;
    button.append(label, distance);
    button.onclick = () => {stopPlayback(); state.step = index; render();};
    row.append(button);
    return row;
  });
  $('episodeTimeline').replaceChildren(...entries);
}
function render() {
  if (state.mode !== 'replay') return;
  const current = selectRecordedStep(state.recording, state.episode, state.step);
  if (!current) {
    $('sandboxMessage').textContent = 'Recording unavailable or empty; no simulated frames are invented.';
    return;
  }
  renderStage(current.episode, current.step, current.index);
  renderMotor(current.episode, current.step, current.index);
  renderTimeline(current.episode, current.index);
}
function selectEpisode(index) {
  stopPlayback();
  state.episode = index;
  state.step = 0;
  render();
}
async function loadRecording(id) {
  stopPlayback();
  state.recording = null;
  $('sandboxMessage').textContent = 'Loading validated recorded data…';
  try {
    const response = await fetch('/api/v2/recordings/' + encodeURIComponent(id));
    if (!response.ok) throw Error('Recorded experiment is unavailable (HTTP ' + response.status + ').');
    const recording = await response.json();
    if (recording.schema_version !== 2 || recording.source !== 'recorded_validation_artifact') {
      throw Error('Unsupported recording contract; no replay loaded.');
    }
    state.recording = recording;
    state.recordingId = id;
    state.episode = 0;
    state.step = 0;
    $('episodeSelect').replaceChildren(...recording.episodes.map((episode, index) => {
      const option = new Option(`Episode ${index + 1} · ${episode.success ? 'success' : 'no success'}`, String(index));
      return option;
    }));
    $('sandboxMessage').textContent = 'Validated recorded cursor movement · individual probe clocks not stored.';
    render();
  } catch (error) {
    $('sandboxMessage').textContent = String(error?.message ?? error);
  }
}
async function bootstrap() {
  $('sandboxConnection').textContent = 'Checking…';
  const [sandboxResult, recordingResult] = await Promise.allSettled([
    fetch('/api/v2/sandbox/status').then(async response => {if (!response.ok) throw Error('Sandbox status unavailable'); return response.json();}),
    fetch('/api/v2/recordings').then(async response => {if (!response.ok) throw Error('Recording catalog unavailable'); return response.json();}),
  ]);
  if (sandboxResult.status === 'fulfilled') {
    const sandbox = sandboxResult.value;
    $('sandboxConnection').textContent = sandbox.connection === 'not_configured' ? 'Not connected' : String(sandbox.connection).replaceAll('_', ' ');
    $('sandboxControlStatus').textContent = sandbox.guest_actions_enabled ? 'GUEST CONTROL ENABLED' : 'GUEST ACTIONS OFF';
  } else {
    $('sandboxConnection').textContent = 'Status unavailable';
    $('sandboxMessage').textContent = 'Cannot confirm a QEMU connection; guest actions unavailable.';
  }
  if (recordingResult.status === 'fulfilled') {
    const items = recordingResult.value.recordings.filter(item => item.available);
    $('recordingSelect').replaceChildren(...items.map(item => new Option(item.label, item.recording_id)));
    if (items.length) {
      await loadRecording(items[0].recording_id);
      // When the guest is offline, show the real recorded fallback immediately.
      // The disconnected QEMU status stays visible in the status ribbon.
      if (sandboxResult.status === 'fulfilled' &&
          !sandboxResult.value.vm_frame_available && state.recording) {
        setMode('replay');
      }
    } else {
      $('recordingSelect').replaceChildren(new Option('No validated recordings installed', ''));
      $('openReplay').disabled = true;
    }
  } else {
    $('recordingSelect').replaceChildren(new Option('Recording catalog unavailable', ''));
    $('openReplay').disabled = true;
  }
}
$('modeLive').onclick = () => setMode('guest');
$('modeReplay').onclick = () => setMode('replay');
$('openReplay').onclick = () => setMode('replay');
$('recordingSelect').onchange = event => {if (event.target.value) void loadRecording(event.target.value);};
$('episodeSelect').onchange = event => selectEpisode(Number(event.target.value));
$('stepScrub').oninput = event => {stopPlayback(); state.step = Number(event.target.value); render();};
$('replayPrev').onclick = () => {stopPlayback(); state.step = Math.max(0, state.step - 1); render();};
$('replayNext').onclick = () => {
  stopPlayback();
  const episode = state.recording?.episodes?.[state.episode];
  state.step = Math.min((episode?.steps?.length ?? 1) - 1, state.step + 1);
  render();
};
$('replayPlay').onclick = () => {
  if (state.playing) {stopPlayback(); return;}
  const episode = state.recording?.episodes?.[state.episode];
  if (!episode?.steps?.length) return;
  if (state.step >= episode.steps.length - 1) state.step = 0;
  state.playing = true;
  $('replayPlay').textContent = 'Pause';
  $('replayPlay').setAttribute('aria-label', 'Pause recorded steps');
  state.timer = setInterval(() => {
    if (state.step >= episode.steps.length - 1) {stopPlayback(); return;}
    state.step++;
    render();
    if (state.step >= episode.steps.length - 1) stopPlayback();
  }, 600); // UI playback cadence only; not a recovered neural or experiment clock.
};
$('sandboxFit').onclick = () => {if (state.mode === 'replay') render();};
$('sandboxFullscreen').onclick = async () => {
  const stage = $('sandboxViewport');
  if (document.fullscreenElement) await document.exitFullscreen();
  else if (stage.requestFullscreen) await stage.requestFullscreen();
  render();
};
$('openProtocol').onclick = () => {
  const drawer = $('protocol');
  drawer.open = !drawer.open;
  $('openProtocol').setAttribute('aria-expanded', String(drawer.open));
  if (drawer.open) drawer.scrollIntoView({behavior:'smooth',block:'nearest'});
};
$('protocol').addEventListener('toggle', event => {
  $('openProtocol').setAttribute('aria-expanded', String(event.target.open));
});
initPathwayInspector();
addEventListener('resize', () => {if (state.mode === 'replay') render();});
document.addEventListener('visibilitychange', () => {if (document.hidden) stopPlayback();});
window.addEventListener('flybrain:live-neural-sample', event => {
  state.liveSamples = appendNeuralSample(state.liveSamples, event.detail);
  if (state.mode === 'guest') renderLiveTimeline();
});
$('sandboxViewport').addEventListener('keydown', event => {
  if (event.target !== $('sandboxViewport')) return;
  if (event.code === 'Space' && state.mode === 'replay') {
    event.preventDefault();
    $('replayPlay').click();
  } else if (event.key === 'Escape' && document.fullscreenElement) {
    event.preventDefault();
    void document.exitFullscreen();
  }
});
void bootstrap();
