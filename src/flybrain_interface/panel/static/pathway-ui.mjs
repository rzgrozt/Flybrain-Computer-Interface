import {appendPathwaySample} from './pathway-state.mjs';

const $ = id => document.getElementById(id);
const ACTIVE = new Set(['starting', 'running', 'pausing', 'paused', 'resuming']);
const display = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';

export function initPathwayInspector() {
  const state = {
    target: null, selection: null, route: null, token: 0, samples: [],
    simulator: 'idle', session: null, disconnected: false,
  };
  const setSource = (text, kind) => {
    $('pathwayLiveSource').textContent = text;
    $('pathwayLiveSource').className = 'source-pill ' + kind;
  };
  const setStatus = text => { $('pathwayMeasurementStatus').textContent = text; };
  const clearMeasurements = () => {
    state.samples = [];
    $('pathwayMeanVoltage').textContent = '—';
    $('pathwayMeanDrive').textContent = '—';
    $('pathwaySpikes').textContent = '—';
    $('pathwayNeuronSamples').replaceChildren();
    for (const id of ['pathwayVoltageTrace', 'pathwayDriveTrace', 'pathwaySpikeTrace']) {
      const canvas = $(id);
      const context = canvas.getContext('2d');
      if (context) context.clearRect(0, 0, canvas.width, canvas.height);
    }
    setSource('NOT OBSERVED', 'unavailable');
  };
  const publishSelection = (selection, overlay) => {
    window.flybrainSelectedPathway = selection;
    window.flybrainPathwayOverlay = overlay;
    window.dispatchEvent(new CustomEvent('flybrain:pathway-selection', {detail: selection}));
    window.dispatchEvent(new CustomEvent('flybrain:pathway-overlay', {detail: overlay}));
  };

  function drawTrace(id, key, color) {
    const canvas = $(id);
    const width = Math.max(30, canvas.clientWidth || 100);
    const height = Math.max(20, canvas.clientHeight || 55);
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, width, height);
    const samples = state.samples;
    if (!samples.length) return;
    const values = samples.map(sample => sample[key]);
    let minimum = Math.min(...values);
    let maximum = Math.max(...values);
    const spread = Math.max(0.001, maximum - minimum);
    minimum -= spread * .12;
    maximum += spread * .12;
    const x = index => 6 + index * (width - 12) / Math.max(1, samples.length - 1);
    const y = value => height - 6 - (value - minimum) / (maximum - minimum) * (height - 12);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    samples.forEach((sample, index) => {
      // Never draw an unobserved interval as if adjacent frames were continuous.
      if (index === 0 || sample.chunk !== samples[index - 1].chunk + 1 ||
          sample.session_id !== samples[index - 1].session_id) ctx.moveTo(x(index), y(values[index]));
      else ctx.lineTo(x(index), y(values[index]));
    });
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x(samples.length - 1), y(values.at(-1)), 2.5, 0, 2 * Math.PI);
    ctx.fill();
  }

  function showSample() {
    const sample = state.samples.at(-1);
    if (!sample || !state.route) return;
    const running = state.simulator === 'running';
    const paused = state.simulator === 'paused';
    setSource(running && !state.disconnected ? 'LIVE SIMULATED SAMPLE' : 'LAST SIMULATED SAMPLE',
      state.disconnected ? 'unavailable' : 'measured');
    setStatus((state.disconnected ? 'Disconnected · ' : paused ? 'Paused · ' : '') +
      'Recorded neural state at simulation t=' + display(sample.simulated_time_s) +
      ' s · chunk ' + sample.chunk + ' · ' + display(sample.bin_duration_s * 1000, 1) +
      ' ms bin · ' + state.samples.length + '/60 retained samples. Missing frames are not interpolated.');
    $('pathwayMeanVoltage').textContent = display(sample.population_voltage_mv) + ' mV';
    $('pathwayMeanDrive').textContent = display(sample.population_drive_mv) + ' mV';
    $('pathwaySpikes').textContent = String(sample.spike_counts.reduce((sum, value) => sum + value, 0));
    const tiles = state.route.neurons.map((neuron, index) => {
      const tile = document.createElement('div');
      tile.className = 'pathway-sample';
      const name = document.createElement('strong');
      name.textContent = (neuron.instance || neuron.type || 'Neuron') + ' · #' + neuron.neuron_index;
      const metrics = document.createElement('span');
      metrics.textContent = 'V ' + display(sample.voltage_mv[index]) + ' mV\n' +
        'Drive ' + display(sample.drive_mv[index], 5) + ' mV\n' +
        'Spikes ' + sample.spike_counts[index] + ' / bin';
      metrics.style.whiteSpace = 'pre-line';
      tile.append(name, metrics);
      return tile;
    });
    $('pathwayNeuronSamples').replaceChildren(...tiles);
    drawTrace('pathwayVoltageTrace', 'population_voltage_mv', '#50d6db');
    drawTrace('pathwayDriveTrace', 'population_drive_mv', '#aa9eff');
    drawTrace('pathwaySpikeTrace', 'population_spike_rate_hz', '#ffbd7c');
  }

  function renderRoute() {
    const route = state.route;
    if (!route) {
      $('pathwayFlow').replaceChildren();
      return;
    }
    const elements = [];
    route.neurons.forEach((neuron, index) => {
      if (index) {
        const edge = document.createElement('div');
        edge.className = 'pathway-edge';
        const arrow = document.createElement('b');
        arrow.textContent = '→';
        const contacts = document.createElement('span');
        contacts.textContent = route.synapse_counts[index - 1] + ' contacts';
        edge.append(arrow, contacts);
        elements.push(edge);
      }
      const tile = document.createElement('div');
      tile.className = 'pathway-node' + (index === route.neurons.length - 1 ? ' target' : '');
      const hop = document.createElement('small');
      hop.textContent = index === 0 ? 'UPSTREAM VISUAL' :
        index === route.neurons.length - 1 ? 'DESCENDING TARGET' : 'RELAY · HOP ' + index;
      const label = document.createElement('strong');
      label.textContent = neuron.instance || neuron.type || 'Unnamed neuron';
      const details = document.createElement('span');
      details.textContent = '#' + neuron.neuron_index + ' · ' +
        (neuron.transmitter || 'unknown transmitter') + ' · body ' + neuron.body_id;
      tile.append(hop, label, details);
      elements.push(tile);
    });
    $('pathwayFlow').replaceChildren(...elements);
  }

  async function queueObservation(selection) {
    if (!ACTIVE.has(state.simulator)) {
      setStatus('Route selected and armed for the next live simulation. Start an experiment using the existing protocol controls.');
      return;
    }
    try {
      const response = await fetch('/api/v2/pathways/observe', {
        method: 'POST', headers: {'content-type': 'application/json'},
        body: JSON.stringify({observation: selection && {
          target_neuron_index: selection.target_neuron_index,
          path_index: selection.path_index,
        }}),
      });
      const payload = await response.json();
      if (!response.ok) throw Error(payload.detail || 'Cannot attach pathway observation.');
      if (state.selection === selection) {
        setStatus('Observation queued for the next simulated chunk; no measurement has been received yet.');
      }
    } catch (error) {
      if (state.selection === selection) {
        setStatus('Cannot attach live observation: ' + String(error.message || error));
      }
    }
  }

  async function selectRoute(index) {
    const target = state.target;
    if (!target || !Number.isInteger(index) || !target.paths[index]) return;
    const token = ++state.token;
    state.route = target.paths[index];
    const neurons = state.route.neurons;
    const selection = {
      target_neuron_index: target.target_neuron_index,
      path_index: index,
      neuron_indices: neurons.map(item => item.neuron_index),
    };
    state.selection = selection;
    clearMeasurements();
    renderRoute();
    $('pathwayObserve').disabled = false;
    $('pathwayClear').disabled = false;
    setStatus('Verified ' + (neurons.length - 1) +
      '-hop route · measuring only if the worker is configured to watch these neural indices.');
    publishSelection(selection, null); // Drop the previous overlay while coordinates resolve.
    if (ACTIVE.has(state.simulator)) void queueObservation(selection);
    try {
      const response = await fetch('/api/anatomy/map', {
        method: 'POST', headers: {'content-type': 'application/json'},
        body: JSON.stringify({neuron_indices: selection.neuron_indices}),
      });
      if (!response.ok) throw Error('Atlas coordinates unavailable (HTTP ' + response.status + ')');
      const mapped = await response.json();
      if (token !== state.token) return;
      const byIndex = new Map(mapped.entries.map(entry => [entry.neuron_index, entry]));
      const overlay = {
        nodes: neurons.map(neuron => byIndex.get(neuron.neuron_index)).filter(Boolean),
        edges: state.route.synapse_counts.map((count, step) => ({
          source_neuron_index: neurons[step].neuron_index,
          target_neuron_index: neurons[step + 1].neuron_index,
          synapse_count: count,
        })),
      };
      window.flybrainPathwayOverlay = overlay;
      window.dispatchEvent(new CustomEvent('flybrain:pathway-overlay', {detail: overlay}));
      $('pathwayStatus').textContent = target.target_instance + ' · ' + (neurons.length - 1) +
        ' directed hops · ' + mapped.visible_soma_count + '/' + neurons.length +
        ' actual measured somata highlighted. ' + mapped.without_visible_soma_count +
        ' lack a displayed brain-atlas soma. Edges are anatomy only.';
    } catch (error) {
      if (token === state.token) $('pathwayStatus').textContent =
        'Verified indices selected; Brain View mapping failed: ' + String(error.message || error);
    }
  }

  async function selectTarget(index) {
    const previous = state.selection;
    ++state.token;
    state.route = null;
    state.target = null;
    state.selection = null;
    clearMeasurements();
    publishSelection(null, null);
    renderRoute();
    $('pathwayRoute').disabled = true;
    $('pathwayObserve').disabled = true;
    $('pathwayClear').disabled = true;
    if (!Number.isInteger(index)) {
      if (previous && ACTIVE.has(state.simulator)) void queueObservation(null);
      return;
    }
    const token = state.token;
    $('pathwayStatus').textContent = 'Loading validated route endpoints…';
    try {
      const response = await fetch('/api/v2/pathways/' + index);
      if (!response.ok) throw Error('Verified target unavailable (HTTP ' + response.status + ')');
      const target = await response.json();
      if (token !== state.token) return;
      if (target.schema_version !== 2 || target.source !== 'verified_connectome_anatomy' ||
          target.target_neuron_index !== index || !Array.isArray(target.paths) ||
          target.paths.length > 12) throw Error('Unsupported anatomical pathway contract');
      state.target = target;
      $('pathwayRoute').replaceChildren(...target.paths.map((path, routeIndex) =>
        new Option('Route ' + (routeIndex + 1) + ' · ' + (path.neurons.length - 1) +
          ' hops · contacts ' + path.synapse_counts.join('/'), String(routeIndex))));
      $('pathwayRoute').disabled = !target.paths.length;
      if (target.paths.length) await selectRoute(0);
      else $('pathwayStatus').textContent = 'Target has no verified two- or three-hop routes.';
    } catch (error) {
      if (token === state.token) {
        $('pathwayStatus').textContent = String(error.message || error);
        if (previous && ACTIVE.has(state.simulator)) void queueObservation(null);
      }
    }
  }

  $('pathwayTarget').onchange = event => {
    const value = event.target.value;
    void selectTarget(value === '' ? NaN : Number(value));
  };
  $('pathwayRoute').onchange = event => { void selectRoute(Number(event.target.value)); };
  $('pathwayExplore').onclick = () => {
    $('brainViewport').scrollIntoView({behavior: 'smooth', block: 'center'});
  };
  $('pathwayObserve').onclick = () => {
    if (state.selection) void queueObservation(state.selection);
  };
  $('pathwayClear').onclick = () => {
    ++state.token;
    if (ACTIVE.has(state.simulator)) void queueObservation(null);
    state.target = null;
    state.route = null;
    state.selection = null;
    $('pathwayTarget').value = '';
    $('pathwayRoute').replaceChildren(new Option('Select a target', ''));
    $('pathwayRoute').disabled = true;
    $('pathwayObserve').disabled = true;
    $('pathwayClear').disabled = true;
    clearMeasurements();
    publishSelection(null, null);
    renderRoute();
    $('pathwayStatus').textContent = 'Pathway highlight cleared.';
    setStatus('No route currently observed.');
  };
  window.addEventListener('flybrain:simulation-status', event => {
    if (state.session && event.detail.experiment_id &&
        state.session !== event.detail.experiment_id) clearMeasurements();
    state.simulator = event.detail.status;
    state.session = event.detail.experiment_id;
    if (['idle', 'resetting', 'failed', 'stopped'].includes(state.simulator) && state.samples.length) {
      setSource('LAST SIMULATED SAMPLE', 'unavailable');
      setStatus('Simulation no longer active · last sample retained as history, not live.');
    }
    if (state.simulator === 'paused' && state.samples.length) showSample();
  });
  window.addEventListener('flybrain:neural-disconnected', () => {
    state.disconnected = true;
    if (state.samples.length) showSample();
    else setSource('LIVE FEED DISCONNECTED', 'unavailable');
  });
  window.addEventListener('flybrain:live-neural-sample', event => {
    state.disconnected = false;
    const payload = event.detail;
    if (!state.selection) return;
    const next = appendPathwaySample(state.samples, payload, state.selection);
    if (next !== state.samples) {
      state.samples = next;
      showSample();
    }
  });
  window.addEventListener('resize', () => { if (state.samples.length) showSample(); });

  void (async () => {
    try {
      const [catalogResponse, statusResponse] = await Promise.all([
        fetch('/api/v2/pathways'), fetch('/api/status'),
      ]);
      if (statusResponse.ok) {
        const status = await statusResponse.json();
        state.simulator = status.status;
        state.session = status.experiment_id;
      }
      if (!catalogResponse.ok) throw Error('Verified pathway catalog unavailable');
      const catalog = await catalogResponse.json();
      if (catalog.schema_version !== 2 || !catalog.available || !Array.isArray(catalog.targets)) {
        throw Error('No verified anatomical pathway catalog is installed');
      }
      $('pathwayTarget').replaceChildren(
        new Option('Select descending neuron', ''),
        ...catalog.targets.map(target => new Option(
          target.target_instance + ' · graph #' + target.target_neuron_index +
          ' · body ' + target.target_body_id, String(target.target_neuron_index),
        )),
      );
      $('pathwayTarget').disabled = false;
      if (catalog.targets.length) {
        const first = catalog.targets[0].target_neuron_index;
        $('pathwayTarget').value = String(first);
        await selectTarget(first);
      } else $('pathwayStatus').textContent = 'No verified descending targets available.';
    } catch (error) {
      $('pathwayTarget').replaceChildren(new Option('Pathway catalog unavailable', ''));
      $('pathwayStatus').textContent = String(error.message || error);
    }
  })();
}
