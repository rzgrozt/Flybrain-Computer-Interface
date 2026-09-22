import assert from 'node:assert/strict';
import {appendPathwaySample, MAX_PATHWAY_SAMPLES} from '../../src/flybrain_interface/panel/static/pathway-state.mjs';

const selected = {
  target_neuron_index: 131957, path_index: 0, neuron_indices: [78761, 35, 412, 131957],
};
const make = (chunk, changes = {}) => ({
  kind: 'telemetry',
  experiment_id: 'run-one',
  chunks: chunk,
  pathway_measurement: {
    source: 'simulated_neural_measurement',
    target_neuron_index: selected.target_neuron_index,
    path_index: selected.path_index,
    neuron_indices: [...selected.neuron_indices],
    sampled_chunk: chunk,
    simulated_time_s: chunk * .02,
    bin_duration_s: .02,
    voltage_mv: [-55, -54, -52, -51],
    synaptic_drive_mv: [.001, .002, .003, .004],
    spike_counts: [0, 1, 2, 0],
    ...changes,
  },
});
let history = appendPathwaySample([], make(1), selected);
assert.equal(history.length, 1);
assert.equal(history[0].population_voltage_mv, -53);
assert.equal(history[0].population_spike_rate_hz, 37.5);
assert.deepEqual(history[0].spike_counts, [0, 1, 2, 0]);
assert.equal(appendPathwaySample(history, make(1), selected), history, 'deduplicate chunk');
assert.equal(appendPathwaySample(history, make(2, {path_index: 1}), selected), history);
assert.equal(appendPathwaySample(history, {...make(2), kind:'heartbeat'}, selected), history);
assert.equal(appendPathwaySample(history, make(2, {spike_counts: null}), selected), history);
assert.equal(appendPathwaySample(history, make(2, {voltage_mv: [0, NaN, 0, 0]}), selected), history);
assert.equal(appendPathwaySample(history, make(2, {neuron_indices: [78761, 35, 412]}), selected), history);
assert.equal(appendPathwaySample(history, make(2, {bin_duration_s: Infinity}), selected), history);
for (let index = 2; index <= 100; index++) history = appendPathwaySample(history, make(index), selected);
assert.equal(history.length, MAX_PATHWAY_SAMPLES);
assert.equal(history[0].chunk, 41);
assert.equal(history.at(-1).chunk, 100);
const restarted = appendPathwaySample(history, {...make(1), experiment_id: 'new-run'}, selected);
assert.equal(restarted.length, 1);
assert.equal(restarted[0].session_id, 'new-run');
assert.equal(appendPathwaySample(restarted, {kind:'status'}, selected), restarted);
assert.equal(appendPathwaySample(restarted, make(2), null), restarted);
console.log('PASS bounded pathway sample state, provenance and session reset');
