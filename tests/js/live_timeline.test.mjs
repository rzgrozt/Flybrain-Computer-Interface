import assert from 'node:assert/strict';
import test from 'node:test';
import {appendNeuralSample} from '../../src/flybrain_interface/panel/static/live-timeline.mjs';

const sample = (session, chunk, overrides = {}) => ({
  kind: 'telemetry', experiment_id: session, chunks: chunk,
  simulated_time_s: chunk * 0.02, chunk_spikes: chunk * 3, status: 'running',
  ...overrides,
});

test('bounded live history carries only observed neural snapshots', () => {
  let history = [];
  for (let index = 1; index <= 150; index++) {
    history = appendNeuralSample(history, sample('a', index));
  }
  assert.equal(history.length, 80);
  assert.equal(history.at(0).step_id, 71);
  assert.equal(history.at(-1).step_id, 150);
  assert.equal(history.at(-1).motor, null);
  assert.equal(history.at(-1).reward, null);
  assert.equal(history.at(-1).timestamp_utc, null);
  assert.equal(appendNeuralSample(history, sample('a', 150)), history);
});

test('status and heartbeat cannot become fabricated samples', () => {
  const history = appendNeuralSample([], sample('run', 1));
  assert.equal(appendNeuralSample(history, {...sample('run', 2), kind: 'heartbeat'}), history);
  assert.equal(appendNeuralSample(history, {kind: 'status', status: 'paused'}), history);
  assert.equal(appendNeuralSample(history, sample('run', 2, {simulated_time_s: NaN})).length, 1);
});

test('new sessions discard old neural samples rather than mixing clocks', () => {
  let history = appendNeuralSample([], sample('first', 1));
  history = appendNeuralSample(history, sample('second', 1));
  assert.equal(history.length, 1);
  assert.equal(history[0].session_id, 'second');
  assert.equal(history[0].step_id, 1);
});
