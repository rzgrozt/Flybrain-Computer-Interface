import assert from 'node:assert/strict';
import test from 'node:test';
import {clampNormalized, replayDirection, scoreRows, selectRecordedStep} from '../../src/flybrain_interface/panel/static/replay-state.mjs';

test('recorded episodes select bounded steps without synthesizing clocks', () => {
  const recording = {episodes: [{steps: [
    {simulated_time_s: null, motor_command: {dx: -0.1, dy: 0}},
    {simulated_time_s: null, motor_command: {dx: -0.1, dy: -0.1}},
  ]}]};
  const first = selectRecordedStep(recording, 0, -100);
  assert.equal(first.index, 0);
  assert.equal(first.step.simulated_time_s, null);
  const last = selectRecordedStep(recording, 80, 80);
  assert.equal(last.index, 1);
  assert.equal(last.total, 2);
  assert.equal(selectRecordedStep({episodes: []}, 0, 0), null);
});

test('virtual-stage points are normalized and motor direction is derived from recorded delta', () => {
  assert.equal(clampNormalized(-1), 0);
  assert.equal(clampNormalized(1.5), 1);
  assert.equal(clampNormalized(NaN), 0.5);
  assert.equal(replayDirection({motor_command: {dx: 0, dy: 0}}), null);
  assert.equal(replayDirection({motor_command: {dx: -1, dy: 0}}), 180);
  assert.equal(replayDirection({motor_command: {dx: 0, dy: 1}}), 90);
});

test('RBF scores remain signed scores, not probabilities', () => {
  const rows = scoreRows({
    scores_are_probabilities: false,
    scores: [1.05885, -0.2],
    score_labels: ['LEFT', 'RIGHT'],
  });
  assert.deepEqual(rows.map(row => row.label), ['LEFT', 'RIGHT']);
  assert.equal(rows[0].value, 1.05885);
  assert.equal(rows[1].value, -0.2);
  assert.equal(rows[0].magnitude, 1);
  assert.equal(scoreRows({scores_are_probabilities: true, scores: [0.9, 0.1], score_labels: ['LEFT', 'RIGHT']}).length, 0);
});
