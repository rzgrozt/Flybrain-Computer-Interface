import assert from 'node:assert/strict';
import test from 'node:test';
import {activityIsStale,shouldClearActivity} from '../../src/flybrain_interface/panel/static/brain-state.mjs';

test('activity becomes stale only after a bounded telemetry gap',()=>{
  assert.equal(activityIsStale(0,10_000),false);
  assert.equal(activityIsStale(1_000,3_000),false);
  assert.equal(activityIsStale(1_000,3_001),true);
});

test('idle and disconnected states clear activity',()=>{
  for(const state of ['idle','offline','failed','stopped','disconnected']){
    assert.equal(shouldClearActivity(state),true);
  }
  assert.equal(shouldClearActivity('paused'),false);
  assert.equal(shouldClearActivity('running'),false);
});
