import test from 'node:test';
import assert from 'node:assert/strict';
import {parseIndices} from '../../src/flybrain_interface/panel/static/indices.mjs';

test('empty target input is rejected instead of becoming neuron zero', () => {
  assert.throws(
    () => parseIndices('  ', {required: true, label: 'Target neurons'}),
    /Target neurons is required/,
  );
});

test('empty watchlist is allowed', () => {
  assert.deepEqual(parseIndices('', {label: 'Watchlist'}), []);
});

test('valid indices are parsed without discarding tokens', () => {
  assert.deepEqual(parseIndices('0, 12,900'), [0, 12, 900]);
});

test('blank, negative, decimal, and nonnumeric tokens are rejected', () => {
  for (const value of ['1,,2', '-1', '1.5', '2,fly']) {
    assert.throws(() => parseIndices(value, {label: 'Targets'}));
  }
});
