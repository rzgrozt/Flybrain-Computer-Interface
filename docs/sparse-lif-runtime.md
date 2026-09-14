# Sparse LIF runtime

The runtime implements the same baseline equations and default parameters as the
Brian2 reference adapter:

```text
dv/dt = (v_rest - v + g) / tau_membrane
dg/dt = -g / tau_synapse
```

The coupled linear state update is evaluated analytically at each 0.1 ms grid
point. It is not an Euler approximation. The default backend executes that exact
update in a serial Numba kernel with `fastmath=False`; a NumPy implementation is
retained as an independent fallback. Recurrent spikes are held in a sparse 18-step
ring and delivered through the source-major graph after the configured 1.8 ms
delay.

## Scheduling semantics

Each grid point follows the Brian2 reference schedule:

1. analytically update non-refractory membrane and synaptic state;
2. identify threshold crossings;
3. deliver recurrent events due at this grid point;
4. apply deterministic artificial stimulation;
5. reset neurons that crossed threshold and schedule their outgoing events.

Variables marked `unless refractory` are frozen, including suppression of incoming
synaptic writes during the refractory interval. Integration and input reception
resume exactly at `spike_time + refractory_period`. This boundary is covered by a
dedicated regression test.

The delay ring stores only arrays of spiking source indices. It does not allocate a
dense neuron-by-delay matrix. The caller can request voltage and synaptic-drive
traces for a watchlist without recording all 166,700 neurons.

## Continuous stepping and bounded recording

`advance_step()` and `advance_chunk()` preserve membrane voltage, synaptic drive,
refractory counters, the recurrent delay ring, and absolute simulation time. A
chunk's artificial stimulus times are relative to the start of that chunk, while
sample and recorded spike times in `ChunkResult` are absolute since the last
explicit `reset()`.

The default chunk recorder returns total spikes and configured population rates but
keeps no per-neuron counts, spike events, or voltage history. `ChunkRecording` can
request a fixed watchlist, one chunk of per-neuron counts, and at most a specified
number of spike events. Excess events contribute to totals and
`dropped_spike_events` but are not retained. Runtime memory therefore scales with
the fixed neural state, delay ring, chunk length times watchlist size, and the
explicit event cap; the simulator does not accumulate completed chunk history.

`run()` remains the finite-duration compatibility path used for Brian2 validation:
it resets first and returns complete per-neuron spike histories and requested
watchlist traces.

## Validation

Run:

```bash
uv run python -m flybrain_interface.experiments.validate_runtime
```

The harness deterministically selects three 64-neuron induced MaleCNS neighborhoods
around distinct strong excitatory edges. It runs two input cases through Brian2 and
both NumPy and Numba runtime backends, requiring agreement in:

- per-neuron spike counts;
- spike times to 1e-12 seconds;
- membrane trajectories to 1e-10 mV;
- synaptic-drive trajectories to 1e-10 mV.

It also performs a ten-step, no-stimulus smoke run with the complete 166,700-neuron
state vector. This is a construction and state-update check, not a claim of
real-time performance.

For performance measurement, use:

```bash
uv run python -m flybrain_interface.experiments.benchmark_runtime
```

The benchmark compares NumPy and Numba backends and records host/runtime and
dataset/configuration provenance, first and warm preparation time, repeated
minimum/median/maximum timings, per-phase costs, loading and construction time,
peak process memory, spikes, visited edges, and simulated-time/wall-time ratio. It
uses increasing quiet durations to expose fixed overhead and bounded active runs to
avoid an uncontrolled activity cascade. Its dense-input case is explicitly an
engineering stress test, not a physiological activity model.

On the development host, the summary-only full-network measurement reduced the
100-step quiet median from 0.464 to 0.188 ms/step and the sparse-input median from
0.730 to 0.225 ms/step. The corresponding simulated/wall ratios were 0.533 and
0.444, so neither case is real time yet. Spike and visited-edge counts were
identical across all repeats and backends. These are host-specific engineering
measurements, not biological-performance claims.

## Current limitation

The membrane and conductance update still scans the complete neuron state each
0.1 ms step. Numba makes that scan cheaper, but the current implementation remains
slower than real time on the development host. Under dense engineering load,
source-edge propagation becomes the dominant measured phase. Optimization must
continue to preserve the NumPy and Brian2 equivalence gates.
