# Sparse LIF runtime

The runtime implements the same baseline equations and default parameters as the
Brian2 reference adapter:

```text
dv/dt = (v_rest - v + g) / tau_membrane
dg/dt = -g / tau_synapse
```

The coupled linear state update is evaluated analytically at each 0.1 ms grid
point. It is not an Euler approximation. Recurrent spikes are held in a sparse
18-step ring and delivered through the source-major graph after the configured
1.8 ms delay.

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

## Validation

Run:

```bash
uv run python -m flybrain_interface.experiments.validate_runtime
```

The harness deterministically selects a 64-neuron induced MaleCNS neighborhood
around a strong excitatory edge. It runs identical stimulation through Brian2 and
the sparse runtime, requiring agreement in:

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

The benchmark records host/runtime and dataset/configuration provenance, repeated
minimum/median/maximum timings, loading and construction time, peak process memory,
spikes, visited edges, and simulated-time/wall-time ratio. It uses increasing quiet
durations to expose fixed overhead and bounded active runs to avoid an uncontrolled
activity cascade. Its dense-input case is explicitly an engineering stress test,
not a physiological activity model.

## Current limitation

Synaptic propagation is event-driven, but membrane and conductance decay still scan
the complete neuron state each 0.1 ms step. Initial profiling shows that this dense
state update, not graph traversal, is now the dominant full-network cost. The next
optimization should compile and benchmark the state kernel while retaining this
Brian2 trace suite as the semantic gate.
