# Local experiment observation panel

The panel is a local laboratory surface for declaring a bounded artificial
stimulus, controlling a MaleCNS simulation, and inspecting bounded telemetry. It
is not a desktop interface for the connectome to learn to operate. It contains
no external action-selection policy and exposes no mouse, keyboard, arbitrary
code, model-inference, plasticity, or reward controls.

## Architecture and safety boundary

The FastAPI process owns the HTTP/WebSocket presentation layer. A spawned,
isolated process owns the memory-mapped graph, NumPy/Numba LIF simulator, delayed
event ring, and experiment manifest. The neural loop advances independently in
bounded chunks and checks a 16-entry command queue only between chunks.

Worker telemetry uses a one-entry multiprocessing queue. The server drains its
latest value and gives each WebSocket a one-entry queue. Obsolete telemetry is
replaced, not accumulated, and neither the simulation worker nor server waits for
a browser. The UI exposes cumulative worker-IPC and browser-frame replacement
counts so backpressure remains measurable.

The explicit states are `idle`, `running`, `paused`, `completed`, `failed`, and
`stopped`. Pause takes effect at a chunk boundary. It does not reset the
simulator, so membrane voltage, synaptic drive, refractory counters, delayed
events, and neural-grid time are preserved for resume. Reset closes the current
manifest and discards that in-memory experiment state. Server shutdown asks the
worker to stop, waits up to five seconds, and terminates it only as a fallback.

## Install and launch

Use the existing normalized data in place; do not copy it into the worktree:

```bash
uv sync --all-groups
uv run flybrain-panel \
  --data-directory /home/ruzgar/Flybrain-Computer-Interface/data/processed/malecns-v1.0
```

Open <http://127.0.0.1:8000>. The first release accepts loopback hosts only. To
select another loopback port and ignored local output directory:

```bash
uv run flybrain-panel \
  --data-directory /home/ruzgar/Flybrain-Computer-Interface/data/processed/malecns-v1.0 \
  --output-directory runs/panel \
  --host 127.0.0.1 \
  --port 8765
```

The dependency-free browser UI is served by FastAPI; no Node.js or separate
frontend build is required.

## Experiment definition

The UI accepts explicit neural indices for the artificial stimulus, engineering
observation population, and voltage/drive watchlist. These assignments do not
claim biological function. Inputs are validated against the loaded graph in the
worker. Limits are deliberately small:

- experiment duration: 0.01–60 simulated seconds;
- chunk duration: 1–250 ms and no longer than the run;
- telemetry: 1–30 Hz;
- artificial stimulus targets: at most 256 unique neuron indices;
- watchlist: at most 32 unique neuron indices;
- observation populations: at most 16, each with at most 4,096 indices;
- stimulus amplitude: greater than 0 and at most 100 mV;
- all timing must align to the 0.1 ms neural grid.

The seed is recorded even though this deterministic first stimulus does not draw
random values. It reserves the reproducibility boundary for later stochastic
experimental components.

`preserve` is the default subnormal-drive policy and retains exact IEEE-754 decay.
The optional `zero` policy explicitly sets drive smaller than the normal float64
range to zero. That approximation can prevent severe long sparse-run slowdown,
but is not scientifically interchangeable with exact preservation. The choice
is visible in the UI and persisted in every manifest.

## Telemetry and output

The panel shows simulated time, elapsed and active wall time, simulation/wall
ratio, current worker RSS, pending delayed spikes, total and per-chunk activity,
population spike rate in Hz per neuron, selected-neuron voltage and drive in mV,
visited edges, neural-loop time, and dropped telemetry counts. Browser chart
history is bounded to the latest 120 received samples. The worker does not retain
spike-event history for panel runs.

Each run writes `runs/panel/<experiment-id>/manifest.json` by default. `runs/` is
ignored. The manifest is atomically updated and contains normalized dataset
identity and available hashes, sign policy, all Shiu LIF parameters, seed,
artificial stimulus, observation populations, watchlist, telemetry limits,
subnormal policy and approximation statement, software version, final state, and
summary results. The normalized dataset remains read-only and is never copied or
mutated.

## Validation and measured overhead

Run the complete equivalence and panel suite:

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

Reproduce the bounded four-condition timing comparison:

```bash
uv run python -m flybrain_interface.experiments.benchmark_panel \
  --data-directory /home/ruzgar/Flybrain-Computer-Interface/data/processed/malecns-v1.0 \
  --duration-s 0.1 \
  --repetitions 3
```

On 2026-09-14, three full-MaleCNS exact-policy repetitions measured median neural
loop times of 0.2357 s headless, 0.2440 s with the server relay and no browser,
0.2510 s with a draining client, and 0.2524 s with a deliberately stalled client.
Those are changes of 0%, +3.54%, +6.49%, and +7.10% relative to the headless
median. The stalled clients caused 18 browser-frame replacements and no worker
IPC drops. A separate 0.1 simulated-second API smoke run produced 603 spikes,
visited 216,733 edges, took 0.3886 s in the neural loop, and ended with zero
pending delayed events. These short-run measurements include normal system noise
and do not establish real-time performance.
