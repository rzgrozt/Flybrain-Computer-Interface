# FlyBrain Computer Interface

FlyBrain Computer Interface is a research project exploring whether the MaleCNS
v1.0 *Drosophila* connectome can act as an adaptive controller for a computer.

The architectural rule is simple: supporting software may translate sensory
information and provide reinforcement, but only connectome activity may select
computer actions.

## Current status

The first executable foundation is in place:

- target-by-source SciPy CSR connectivity for CPU-first sparse propagation;
- a small Brian2 reference backend using the published Shiu et al. LIF defaults;
- deterministic synthetic spike stimulation and a propagation benchmark;
- typed boundaries for sensory input, semantic goals, neural output, fixed motor
  decoding, scalar reward, and non-blocking telemetry;
- optional semantic and visual-support registries that are disabled by default.
- a verified MaleCNS v1.0 acquisition and memory-bounded normalization pipeline.
- a read-only full-graph loader with memory-mapped edges, annotation-based
  population selection, and explicit transmitter-sign policies.
- a source-major outgoing index and event-driven spike propagation validated
  against the full target-major matrix operation.
- a deterministic sparse-event LIF engine with delayed delivery, exact state
  updates, refractory handling, watchlist traces, and Brian2 equivalence tests.

The real MaleCNS graph is connected to the sparse LIF runtime. Brian2 remains the
transparent reference for induced-subgraph validation rather than full-network
execution. No plasticity has been implemented and no desktop-control API is
connected.

## Development setup

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

Run the local checks with:

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

Run the deterministic reference benchmark with:

```bash
uv run python -m flybrain_interface.experiments.synthetic_reference --runs 3
```

The command prints measured runtime, per-neuron spikes, population rates, and the
output of the fixed artificial motor decoder. It performs no operating-system action.

## MaleCNS data pipeline

The official CC-BY MaleCNS v1.0 annotations, neurotransmitter predictions, and
connection weights are locked by generation, size, GCS checksums, and SHA-256 in the
tracked source manifest. Raw and normalized data remain ignored by Git.

```bash
uv run flybrain-data download
uv run flybrain-data build --memory-limit 4GB --threads 4
uv run flybrain-data validate
```

The normalized local dataset contains 166,700 classified neurons, 25,582,938
directed edges, and 124,177,617 synaptic contacts in Parquet plus target-major and
source-major memory-mappable sparse arrays. See
[`docs/data-pipeline.md`](docs/data-pipeline.md) for the selection rule, output
schema, and limitations.

Inspect and benchmark that full graph with:

```bash
uv run python -m flybrain_interface.experiments.full_graph
```

This maps both sparse orientations read-only, reports transmitter/sign populations,
and compares full-matrix propagation with an outgoing-edge spike event. See
[`docs/runtime-graph.md`](docs/runtime-graph.md) for the assumptions and API.

Validate the neural runtime against Brian2 on a deterministic real-data subgraph:

```bash
uv run python -m flybrain_interface.experiments.validate_runtime
```

The command compares spike counts, spike times, membrane voltage, and synaptic
drive across multiple real-data neighborhoods, input cases, and both runtime
backends. It then runs a summary-only full-network runtime smoke benchmark. See
[`docs/sparse-lif-runtime.md`](docs/sparse-lif-runtime.md).

Run the reproducible full-network performance benchmark with:

```bash
uv run python -m flybrain_interface.experiments.benchmark_runtime
```

It compares the exact NumPy fallback with the default serial Numba state kernel in
quiet, deterministic sparse-input, and explicitly non-physiological
engineering-stress scenarios. The simulated-to-wall ratio is defined as simulated
seconds divided by wall seconds; values below 1 are slower than real time.

Run bounded, chunked full-graph stability measurements with:

```bash
uv run python -m flybrain_interface.experiments.stability_runtime \
  --output artifacts/benchmarks/stability.json
```

This records memory residency, finite state, population activity, delayed-event
backlog, and latency distributions under explicit wall-time, RSS, spike, and edge
budgets. See [`docs/stability-benchmark-2026-09-14.md`](docs/stability-benchmark-2026-09-14.md)
for the latest host-specific results and limitations.

The exact runtime preserves IEEE float64 subnormals by default. For long sparse
runs, an explicit `subnormal_drive_policy="zero"` approximation avoids the measured
CPU subnormal-arithmetic cliff without changing global floating-point state. See
[`docs/subnormal-drive-policy-2026-09-14.md`](docs/subnormal-drive-policy-2026-09-14.md)
for the measured tradeoff and opt-in commands.

## Local experiment panel

A lightweight local FastAPI/WebSocket panel can start, pause, resume, and reset a
bounded MaleCNS experiment while showing compact population and watchlist
telemetry from an isolated simulation worker:

```bash
uv sync --all-groups
uv run flybrain-panel \
  --data-directory /home/ruzgar/Flybrain-Computer-Interface/data/processed/malecns-v1.0
```

Open <http://127.0.0.1:8000>. The panel binds to loopback only and writes manifests
under the ignored `runs/panel/` directory. It defaults to exact subnormal-drive
preservation; the optional zeroing approximation is always explicit and recorded.
See [`docs/experiment-panel.md`](docs/experiment-panel.md) for controls, limits,
architecture, reproducibility fields, validation, and measured UI overhead.

## Architectural boundaries

Semantic translators can return only structured goals. Optional visual evaluators
can return only scalar reward signals. Neither interface exposes motor commands or
cursor coordinates. The fixed motor decoder accepts only neural readout, preserving
the rule that external models cannot select actions.

Future LLM or VLM adapters can be registered behind these narrow interfaces without
editing the simulation backend. No generative model or provider SDK is installed by
default, and the base configuration keeps both optional support paths disabled.

## Data policy

Large downloaded and derived datasets are not stored in Git. Source identity,
versions, hashes, and normalization settings belong in `data/manifests/`.
Downloaded inputs belong in `data/raw/`, and normalized outputs belong in
`data/processed/`.

## Package boundaries

- `connectome_data`: acquisition, integrity checks, indexing, and normalization
- `simulation`: transparent reference and optimized runtime backends
- `sensory`: deterministic visual and semantic sensory transducers
- `motor`: fixed decoding of connectome outputs into actions
- `plasticity`: biologically constrained learning mechanisms
- `reward`: evaluators that emit reinforcement but never actions
- `environment`: isolated computer or training environments
- `telemetry`: non-blocking scientific observability
- `experiments`: reproducible experiment definitions and execution
- `validation`: reference/runtime comparisons and scientific controls
- `ai_support`: optional translation and evaluation fallbacks outside action selection

Reviewed upstream sources and scientific limitations are recorded in
[`docs/upstream.md`](docs/upstream.md).
