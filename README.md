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

The real MaleCNS graph can now be prepared locally, but it is not yet connected to
the Brian2 backend. No plasticity has been implemented and no desktop-control API is
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
directed edges, and 124,177,617 synaptic contacts in both Parquet and target-major
memory-mappable CSR arrays. See [`docs/data-pipeline.md`](docs/data-pipeline.md) for
the selection rule, output schema, and limitations.

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
