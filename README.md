# FlyBrain Computer Interface

FlyBrain Computer Interface is a research project exploring whether the MaleCNS
v1.0 *Drosophila* connectome can act as an adaptive controller for a computer.

The architectural rule is simple: supporting software may translate sensory
information and provide reinforcement, but only connectome activity may select
computer actions.

## Current status

The repository currently provides the project scaffold for Phase A: acquiring,
validating, indexing, and normalizing MaleCNS data. Simulation, sensory, motor,
plasticity, reward, and telemetry packages are explicit boundaries for later
phases; they do not yet contain behavioral implementations.

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
