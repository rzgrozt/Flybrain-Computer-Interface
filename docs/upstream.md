# Upstream foundations

The first reference backend uses maintained upstream software instead of replacing
neuroscience infrastructure.

## Brian2

- Package: `brian2`
- Role: transparent CPU reference simulator
- Version: resolved in `uv.lock`
- Source: <https://github.com/brian-team/brian2>
- Documentation: <https://brian2.readthedocs.io/>

## Shiu Drosophila brain model

- Repository: <https://github.com/philshiu/Drosophila_brain_model>
- Reviewed commit: `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`
- License: MIT, copyright Philip Shiu and Nico Spiller
- Role: reference equations and published parameter defaults

The upstream repository is cloned under ignored `external/` for inspection. It is
not vendored or published with this project. The local reference class re-expresses
the small set of documented equations and parameters needed for validation and
records where they came from.

## Current scientific boundary

Shiu's implementation targets FlyWire adult-brain data, not MaleCNS v1.0. The
synthetic benchmark in this repository therefore validates Brian2 integration,
sparse edge construction, deterministic stimulation, and module boundaries only.
It is not evidence of a working whole-brain or MaleCNS simulation.

## MaleCNS v1.0

- Official downloads: <https://male-cns.janelia.org/download/>
- Dataset: `male-cns:v1.0`
- Dataset UUID: `4b2087c0fbe046bfaf0d60bc970e3e5d`
- License: CC-BY-4.0
- Role: authoritative neuron annotations, neurotransmitter predictions, and
  segment-to-segment connection weights

The source files and normalized outputs are not committed. Their immutable metadata
and hashes are recorded in `data/manifests/`.

## fly-connectome-template

- Repository: <https://github.com/cobanov/fly-connectome-template>
- Reviewed and adapted commit: `38f55332055328d38c29e72474c4ad5b6876101f`
- Version at review: `0.1.0`
- License: custom Cobanov Template Attribution License 1.0, attribution required,
  source-available and not represented as OSI approved
- Role: measured-soma atlas export/manifest, rigid and uniform anatomical display
  transform, Three.js point-cloud approach, and exact body-ID activity boundary

The project does not copy the sample React application, example replay, environment,
or Flybody animation. It adapts the maintained `BrainScene` approach into a small
TypeScript module for the existing panel, adds incremental activity-buffer updates,
selected-group highlighting, and a bounded sparse-graph relationship overlay. The
upstream license, attribution, atlas notice, and third-party notices are preserved
under `third_party/fly-connectome-template/` and in the packaged web UI.
