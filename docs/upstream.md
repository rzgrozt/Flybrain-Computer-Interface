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
