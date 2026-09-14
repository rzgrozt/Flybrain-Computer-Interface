# Full-graph stability benchmark: 2026-09-14

This measurement used the complete MaleCNS v1.0 graph (166,700 neurons and
25,582,938 directed edges), the default 0.1 ms Shiu-style LIF configuration, and
the fused serial Numba backend. The ratio convention is simulated seconds divided
by steady-state execution seconds; values below 1 are slower than real time.

## Results

Each ordinary scenario used three independent deterministic repeats. State was
advanced in fixed 10 ms chunks with summary-only bounded recording. Median values
are across repeats; chunk p95 is the median of the three per-run p95 values.

| Input | Requested simulated time | Completed | Median execution | Simulated/wall | Median / p95 chunk | Spikes | Visited edges | Maximum sampled RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 100 ms | 3/3 | 0.239 s | 0.418 | 23.6 / 26.6 ms | 0 | 0 | 376.6 MiB |
| sparse, 128 excitatory neurons once at t=0 | 100 ms | 3/3 | 0.235 s | 0.426 | 23.2 / 26.3 ms | 144 | 41,773 | 397.1 MiB |
| none | 1 s | 3/3 | 2.427 s | 0.412 | 23.6 / 28.6 ms | 0 | 0 | 397.2 MiB |
| sparse, 128 excitatory neurons once at t=0 | 1 s | 3/3 | 2.388 s | 0.419 | 23.1 / 27.0 ms | 144 | 41,773 | 397.4 MiB |
| none | 10 s | 3/3 | 24.362 s | 0.411 | 23.3 / 30.2 ms | 0 | 0 | 398.7 MiB |
| sparse, 128 excitatory neurons once at t=0 | 10 s | 0/3; stopped at 8.44–8.50 s | 116.911 s before stop | 0.0725 | 214.1 / 227.6 ms | 144 | 41,773 | 399.9 MiB |
| denser engineering stress, 4,096 excitatory neurons once at t=0 | 100 ms | 3/3 | 1.050 s | 0.0953 | 101.7 / 125.0 ms | 52,512 | 17,757,885 | 578.6 MiB |

The denser case is an engineering stress test, not a physiological activity
assumption. Its deterministic population totals were 31,292 excitatory, 16,127
inhibitory, and 5,093 separated-or-unknown-sign spikes per run. Activity persisted
through the 100 ms observation window: the final chunk contained 4,920 spikes and
827 emitted spikes remained in the delay ring. This observation alone is neither a
biological success nor evidence of pathology.

The sparse probe produced 144 spikes (128 excitatory and 16 inhibitory) in the
first 10 ms chunk, then no further spikes. Its delayed-event ring was empty by
100 ms. The silent no-input result demonstrates the configured resting-state
behavior, but silence is not by itself a sufficient correctness criterion.

## Budget-limited sparse 10-second result

The per-run budgets, fixed before extension from the 100 ms pilot, were 120 wall
seconds, 2 GiB RSS, 50 million spikes, and 5 billion visited edges. All sparse
10-second repeats reached the wall-time limit after only 8.44–8.50 simulated
seconds, so this scenario did **not** complete and must not be cited as a successful
10-second run.

Chunk latency first exceeded 100 ms at 3.53–3.54 simulated seconds and then settled
near 214 ms median, even though spiking and propagation had ended. At the stop,
synaptic-drive extrema were approximately ±1.24e-322 and state-update time accounted
for about 112 of 117 execution seconds in a representative repeat. The onset timing
and subnormal magnitudes are consistent with CPU handling of gradually underflowing
float64 conductance values. This is a measured engineering inference, not a change
to model semantics.

A narrowly scoped next experiment should compare an explicitly documented
near-zero conductance floor or hardware flush-to-zero mode against the current
NumPy and Brian2 equivalence gates. It must preserve spike timing, watched
trajectories within existing tolerances, delayed-event behavior, and the declared
model approximation before adoption.

## Memory and load observations

Initial data loading took 0.106 seconds, representative simulator construction
0.017 seconds, and Numba compilation/warmup 0.166 seconds. RSS rose from 149.1 MiB
before load to 313.7 MiB after load and 372.4 MiB after warmup. The process high
water reached 578.6 MiB, well under the 2 GiB budget.

The first sparse run faulted about 18.7 MiB of additional file-backed graph pages;
later cached sparse runs showed negligible file-backed growth. The first denser
stress run faulted about 177.8 MiB of additional file-backed pages while anonymous
RSS grew by about 1.0 MiB. Later stress repeats grew by about 1.6 MiB total. The
10-second no-input repeats grew by about 0.45 MiB each, attributable at least in
part to the benchmark's bounded scalar latency and memory samples rather than
neural histories. These figures distinguish residency from anonymous allocation;
RSS alone would incorrectly present mapped pages as a heap leak.

Under the denser engineering load, source-major synaptic propagation was the
largest measured execution phase (about 0.69 of 1.05 seconds in a representative
run), versus about 0.31 seconds for state updates. The next load optimization should
therefore be a compiled, source-major propagation kernel over the existing outgoing
arrays, validated against full-matrix propagation and both reference-equivalence
gates. No such kernel was implemented in this testing task.

## Reproduction and provenance

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src
uv run python -m flybrain_interface.experiments.validate_runtime \
  --data-directory /home/ruzgar/Flybrain-Computer-Interface/data/processed/malecns-v1.0
uv run python -m flybrain_interface.experiments.stability_runtime \
  --data-directory /home/ruzgar/Flybrain-Computer-Interface/data/processed/malecns-v1.0 \
  --lock data/manifests/malecns-v1.0-normalized-v2.json \
  --durations 0.1 1 10 --repeats 3 \
  --output artifacts/benchmarks/stability-full.json
```

The host was Linux 7.2.3 on an Intel Core i5-11400H (12 logical CPUs) with
15.4 GiB RAM, CPython 3.13.14, NumPy 2.3.5, Numba 0.67.0, Brian2 2.10.1, and SciPy
1.18.1. Seed 0 selected the fixed input populations. The normalized-manifest hash
was `45ad93d6f6d07906a19c1b1086f18e86c7ca3157c27f281232e0a8a8d710bdf5`; the
transmitter policy was `inhibitory-glutamate-modulators-separated-v1`. The full
machine-readable result is local at
`artifacts/benchmarks/stability-full.json` and remains intentionally ignored by Git.

A subsequent opt-in numerical policy removed the measured subnormal slowdown and
completed the sparse 10-second case without changing the exact default. See
[`subnormal-drive-policy-2026-09-14.md`](subnormal-drive-policy-2026-09-14.md).
