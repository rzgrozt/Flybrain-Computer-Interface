# Subnormal synaptic-drive policy: 2026-09-14

## Outcome

The long-run latency cliff was caused by float64 subnormal arithmetic in the fused
Numba state-update kernel. The exact IEEE behavior remains the default. An explicit
opt-in `zero` policy prevents subnormal drive inputs and decay results without
changing process-wide floating-point state, enabling the previously budget-limited
sparse 10-second benchmark to complete in all three repeats.

The policy uses the float64 minimum-normal boundary, not a biological or
empirically tuned epsilon. With the default synaptic decay, a drive smaller than
`2.270023332107551e-308 mV` before an update is set to zero; otherwise its decayed
result remains at least the smallest normal float64 value,
`2.2250738585072014e-308 mV`.

## Causality diagnostic

The diagnostic ran 100 fused state updates over 166,700 neurons without graph
loading, spike propagation, or recording. Each case used seven repeats.

| Initial drive class | Exact median | `zero` median | Exact slowdown vs exact zero |
| --- | ---: | ---: | ---: |
| zero | 15.44 ms | 20.57 ms | 1.00 |
| ordinary normal (`1e-100`) | 15.46 ms | 20.60 ms | 1.00 |
| smallest normal | 910.00 ms | 20.56 ms | 58.9 |
| subnormal | 910.46 ms | 20.54 ms | 59.0 |

This isolates the slowdown from propagation and activity. The `zero` kernel pays a
guard cost for ordinary values, but avoids the approximately 59-fold exact-kernel
penalty once drives become subnormal. A separate compiled exact kernel keeps the
default path's implementation and ordinary-value performance unchanged.

Reproduce the diagnostic with:

```bash
uv run python -m flybrain_interface.experiments.diagnose_subnormal \
  --output artifacts/benchmarks/subnormal-diagnostic.json
```

## Full-graph before and after

Both runs used seed 0, three repeats, 10 ms chunks, the full 166,700-neuron /
25,582,938-edge graph, and the previously declared per-run limits: 120 wall
seconds, 2 GiB RSS, 50 million spikes, and 5 billion visited edges. The ratio is
simulated seconds divided by steady-state execution seconds.

| Scenario | Exact baseline | Opt-in `zero` policy | Outcome |
| --- | ---: | ---: | --- |
| no input, 100 ms | 0.239 s / 0.418 | 0.248 s / 0.403 | 3/3 complete |
| sparse input, 100 ms | 0.235 s / 0.426 | 0.247 s / 0.405 | 3/3 complete |
| no input, 1 s | 2.427 s / 0.412 | 2.446 s / 0.409 | 3/3 complete |
| sparse input, 1 s | 2.388 s / 0.419 | 2.449 s / 0.408 | 3/3 complete |
| no input, 10 s | 24.362 s / 0.411 | 24.524 s / 0.408 | 3/3 complete |
| sparse input, 10 s | stopped at 8.44–8.50 s | 24.738 s / 0.404 | 3/3 complete |
| denser engineering stress, 100 ms | 1.050 s / 0.095 | 0.826 s / 0.121 | 3/3 complete |

At the former onset window, representative sparse-run chunk latency changed from
27.7 ms at 3.5 seconds to 219.7 ms at 3.6 seconds under exact semantics. With the
`zero` policy it was 24.3 ms and 24.5 ms respectively, and stayed around 24–25 ms
through 10 seconds. The corrected sparse run's median chunk latency was 24.5 ms and
median p95 was 25.0 ms, versus 214.1 ms and 227.6 ms in the budget-stopped exact
runs.

All matched scenarios retained deterministic spike and propagation results. The
sparse case remained at 144 spikes, 41,773 visited edges, no pending events, and no
nonfinite state. Its activity still ended in the first 10 ms chunk. The denser
engineering stress case remained at 52,512 spikes, 17,757,885 visited edges, and
827 pending delayed events after 100 ms. Its measured performance did not regress,
although the improvement in this run should be treated as host/cache variability
rather than a propagation optimization.

Maximum sampled RSS was 410.4 MiB for the corrected sparse 10-second runs and
589.1 MiB across the suite, below the 2 GiB limit. Each sparse 10-second repeat grew
by approximately 0.45 MiB, consistent with the bounded scalar measurement history.

## Numerical and operational tradeoff

The policy is an explicit numerical approximation: exact subnormal drive values are
replaced with zero. The maximum omitted pre-update magnitude is below
`2.270023332107551e-308 mV`; the corresponding one-step voltage term is below
`1.121e-310 mV`, compared with an ulp near the resting voltage of approximately
`7.105e-15 mV`. The long-decay regression observed:

- exact and `zero` voltage arrays bit-identical;
- identical spike counts and threshold behavior;
- only the synaptic-drive value differing, by less than the float64 minimum-normal
  value;
- identical state under continuous and split-chunk execution for the `zero` policy.

These tests and magnitudes support the policy for this configuration, but do not
prove bitwise equivalence for every possible artificial state or future model.
Consequently `preserve` remains the default, and experiments must opt in explicitly:

```bash
uv run python -m flybrain_interface.experiments.stability_runtime \
  --data-directory /absolute/path/to/data/processed/malecns-v1.0 \
  --subnormal-drive-policy zero \
  --output artifacts/benchmarks/stability-subnormal-zero-full.json
```

No global FTZ/DAZ mode, `fastmath`, lower precision, timestep change, or arbitrary
epsilon was introduced. This avoids leaking thread/process floating-point mode into
Brian2, NumPy, or unrelated components. Both exact and `zero` policies passed the
12-case real-data Brian2/NumPy/Numba equivalence gate at the existing tolerances:
zero spike-time error, maximum voltage error `2.274e-13 mV`, and maximum drive error
`5.684e-14 mV`.

## Source basis

- [Intel FTZ/DAZ documentation](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-0/set-the-ftz-and-daz-flags.html)
  states that FTZ replaces denormal results with zero, DAZ treats denormal inputs as
  zero, and both are incompatible with strict IEEE behavior. It also notes that a
  process-level setting affects subsequently created threads.
- [LLVM Language Reference](https://llvm.org/docs/LangRef.html#denormal-fp-math)
  distinguishes the default `ieee,ieee` handling from modes that permit flushed
  results or zero-valued inputs.
- [NumPy floating-point error handling](https://numpy.org/doc/stable/reference/generated/numpy.seterr.html)
  controls reporting of underflow; it is not a mechanism for removing subnormal
  arithmetic or its CPU cost.

The complete local outputs remain ignored by Git:

- `artifacts/benchmarks/subnormal-diagnostic.json`
- `artifacts/benchmarks/stability-full.json` (exact baseline)
- `artifacts/benchmarks/stability-subnormal-zero-full.json` (opt-in policy)
