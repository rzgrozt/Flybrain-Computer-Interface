# Live anatomical Brain View

The Brain View is a laboratory visualization inside the existing local panel. It
renders measured MaleCNS v1.0 soma coordinates and applies bounded output from the
existing isolated simulation worker. It does not render neurites, synapse locations,
full membrane state, a biological recording, or a behavioral policy.

## Anatomy, identity, and coverage

The atlas is adapted from `cobanov/fly-connectome-template` at commit
`38f55332055328d38c29e72474c4ad5b6876101f`. Its three binary exports are pinned by
SHA-256 in the original manifest and verified server-side before a join is exposed.
The atlas contains 140,024 measured soma positions; the renderer displays the
124,289 classified optic, central, and descending somata. VNC-associated and
unclassified somata are intentionally omitted. Positions are 8 nm voxel coordinates
from the official MaleCNS source.

At load, the server validates unique atlas IDs, unique graph body IDs, contiguous
stable graph indices, dataset identity, asset hashes, and exact join coverage. Body
IDs cross JSON as decimal strings, so JavaScript number precision cannot alter them.
The packaged atlas uses uint32 IDs and the renderer indexes activity by the exact ID,
never by spatial proximity. Missing positions are not fabricated.

For the current normalized graph, the verified join reports:

- 124,289 displayed measured somata (74.56% of 166,700 graph neurons);
- 15,363 graph neurons with a measured VNC-associated or unclassified soma omitted
  from this brain view;
- 27,048 graph neurons without a measured soma in this atlas.

Native `(x, y, z)` becomes `(x, -y, -z)`, is centered on its bounding box, and uses
one uniform scale. Drag/orbit rotations are rigid. The XY button restores the native
projection. Axis-wise stretching is never used.

## Live signal and clocks

The experiment request explicitly selects at most 4,096 visualization indices. Each
simulation chunk aggregates emitted simulator spikes for only that bounded selection.
The latest telemetry frame contains nonzero selected counts only; no spike-event
history or global voltage vector is streamed. Brightness is:

`clamp((spikes / simulation_bin_seconds) / 50 Hz, 0, 1)`

The default bin is 20 ms of simulation time. This normalized value is display
brightness—not membrane voltage, synaptic current, fluorescence, or biological
measurement. The UI separately labels artificial stimulus targets, emitted simulated
spikes, watchlist voltage/drive, and the count of modeled delayed events. A windowed
frame does not establish exact within-bin event order. Replaced worker/browser frames
are counted, and no sequence is inferred across dropped frames.

The neural clock and wall clock remain separate in the panel. A simulation/wall ratio
below 1 means the simulator is slower than biological real time. Pausing the simulator
preserves its state; pausing rendering only stops WebGL draws. On idle, reset,
disconnect, or stale telemetry, activity is cleared rather than invented or replayed.
The view reports mean and maximum browser CPU time spent submitting WebGL draws; this
does not measure asynchronous GPU completion time.

## Selected graph overlay

The opt-in overlay reads the actual source-major sparse graph and returns the strongest
outgoing relationships from at most 64 requested seed indices. Controls cap the view
at 256 nodes and 2,048 edges and apply a minimum synaptic-contact count. Ties are
deterministic by stable source and target index. Missing or view-excluded coordinates
are counted and omitted.

Lines are directed graph relationships drawn as straight segments between soma
markers. They are not anatomical neurites, axon paths, or synapse positions. The
overlay does not render all 25.6 million edges and does not animate transmission.

## Build and verification

The committed static bundle keeps normal panel launch independent of Node.js. To
rebuild it reproducibly:

```bash
cd frontend/brain-view
npm ci
npm run build
```

Three.js and build-tool versions are locked in `frontend/brain-view/package-lock.json`.
The original custom license and relevant notices are preserved in the UI and under
`third_party/fly-connectome-template/`. The adaptation omits the upstream sample
React environment, replay data, and Flybody model.

Run Python and JavaScript checks from the repository root:

```bash
uv run pytest
uv run ruff check .
uv run mypy src
node --test tests/js/*.test.mjs
```

This is an observability surface only. The R1-R6, ORN_DA1, JO-A1 and descending
presets are annotation-grounded selections useful for inspection, not proven
functional mappings. Prior deterministic stimulation showed repeatability and
distinct overlapping descending patterns for selected olfactory/auditory conditions;
it did not establish biological behavior, the absence of a visual pathway, or the
time at which activity ceases.

## Short host measurements (2026-09-14)

The verified real-data atlas join took 0.073 s. A one-seed, 128-node/512-edge
neighborhood request returned 25 visible relationships in 0.2 ms. In the browser,
the 124,289-point view reported 0.17 ms mean and 14.10 ms maximum CPU WebGL-submit
time over an initial 351 draws; GPU completion was not measured. The production
Three.js bundle is 630.20 kB (142.30 kB gzip), excluding the 2.3 MB atlas assets.

A one-repetition, 0.1 simulated-second exact-policy comparison measured neural-loop
times of 0.2135 s headless, 0.2057 s with the server and no client, 0.2124 s with a
draining client, and 0.2070 s with a slow client. The slow client replaced six
browser frames; worker IPC replaced none. These differences are within ordinary
single-run timing noise, but confirm that a stalled observer did not stall the
simulation. A 0.1 simulated-second DA1-preset browser smoke produced live bounded
activity, completed with no console warnings/errors, and used 593 MiB worker RSS.
