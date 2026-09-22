# FlyBrain Control Center — Panel v2 specification

## Verified baseline (2026-09-22)

The current panel is FastAPI + an isolated sparse-LIF worker with a one-slot worker
telemetry queue and one-slot queues per WebSocket observer. Its vanilla-JS UI embeds
the pinned Three.js soma atlas. Existing `/api/status`, `/ws/telemetry`,
`/api/anatomy`, `/api/anatomy/map`, `/api/anatomy/neighborhood`,
`/api/presets/*`, and experiment start/pause/resume/reset remain authoritative.
The atlas displays 124,289 identified, measured soma positions; it does not
depict all 166,700 neurons, synapse locations, or real neurites.

The separately validated 2D virtual cursor controller composes a horizontal
temporal-voltage probe and a vertical temporal-voltage/drive probe. These are
**separate MaleCNS simulations per control step**, not simultaneous recordings
from one unified retinal frame. Its frozen RBF scores are not calibrated
probabilities. The recorded experiment artefact supplies cursor positions,
targets, decoder scores, per-axis total simulated spikes and distance changes;
it does not supply per-neuron traces or live region-level motor activation.

No QEMU guest display is connected by the panel baseline. Existing uinput /
Wayland experiments are optional host adapters and must never become the panel
default.

## Information architecture

- **Persistent status ribbon**: actual simulator state and simulated time;
  worker ratio/activity/backpressure; explicitly disconnected sandbox status;
  experiment identity; source (live vs recorded); control/recording availability.
  An emergency-stop-looking control must never claim to stop a VM until a VM
  control backend exists. Existing reset controls affect the simulation only.
- **Main stage, desktop**: dominant Sandbox / recorded virtual-cursor stage
  (roughly 5 columns), measured soma Brain View (4), descending motor readout
  (3). At narrower widths the sandbox remains first, followed by brain and motor.
- **Second stage**: anatomical Pathways; bounded Neural Detail with existing
  population-rate, voltage and drive samples; a separate Episode Timeline.
- **Protocol drawer**: preserve existing validated preset, stimulus, watchlist
  and run controls; expanded on demand. Diagnostics remain visible below it.
- **Source badges**: LIVE SIMULATION, RECORDED EXPERIMENT, ANATOMY ONLY,
  DERIVED, NOT AVAILABLE. Do not equate recorded step-level total spikes with
  neuron-level activity or anatomical connectivity with causal attribution.

## Style and behavior

Dark graphite `#0b1119`, raised slate `#151f2b`, fine borders `#2a3a48`,
off-white `#eff4fa`, secondary `#97a8ba`; restrained cyan `#50d6db`
for selected data, violet `#aa9eff` for anatomical selections; warm
`#ffbd7c` for actual simulated activity. Fixed numeric figures use a
tabular monospace. Spacing follows an 8 px grid; cards use 12–16 px radii,
subtle border differentiation and accessible focus rings. Motion is limited
to clearly labeled live indicators and bounded chart updates, disabled under
`prefers-reduced-motion`. Native UI controls have labels and are keyboard
operable. High-density widescreen displays use 12 columns; sub-1100 px
layouts stack brain/motor; sub-720 px layouts become single-column.

Each panel defines non-deceptive empty, loading, paused, disconnected, error and
recorded-source states. Live/replay modes have a persistent contrasting badge;
replay scrub changes the recorded virtual screen, motor readout and timeline
together while independently running live simulation remains clearly labeled.
`Escape` closes expanded controls or fullscreen; `Space` may toggle
*replay playback only while the replay stage has keyboard focus*.
The browser must never steal typing from experiment form fields.

## Data flow and boundaries

```text
Existing simulation worker ── bounded latest-value IPC ── FastAPI
           |                                         |
           |                               /ws/telemetry
           |                                         |
     existing manifests                       live UI + atlas
                                                     |
Recorded validated experiment JSON ── read-only /api/v2/recordings
                                                     |
                                   bounded replay controller (browser)
                                                     |
                     recorded cursor + axis scores + event timeline

MaleCNS static anatomy ── cached /api/anatomy/* ── bounded pathway view

Future QEMU guest ── local display adapter ── sandbox viewport
                         |
              disabled-by-default guest pointer action
                         |
              never route through host Wayland/uinput
```

New recordings endpoints return a versioned, bounded, typed presentation-neutral
contract including a recording ID, source labels, provenance, episode ID and
per-step IDs. Separate horizontal/vertical probe identifiers and their true
simulation-time availability must be explicit; an absent clock is `null`,
not the enclosing cursor step's timestamp. Cursor coordinates and targets are
recorded environment state, forbidden as motor-decoder inputs.

Telemetry rates are targets until separately benchmarked. Keep existing bounded
queues and avoid shipping full connectome state. Emit region aggregation only
after verifying source annotations; sampled pathway voltage/drive only when
the runtime or recording provides those quantities.

## Increments

1. **Foundation:** preserve FastAPI/panel lifecycle; add a read-only, versioned
   recordings and sandbox-status boundary, responsive stage, clear source badges,
   real live dashboard metrics, measured Brain View, preserved protocol drawer
   and synchronized *recorded* virtual-cursor/motor/timeline.
2. **Measured motor and brain telemetry:** integrate the actual neural motor
   experiment controller, bounded anatomy-grounded region summaries and sampled
   voltage/drive. Retain distinct probe clocks. Add ring buffers and replay
   recording without modifying the validated numerical kernels.
3. **Pathways:** load anatomical hop metadata independently, inspect visual-to-DN
   chains and select one to highlight in the atlas; overlay recorded or live
   activity only where actual sampled telemetry exists.
4. **Sandbox:** add a local QEMU/KVM adapter and loopback-only, authenticated or
   local-socket noVNC transport; show real frames only after a successful
   handshake. Disconnect must disable guest actions; host adapters remain
   completely separate.
5. **Experiment orchestration:** distinguish simulation pause, guest pause and
   replay pause. Add bounded persisted event recording and synchronized scrubbing.

## Verified pathway inspector and bounded observation (2026-09-22)

The Connectivity card now calls the original read-only
`GET /api/v2/pathways` and `GET /api/v2/pathways/{target_index}`
endpoints. The installed, independently validated artifact exposes ten
individual DNa02, DNg13, DNp09 and MDN descending targets. Selecting one
of its at-most-twelve sampled, two- or three-hop routes maps **actual
stable neuron indices** through `POST /api/anatomy/map`; violet route
somata and edges have a separate overlay, independent of the generic
anatomical-neighborhood renderer. Synapse counts label *anatomical*
contacts, never signal transmission. Atlas exclusions are displayed, not
silently relocated.

The currently selected route is included as `pathway_observation:
{target_neuron_index, path_index}` when a new protocol is started. While
a worker is running or paused, selecting a different route or pressing
**Observe live** sends `POST /api/v2/pathways/observe` with
`{"observation":{"target_neuron_index":...,"path_index":...}}`; sending
`{"observation":null}` stops observation. The backend resolves the
identifiers *again* against the verified artifact and rejects arbitrary
indices. Changes take effect at the next worker chunk boundary and do
not modify neural state or restart the running experiment. A selection
made while idle is armed for the next experiment, not silently
presented as current activity.

Only the selected route's **three or four verified neurons** are added
to the existing bounded `watch_indices` and `visualization_indices`.
The worker records end-of-chunk simulated membrane voltage and synaptic
drive plus the number of emitted spikes *within that chunk* for each
watched neuron. Original user-configured watchlist telemetry remains
unchanged. Legacy `/ws/telemetry` carries a separate
`pathway_measurement`; versioned `/ws/v2/telemetry` projects that
measurement to `observed_pathway`. Both retain
`pathway_activation: null` conceptually: observing the state of adjacent
neurons is **not** proof that a particular synapse transmitted a signal,
nor proof of a causal motor contribution.

The browser retains at most **60 sampled frames** for the selected
route, with per-neuron latest values, mean voltage and drive, per-bin
population spikes, and separate short voltage/drive/firing-rate graphs.
It resets on route or experiment change and does not connect lines
across dropped sample chunks. A missing, paused, disconnected or
previously recorded source is explicitly labeled. The UI ring buffer
is transient and is **not** a persistent per-neuron trace archive.
Historical two-axis virtual-cursor artifacts still have no such traces,
and the QEMU guest remains disconnected. These measurements are of the
current isolated sparse-LIF simulation only.

## Acceptance and nonclaims

Panel launch and historical experiment endpoints keep working. The prominent
sandbox view reports **VM NOT CONNECTED** without live video until a guest is
actually attached. Recorded 2D episodes replay with independent per-axis scores
and real distance traces. Brain activity is displayed only for selected simulated
spikes; no replay motor step invents a brain region heatmap. Pathways that lack
temporal measurements are labeled anatomical only. The default exposes no host
pointer action or writable VM action endpoint. Tests cover replay normalization,
explicit source labels and sandbox-disabled state.
