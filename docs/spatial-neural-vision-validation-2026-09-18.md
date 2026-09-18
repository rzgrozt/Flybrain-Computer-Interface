# Spatial graded R1-R6 propagation through MaleCNS

Date: 2026-09-18. Dataset target: `male-cns:v1.0`.

This increment connects the validated screen geometry to the full MaleCNS graph.
Spatial screen luminance is sampled at measured viewing directions, converted into
graded differential R1-R6 histamine release, and propagated through the real signed
outgoing connections of the mapped R1-R6 neurons.

This is the first implementation in the project with the complete path:

`screen pixel -> measured visual ray -> optic column -> R1-R6 group -> graded
histamine release -> real MaleCNS edges -> neural state`.

It is still a neural-engineering validation, not a behavioral result.

## Inputs and source selection

The experiment uses the previously validated resources:

- official MaleCNS optic-column workbook;
- connectivity-grounded R1-R6 to column assignments;
- Zhao et al. 2025 measured viewing-direction table;
- 120 x 90 degree perspective virtual-screen geometry.

Only R1-R6 assignments classified as `unique_high` or `dominant` are used as
spatial sources. At the canonical screen geometry:

- visible high-confidence source columns: 113;
- visible high-confidence R1-R6 neurons: 374;
- watched official L1 neurons: 113.

All selected R1-R6 sources have presynaptic sign `-1` under the default
`inhibitory-glutamate-modulators-separated-v1` policy, consistent with their
histaminergic inhibitory output. The experiment aborts rather than silently
continuing if that condition is not true.

The measured-direction CSV is re-verified against its tracked SHA-256 lock before
each canonical run.

## Parallel graded photoreceptor model

`simulate_graded_photoreceptor_channels()` extends the earlier uniform graded
photoreceptor model to independent spatial channels.

Each channel:

1. starts adapted at background luminance 0.5;
2. follows sampled screen luminance through the 15 ms photoreceptor time constant;
3. keeps the existing tonic-release floor of 0.1;
4. reports histamine release relative to the adapted tonic baseline.

An all-background sequence therefore produces an exactly zero release delta in all
channels.

For the canonical 0.5 background:

- steady full-dark release delta approaches -0.45;
- steady full-bright release delta approaches +0.45.

The experiment normalizes this differential range before applying the existing
50 Hz equivalent-rate engineering scale. This 50 Hz value is unchanged from the
earlier graded-drive experiment; it was not retuned to make the spatial result pass.

## Real connectome projection

R1-R6 neurons sharing one inferred optic column are grouped into one projected-drive
channel. Their outgoing contacts are aggregated once with
`MemoryMappedConnectome.project_sources()`.

For a dark bar:

- luminance falls;
- graded histamine release falls;
- release delta is negative;
- R1-R6 signed contact weights are negative;
- their product produces positive drive in downstream targets such as L1.

For a bright bar the sign reverses.

The path therefore obtains its light/dark sign from the measured visual input,
graded photoreceptor model, transmitter policy and actual connectome edges rather
than from a hand-coded L1 response.

## Canonical protocol

Configuration:
`configs/spatial-neural-vision-v1.json`.

Each condition lasts 12 frames at 60 Hz = 0.2 s:

- 3 frames adapted background;
- 6 frames stimulus;
- 3 frames recovery.

Screen:

- 256 x 144 validation raster;
- 120 degree horizontal FOV;
- 90 degree vertical FOV;
- background intensity 0.5;
- dark bar 0.0;
- bright bar 1.0;
- bar width 12%.

Conditions:

- adapted baseline;
- five horizontal dark-bar locations;
- five vertical dark-bar locations;
- one bright horizontal center control.

The LIF runtime remains unchanged:

- dt: 0.1 ms;
- backend: Numba;
- subnormal policy: preserve;
- R1-R6 projected-drive delay: 1.8 ms;
- equivalent-rate scale: 50 Hz.

## Canonical full-graph result

Run:

```bash
.venv/bin/python -m flybrain_interface.experiments.spatial_neural_vision \
  --data-directory data/processed/malecns-v1.0 \
  --output /tmp/spatial-neural-vision-v1.json
```

All predeclared gates pass.

Geometry carried by L1 synaptic drive:

| Metric | Result |
| --- | ---: |
| horizontal drive-centroid correlation | 0.994948 |
| vertical drive-centroid correlation | 0.998940 |
| horizontal centroid monotonic | yes |
| vertical centroid monotonic | yes |
| dark active-L1 sign fraction | 1.00 in every run |
| bright active-L1 sign fraction | 1.00 |
| adapted baseline max L1 drive | 0.0 mV |

The largest network-spike count in any 0.2 s condition was 91, far below the
predeclared 200,000-spike runaway guard.

Representative dark-bar runs:

| Condition | Active source R1-R6 | L1 drive centroid | Network spikes |
| --- | ---: | ---: | ---: |
| horizontal 0.15 | 40 | 0.1869 | 40 |
| horizontal 0.50 | 121 | 0.5148 | 91 |
| horizontal 0.85 | 67 | 0.8596 | 19 |
| vertical 0.15 | 86 | 0.1445 | 79 |
| vertical 0.50 | 12 | 0.5262 | 0 |
| vertical 0.85 | 60 | 0.8519 | 74 |

The adapted baseline produces:

- zero projected channels;
- zero projected target updates;
- zero recurrent visited edges;
- zero network spikes;
- zero L1 drive.

The bright-center control produces negative L1 drive, zero L1 spikes and zero
network spikes.

## Important physiology limitation: point-LIF L1 saturation

The spatial mapping and graded R1-R6 transduction work, but the downstream L1 cells
are still represented by the generic point-LIF model.

Dark bars can therefore make some L1 neurons cross the artificial point-neuron
threshold:

| Condition | Spiking watched L1 | Watched L1 spikes |
| --- | ---: | ---: |
| horizontal 0.15 | 4 | 15 |
| horizontal 0.325 | 2 | 3 |
| horizontal 0.50 | 15 | 39 |
| horizontal 0.675 | 1 | 4 |
| horizontal 0.85 | 4 | 7 |
| vertical 0.15 | 11 | 30 |
| vertical 0.325 | 0 | 0 |
| vertical 0.50 | 0 | 0 |
| vertical 0.675 | 5 | 14 |
| vertical 0.85 | 9 | 27 |

Peak positive L1 drive reaches roughly 18 mV in the strongest conditions, and peak
recorded L1 membrane voltage approaches the 7 mV-above-rest LIF threshold before
reset.

This should **not** be interpreted as evidence that biological L1 neurons fire
action potentials. It reproduces the known abstraction mismatch: the real early
visual pathway is graded, whereas the generic MaleCNS runtime currently treats L1
as a spiking LIF neuron.

The spatial centroid validation is based on the continuous L1 synaptic-drive field,
not on L1 spike count, so the spatial mapping result does not depend on this
artificial spiking.

## Current boundary and next model increment

The project can now truthfully perform:

`screen -> measured fly viewing directions -> spatial R1-R6 graded release ->
real MaleCNS contacts -> spatially organized downstream drive`.

The next scientifically justified step is **not cursor control yet**.

The next model increment should retain the spatial mapping while replacing the
point-LIF approximation for the first graded visual layers:

1. compute per-column graded R1-R6 state;
2. compute per-column graded/transient L1 and L2 state and sustained L3 state;
3. project those graded L1/L2/L3 states through their real MaleCNS outgoing edges;
4. compare that hybrid spatial early-vision frontend against the direct
   R1-R6-to-point-LIF baseline documented here;
5. verify that spatial centroids remain accurate while artificial L1 spiking
   disappears.

Only after that comparison should spatial visual activity be treated as a stable
input for downstream motor-readout or learning experiments.
