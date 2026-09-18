# Measured eye map and spatial screen validation

Date: 2026-09-18. Dataset target: `male-cns:v1.0`.

This increment extends structural R1-R6 column membership into measured visual
direction and then validates a deterministic virtual-screen sampler. It does not yet
claim that spatial stimuli have been propagated through the full connectome.

## Primary visual-direction source

Measured ommatidium directions come from the data accompanying Zhao et al. (2025),
*Eye structure shapes neuron function in Drosophila motion vision*:

- paper: https://www.nature.com/articles/s41586-025-09276-5
- code/data repository: https://github.com/reiserlab/eyemap_T4
- pinned commit: `99d2a43123db636cedb55af9ff31a59657e7d17e`

The source manifest is
`data/manifests/eyemap-zhao2025-v1.json`. Primary RData files remain ignored under
`data/raw/eyemap-zhao2025-v1/`.

Fetch the locked files with:

```bash
.venv/bin/python scripts/fetch_eyemap_primary.py
```

Build the local measured-direction table with:

```bash
.venv/bin/python scripts/build_eyemap_directions.py
```

R is required only for this reproducible build step because the pinned primary data
are RData objects. Runtime use of the derived CSV does not require R.

## Coordinate validation

The upstream eye-map code represents ommatidium viewing directions as unit vectors
and converts them to geographic visual coordinates by flipping the Y axis for the
inside-out viewing convention. The local runtime therefore stores each ray as:

- `forward = x`
- `right = -y`
- `up = z`
- `azimuth = atan2(right, forward)`
- `elevation = asin(up)`

The MaleCNS column ROI naming convention is converted to centered p/q coordinates as:

- `p = hex2 - 19`
- `q = hex1 - 18`

The pinned source provides a direct right-eye anchor between its manual eye map and
medulla p/q coordinates. Matching the newer microCT table to those anchors requires:

- right eye: `p = raw_p + 1; q = raw_q`
- left eye: `p = raw_p; q = raw_q`

Validation results:

- 778 right-eye anchors checked;
- missing anchors: 0;
- maximum direction-vector mismatch:
  `1.41e-15`;
- bilateral common p/q coordinates: 836;
- left/right mirrored direction median error: 2.14 degrees;
- mean mirror error: 2.30 degrees;
- maximum mirror error: 8.90 degrees.

The bilateral check is a validation of the coordinate alignment, not an assertion
that the two biological eyes are perfectly symmetric.

## Measured-direction coverage

The deterministic derived CSV has:

- records: 1,678;
- SHA-256:
  `ef1b36c62a57d3b356249e899468da0de3a99604bba2a0df96f09c0e2a399b39`;
- official column coverage: 94.70%;
- left: 835 / 880 official columns;
- right: 843 / 892 official columns.

Rebuilding the CSV from the pinned RData reproduced the exact same SHA-256.

Combining this table with the connectivity-grounded R1-R6 mapping yields:

- assigned R1-R6: 3,227;
- assigned R1-R6 with a measured viewing direction: 3,184 (98.67%);
- high-confidence R1-R6 (`unique_high + dominant`): 3,213;
- high-confidence R1-R6 with measured direction: 3,170 (98.66%).

Missing measured directions are retained as missing; no interpolation or soma-based
receptive-field fabrication is used.

## Virtual screen model

The screen interface is an explicit engineering model, not a biological property.
A perspective monitor is centered on the fly's forward axis. A measured unit ray is
projected into normalized screen coordinates:

```text
x = (right / forward) / tan(horizontal_fov / 2)
y = (up / forward)    / tan(vertical_fov / 2)

u = (x + 1) / 2
v = (1 - y) / 2
```

Rays with `forward <= 0` or outside the configured frustum are not on the monitor.
Screen-frame values are sampled by deterministic bilinear interpolation. Off-screen
mapped neurons use the adapted background intensity rather than receiving an
artificial stimulus.

The canonical geometry-validation screen is intentionally versioned in
`configs/spatial-vision-v1.json`:

- horizontal FOV: 120 degrees;
- vertical FOV: 90 degrees;
- validation raster: 256 x 144;
- adapted background: 0.5;
- bright bar: 1.0;
- bar width: 12% of the relevant screen dimension;
- five positions: 0.15, 0.325, 0.5, 0.675, 0.85.

These FOV values are interface parameters and can later be changed to represent a
different virtual monitor geometry.

At 120 x 90 degrees:

- measured optic columns projected onto the monitor: 620;
- high-confidence mapped R1-R6 on the monitor: 374.

## Moving-bar geometry validation

Run:

```bash
.venv/bin/python -m flybrain_interface.experiments.spatial_vision \
  --data-directory data/processed/malecns-v1.0 \
  --output /tmp/spatial-vision-v1.json
```

The experiment moves a bright vertical bar left-to-right and a bright horizontal
bar top-to-bottom. It measures the intensity-weighted activity centroid over:

1. all measured visible columns; and
2. the same columns weighted by their number of high-confidence assigned R1-R6
   photoreceptors.

Canonical results:

| Sweep | Column centroid correlation | R1-R6 centroid correlation |
| --- | ---: | ---: |
| horizontal | 0.999997 | 0.996248 |
| vertical | 0.999934 | 0.998948 |

All four centroids move strictly monotonically in the expected screen direction.

The predeclared gates require:

- at least 500 visible measured columns;
- at least 300 visible high-confidence R1-R6;
- column-centroid correlation >= 0.99 on both axes;
- R1-R6-centroid correlation >= 0.99 on both axes;
- monotonic movement on both axes.

All ten gate checks pass.

## Current boundary

The project can now truthfully perform:

`screen frame -> measured viewing direction -> optic column -> mapped R1-R6`

for the measured/high-confidence portion of MaleCNS.

This does **not** yet mean the connectome is seeing the screen. The next increment
must convert each spatially sampled R1-R6 intensity trace into graded photoreceptor
release and deliver that release through the real signed R1-R6 outgoing MaleCNS
connections. The existing uniform graded visual model remains the reference for
sign, adapted baseline, and temporal dynamics.

Only after that neural propagation is validated should moving targets be connected
to motor readout or learning.
