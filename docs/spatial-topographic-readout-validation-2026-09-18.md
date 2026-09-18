# Spatial topographic subthreshold readout validation

Date: 2026-09-18. Dataset target: `male-cns:v1.0`.

This experiment follows the negative high-dimensional ridge result in
`spatial-subthreshold-readout-validation-2026-09-18.md`.

The previous readout showed that downstream voltage/drive state retained strong
position correlation, but a 256-dimensional ridge model trained on only five anchors
overfit and failed the predeclared held-out MAE gate. This increment changes the
representation, not the canonical visual frontend.

## Question

Can the same canonical subthreshold MaleCNS state support a reliable held-out spatial
readout when it is summarized through the known retinotopic anatomy rather than fit as
256 unrelated features?

## Frozen visual and neural frontend

The following are unchanged:

- 120 x 90 degree virtual-screen geometry;
- pinned measured eye map;
- connectivity-grounded R1-R6 column assignment;
- graded R1-R6/L1/L2/L3 frontend;
- 694 clamped early-vision neurons;
- 50 Hz equivalent projected-drive scale;
- 0.1 ms neural timestep;
- the same 256 strongest non-clamped downstream targets;
- five training positions and four unseen intermediate test positions per axis.

No visual gain, LIF threshold, held-out positions, or canonical frontend parameter was
changed after the earlier experiment.

Configuration:
`configs/spatial-topographic-readout-v1.json`.

## Anatomical backprojection

For each of the 256 watched downstream neurons, the experiment reconstructs a fixed
distribution over the 113 visible optic columns using the absolute anatomical contact
weights arriving from the spatial L1/L2/L3 sources.

For target neuron `t` and visual column `c`, contacts from L1/L2/L3 are summed and
then normalized across columns for that target. This gives a fixed
target-to-visual-column backprojection matrix.

The matrix is based only on:

- measured visual-column screen coordinates;
- fixed MaleCNS L1/L2/L3 connectivity;
- the preselected downstream watchlist.

It does not use training or test labels to assign a downstream neuron to a screen
position.

The watched targets span nearly the full visible field:

- target receptive-field horizontal range: 0.1082 to 0.9710;
- target receptive-field vertical range: 0.0195 to 0.9950;
- minimum total lamina contact weight among watched targets: 133;
- median total lamina contact weight: 150.

## Low-dimensional readouts

For every stimulus, the absolute magnitude of each target's signed peak voltage (or
drive) is backprojected into visual-column activity.

Two deliberately small representations are evaluated.

### Primary: one-dimensional activity centroid

For the relevant axis, a single activity centroid is computed on the measured visual
coordinates. A two-parameter affine calibration is fitted using only the five training
positions.

Predeclared voltage-centroid gates:

- held-out MAE <= 0.10;
- held-out correlation >= 0.80;
- held-out predictions strictly monotonic.

### Secondary: four-bin pooled profile

Backprojected activity is pooled into four fixed coordinate bins and decoded with the
same dependency-free ridge implementation used by the previous experiment. This is a
secondary characterization rather than the primary gate.

## Canonical result

Run:

```bash
.venv/bin/python -m flybrain_interface.experiments.spatial_topographic_readout \
  --data-directory data/processed/malecns-v1.0 \
  --output /tmp/spatial-topographic-readout-v1.json
```

Overall result: **all predeclared primary gates passed**.

Measured simulation wall time was about 13.98 seconds.

### Voltage centroid

| Axis | Held-out MAE | Correlation | Monotonic |
| --- | ---: | ---: | --- |
| Horizontal | 0.0162 | 0.9971 | yes |
| Vertical | 0.0355 | 0.9866 | yes |

Horizontal unseen positions:

| Actual | Raw neural centroid | Predicted |
| ---: | ---: | ---: |
| 0.2375 | 0.2428 | 0.2281 |
| 0.4125 | 0.4401 | 0.4238 |
| 0.5875 | 0.5722 | 0.5548 |
| 0.7625 | 0.7702 | 0.7513 |

Vertical unseen positions:

| Actual | Raw neural centroid | Predicted |
| ---: | ---: | ---: |
| 0.2375 | 0.2218 | 0.2113 |
| 0.4125 | 0.3357 | 0.3266 |
| 0.5875 | 0.6184 | 0.6128 |
| 0.7625 | 0.7617 | 0.7579 |

The affine calibration is close to identity on both axes:

- horizontal slope 0.9920, intercept -0.0128;
- vertical slope 1.0126, intercept -0.0134.

This is substantially more stable than the previous 256-dimensional ridge readout.

### Synaptic-drive centroid

Drive-state decoding independently gives nearly the same result:

| Axis | Held-out MAE | Correlation | Monotonic |
| --- | ---: | ---: | --- |
| Horizontal | 0.0155 | 0.9975 | yes |
| Vertical | 0.0353 | 0.9867 | yes |

The close voltage/drive agreement is expected in this deterministic, quiet,
subthreshold regime.

### Four-bin voltage profile

The secondary low-dimensional profile also generalizes:

| Axis | Held-out MAE | Correlation | Monotonic |
| --- | ---: | ---: | --- |
| Horizontal | 0.0185 | 0.9948 | yes |
| Vertical | 0.0311 | 0.9845 | yes |

All four pooled profile features are active.

### Spike-only control

The spike state remains completely uninformative:

- zero network spikes in all 18 position runs;
- no available spike centroid;
- zero active pooled spike features;
- pooled spike prediction collapses to 0.5.

The successful result therefore comes from continuous neural state, not hidden spike
activity.

## Interpretation

This experiment resolves the specific failure of the previous readout.

The result supports the following claims:

1. spatial information survives the graded early-vision frontend in later MaleCNS
   subthreshold state;
2. that information is topographically organized with respect to the measured visual
   field;
3. a one-dimensional anatomically grounded summary generalizes to unseen positions
   far better than a 256-dimensional sample-starved ridge fit;
4. the useful signal is continuous voltage/drive state, not spikes.

The result does **not** mean that MaleCNS itself emits an explicit Cartesian screen
coordinate. The centroid is an external scientific readout built from measured
retinotopy and fixed anatomical connectivity. It is evidence that the neural state
contains a stable spatial representation.

It also does not yet demonstrate motor behavior, learning, target selection, or
closed-loop cursor control.

## Architecture consequence

The project no longer needs to require a spike cascade merely to establish that visual
position is available downstream.

The next motor-control increment should therefore add an explicit continuous-state
field to the typed `NeuralReadout` boundary and characterize anatomically grounded
descending/motor populations in that representation. A fixed decoder can then consume
only neural state, while reward remains a separate scalar teacher signal.

The topographic visual decoder should remain a diagnostic/observability tool. It
should **not** be wired directly to cursor actions, because that would bypass the
connectome's motor pathway.

## Increment conclusion

The low-dimensional topographic readout increment is successful.

The previous bottleneck was not absence of spatial information. It was the choice of
a high-dimensional readout with only five training examples.

The validated path is now:

`screen -> measured retinotopy -> graded early vision -> real MaleCNS connectivity
-> structured continuous downstream state`.

The next unresolved boundary is:

`continuous neural state -> anatomically grounded motor readout -> cursor action`.
