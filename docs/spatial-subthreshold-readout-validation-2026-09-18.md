# Spatial subthreshold held-out readout validation

Date: 2026-09-18. Dataset target: `male-cns:v1.0`.

This increment asks one narrow question after the canonical spatial graded-lamina
experiment:

> Does the downstream **subthreshold continuous neural state** carry enough spatial
> information to decode an unseen bar position without changing the visual gain or
> forcing spikes?

The canonical graded-lamina frontend is unchanged. In particular:

- the 120 x 90 degree screen geometry is unchanged;
- the measured eye-map and R1-R6 column mapping are unchanged;
- the R1-R6/L1/L2/L3 graded frontend is unchanged;
- the 50 Hz equivalent projected-drive scale is unchanged;
- the same 256 anatomically strongest non-clamped downstream targets are used;
- clamped early-vision neurons remain non-spiking.

No post-hoc gain adjustment is made.

## Protocol

Configuration:
`configs/spatial-subthreshold-readout-v1.json`.

For each axis independently, the tiny readout is trained on five canonical bar
positions:

- 0.15
- 0.325
- 0.50
- 0.675
- 0.85

It is then tested on four **unseen intermediate positions**:

- 0.2375
- 0.4125
- 0.5875
- 0.7625

Features are extracted from the same 256-neuron downstream watchlist.

Three feature families are compared:

1. signed peak membrane-voltage delta;
2. signed peak synaptic drive;
3. spike counts.

The readout is a dependency-free standardized ridge regressor with alpha = 1.0.
Only the five training positions are used to fit the readout. Test positions are
not used for fitting.

The predeclared voltage-readout gates are:

- held-out MAE <= 0.10;
- held-out correlation >= 0.80;
- held-out predictions must be strictly monotonic.

## Runtime behavior

The full experiment consists of 18 full MaleCNS runs:

- 9 horizontal positions;
- 9 vertical positions.

The measured simulation wall time was about **11.02 seconds**.

The CLI now writes the detailed result to the requested JSON file but prints only a
compact summary, avoiding the very large terminal payload that previously made the
tool interaction appear much slower than the simulation itself.

## Canonical result

Run:

```bash
.venv/bin/python -m flybrain_interface.experiments.spatial_subthreshold_readout \
  --data-directory data/processed/malecns-v1.0 \
  --output /tmp/spatial-subthreshold-readout-v1.json
```

Overall result: **predeclared gate set failed**.

### Voltage-state readout

Horizontal held-out result:

- correlation: **0.9071** — pass;
- MAE: **0.9113** — fail;
- monotonic: **no** — fail;
- active features: 210 / 256.

Predictions:

| Actual | Predicted |
| ---: | ---: |
| 0.2375 | -0.9096 |
| 0.4125 | 0.5288 |
| 0.5875 | 0.2920 |
| 0.7625 | 2.8489 |

Vertical held-out result:

- correlation: **0.8626** — pass;
- MAE: **0.4138** — fail;
- monotonic: **yes** — pass;
- active features: 194 / 256.

Predictions:

| Actual | Predicted |
| ---: | ---: |
| 0.2375 | 0.1918 |
| 0.4125 | 0.3837 |
| 0.5875 | 0.5611 |
| 0.7625 | 2.3167 |

The training positions are fitted almost perfectly on both axes
(correlation > 0.99999, MAE around 0.001), while held-out calibration is poor.
That pattern is consistent with a high-dimensional readout fitted from only five
training examples: spatial ordering exists in the state, but this specific ridge
decoder does not generalize robustly.

### Synaptic-drive readout

Drive-state results are nearly identical to voltage-state results.

Horizontal:

- correlation: 0.9070;
- MAE: 0.9106;
- monotonic: no.

Vertical:

- correlation: 0.8615;
- MAE: 0.4136;
- monotonic: yes.

This close agreement is expected because the downstream voltage response in this
quiet regime is largely driven by the same projected subthreshold synaptic pattern.

### Spike-only readout

Spike readout contains **no usable spatial information** in the canonical hybrid:

- horizontal active spike features: 0 / 256;
- vertical active spike features: 0 / 256;
- every prediction: 0.5;
- test correlation: undefined;
- test MAE: 0.175.

This cleanly confirms that, at the unchanged 50 Hz canonical scale, useful spatial
activity exists only in the continuous subthreshold state, not in spike counts.

## Interpretation

The result is deliberately not converted into a success by changing the ridge alpha,
gain, gate thresholds, or test positions after seeing the outcome.

What the experiment supports:

- downstream continuous state is **not random with respect to screen position**;
- held-out voltage/drive predictions retain strong rank-like correlation with
  position on both axes;
- spike counts carry no information because the canonical hybrid remains silent.

What the experiment does **not** support:

- this five-example high-dimensional ridge readout is not yet a reliable continuous
  position decoder;
- the current subthreshold population representation cannot yet be treated as a
  calibrated cursor-coordinate signal;
- motor control should not be connected to this decoder yet.

## Increment conclusion

This increment is complete with a scientifically useful negative result:

1. the graded-lamina hybrid removed artificial early-vision spikes;
2. spatial information survives downstream as a structured subthreshold pattern;
3. spike-only decoding fails completely;
4. a naive 256-dimensional ridge decoder trained on only five anchors overfits and
   fails the predeclared held-out MAE criterion.

A future readout experiment should be separately versioned rather than altering this
result. Reasonable future candidates include a low-dimensional anatomically pooled
representation, centroid/topographic features, more training positions, or a
regularization-selection protocol defined before evaluation.
