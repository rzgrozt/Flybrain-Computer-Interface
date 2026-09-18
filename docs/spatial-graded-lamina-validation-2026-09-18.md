# Spatial graded-lamina hybrid validation

Date: 2026-09-18. Dataset target: `male-cns:v1.0`.

This experiment is the follow-up to
`spatial-neural-vision-validation-2026-09-18.md`. The earlier spatial path used
graded R1-R6 release but still allowed L1/L2/L3 to be simulated as generic spiking
LIF neurons. Strong dark stimuli therefore produced artificial L1 spikes.

This increment replaces the first lamina layers with a spatial graded frontend while
preserving the full MaleCNS graph and the existing point-LIF runtime for the rest of
the network.

## Model boundary

The externally managed early-vision set contains:

- 374 high-confidence spatially mapped R1-R6 neurons;
- 113 direct L1 cartridge neurons;
- 110 direct L2 cartridge neurons;
- 97 direct L3 cartridge neurons.

The union contains 694 unique neurons.

For the 113 visible R1-R6 column groups:

- L1 mapping is unique in 113 / 113 columns;
- L2 mapping is unique in 110 / 113 columns, with 3 missing;
- L3 mapping is unique in 97 / 113 columns, with 16 missing;
- no mapped column has competing L1, L2 or L3 direct targets.

The direct L1 target also matches the official MaleCNS optic-column workbook in every
visible column.

Under the default transmitter policy:

- L1 sources are glutamatergic and have sign -1;
- L2 sources are cholinergic and have sign +1;
- L3 sources are cholinergic and have sign +1.

The experiment verifies these signs rather than hard-coding a replacement sign.

## Parallel graded frontend

`simulate_graded_early_vision_channels()` applies the same equations as the
previous scalar graded reference independently to every spatial column:

1. graded R1-R6 photoreceptor state and tonic histamine release;
2. transient fast-minus-slow L1 state;
3. transient L2 state with the existing slow component;
4. sustained L3 state.

A single-channel regression test confirms that the parallel implementation matches
the earlier scalar implementation numerically to 1e-14 for photoreceptor state,
histamine release and L1/L2/L3 outputs.

The parameters are unchanged from the prior early-vision reference and remain
engineering priors rather than a fitted electrophysiological model.

## Clamped visual neurons

The generic `SparseLIFSimulator` now supports an optional `clamped_indices` set.
This feature is disabled by default.

When a neuron is clamped:

- membrane voltage is held at the configured resting potential;
- internal synaptic drive is held at zero;
- refractory state is held at zero;
- external pulse input cannot make it spike;
- recurrent or projected input cannot make it spike.

The graded frontend therefore owns the R1-R6/L1/L2/L3 state, while the underlying
MaleCNS nodes remain present as anatomical sources for their real outgoing
connections.

Regression tests cover both NumPy and Numba backends. A clamped neuron remains at
rest with zero drive and zero spikes even under a 100 mV artificial external pulse
plus projected drive. Empty/default clamp configuration leaves the prior runtime
semantics unchanged.

## Graded output projection

Per-column L1/L2/L3 signals are normalized by their configured gain, passed through
the unchanged 50 Hz equivalent-rate engineering scale, delayed by 1.8 ms and
projected through each real MaleCNS source neuron's outgoing contacts.

No R1-R6/L1/L2/L3 spikes are fabricated.

Configuration:
`configs/spatial-graded-lamina-v1.json`.

The visual geometry, timing and gain are intentionally identical to the preceding
spatial R1-R6 experiment:

- 120 x 90 degree virtual screen;
- 256 x 144 validation raster;
- 0.5 adapted background;
- 12 frames at 60 Hz = 0.2 s;
- 3 baseline + 6 stimulus + 3 recovery frames;
- 50 Hz equivalent projected-drive scale;
- 0.1 ms neural dt.

The gain was not increased after seeing the result.

## Predeclared canonical gates

The versioned config declared these checks before the canonical graded-lamina run:

- adapted baseline must produce zero network spikes;
- clamped visual neurons must produce zero spikes;
- clamped visual neurons must show no LIF voltage drift;
- every non-baseline stimulus must create graded projected channels;
- network activity must remain below the runaway budget;
- at least one dark-bar condition must produce a downstream **spike**.

The final condition intentionally distinguishes subthreshold propagation from
threshold-crossing recurrent propagation.

## Canonical result

Run:

```bash
.venv/bin/python -m flybrain_interface.experiments.spatial_graded_lamina \
  --data-directory data/processed/malecns-v1.0 \
  --output /tmp/spatial-graded-lamina-v1.json
```

Result: **canonical gate set failed**.

Passed:

- adapted baseline quiet;
- clamped visual spikes = 0;
- clamped visual voltage drift = 0;
- all stimulus conditions projected;
- network spike budget respected.

Failed:

- `dark_reaches_downstream`: no condition produced a downstream LIF spike.

This failure is retained as the canonical result. The gain and the gate are not
changed post hoc.

## Subthreshold downstream propagation

Absence of spikes does **not** mean that the graded signal stops at L1/L2/L3.

The experiment separately watches the 256 non-clamped downstream neurons receiving
the largest summed absolute anatomical contact weight from the spatial L1/L2/L3
frontend. A response is counted as subthreshold-active at >=0.01 mV peak absolute
voltage change.

Selected dark-bar results:

| Condition | Responsive / 256 | Max abs drive | Max abs voltage |
| --- | ---: | ---: | ---: |
| horizontal 0.15 | 42 | 3.949 mV | 3.154 mV |
| horizontal 0.325 | 35 | 4.273 mV | 3.416 mV |
| horizontal 0.50 | 69 | 8.054 mV | 6.437 mV |
| horizontal 0.675 | 28 | 1.747 mV | 1.395 mV |
| vertical 0.50 | 30 | 2.043 mV | 1.637 mV |
| vertical 0.675 | 63 | 3.770 mV | 3.471 mV |
| vertical 0.85 | 59 | 3.654 mV | 3.364 mV |

The bright horizontal center control also gives:

- 69 / 256 responsive downstream neurons;
- max absolute drive 8.054 mV;
- max absolute voltage delta 6.437 mV;
- positive extreme +5.290 mV;
- negative extreme -6.437 mV;
- zero spikes.

The strongest center dark-bar response is about 6.437 mV from the -52 mV rest
potential. The generic LIF threshold is 7 mV above rest. Thus the canonical graded
signal can approach threshold closely while remaining subthreshold.

Dominant watched targets are optic-lobe intrinsic types including Dm6, Dm12, Dm19,
Dm4, Dm9, Dm20, Tm1, Tm2 and Mi1.

## Interpretation

This result resolves one problem and exposes the next one.

Resolved:

- spatial R1-R6 input no longer requires fabricated photoreceptor spikes;
- L1/L2/L3 no longer produce artificial point-LIF spikes;
- strong, condition-dependent signals reach later optic-lobe circuitry through real
  MaleCNS topology.

Unresolved:

- the generic downstream LIF network remains very quiescent at the unchanged 50 Hz
  equivalent scale;
- spatial signals are strong but mostly subthreshold;
- no recurrent spike cascade is triggered in the canonical run.

It would be scientifically weak to multiply the gain until a spike appears simply
because the predeclared spike gate failed.

## Next experiment

The next experiment should test whether the **subthreshold population state itself
contains a stable spatial representation** before changing excitability.

A useful versioned analysis should:

1. keep this exact 50 Hz canonical graded-lamina frontend unchanged;
2. extract signed downstream voltage/drive vectors for horizontal and vertical bar
   positions;
3. quantify whether nearby screen positions produce nearby neural population states;
4. test a tiny held-out linear/readout model on unseen intermediate bar positions;
5. compare voltage-state decoding with spike-only decoding.

If subthreshold state reliably carries screen position, the project can justify a
motor/readout interface that observes continuous neural state rather than requiring
every useful signal to cross the generic LIF spike threshold.

If spatial information is not recoverable even subthreshold, the next step should
instead be a separately versioned excitability/calibration study.
