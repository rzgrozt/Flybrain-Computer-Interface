# Graded early-vision reference and projected MaleCNS drive

Date: 2026-09-18. Dataset target: `male-cns:v1.0`.

This increment addresses the physiology mismatch identified by the
2026-09-15 visual-pathway validation without replacing the existing 166,700-neuron
point-LIF baseline. It adds two explicit, separable pieces:

1. a qualitative graded R1-R6 -> L1/L2/L3 reference model; and
2. an optional projected-drive path that lets non-spiking early-vision activity
   influence the rest of MaleCNS through the real signed outgoing connectome.

The unchanged point-LIF visual path remains available as the comparison baseline.

## Biological constraint

Adult Drosophila R1-R6 photoreceptors are non-spiking sensory neurons whose axon
terminals release histamine tonically as a graded function of photoreceptor
depolarization. Increased luminance therefore increases inhibitory histaminergic
input to first-order lamina neurons. L1 and L2 respond comparatively rapidly and
transiently, whereas L3 carries a slower, more sustained luminance component.

Useful sources:

- Hardie/photoreceptor context and early visual review:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9115102/
- Graded photoreceptor-to-interneuron transfer and tonic histamine release:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC2216927/
- Drosophila photoreceptor axons as non-spiking, tonic histamine-releasing terminals:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3432082/
- L1/L2 transient versus L3 sustained temporal organization:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11071408/
- Modern L1/L2 flash-response characterization:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11769683/

These papers constrain the **sign and qualitative temporal form** of the reference.
The current time constants and gains remain explicit engineering priors, not a fit to
an electrophysiological dataset.

## Graded reference model

`sensory/vision/graded.py` uses an independent 60 Hz frame clock and the existing
0.1 ms neural grid.

1. Mean frame luminance drives a first-order photoreceptor state.
2. Histamine release remains nonzero at the adapted background through an explicit
   tonic-release floor.
3. L1 and L2 are fast-minus-slow filters, producing sign-inverted transient
   responses to luminance changes.
4. L2 has a small additional slow component so a light flash can produce an
   off-rebound.
5. L3 is a slower sign-inverted low-pass response, preserving a sustained luminance
   component.
6. Lamina outputs are voltage **deltas relative to the adapted background**, not
   absolute membrane voltages and not a claim about exact biological amplitudes.

The fixed configuration is
[`configs/early-vision-v1.json`](../configs/early-vision-v1.json).

## Predeclared stimuli and qualitative gates

Five spatially uniform 12-frame conditions are declared:

- adapted 0.5-luminance baseline;
- 0.5 -> 1.0 -> 0.5 light step;
- 0.5 -> 0.0 -> 0.5 dark step;
- one-frame light flash;
- one-frame dark flash.

At 60 Hz the single-frame flash is 16.67 ms, close to but not identical to the
20 ms flash protocols used in some published physiology experiments.

The graded reference must satisfy these deterministic qualitative gates:

- adapted background has tonic release but effectively zero L1/L2/L3 delta;
- a light increment hyperpolarizes L1, L2 and L3;
- a light decrement depolarizes L1, L2 and L3;
- L1 and L2 late-step magnitude is small relative to their peak;
- L3 retains a substantial late-step response;
- L2 shows a post-light rebound.

**All seven gates pass.**

## Why the unchanged point-LIF path is inadequate here

The canonical comparison drives all 3,377 reconstructed R1-R6 neurons with the
existing pulse-density encoder.

At adapted 0.5 luminance, the point-LIF path produces 81,048 R1-R6/network spikes
over the 0.2 s run even though the graded reference is exactly at its adapted
baseline. The strongest-contact representative L1/L2/L3 samples remain non-spiking,
but are driven strongly below rest:

- L1 peak hyperpolarization: about 45 mV;
- L2 peak hyperpolarization: about 48 mV;
- L3 peak hyperpolarization: about 14 mV.

This reproduces the core mismatch: a luminance value that should be represented as
an adapted tonic operating point becomes a large train of inhibitory point-neuron
events.

The point-LIF visual baseline remains unchanged for reproducibility and comparison.

## Projected graded-drive interface

The new generic projected-drive path does **not** force L1/L2/L3 to emit artificial
spikes.

`MemoryMappedConnectome.project_sources()` first collapses a fixed graded source
population into a sparse target projection using the real MaleCNS outgoing edges,
contact counts, and the configured presynaptic transmitter signs.

For the R1-R6 direct lamina populations:

| Graded source | Source neurons | Outgoing edge rows | Unique projected targets |
| --- | ---: | ---: | ---: |
| L1 | 809 | 13,648 | 8,060 |
| L2 | 814 | 35,245 | 15,191 |
| L3 | 782 | 49,916 | 16,038 |

This aggregation is computed once. The runtime then updates only the unique targets
per active channel instead of traversing roughly 98,800 original source-edge rows on
every 0.1 ms step.

The canonical engineering conversion is:

`per-step amplitude = normalized graded activity × synapse_scale × equivalent_rate × dt`

with:

- equivalent-rate scale: 50 Hz;
- activity clipped to [-1, 1];
- numerical activity below 1e-12 zeroed;
- 1.8 ms explicit axonal delay;
- differential activity around the adapted baseline, not a tonic DC offset.

The 50 Hz number is an interpretable engineering scale: full normalized graded
activity has the same mean per-step synaptic-event amplitude as a 50 Hz spike source.
It is **not** a biological firing-rate claim.

Projected drive writes into the normal `synaptic_drive_mv` state. Refractory targets
suppress these writes using the same rule as recurrent synaptic writes. Recurrent
LIF equations, graph weights, transmitter signs, delayed spike propagation,
Brian2-equivalent baseline semantics, and the old visual experiment are unchanged.

The runtime reports projected target updates separately from recurrent
`visited_edges`, so the new continuous-drive workload does not silently redefine the
old event-propagation metric.

## Canonical 50 Hz full-graph result

The optional projected path produces the following whole-network responses:

| Condition | Network spikes | Active region | Descending spikes |
| --- | ---: | --- | ---: |
| adapted baseline | 0 | none | 0 |
| light step | 41 | optic-lobe intrinsic | 0 |
| dark step | 255 | optic-lobe intrinsic | 0 |
| light flash | 4 | optic-lobe intrinsic | 0 |
| dark flash | 22 | optic-lobe intrinsic | 0 |

This is the first current implementation in which:

- the adapted visual baseline is genuinely quiet at the whole-network level;
- changing luminance produces condition-specific recurrent MaleCNS activity;
- the signal enters the connectome without fabricating R1-R6 or L1/L2/L3 spikes.

At the canonical scale the response remains inside `ol_intrinsic` neurons. Dominant
types include Dm19/Dm6/Dm17 for the light condition and Dm20/Dm19/Dm12/Dm4/Dm6 for
the dark condition.

The result is an engineering characterization, not a behavioral or physiological
validation of those cell-type response magnitudes.

## Exploratory gain sensitivity

After the canonical 50 Hz run was fixed and measured, a separate exploratory sweep
tested 25, 50, 100 and 200 Hz equivalent-rate scales. These values were **not**
predeclared gates and were not used to retroactively tune the canonical result.

Light step:

| Equivalent rate | Network spikes | Propagation |
| --- | ---: | --- |
| 25 Hz | 17 | ol_intrinsic only |
| 50 Hz | 41 | ol_intrinsic only |
| 100 Hz | 84 | ol_intrinsic only |
| 200 Hz | 167 | ol_intrinsic only |

Dark step:

| Equivalent rate | Network spikes | Propagation |
| --- | ---: | --- |
| 25 Hz | 39 | ol_intrinsic only |
| 50 Hz | 255 | ol_intrinsic only |
| 100 Hz | 1,025 | ol_intrinsic + 2 visual_centrifugal cells |
| 200 Hz | 2,527 | ol_intrinsic + 4 visual_centrifugal cells |

No descending neuron spiked at any tested scale.

At 200 Hz, the four active visual-centrifugal neurons were all annotated `aMe30`.
Three of the four have direct anatomical outgoing edges to descending neurons,
including DNpe021, DNp27, DNpe006, DNp101 and DNc01 targets. The remaining active
aMe30 reaches many descending cells within two directed graph hops.

Therefore the absence of descending spikes is not explained by a missing anatomical
route.

## Descending subthreshold check

The 200 Hz dark-step exploratory run was repeated while recording all 1,314
descending neurons.

The strongest depolarizing descending response was DNc01 (body ID 512245):

- peak voltage delta above rest: about +0.422 mV;
- peak synaptic drive: about +0.915 mV;
- no descending neuron exceeded +1 mV above rest;
- no descending neuron spiked.

The LIF threshold is 7 mV above the -52 mV resting potential. The current visual
signal therefore reaches descending circuitry anatomically but remains far below
point-LIF firing threshold.

This argues against simply multiplying the visual gain until a descending spike is
forced. Such a change would be a calibration choice rather than evidence that the
early-vision model is correct.

## Reproduction

Run the canonical comparison with:

```bash
.venv/bin/python -m flybrain_interface.experiments.early_vision \
  --data-directory data/processed/malecns-v1.0 \
  --output /tmp/early-vision-v3.json
```

The result records:

- graded qualitative-gate outputs;
- unchanged point-LIF visual baseline;
- projected-drive channel anatomy and update counts;
- whole-network spike counts;
- descending response;
- active-neuron summaries by superclass, class and type;
- software and dataset lock information.

## Validation

The implementation includes regression checks for:

- signed source-projection aggregation;
- read-only aggregate projection arrays;
- zero projected drive preserving the previous runtime state exactly;
- projected drive entering synaptic state without a source spike;
- refractory suppression of projected-drive writes;
- neural-grid and network-bound validation;
- deterministic graded background and light/dark response signs;
- predeclared graded qualitative gates.

The full project test suite, Ruff and mypy are run after this experiment update.

## Decision boundary after this increment

This increment is sufficient to show that the project now has a controlled
**frame luminance -> graded early vision -> real MaleCNS topology -> recurrent neural
activity** path without relying on fabricated photoreceptor spikes.

It is **not** yet a spatial visual system. Uniform fields do not tell the connectome
where an object is.

The next main development step should therefore be the retinotopic mapping problem:

1. infer R1-R6 column membership from validated connectivity to optic-column-labelled
   cells such as L1, while keeping uncertainty explicit;
2. establish left/right eye orientation and coverage;
3. validate mapping consistency against R7/R8/L1 column resources;
4. only then introduce spatial patterns such as moving bars, targets and screen
   patches.

The descending result also leaves a later modeling choice open: motor readout may
need to use graded/subthreshold descending state or an explicitly justified tonic
excitability model rather than demanding point-LIF spikes. That should be tested
after the spatial visual mapping is valid, not solved now by arbitrarily increasing
visual gain.
