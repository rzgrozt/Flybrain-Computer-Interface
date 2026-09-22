# Hop-2 steering relay attribution and causal test — 2026-09-21

## Goal

Identify which hop-2 relay groups feeding DNa02/DNg13 preserve versus destabilize
held-out screen-position geometry, then test restricted relay interventions causally
without changing the visual frontend, connectome topology, or graded gain.

## Attribution result

Hop-2 steering layers were partitioned by anatomical superclass and type under the
existing graded relay model (gain 0.25).

Superclass summary across horizontal and vertical DNa02/DNg13 targets:

- visual_centrifugal: 5/6 preserve, preserve fraction 0.833
- cb_intrinsic: 5/8 preserve, preserve fraction 0.625
- visual_projection: 4/8 preserve, preserve fraction 0.500
- ascending_neuron: 3/8 preserve, preserve fraction 0.375
- descending_neuron: 2/8 preserve, preserve fraction 0.250

Type-level examples:

Strong/preserving candidates:
- PS077: 3/4 preserve
- PS230: 3/4 preserve
- PS194: 2/2 preserve
- PS019: 2/2 preserve

Consistently unstable candidates:
- PLP301m: 0/8 preserve
- PLP300m: 0/8 preserve
- AN06B009: 0/8 preserve
- LLPC1: 0/4 preserve
- LC19: 0/6 preserve
- LoVP93: 0/4 preserve

The attribution therefore showed that hop-2 failure is not a uniform property of the
entire layer. Spatially useful and unstable subpopulations coexist.

## Horizontal causal intervention

A bounded causal experiment compared four variants:

1. baseline graded relay model
2. remove hop-2 LLPC1/LC19/LoVP93 graded sources
3. add hop-2 PS077/PS230 as graded cb-intrinsic sources
4. combine removal and addition

All other relays, gain, visual input, and network parameters were unchanged.

### Baseline

- hop-2 pass count: 1/4
- final target pass count: 1/4
- mean hop-2 best held-out MAE: 0.1491
- mean final-target best held-out MAE: 0.1555

### Remove LLPC1/LC19/LoVP93

- hop-2 pass count: 1/4
- final target pass count: 2/4
- mean hop-2 best held-out MAE: 0.1448
- mean final-target best held-out MAE: 0.1331
- total network spike counts were also lower than baseline across the tested positions

This is a genuine causal improvement at the final descending-target level.

### Add PS077/PS230

- hop-2 pass count: 0/4
- final target pass count: 1/4
- mean hop-2 best held-out MAE: 0.2374
- mean final-target best held-out MAE: 0.1891

Adding anatomically preserving-looking cb-intrinsic groups as graded sources made the
system worse. Attribution quality therefore does not imply that forcing a group into
graded transmission is beneficial.

### Combined

- hop-2 pass count: 0/4
- final target pass count: 0/4
- mean hop-2 best held-out MAE: 0.1949
- mean final-target best held-out MAE: 0.1682

The combined intervention was not useful.

## Vertical independent validation

The removal-only intervention was then tested independently on the vertical axis.

### Baseline vertical

- hop-2 pass count: 0/4
- final target pass count: 3/4
- mean hop-2 best held-out MAE: 0.1686
- mean final-target best held-out MAE: 0.1543

### Remove LLPC1/LC19/LoVP93 vertical

- hop-2 pass count: 3/4
- final target pass count: 2/4
- mean hop-2 best held-out MAE: 0.1627
- mean final-target best held-out MAE: 0.1465

This is strong evidence that the selected visual-projection types causally distort
hop-2 spatial geometry. Removing their graded release restores held-out spatial
structure in the hop-2 layer on the vertical axis.

However, the final DN pass count falls from 3/4 to 2/4 even while final-target mean MAE
improves slightly. Therefore a cleaner hop-2 spatial code is not sufficient by itself to
produce a uniformly better descending motor representation.

## Interpretation

Two distinct problems are now separated:

1. Hop-2 relay mixture problem
   - LLPC1/LC19/LoVP93 graded release contributes causally to unstable spatial geometry.
   - removing these types can improve hop-2 representation and, on horizontal steering,
     improve final DN generalization.

2. Hop-2 to descending-neuron transformation problem
   - improving hop-2 geometry does not guarantee that DNa02/DNg13 final states inherit
     that improvement.
   - the final projection therefore performs another nontrivial transformation that can
     preserve, invert, or collapse the improved intermediate representation.

The project is no longer blocked by visual propagation. The current bottleneck is now
localized to the structure of the hop-2 relay mixture and the final hop-2→DN mapping.

## Next increment

Keep the remove-only relay policy as an experimental candidate, not a new default yet.

The next experiment should inspect hop-2→DN edge structure for DNa02/DNg13:
- excitatory versus inhibitory contact balance
- contact-weight asymmetry
- which hop-2 types dominate each DN target
- whether spatial-preserving versus unstable relay groups have systematically different
  signed/contact-weight projections onto left versus right steering DNs

Then test a restricted final-edge/readout hypothesis rather than adding a more complex
cursor decoder.
