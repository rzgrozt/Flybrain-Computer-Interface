# Spatial motor hop-geometry localization — 2026-09-21

## Question

Where does screen-position structure stop generalizing along the real MaleCNS shortest
visual-to-descending pathways after graded relay transmission restores propagation?

## Method

- Dataset: MaleCNS v1.0.
- Visual frontend: measured Zhao eye map with graded R1–R6/L1/L2/L3 channels.
- Graded intermediate relay model: gain 0.25, activation scale 7 mV.
- Relay scope: shortest effective pathways through `ol_intrinsic`,
  `visual_projection`, and `visual_centrifugal` neurons.
- Motor targets: DNa02 L/R, DNg13 L/R, DNp09 L/R, and four MDNs.
- Train positions: 0.15, 0.50, 0.85.
- Held-out positions: 0.325, 0.675.
- At every pathway hop, voltage, synaptic-drive, and spike vectors are evaluated as
  three predeclared feature modalities.
- A hop passes if at least one modality reaches held-out MAE <= 0.20 and side accuracy
  = 1.0. No feature modality is selected from the held-out set for the gate.

The neural simulation itself was run once. A methodological correction was then applied
only to the post-processing rule: the first draft selected the lowest held-out MAE
modality before gating. The corrected rule evaluates all three predeclared modalities
independently and declares failure only when none passes.

## Main result

Across the 20 axis × motor-target combinations, the first generalization failure is:

- hop 1: 7 / 20
- hop 2: 11 / 20
- hop 3: 2 / 20

Therefore hop 2 is the dominant location where spatial geometry stops generalizing.

### Horizontal

First failure counts across 10 descending targets:

- hop 1: 2
- hop 2: 7
- hop 3: 1

Notable steering pathways:

- DNa02_L retains held-out spatial decoding through hops 1 and 2, then fails at the
  descending target (hop 3).
- DNa02_R fails first at hop 2.
- DNg13_L fails first at hop 2.
- DNg13_R fails first at hop 2.
- DNp09_R also fails first at hop 2.
- DNp09_L fails already at hop 1.

For DNa02_L, spike state is especially informative:
- hop 1 spike MAE: 0.0216, side accuracy 1.0
- hop 2 spike MAE: 0.0709, side accuracy 1.0
- hop 3 has no modality satisfying both gates

This demonstrates that the visual-to-steering path can preserve spatial ordering well
into the intermediate network before the final descending representation becomes
unstable.

### Vertical

First failure counts across 10 descending targets:

- hop 1: 5
- hop 2: 4
- hop 3: 1

The DNa02/DNg13 steering targets again show a strong hop-2 pattern:
- DNa02_L: first failure hop 2
- DNa02_R: first failure hop 2
- DNg13_L: first failure hop 2
- DNg13_R: first failure hop 2

DNp09_R is an exception: all three modalities pass at hop 2, then the pathway fails at
the descending target at hop 3.

## Important interpretation

Spatial information is not simply erased monotonically with each hop. Some pathways
fail the held-out gate at an intermediate layer but become decodable again at a later
layer or target. This means the problem is better described as an unstable
transformation/aggregation of spatial state rather than simple signal disappearance.

The most reproducible common bottleneck is nevertheless hop 2, especially for the
DNa02/DNg13 steering pathways. Hop 1 frequently retains useful position information
through spike or drive patterns, while hop 2 often loses side-correct held-out
generalization.

This also explains why directly fitting a cursor decoder to final descending populations
was unstable: the motor targets inherit a transformed representation whose geometry is
not consistently aligned with screen coordinates.

## Current project state

- Retinotopic visual encoding: working.
- Graded propagation through real MaleCNS connectivity: working.
- Descending-neuron activation: working.
- Spatial information in early/intermediate motor pathways: demonstrated.
- Dominant geometry transformation bottleneck: localized to hop 2.
- Stable generalizable cursor policy: not yet demonstrated.

## Next increment

Do not add a more complex cursor decoder yet.

The next experiment should perform hop-2 relay attribution on the steering pathways:
identify which hop-2 neurons or anatomical classes preserve versus invert/destroy
held-out spatial ordering, then test restricted relay subsets causally under the same
unchanged gain=0.25 graded model.

The immediate targets are the hop-2 layers feeding DNa02 and DNg13, because both axes
show the strongest repeated failure there.
