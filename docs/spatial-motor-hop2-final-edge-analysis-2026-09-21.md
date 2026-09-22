# Hop-2 steering relay attribution, causal pruning, and final-edge analysis — 2026-09-21

## Summary

This increment moved the motor bottleneck from a vague hop-2 geometry failure to a
specific causal/anatomical mechanism.

The steering hop-2 layers feeding DNa02 and DNg13 contain a mixture of spatially useful
and unstable relay groups. Removing selected unstable visual-projection graded sources
changes both hop-2 geometry and final descending-neuron generalization. Final-edge
anatomy shows that the unstable group projects with a strong rightward excitatory contact
asymmetry.

However, an independent new-position hold-out shows that neither the generic six-
population ridge decoder nor individual DNa02/DNg13 scalar channels yet form a robust
cursor policy.

## Hop-2 attribution

Superclass preserve fractions across horizontal and vertical DNa02/DNg13 targets:

- visual_centrifugal: 5/6 = 0.833
- cb_intrinsic: 5/8 = 0.625
- visual_projection: 4/8 = 0.500
- ascending_neuron: 3/8 = 0.375
- descending_neuron: 2/8 = 0.250

Consistently unstable type examples:

- PLP301m: 0/8 preserve
- PLP300m: 0/8 preserve
- AN06B009: 0/8 preserve
- LLPC1: 0/4 preserve
- LC19: 0/6 preserve
- LoVP93: 0/4 preserve

Preserving examples included PS077 and PS230.

## Horizontal causal pruning

The baseline gain=0.25 graded model was compared with exact hop-2 relay
interventions.

Baseline:
- hop-2 pass: 1/4
- final DN pass: 1/4
- mean hop-2 held-out MAE: 0.1491
- mean target held-out MAE: 0.1555

Remove LLPC1/LC19/LoVP93 graded release:
- hop-2 pass: 1/4
- final DN pass: 2/4
- mean hop-2 held-out MAE: 0.1448
- mean target held-out MAE: 0.1331

Adding PS077/PS230 as graded cb-intrinsic sources made the model worse:
- hop-2 pass: 0/4
- final DN pass: 1/4
- mean hop-2 held-out MAE: 0.2374
- mean target held-out MAE: 0.1891

The combined add/remove intervention also failed.

## Vertical independent validation

Baseline:
- hop-2 pass: 0/4
- final DN pass: 3/4
- mean hop-2 held-out MAE: 0.1686
- mean target held-out MAE: 0.1543

Remove LLPC1/LC19/LoVP93:
- hop-2 pass: 3/4
- final DN pass: 2/4
- mean hop-2 held-out MAE: 0.1627
- mean target held-out MAE: 0.1465

The removal therefore strongly improves intermediate spatial geometry on an independent
axis, but that improvement is not uniformly inherited by the final descending targets.

## Hop-2 to DN edge anatomy

The final shortest-path edge layer into DNa02/DNg13 was summarized by signed contact
count.

For the unstable type group:

DNa02:
- left target signed contacts: +201
- right target signed contacts: +412
- right-minus-left signed-contact asymmetry: +211

DNg13:
- left target signed contacts: +36
- right target signed contacts: +72
- right-minus-left signed-contact asymmetry: +36

The preserving group is much more balanced for DNa02:
- left signed contacts: +73
- right signed contacts: +55

This provides an anatomical candidate mechanism for the persistent right-steering bias:
the unstable relay group contributes substantially more net excitatory contact to the
right steering targets.

## Independent new-position cursor hold-out

After selecting the removal-only relay policy, a new horizontal experiment used positions
that were not involved in the attribution or causal selection:

Training positions:
- 0.15
- 0.50
- 0.85

New held-out positions:
- 0.2375
- 0.7625

A six-population DN ridge readout still failed:
- held-out MAE: 0.5879
- left target moved left correctly
- right target still moved left incorrectly

The previously promising DNa02_R drive scalar was also independently tested and failed
both new held-out side predictions. DNa02_L and DNg13 scalar channels similarly failed
to generalize to the right held-out target.

## Interpretation

The current system now has several strongly established properties:

1. Retinotopic visual state reaches real descending motor neurons.
2. Hop-2 relay composition causally changes the quality of spatial representation.
3. LLPC1/LC19/LoVP93 graded release contributes to unstable geometry.
4. Their projections show a strong rightward excitatory contact asymmetry.
5. Removing them improves some hop-2 and final-DN metrics.
6. Final DN state is nevertheless nonlinear/non-monotonic enough that a simple fixed
   linear channel does not generalize to new target positions.

This argues against continuing to hand-pick one scalar motor channel.

The next computer-use-oriented step should use a small trainable readout whose only
inputs are anatomically selected descending-neuron state variables. This remains
connectome-grounded because screen/topographic features are never exposed to the motor
readout. The readout should be trained on many target positions and evaluated on strictly
held-out positions before being connected to the virtual cursor.

A suitable first model is a tiny regularized nonlinear readout (for example a two-layer
MLP or RBF/ridge basis) over individual DNa02/DNg13/DNp09/MDN voltage, drive, and spike
features. The experiment should compare it against linear ridge under identical
train/test splits and reject it unless unseen left/right direction generalizes.
