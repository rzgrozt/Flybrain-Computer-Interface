# Spatial motor directionality diagnostic — 2026-09-20

## Question

After anatomically restricted graded visual relays restored activity in DNa02, DNg13,
DNp09 and MDN populations, does that descending state already contain a robust,
generalizable cursor-control direction signal?

## Setup

- Dataset: MaleCNS v1.0.
- Visual frontend: measured Zhao eye map + graded R1–R6/L1/L2/L3 projection.
- Graded relay sensitivity point: gain 0.25, activation scale 7 mV.
- Relay scope: shortest effective paths only, restricted to `ol_intrinsic`,
  `visual_projection` and `visual_centrifugal` intermediate neurons.
- Motor observations: DNa02 L/R, DNg13 L/R, DNp09 and MDN.
- No topographic decoder is connected to the cursor.
- Cursor diagnostics use only descending-neuron state.

## Directionality result

The graded relay model successfully produces descending-neuron activity across spatial
positions, but the simple anatomical channel differences are biased rather than clean
signed steering commands.

On the five-position diagnostic grid:

- Horizontal DNa02 right-minus-left voltage correlation with position: -0.4650.
- Vertical MDN-minus-DNp09 voltage correlation with position: -0.4934.
- Both channels retain position-dependent variation, but neither crosses a stable
  center-referenced sign boundary suitable for direct cursor control.

A six-population ridge calibration trained at positions 0.15, 0.50 and 0.85 did not
generalize to 0.325 and 0.675:

- Horizontal MAE: 0.4204; both held-out targets moved the virtual cursor to the wrong
  side.
- Vertical MAE: 0.2683; only one of two held-out targets moved to the correct side.

## Exploratory feature search and independent hold-out

An exploratory all-split scan on the original five-point grid found aggregate MDN
(`backward`) spike count to be the strongest horizontal scalar feature, with all
20 leave-two-out side classifications correct on that small grid.

Because that feature was selected post hoc, two new horizontal target positions were
generated and simulated: 0.2375 and 0.7625. The decoder was fit only on 0.15, 0.50 and
0.85.

Independent hold-out result:

- Target 0.2375: predicted 0.6033, cursor command points right — incorrect.
- Target 0.7625: predicted 0.3709, cursor command points left — incorrect.

Therefore the apparent 20/20 result on the original grid was not a robust motor policy.

## Interpretation

The current model has crossed the previous propagation bottleneck: visual state reaches
and drives anatomically grounded descending neurons. The remaining problem is now
different. Descending activity is spatially modulated, but the relation between screen
geometry and the selected motor populations is not yet stable enough to serve as a
fixed cursor controller.

This means the next increment should not add more decoder complexity or directly wire
the current channels to the cursor. It should localize where target-position structure
is transformed or lost between the graded visual relay ensemble and the descending
motor populations. Member-level DN metrics are now emitted so this can be measured
without population averaging.

## Current status

- Visual spatial encoding: working.
- Graded propagation through real MaleCNS paths: working at gain 0.25.
- Descending/motor neuron activation: working.
- Position-dependent motor modulation: present.
- Generalizable signed motor policy: not yet demonstrated.
- Anatomically grounded virtual cursor control: intentionally not accepted yet.
