# Anatomically grounded 1D virtual cursor control — 2026-09-21

## Milestone

The project now demonstrates closed-loop horizontal cursor control driven by MaleCNS
descending-neuron state on target/start configurations that were not used to train or
select the final motor readout.

The validated control chain is:

screen-relative target
→ measured/graded visual frontend
→ real MaleCNS connectivity
→ anatomically restricted graded relay propagation
→ broad descending-neuron temporal state
→ tiny trainable motor readout
→ LEFT/RIGHT motor command
→ virtual cursor
→ updated cursor-relative visual input
→ repeat

No screen coordinate, target coordinate, topographic decoder output, or hand-coded
target direction is exposed to the motor readout at inference time.

## Why the readout changed

The earlier fixed readouts over ten named DNs (DNa02/DNg13/DNp09/MDN) were not
sufficiently stable on unseen positions.

Broad recording showed that 1,310 descending neurons are reachable from the lamina
sources within three effective hops. Recording their temporal voltage/drive state
provided a richer motor-side reservoir without exposing visual coordinates to the
decoder.

Several simpler approaches were rejected:

- ten-DN linear position regression: failed held-out direction
- ten-DN RBF position regression: failed held-out direction
- ten-DN binary and three-way classifiers: 4/6 on held-out positions
- broad-DN linear PCA direction model with sparse training: 3/4
- broad-DN supervised feature selection: 2/4
- broad-DN PCA regression: 4/6 side accuracy
- dense broad-DN linear PCA direction model: 5/8 on independent midpoint positions
- PCA k-NN: 5/8 on independent midpoint positions

## Broad temporal representation

The broad state recorder watches 1,310 descending neurons reachable from lamina sources
within three effective hops. Temporal state is summarized in three bins per visual
presentation.

The final readout uses temporal voltage state from this descending population.

The relay model remains the pruned gain=0.25 graded model:
- graded superclasses: ol_intrinsic, visual_projection, visual_centrifugal
- excluded graded types: LLPC1, LC19, LoVP93

This pruning came from the prior hop-2 attribution and causal experiments.

## Independent midpoint development validation

The initial broad-DN artifact contained 14 non-center directional positions. A separate
MaleCNS simulation generated eight new midpoint positions:

0.1333, 0.20, 0.2667, 0.3333, 0.6667, 0.7333, 0.80, 0.8667

A whitened PCA + RBF classifier selected only from the original training set achieved:

- train-only LOOCV accuracy: 0.9286
- independent midpoint accuracy: 0.875 (7/8)

The single midpoint error was at 0.6667.

Whitening the PCA latent solved the numerical underflow observed in the first RBF
prototype while preserving the 7/8 validation result.

## Final readout training

After architecture selection, the final RBF readout was fit using all 22 available
directional positions from the original and midpoint artifacts.

Train-only model selection chose:

- modality: temporal_voltage
- PCA components: 6
- RBF gamma: 0.3
- ridge alpha: 0.1
- LOOCV accuracy: 0.9545

No closed-loop validation episode was used for model selection.

## Final closed-loop generalization

Four new start/target pairs were chosen after the static readout architecture was fixed:

- start 0.51 → target 0.13
- start 0.49 → target 0.87
- start 0.19 → target 0.56
- start 0.81 → target 0.44

These target/start pairs were not members of the static training grids.

At every closed-loop step:

1. the target was expressed in cursor-relative visual coordinates
2. that visual state was injected through the same MaleCNS visual frontend
3. the full neural simulation ran
4. only broad descending-neuron temporal state was passed to the RBF readout
5. the readout emitted LEFT or RIGHT
6. the virtual cursor moved
7. the next observation reflected the new cursor-target relation

Result:

- episode success: 4/4
- success fraction: 1.0
- mean distance reduction: 0.3335
- final distances:
  - target 0.13: 0.0465
  - target 0.87: 0.0465
  - target 0.56: 0.0365
  - target 0.44: 0.0365
- success tolerance: 0.07

Every executed step in the final four episodes reduced target distance. No action
reversal was required by hysteresis in the successful traces.

## Interpretation

This is the first robust computer-control milestone in the project.

The cursor is no longer being driven by:
- the diagnostic topographic decoder
- direct screen coordinates
- a semantic model
- a hand-coded target direction

Instead, the action comes from a lightweight trainable readout over the temporal state of
an anatomically constrained MaleCNS descending-neuron reservoir.

The readout is still artificial and trainable, which is consistent with the project
design: the fixed connectome acts as the biological reservoir while small readouts can
learn task-specific control.

## Next step

The next increment should extend the same methodology to the vertical axis:

1. record broad descending temporal state for vertical target positions
2. train and independently validate a vertical UP/DOWN readout
3. run vertical closed-loop cursor episodes
4. combine the independently validated horizontal and vertical readouts into a 2D
   virtual target-seeking controller
5. only after 2D generalization is stable, connect the controller to sandboxed OS-level
   pointer actions
