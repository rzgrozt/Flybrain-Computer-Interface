# Continuous descending motor readout and propagation validation

Date: 2026-09-18. Dataset target: `male-cns:v1.0`.

This increment follows the successful spatial topographic readout. The visual system
now carries reliable position information in continuous downstream MaleCNS state, but
that diagnostic spatial decoder must not be wired directly to cursor actions.

The question here is narrower:

> Can anatomically and functionally grounded descending-neuron populations expose a
> continuous motor state under the unchanged canonical spatial visual frontend?

## Motor populations

The motor candidates are resolved directly from MaleCNS annotations.

Primary steering:

- `DNa02_L`: body ID 523769;
- `DNa02_R`: body ID 10360.

Secondary steering control:

- `DNg13_L`: body ID 11074;
- `DNg13_R`: body ID 512006.

Forward locomotion:

- `DNp09_L`: body ID 10783;
- `DNp09_R`: body ID 11177.

Backward locomotion:

- four `MDN` cells: body IDs 10763, 11288, 11332 and 12348.

These choices are not inferred from this simulation. DNa02 and DNg13 are established
steering-related descending neuron types; bilateral DNa02 differences predict turning.
DNp09/P9 is associated with forward walking and MDN with backward walking.

The artificial-body convention is therefore fixed as:

- horizontal cursor state = right DNa02 minus left DNa02;
- vertical cursor state = backward MDN mean minus forward DNp09 mean;
- positive x means screen-right;
- positive y means screen-down;
- click remains disabled.

This mapping is an artificial-body interface built from biologically grounded motor
channels. It is not a claim that a fly represents Cartesian cursor coordinates.

## Typed continuous-state boundary

`NeuralReadout` now retains the existing spike-rate fields and additionally exposes:

- population mean membrane-voltage delta in mV;
- population mean synaptic drive in mV.

The sparse runtime computes these values at the end of each chunk for configured
populations. The Brian2 reference backend exposes the same fields.

The existing spike-only `FixedPopulationMotorDecoder` remains unchanged. A new
`ContinuousPopulationMotorDecoder` consumes only the continuous descending
population state. It has no access to:

- screen coordinates;
- the topographic diagnostic decoder;
- target coordinates;
- reward;
- semantic/LLM output.

## Sandbox cursor action boundary

`VirtualCursorEnvironment` applies `MotorCommand` values to a normalized [0, 1]
cursor state. It is deliberately sandbox-only and does not move the host operating
system's mouse.

This means the action boundary is implemented, but it is not considered a validated
closed loop until the motor populations receive a usable neural signal.

## Canonical motor-candidate experiment

Configuration:

`configs/spatial-motor-candidate-readout-v1.json`.

The canonical graded spatial frontend is unchanged:

- measured retinal directions;
- connectivity-grounded R1-R6 columns;
- externally managed graded R1-R6/L1/L2/L3;
- 694 clamped early-vision nodes;
- 50 Hz equivalent projected-drive scale;
- original -52 mV rest and -45 mV threshold;
- no tonic bias.

Five horizontal and five vertical bar positions were run while watching all selected
motor candidates.

Predeclared gate:

- each primary motor population must reach at least 0.001 mV absolute peak voltage
  change.

Result: **gate set failed**.

Every watched DNa02, DNg13, DNp09 and MDN cell remained exactly at zero visual-evoked
voltage/drive delta and emitted zero spikes.

Consequently:

- horizontal DNa02 right-minus-left range = 0;
- vertical MDN-minus-DNp09 range = 0;
- position correlations are undefined.

## Anatomical path diagnosis

The absence of a motor signal is not caused by missing anatomical paths.

Shortest directed chemical paths from the spatial L1/L2/L3 source set are:

| Motor candidate | Shortest lamina-to-DN path |
| --- | ---: |
| DNa02_L | 3 hops |
| DNa02_R | 3 hops |
| DNg13_L | 2 hops |
| DNg13_R | 3 hops |
| DNp09_L | 2 hops |
| DNp09_R | 3 hops |
| MDN cells | 3 hops each |

This explains the computational boundary.

The canonical graded frontend creates strong continuous responses in the first
non-clamped optic-lobe targets, but those targets do not spike. The generic LIF graph
propagates recurrent information only when a neuron spikes. A subthreshold signal
therefore cannot cross the second and third chemical synapses into the descending
population.

## Optional tonic-excitability support

The shared LIF configuration now includes `tonic_bias_mv`.

Semantics:

- default is exactly 0.0 mV;
- it shifts the zero-input membrane equilibrium while leaving rest/reset, threshold,
  synaptic scale and delay unchanged;
- configuration rejects a tonic equilibrium at or above threshold;
- NumPy, Numba and Brian2 reference implementations use the same equation;
- the default zero-bias model preserves prior behavior.

This is an explicit model option, not a retroactive change to the canonical visual
validation.

## Predeclared tonic sensitivity v1

Configuration:

`configs/spatial-motor-tonic-sensitivity-v1.json`.

Biases were fixed before the sweep:

- 0.00 mV;
- 0.25 mV;
- 0.50 mV;
- 0.75 mV;
- 1.00 mV;
- 1.25 mV.

Success required, at the same bias:

- horizontal DNa02 channel range >= 0.001 mV;
- vertical MDN-minus-DNp09 channel range >= 0.001 mV;
- no run above 50,000 network spikes.

Result: **sensitivity gate failed**.

| Tonic bias | Max network spikes/run | Horizontal motor range | Vertical motor range |
| ---: | ---: | ---: | ---: |
| 0.00 mV | 0 | 0 | 0 |
| 0.25 mV | 0 | 0 | 0 |
| 0.50 mV | 0 | 0 | 0 |
| 0.75 mV | 1 | 0 | 0 |
| 1.00 mV | 1 | 0 | 0 |
| 1.25 mV | 1 | 0 | 0 |

Thus modest uniform tonic depolarization can produce the first isolated spike but is
not sufficient to carry the spatial signal through the 2-3 hop pathway to the chosen
motor DNs.

## Interpretation

This increment successfully establishes the software and anatomical boundaries, but
not yet functional cursor control.

Completed:

1. continuous population state is a first-class typed neural output;
2. DNa02/DNp09/MDN motor channels are resolved from MaleCNS annotations;
3. a fixed continuous motor decoder exists and cannot access target/screen/reward
   information;
4. a deterministic virtual cursor environment exists;
5. the visual-to-motor path failure has been localized to recurrent spike propagation,
   not missing visual information or missing anatomical paths;
6. a modest tonic-excitability sensitivity test was performed without changing the
   canonical result.

Not yet supported:

- claiming that the fly connectome currently moves the cursor;
- wiring the successful topographic diagnostic position decoder directly to actions;
- selecting a larger tonic bias post hoc and declaring it canonical;
- reward learning before a controllable motor signal exists.

## Next experiment

The next experiment should be separately versioned and should target the propagation
bottleneck itself.

Two defensible directions are:

1. a broader predeclared excitability regime that remains strictly subthreshold at
   zero input and includes runaway/spontaneous-activity gates; or
2. an anatomically restricted pathway experiment that identifies the exact 2-3 hop
   intermediates between graded optic-lobe output and DNa02/DNp09/MDN, then tests
   whether those intermediate populations require a different neural model or tonic
   operating point.

The second route is preferable before increasing a global bias further, because it
can determine exactly where propagation stops and avoids activating the full MaleCNS
network merely to force a motor response.
