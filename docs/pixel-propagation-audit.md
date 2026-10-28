# Pixel propagation audit

This experiment extends the [fixed motor pixel baseline](pixel-fixed-motor.md)
with the existing anatomical pathway-localization tools, controlled neural test
pulses, and the existing Brian2 reference. It changes no synaptic weights, neural
thresholds, tonic biases, transmitter policies, or graded relay settings.

```bash
uv run python -m flybrain_interface.experiments.pixel_propagation_audit \
  --output artifacts/pixel-propagation-audit-v1.json
```

It uses the same local data prerequisites as the baseline. The configuration is
`configs/pixel-propagation-audit-v1.json`. Outputs remain ignored under `artifacts/`.
No host action is performed. Decoder commands are recorded numerical values.

## Observation

The existing graph code reconstructs all shortest effective chemical paths from
the 320 spatial lamina sources to the ten configured motor-candidate neurons.
The union contains 2,481 non-lamina neurons, within a 4,096-neuron watch bound.
Every watched neuron has peak positive/negative voltage change, absolute drive,
and total spikes recorded across persistent 20 ms chunks. Raw traces are discarded
after each chunk. Layer membership is relative to a target's shortest paths;
layers from different targets can overlap and must not be summed as disjoint
populations. Exact indices and annotations distinguish repeated instance names.

The same 2D dark-square sequence is replayed in three conditions: pixels, blank
background, and pixels with projected visual drive disconnected. There are no
coordinates or trained readouts in the observation-to-motor path.

## Default result — 2026-10-28

All ten motor candidates are anatomically reachable within two or three hops.
None spikes under the default pixels. Across the complete pathway watchlist:

- first-hop depolarization reaches 1.58930 mV and hyperpolarization 1.32129 mV;
- the largest first-hop absolute drive is 1.99017 mV;
- no first-hop neuron spikes; the configured resting-to-threshold gap is 7 mV;
- every selected path's second and later layers have zero drive and zero spikes;
- blank and disconnected-vision controls produce zero network spikes.

Eight target pathways have first-hop drive under this particular square. The
DNp09_L path and one MDN_L path receive no first-hop drive for this stimulus, so
their first measured loss is earlier. These are stimulus-specific observations,
not evidence that those paths cannot respond to other images.

The 1.58930 mV maximum exceeds the earlier 0.32755 mV baseline observation because
the audit watches all shortest-path relays rather than only sixteen strongest
lamina targets. The input and neural parameters are unchanged.

## Controlled stimulation

Two target routes are selected by the existing contact-count ranking, independent
of measured responses: one to DNa02_L and one to DNp09_L. At each non-lamina node,
three explicit 8 mV voltage pulses at 1, 5, and 9 ms are applied in a separate
20 ms full-graph run starting from rest, with no pixels. These are engineering
positive controls, not physiological stimulation assumptions, perception, or a
behavioural policy.

| Stimulated node | Local spikes | Observed downstream effect | Fixed motor output |
| --- | ---: | --- | --- |
| MeVPLp1_R, hop 1 toward DNa02_L | 3 | PS059_L spikes once; DNa02_L is inhibited | zero |
| PS059_L, hop 2 toward DNa02_L | 3 | DNa02_L hyperpolarizes about 30.16 mV | zero |
| DNa02_L, hop 3 | 3 | Direct DN positive control | dx = −1, dy = 0 |
| MeVPMe2_R, hop 1 toward DNp09_L | 3 | DNp09_L hyperpolarizes about 0.247 mV | zero |
| DNp09_L, hop 2 | 3 | Direct DN positive control; other DNs also respond | dx = −0.5, dy = −0.75 |

Inhibition is expected on the named PS059 and MeVPMe2 outputs under the declared
GABA/glutamate sign policy. Anatomical reachability does not imply an excitatory
motor route. Full-graph stimulation can recruit collateral pathways; a response
cannot be attributed uniquely to one edge from these measurements alone.

## Reference validation

Each selected route's non-lamina spiking suffix is extracted using the existing
induced-connectome loader, retaining real signed contacts among selected nodes.
The same pulse protocol is applied successively to each local node. Both NumPy
and Numba runtime traces are compared to Brian2 for five cases, ten comparisons.
All comparisons pass: spike counts match, spike times agree within 1e-12 s, and
voltage/drive trajectories agree within 1e-10 mV. The measured maximum voltage
error was approximately 2.8e-13 mV.

This validates the spiking suffix and test-pulse dynamics. It does not independently
validate the graded sensory approximation, every full-graph connection, biological
transmitter effects, or a learned policy. Full-graph collateral responses are not
expected to equal an isolated induced suffix.

The output includes config/code/data hashes, versions, LIF parameters, sign policy,
retinal provenance, watched measurements, annotated selected routes, intervention
responses, and reference errors. Integrity failures produce a nonzero CLI exit.
Absence of sensory-driven motor activity remains an experimental result.

## Interpretation

The new pixels reproduce the previously localized graded-to-spiking propagation
gap. The selected downstream circuitry transmits controlled spikes, and the fixed
motor decoder responds to direct DN activity. The first-hop sensory response is
subthreshold under this model and stimulus; this blocks recurrent spike delivery.

The next modelling investigation should test the sensory-to-relay representation
and excitability assumptions against upstream implementations and physiological
evidence. Simply increasing gain until the cursor moves would not establish a
biological explanation. Keep this unchanged baseline and its controls for any
future comparison.
