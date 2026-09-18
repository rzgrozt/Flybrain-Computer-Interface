# Spatial motor-pathway localization validation

Date: 2026-09-18. Dataset target: `male-cns:v1.0`.

This experiment follows the continuous motor-readout increment. The earlier motor
characterization established that DNa02, DNp09 and MDN populations remained silent
under the canonical spatial visual frontend, even though graph-theoretic paths from
L1/L2/L3 existed.

The goal here is to identify the exact intermediate neurons on those paths and measure
where the canonical visual signal disappears.

## Path definition

Only **effective fast-chemical** paths are considered.

An edge is traversable only when the presynaptic neuron has a non-zero fast synaptic
sign under the repository's fixed transmitter-sign policy. Dopamine, serotonin,
octopamine and other `sign=0` sources therefore cannot create a runtime-propagating
shortcut.

This distinction corrects one result from the previous coarse hop analysis:
`DNg13_L` had appeared two hops from the lamina source set when all anatomical
chemical edges were counted. Under the actual runtime sign policy, its shortest
effective path is three hops.

Configuration:
`configs/spatial-motor-pathway-localization-v1.json`.

## Fixed conditions

The experiment does not change the visual or neural model:

- measured Zhao eye map;
- connectivity-grounded R1-R6 columns;
- graded R1-R6/L1/L2/L3 frontend;
- 694 clamped early-vision neurons;
- 50 Hz equivalent projected-drive scale;
- -52 mV rest;
- -45 mV spike threshold;
- 0.275 mV synapse scale;
- 0 mV tonic bias;
- 0.1 ms neural timestep.

Three positions per axis are used: 0.15, 0.50 and 0.85.

For every selected motor DN, all neurons lying on a shortest effective path of at most
three chemical hops are reconstructed. The union of those non-lamina path neurons is
then recorded simultaneously.

## Anatomy-only result

All 10 selected motor neurons are reachable through effective fast-chemical paths.

The source set contains 320 spatial L1/L2/L3 neurons.

Shortest-path union:

| Layer | Unique neurons |
| --- | ---: |
| hop 1 | 1,755 |
| hop 2 | 717 |
| hop 3 | 9 |

The bounded neural watchlist contains 2,481 unique non-lamina pathway neurons, below
the configured 4,096-neuron limit.

Shortest effective distances:

| Target | Distance |
| --- | ---: |
| DNa02_L | 3 |
| DNa02_R | 3 |
| DNg13_L | 3 |
| DNg13_R | 3 |
| DNp09_L | 2 |
| DNp09_R | 3 |
| MDN_R 10763 | 3 |
| MDN_L 11288 | 3 |
| MDN_R 11332 | 3 |
| MDN_L 12348 | 3 |

## Representative highest-contact-score chains

The ranked chains below are diagnostic summaries. Ranking uses the contact counts along
the shortest path; it is not a claim that the listed chain is the unique biological
route.

| Target | Representative shortest chain | Contacts |
| --- | --- | --- |
| DNa02_L | L3_R -> MeVPLp1_R -> PS059_L -> DNa02_L | 1, 73, 244 |
| DNa02_R | L3_R -> MeVPLp1_R -> PS059_R -> DNa02_R | 1, 130, 263 |
| DNg13_L | L3_R -> MeVPLp1_R -> PS059_L -> DNg13_L | 1, 73, 9 |
| DNg13_R | L3_R -> Tm5c_R -> LoVP90b_R -> DNg13_R | 40, 6, 15 |
| DNp09_L | L1_R -> MeVPMe2_R -> DNp09_L | 1, 2 |
| DNp09_R | L3_R -> MeVP26_R -> aMe_TBD1_R -> DNp09_R | 1, 286, 22 |
| MDN_R 10763 | L3_R -> MeVP26_R -> pIP1_R -> MDN_R | 1, 20, 7 |
| MDN_L 11288 | L2_L -> Tm4_L -> LT51_L -> MDN_L | 38, 7, 2 |
| MDN_R 11332 | L3_L -> MeLo3a_L -> LT51_L -> MDN_R | 5, 1, 64 |
| MDN_L 12348 | L3_L -> MeLo3a_L -> LT51_L -> MDN_L | 5, 1, 64 |

## Canonical neural result

Run:

```bash
.venv/bin/python -m flybrain_interface.experiments.spatial_motor_pathway \
  --data-directory data/processed/malecns-v1.0 \
  --output /tmp/spatial-motor-pathway-localization-v1.json
```

All six canonical runs produced exactly zero network spikes.

Despite that global silence, hop-1 neurons on the actual motor pathways receive strong
graded visual drive.

Examples of maximum hop-1 voltage response:

| Motor pathway | Axis | Maximum hop-1 voltage delta |
| --- | --- | ---: |
| DNp09_R | horizontal | 4.534 mV |
| DNa02_R | horizontal | 3.797 mV |
| DNa02_R | vertical | 2.547 mV |
| DNa02_L | horizontal | 2.443 mV |
| DNp09_R | vertical | 2.441 mV |
| MDN_R 10763 | horizontal/vertical | 1.806 mV |
| DNp09_L | vertical | 0.019 mV |

The largest pathway-specific hop-1 response, 4.534 mV, remains below the 7 mV
rest-to-threshold distance.

Examples of strongly responding hop-1 relay neurons include:

- Dm19_R body 12021: 4.534 mV on a DNp09_R shortest-path layer;
- Dm12_R body 40659: 3.797 mV on a DNa02_R shortest-path layer;
- Tm5c_R body 63698: 2.443 mV on DNa02/DNp09-related layers;
- Dm12_L body 57089: 2.547 mV on the DNa02_R layer;
- Mi9_L body 105528: 2.127 mV on DNa02-related layers;
- Dm9_R body 36034: 1.806 mV on an MDN_R shortest-path layer.

These are not yet declared causal motor relays. They are the strongest canonical
visual responders among neurons that also lie on a shortest effective path to the
selected motor outputs.

## Exact propagation break

The result is identical across all ten motor targets and both tested axes:

1. hop 1 receives non-zero synaptic drive and subthreshold voltage;
2. hop 1 emits zero spikes;
3. hop 2 receives exactly zero synaptic drive;
4. all later hops, including the motor DN, remain exactly at baseline.

Therefore:

**last non-zero-drive layer = hop 1**

**first zero-drive layer = hop 2**

For DNa02_L, for example:

- horizontal hop 1: 705 path neurons, 459 with >=0.001 mV response,
  maximum voltage 2.443 mV, maximum drive 2.653 mV, zero spikes;
- hop 2: 216 neurons, exactly zero voltage/drive change;
- hop 3 DNa02_L: exactly zero voltage/drive change.

For DNa02_R:

- horizontal hop 1: 1,175 path neurons, 852 with >=0.001 mV response,
  maximum voltage 3.797 mV, maximum drive 4.124 mV, zero spikes;
- hop 2 and DNa02_R: exactly zero.

DNp09_L is especially informative because it is only two effective hops away:

- its hop-1 layer contains only two neurons;
- one receives measurable visual input;
- maximum voltage is 0.0097 mV horizontally and 0.0194 mV vertically;
- neither spikes;
- DNp09_L itself therefore receives exactly zero drive.

## Interpretation

The visual-to-motor failure is now localized much more precisely.

The problem is **not**:

- missing retinotopic information;
- missing anatomical connections to motor DNs;
- the final motor decoder;
- a readout dimensionality problem.

The failure occurs at the first ordinary spiking relay after the externally managed
graded lamina frontend.

The current hybrid model represents R1-R6/L1/L2/L3 as biologically motivated graded
channels, but immediately hands their outputs to generic point-LIF neurons. Those
first recurrent neurons can carry several millivolts of visual information without
crossing threshold. Because recurrent transmission in the current runtime is
spike-triggered, their continuous state is discarded at the next synapse.

## Consequence for the next experiment

A global gain increase is still not justified.

The next targeted experiment should operate only on the identified hop-1 relay set and
ask which modeling assumption is responsible for the failed relay. Candidate tests
should be predeclared and compared against the unchanged canonical baseline, for
example:

- restricted tonic depolarization applied only to shortest-path hop-1 relays;
- a restricted lower-threshold sensitivity analysis for those relays;
- a graded-transmission variant for biologically appropriate optic-lobe interneuron
  classes rather than treating every post-lamina neuron as an all-or-none point-LIF
  unit.

The first targets for that experiment can now be selected from the actual path
membership plus measured visual response rather than from the whole 166,700-neuron
connectome.

## Increment conclusion

This increment succeeds as a localization experiment.

The validated chain is now:

`graded lamina -> motor-path hop 1: strong subthreshold visual state -> no spikes
-> motor-path hop 2: zero drive -> descending motor neuron: zero state`.

That is the current mechanistic bottleneck between the working spatial visual
representation and the already implemented motor decoder / virtual cursor boundary.
