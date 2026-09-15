# Visual pathway validation and uniform-luminance encoder

Date: 2026-09-15. Dataset: `male-cns:v1.0`. This increment diagnoses the
previous silent descending result and adds the smallest deterministic visual input
supported without inventing retinal coordinates. It characterizes a fixed model;
it does not claim literal photoreceptor physiology, biological causality, vision,
behavior, or motor control.

## Why the earlier R1-R6 run was silent

The earlier condition selected the first 32 of 3,377 exact R1-R6 annotation matches
after ascending stable body-ID sort. The reproduced 8 mV, 5 ms, 20 ms pulse train
made all 32 inputs spike four times (128 input/network spikes). Those neurons have
155 outgoing chemical edges and 3,245 contacts in the normalized graph. All 32 are
annotated histaminergic, and the recorded sign policy treats histamine as inhibitory.

The connectivity is anatomically coherent rather than absent. Across all 3,377
R1-R6 neurons, the graph contains 3,280 edges/103,405 contacts to 809 L1 neurons,
3,280/107,646 to 814 L2 neurons, and 3,076/23,510 to 782 L3 neurons. This agrees
with established lamina organization in which R1-R6 strongly contact L1, L2 and L3
([review with connectomic context](https://academic.oup.com/genetics/article/224/2/iyad064/7147603)).

The failure is at the model-activity boundary. The point-neuron network starts at a
silent resting state. R1-R6 spikes therefore deliver negative drive and
hyperpolarize L1/L2/L3; inhibition cannot make those quiescent LIF neurons spike.
The reproduced sample caused no L1/L2/L3 or descending spikes, but it did produce
measured subthreshold responses in 31 neurons of each direct target type. Maximum
hyperpolarization was 10.92 mV (L1), 13.19 mV (L2), and 7.13 mV (L3), beginning
2.0 ms after stimulus onset. A spike-only readout had hidden this real modeled
first-layer effect.

This is also a physiology mismatch, not a reason to flip signs or force spikes.
Fly photoreceptors communicate with graded voltage and tonic histamine release;
histamine gates chloride channels in L1/L2
([primary receptor study](https://www.sciencedirect.com/science/article/pii/S002192581972064X)),
and graded photoreceptor-to-interneuron transfer changes with background intensity
([electrophysiology](https://pmc.ncbi.nlm.nih.gov/articles/PMC2216927/)). The current
runtime represents neurons as spiking point LIF units. Its graph is a chemical
synapse table ([official MaleCNS download documentation](https://male-cns.janelia.org/download/))
and does not represent relevant electrical coupling: R1-R6 axons have documented
within-cartridge gap junctions
([primary ultrastructure](https://pubmed.ncbi.nlm.nih.gov/737722/)), while L1 and L2
are also electrically coupled. These omissions remain explicit.

## Mapping decision

The local MaleCNS neuron table identifies R1-R6 type, instance side and root side,
but contains no optic-column or receptive-field coordinate. It has 1,112 left and
2,265 right R1-R6 reconstructions. The official MaleCNS supplemental
[optic-column assignment](https://github.com/flyconnectome/2025malecns#optic-lobe-column-assignment)
provides column IDs for L1, R7 and R8—not individual R1-R6 cells. An R1-R6 column
might later be inferred through connectivity to L1 and validated against that table,
but this experiment does not treat that inference as established retinal geometry.
Soma XYZ is never used as retinal position.

[Flyvis](https://github.com/TuragaLab/flyvis) is a reusable, published model with
flash and moving-edge stimuli on a hexagonal visual system
([Nature paper](https://www.nature.com/articles/s41586-024-07939-3)). It uses a
different connectome-constrained, task-optimized dynamical model. Its moving-bar
renderer is therefore a useful future reference, not a drop-in source of MaleCNS
R1-R6 receptive fields or calibrated parameters. No moving bar is implemented here.

## Deterministic uniform-field transduction

The encoder accepts normalized image frames, reduces each frame to mean luminance,
and applies the same value to every reconstructed R1-R6 neuron. At 60 Hz it converts
luminance to 0–4 evenly spaced, fixed 8 mV artificial pulses per frame. Frame
timestamps remain on the independent 60 Hz clock; pulse timestamps are deterministically
rounded to the nearest 0.1 ms neural step, half upward. Inputs, frame timestamps,
per-frame luminance, pulse counts, emitted timestamps and a frame SHA-256 are
recorded. This pulse-density code is an engineering approximation to graded output,
not a physiological calibration.

Three controls use six 2×2 uniform frames (100 ms after a 50 ms baseline):

- no input: six frames at 0;
- static half luminance: six frames at 0.5, two pulses per frame;
- dark-to-bright change: three frames at 0 then three at 1, four pulses per bright
  frame.

Static and changing inputs each schedule exactly 12 pulses per photoreceptor, so
their engineered dose is matched while temporal structure differs. Both repeats
were exactly equal in neural outputs and encoded timestamps.

## Full-graph results

| Condition | R1-R6 spikes | L1 median/max hyperpolarization mV | L2 median/max | L3 median/max | DN spikes |
| --- | ---: | ---: | ---: | ---: | ---: |
| no input | 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 |
| static half | 40,524 | 20.71 / 52.20 | 21.13 / 55.06 | 3.87 / 32.33 | 0 |
| dark → bright | 40,524 | 36.51 / 92.02 | 37.25 / 97.07 | 6.83 / 56.99 | 0 |

All 809 direct L1, 814 L2 and 782 L3 targets crossed the declared 0.01 mV
subthreshold-response threshold in both nonzero conditions, but none spiked. The
static response began 6.2 ms after the 50 ms stimulus start; the change response
began 54.1 ms after start, following its first bright frame. Both remained above
0.01 mV through the 250 ms observation boundary. The larger peak for the matched-dose
step reflects temporal concentration of this engineered drive, not a validated
biological contrast response. Extreme voltages (down to −149.07 mV) show that the
pulse amplitude/contact scaling is not physiologically calibrated and should not be
interpreted quantitatively.

Peak resource use was 40,524 network spikes, 177,420 visited edges, 527.3 MiB RSS,
and 5.94 s total wall time. No event, spike, edge, memory or wall-time budget was
truncated or exceeded. Raw results remain ignored at
`/tmp/visual-pathway-validation-v1.json`.

Reproduce:

```bash
uv run python -m flybrain_interface.experiments.visual_pathway \
  --data-directory /home/ruzgar/Flybrain-Computer-Interface/data/processed/malecns-v1.0 \
  --output /tmp/visual-pathway-validation-v1.json
```

The config is
[`configs/visual-pathway-validation-v1.json`](../configs/visual-pathway-validation-v1.json).
The result records the exact 3,377 input body IDs, every direct target ID/contact
count, dataset lock hashes, sign policy, encoder parameters, timestamps, hashes,
responses and budgets.

## Panel and next experiment

The laboratory panel offers a **Uniform field · all R1-R6** preset. It uses the
actual 3,377 input IDs, observes the full direct L1/L2/L3 populations, watches the
strongest-contact member of each, and limits the Brain View selection to the inputs
plus 200 strongest targets per group (3,977 total). Its regular pulse train is an
interactive preview of the engineering proxy; the offline runner above is the
canonical independent-frame-clock protocol. Missing R1-R6 soma markers and any
non-response remain visible in the panel's coverage and activity status.

The next justified experiment is not a cursor. It is a calibrated early-vision
model comparison: add a tonic/graded photoreceptor and lamina representation as an
explicit alternative to the unchanged LIF baseline, then test sign, gain and decay
sensitivity against published flash responses. A moving bar should wait for a
validated R1-R6-to-column-to-visual-field mapping, including eye orientation and
coverage asymmetry.
