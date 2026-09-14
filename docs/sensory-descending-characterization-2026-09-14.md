# Controlled sensory-to-descending characterization

Date: 2026-09-14. Dataset: `male-cns:v1.0` (166,700 neurons; 25,582,938
normalized chemical edges). This experiment asks a deliberately narrow question:
under the repository's fixed sparse LIF model, how do annotation-defined sensory
populations change annotation-defined descending-neuron spiking? It does not infer
behavior, establish biological causality, or validate a cursor controller.

## Populations and provenance

The input queries are exact MaleCNS annotation matches:

| Key | Query | Available | Selected |
| --- | --- | ---: | ---: |
| `r1_r6` | `superclass=ol_sensory, class=visual, type=R1-R6` | 3,377 | 32 |
| `orn_da1` | `superclass=cb_sensory, class=olfactory, type=ORN_DA1` | 204 | 32 |
| `jo_a1` | `superclass=cb_sensory, class=mechanosensory, subclass=auditory, type=JO-A1` | 8 | 8 |

Selection is the first `sample_limit` neurons after ascending stable `body_id`
sort. The machine-readable result records every selected body ID and neuron index.
The output queries select all 1,314 `descending_neuron` neurons, the two DNg13
neurons (body IDs 11074 and 512006), and the two DNp01 neurons (10001 and 10010).
The selection rule and all query definitions are fixed in
[`configs/sensory-descending-v1.json`](../configs/sensory-descending-v1.json); the
resolved body IDs are fixed by that config plus the recorded dataset lock.

The [official MaleCNS site](https://male-cns.janelia.org/) identifies v1.0 as the
complete adult male central nervous system release. Its overview explicitly spans
brain, optic lobes, and ventral nerve cord and gives R1-R6 to DNg13 as an example
visual-motor path. The accompanying [MaleCNS paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12636603/)
describes the dataset and annotations. Independent biological context supports
R1-R6 as outer photoreceptors in the image/motion pathway
([review](https://pmc.ncbi.nlm.nih.gov/articles/PMC2701675/)), DA1/Or67d ORNs as
cVA-selective olfactory neurons ([primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2838507/)),
and JO-A/JO-B as auditory sensory pathways
([primary study](https://pubmed.ncbi.nlm.nih.gov/32012264/)). The exact `JO-A1`
subgroup is taken from the dataset annotation; broader JO-A physiology is not
silently assigned to it.

## Protocol

Each run has 50 ms baseline followed by a 20 or 50 ms stimulus and observation to
250 ms. Each selected input neuron receives deterministic pulses every 5 ms at 4
or 8 mV. The full matrix is control plus 3 inputs × 2 amplitudes × 2 durations, with
two repeats: 26 conditions. The simulator resets between conditions, seed is zero,
and the exact `preserve` subnormal policy is used.

The runner reports baseline, stimulus, post-stimulus and total spike counts and
rates; first-spike latency relative to stimulus onset; response duration;
post-stimulus persistence; per-input-neuron normalization; sparse descending
patterns; cosine similarity and active-neuron Jaccard overlap. Silence is explicit
as `no_response=true`, with null latency/duration/persistence. Hard budgets are 90 s
wall time, 100 million visited edges, 200,000 network spikes and recorded events,
and 1 GiB process RSS.

Reproduce from a checkout with the external dataset present:

```bash
uv run python -m flybrain_interface.experiments.sensory_descending \
  --data-directory /home/ruzgar/Flybrain-Computer-Interface/data/processed/malecns-v1.0 \
  --output /tmp/sensory-descending-v1.json
```

The output records dataset lock hashes, sign policy, config SHA-256, software
versions, resolved memberships, condition parameters, measurements, and resource
use. Raw output remains outside Git by design.

## Results

The two repeats were exactly equal for all neural results and sparse patterns.
There was no spontaneous baseline activity. Values below are repeat 0; latency is
from stimulus onset and persistence is from stimulus offset to the final spike.

| Input | mV | ms | All-DN spikes | Latency ms | Persistence ms | DNg13 | DNp01 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 0 | 0 | 0 | — | — | 0 | 0 |
| R1-R6 | 4 | 20 | 0 | — | — | 0 | 0 |
| R1-R6 | 4 | 50 | 0 | — | — | 0 | 0 |
| R1-R6 | 8 | 20 | 0 | — | — | 0 | 0 |
| R1-R6 | 8 | 50 | 0 | — | — | 0 | 0 |
| ORN_DA1 | 4 | 20 | 1,103 | 11.9 | 179.6 | 0 | 0 |
| ORN_DA1 | 4 | 50 | 1,142 | 11.9 | 149.6 | 0 | 0 |
| ORN_DA1 | 8 | 20 | 1,219 | 6.9 | 179.9 | 1 | 0 |
| ORN_DA1 | 8 | 50 | 1,338 | 6.9 | 149.8 | 1 | 0 |
| JO-A1 | 4 | 20 | 429 | 13.7 | 178.4 | 0 | 0 |
| JO-A1 | 4 | 50 | 370 | 13.7 | 147.6 | 0 | 0 |
| JO-A1 | 8 | 20 | 413 | 7.2 | 178.2 | 0 | 0 |
| JO-A1 | 8 | 50 | 610 | 7.2 | 148.0 | 0 | 0 |

ORN_DA1 versus JO-A1 descending patterns were only partly overlapping: active-set
Jaccard was 0.441–0.459 and spike-count cosine similarity was 0.367–0.557 across
the four matched stimulus settings. Similarity involving R1-R6 is undefined because
its descending pattern was silent. Peak use was 88,721 network spikes, 30,533,832
visited edges, 576.5 MiB RSS, and 19.55 s total wall time.

## Interpretation and limits

Within this fixed model, the sampled olfactory and auditory populations produced
distinct but overlapping descending activity, while the sampled visual population
did not reach a descending spike threshold. Higher amplitude shortened first-spike
latency for the responsive inputs. Long persistence is model reverberation, not a
claim of sustained biological state. The isolated DNg13 spike under high-amplitude
ORN_DA1 stimulation is likewise not evidence for an olfactory steering function.

The null R1-R6 result does not show that a biological visual pathway is absent: only
32 of 3,377 R1-R6 neurons were chosen by an ID-based rule, with no retinotopic or
temporal visual encoding. DNp01 silence also does not contradict its established
role in visually evoked escape circuitry; the model uses artificial voltage pulses,
simplified point neurons and normalized chemical connections, and omits electrical
synapses and many physiological details. Descending neurons are diverse command,
population and context-dependent channels
([organization study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6019073/)); spike
counts must not be translated directly into motor direction.

The most informative next experiment is a predeclared sensitivity control with
spatially balanced/retinotopic R1-R6 samples and a sensory encoder, plus an explicit
assessment of omitted electrical connectivity. A cursor decoder should wait until
those controls establish a robust, held-out readout.
