# Pixel input and fixed motor baseline

This experiment measures whether full pixel input reaches descending neurons under
the canonical visual/LIF model. It is an open-loop replay: motor commands are
recorded and do not change the frames, move a cursor, or control the host. There is
no reward, plasticity, model fitting, target detector, target-relative reprojection,
or action hysteresis.

## Reproduce

Acquire the normalized MaleCNS, optic-column workbook, and measured eye directions
using the existing [data](data-pipeline.md) and
[retinotopy](retinotopy-validation-2026-09-18.md) workflows. Then run:

```bash
uv run python -m flybrain_interface.experiments.pixel_fixed_motor \
  --output artifacts/pixel-fixed-motor-v1.json
```

The default stimulus is twelve 256 × 144 grayscale frames at 60 Hz: three adapted
background frames, six frames with a stationary dark 2D square, and three recovery
frames. The square geometry exists only in the stimulus generator; it is never
provided as metadata to the sensory encoder, simulator, or motor decoder.

To replay previously captured pixels, supply `--frames path/to/frames.npy`. Accepted
shapes are `(time, height, width)` and `(time, height, width, 3)` with RGB channel
order. Values must be uint8 or finite floating point in [0, 1]. The config limits
input to 60 frames and 65,536 pixels per frame. Oversized images must be cropped or
scaled before saving; the experiment does not locate or crop around a target.
File loading is memory mapped, with pickled objects disabled. The duration must
align with the neural grid. This command does not capture a live desktop session.

## Reused computation and modelling limits

The existing measured-direction retinal sampler performs bilinear image sampling
at visible optic columns; RGB channels use an arithmetic mean, an engineering
intensity proxy rather than a physiological spectral model. The existing graded
R1–R6/L1/L2/L3 approximation sends input along signed MaleCNS lamina contacts.
Those early-vision neurons remain clamped in the generic LIF runtime. All other
neurons use the unchanged sparse LIF model, without the graded relay overrides
used by the trained readout experiments.

Before integration, the upstream [flyvis](https://github.com/TuragaLab/flyvis)
visual-system model and the [Shiu LIF implementation](https://github.com/philshiu/Drosophila_brain_model)
were checked against the existing project components. This increment reuses the
project's measured MaleCNS image sampler and Shiu-style runtime instead of adding
another visual model or changing simulation equations.

The fixed artificial motor mapping is defined before observing responses:

- horizontal: `0.01 × (DNa02_R rate − DNa02_L rate)`;
- vertical: `0.01 × (MDN rate − DNp09 rate)`;
- each output is clamped to [−1, 1]; no click population is assigned.

Rates are mean spikes per neuron per simulated second. These are artificial body
assignments, not claims that the named cells biologically encode screen directions.
DNg13 populations are recorded as additional candidates but do not drive motion.
Only `NeuralReadout.population_rates_hz` reaches the existing fixed decoder.

Motor outputs are sampled in 20 ms neural chunks. Recurrent state and delayed
events persist across chunks; precomputed visual-drive slices preserve their
original delays. Only the beginning of each independent condition resets the
simulator. All frames and both spatial axes enter the same simulation.

## Controls and initial measurement — 2026-10-28

Three conditions start from the same resting state: input pixels, a uniform adapted
background of equal shape/duration, and the original pixels with visual projected
drive disconnected. A synthetic positive-control test separately confirms that
pixels can produce motor output through neural spikes when a deliberately strong
test connection is present; disconnecting that input removes the output.

The default full MaleCNS run measured:

| Condition | Peak watched downstream voltage change | Peak watched DN voltage change | Network spikes | Motion |
| --- | ---: | ---: | ---: | --- |
| Dark-square pixels | 0.32755 mV | 0 mV | 0 | none |
| Blank background | 0 mV | 0 mV | 0 | none |
| Vision disconnected | 0 mV | 0 mV | 0 | none |

The downstream statistic covers the sixteen strongest anatomical lamina targets;
the motor statistic covers the configured ten candidate DNs. These are bounded
watchlists, not a survey of every neuron's voltage. This result is consistent with
the previously localized subthreshold propagation break. It establishes input
delivery and a negative motor baseline, not learned behaviour or visual targeting.

Neural execution took about 0.47 wall seconds per 0.2 simulated seconds for the
pixel condition on the execution host (ratio about 0.43). This excludes graph
verification, setup, retinal sampling, and compilation; it is not end-to-end or
real-time performance. Timings are host dependent.

The ignored output artifact contains actual per-chunk rates, voltage/drive summaries
and motor commands, independent integrity and signal checks, input/config/code
hashes, package versions, LIF parameters, transmitter policy, and exact DN indices
and body IDs. Normalized files are checksum-verified against the data manifest
before execution. No stochastic operation is used. Failed signal checks remain
recorded without claiming success; failed integrity checks give a nonzero CLI exit.

## Relationship to historical motor results

The PCA/RBF experiments fit a supervised decision rule over DN state. Their
successful target-seeking results belong to a reservoir/readout comparison. The
live desktop probe also extracts target coordinates before synthesizing visual
probes. Neither path is evidence of a fixed artificial body learning through
connectome plasticity.

This baseline does not replace those experiments. It provides a separate,
reproducible starting point for testing biologically justified propagation changes
before adding reinforcement. The current failure to reach DNs must remain visible
when evaluating any such change.

The [propagation audit](pixel-propagation-audit.md) now applies the same pixel
stimulus to all verified shortest-path relays, adds explicit test-pulse controls,
and compares small real spiking subpaths against Brian2.
