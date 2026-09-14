# Runtime graph

The runtime loader opens both normalized sparse orientations with NumPy memory
mapping. The four 25,582,938-element edge arrays therefore remain read-only
file-backed storage. Only the compact neuron annotations, one sign byte per neuron,
and per-step activity/output vectors need ordinary process memory.

```python
from pathlib import Path

from flybrain_interface.connectome_data.runtime import MemoryMappedConnectome

graph = MemoryMappedConnectome.load(Path("data/processed/malecns-v1.0"))
sensory = graph.catalog.select(superclass="sensory")
incoming = graph.incoming(target_index=0)
outgoing = graph.outgoing(source_index=0)
```

`graph.propagate(activity)` evaluates the unsigned contact matrix as
`contacts @ (activity * presynaptic_sign)`. This applies transmitter sign by the
presynaptic neuron and avoids materializing a second, signed 25.6-million-edge
matrix.

`graph.accumulate_spikes(spiking_indices, destination)` follows only source-major
outgoing edges touched by the current spike event. The caller owns and reuses the
dense `float64` destination vector, avoiding one full-sized allocation per neural
step. The method returns the number of non-modulatory edges visited, which is useful
for performance telemetry.

## Default sign policy

The default policy is an explicit modeling approximation:

| MaleCNS transmitter | Fast-current sign |
| --- | ---: |
| acetylcholine | +1 |
| GABA | -1 |
| glutamate | -1 |
| histamine | -1 |
| dopamine | 0 |
| serotonin | 0 |
| octopamine | 0 |
| unclear or missing | 0 |

A value of zero means “separated from ordinary fast transmission,” not “without a
biological effect.” Monoamines will enter through later neuromodulatory mechanisms.
Ground-truth transmitter annotations override consensus predictions when present.

The package also exposes an excitatory-glutamate policy so that the ambiguous sign
of central glutamatergic transmission can be treated as an experimental variable.
The [Shiu whole-brain reference model](https://doi.org/10.1038/s41586-024-07763-9)
motivates the inhibitory-glutamate baseline. However, the MaleCNS transmitter
identity is still a prediction/annotation and does not by itself identify the
postsynaptic receptor or guarantee a universal physiological sign. The underlying
[EM transmitter-classification study](https://doi.org/10.1016/j.cell.2024.03.016)
explicitly notes that glutamate can have either sign in the fly, depending on the
receptor context.

## Scope and next constraint

The source-major representation makes propagation cost depend on the number and
out-degree of neurons that actually spike. The current kernel deliberately uses a
small Python loop over spiking neurons and vectorized NumPy updates over each
neuron's targets. Profiling will determine whether a compiled CPU kernel is needed.

This loader still does not perform neural dynamics, plasticity, or motor control.
The next simulation step is a deterministic sparse LIF state engine with delayed
event delivery, validated against the Brian2 reference on induced subgraphs.
