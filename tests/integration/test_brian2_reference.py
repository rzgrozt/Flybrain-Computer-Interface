"""Small deterministic propagation check for the Brian2 reference backend."""

from flybrain_interface.connectome_data.sparse import SparseConnectivity
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.reference import Brian2ReferenceSimulator


def test_reference_simulator_is_deterministic_and_propagates() -> None:
    connectivity = SparseConnectivity.from_edges(
        neuron_count=4,
        sources=(0, 1, 1),
        targets=(1, 2, 3),
        signed_synapse_counts=(250.0, 250.0, -250.0),
    )
    simulator = Brian2ReferenceSimulator(
        connectivity=connectivity,
        populations={"input": (0,), "output": (2,)},
    )
    stimulus = DeterministicSpikeInput(neuron_indices=(0,), times_s=(0.001,))

    first = simulator.step(stimulus, duration_s=0.015)
    second = simulator.step(stimulus, duration_s=0.015)

    assert first == second
    assert first.neuron_spike_counts == (1, 1, 1, 0)
    assert first.spike_times_s[0][0] < first.spike_times_s[1][0]
    assert first.spike_times_s[1][0] < first.spike_times_s[2][0]
