"""Reference/runtime equivalence on a deterministic signed event graph."""

from __future__ import annotations

import numpy as np

from flybrain_interface.connectome_data.induced import InducedConnectome
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.reference import Brian2ReferenceSimulator
from flybrain_interface.simulation.runtime import SparseLIFSimulator


def test_sparse_runtime_matches_brian2_spikes_and_trajectories() -> None:
    graph = InducedConnectome(
        original_indices=np.arange(3, dtype=np.int64),
        outgoing_indptr=np.array([0, 1, 2, 3], dtype=np.int64),
        target_indices=np.array([1, 2, 1], dtype=np.int32),
        synapse_counts=np.array([250, 250, 100], dtype=np.int32),
        presynaptic_signs=np.array([1, 1, -1], dtype=np.int8),
    )
    stimulus = DeterministicSpikeInput(neuron_indices=(0, 2), times_s=(0.001, 0.001))
    watched = (0, 1, 2)
    reference = Brian2ReferenceSimulator(graph.as_sparse_connectivity()).trace(
        stimulus, duration_s=0.012, watched_indices=watched
    )
    runtime = SparseLIFSimulator(graph).run(
        stimulus, duration_s=0.012, watched_indices=watched
    )

    assert runtime.readout.neuron_spike_counts == reference.readout.neuron_spike_counts
    for actual, expected in zip(
        runtime.readout.spike_times_s,
        reference.readout.spike_times_s,
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        runtime.sample_times_s, reference.sample_times_s, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        runtime.voltage_mv, reference.voltage_mv, atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        runtime.synaptic_drive_mv,
        reference.synaptic_drive_mv,
        atol=1e-10,
        rtol=0,
    )


def test_recurrent_input_is_accepted_at_exact_refractory_boundary() -> None:
    graph = InducedConnectome(
        original_indices=np.arange(2, dtype=np.int64),
        outgoing_indptr=np.array([0, 1, 1], dtype=np.int64),
        target_indices=np.array([1], dtype=np.int32),
        synapse_counts=np.array([100], dtype=np.int32),
        presynaptic_signs=np.array([1, 1], dtype=np.int8),
    )
    # Neuron 1 spikes at 1.1 ms. Neuron 0 spikes at 1.5 ms, and its 1.8 ms
    # delayed output reaches neuron 1 at exactly 3.3 ms: the refractory boundary.
    stimulus = DeterministicSpikeInput(neuron_indices=(1, 0), times_s=(0.001, 0.0014))
    watched = (0, 1)
    reference = Brian2ReferenceSimulator(graph.as_sparse_connectivity()).trace(
        stimulus, duration_s=0.005, watched_indices=watched
    )
    runtime = SparseLIFSimulator(graph).run(
        stimulus, duration_s=0.005, watched_indices=watched
    )

    np.testing.assert_allclose(
        reference.synaptic_drive_mv[33, 1], 27.5, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        runtime.synaptic_drive_mv, reference.synaptic_drive_mv, atol=1e-12, rtol=0
    )
