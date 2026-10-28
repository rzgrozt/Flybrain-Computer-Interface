"""Propagation diagnostics distinguish inhibition, silence and divergence."""

from dataclasses import replace

import numpy as np

from flybrain_interface.connectome_data.induced import InducedConnectome
from flybrain_interface.experiments.pixel_propagation_audit import (
    compare_traces,
    layer_metrics,
)
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.reference import Brian2ReferenceSimulator
from flybrain_interface.simulation.runtime import SparseLIFSimulator


def test_layer_summary_keeps_inhibition_distinct_from_silence() -> None:
    rows = {
        1: {
            "max_voltage_delta_mv": 0.0,
            "min_voltage_delta_mv": -0.5,
            "max_abs_drive_mv": 1.5,
            "spike_count": 0,
        },
        2: {
            "max_voltage_delta_mv": 0.2,
            "min_voltage_delta_mv": 0.0,
            "max_abs_drive_mv": 0.7,
            "spike_count": 3,
        },
    }
    result = layer_metrics((1, 2), rows)
    assert result["max_depolarization_mv"] == 0.2
    assert result["max_hyperpolarization_mv"] == 0.5
    assert result["max_abs_drive_mv"] == 1.5
    assert result["total_spikes"] == 3
    assert result["spiking_neuron_count"] == 1


def test_reference_gate_rejects_voltage_and_spike_divergence() -> None:
    graph = InducedConnectome(
        original_indices=np.arange(2, dtype=np.int64),
        outgoing_indptr=np.array([0, 1, 1], dtype=np.int64),
        target_indices=np.array([1], dtype=np.int32),
        synapse_counts=np.array([250], dtype=np.int32),
        presynaptic_signs=np.array([1, -1], dtype=np.int8),
    )
    stimulus = DeterministicSpikeInput(neuron_indices=(0,), times_s=(0.001,))
    reference = Brian2ReferenceSimulator(graph.as_sparse_connectivity()).trace(
        stimulus, duration_s=0.012, watched_indices=(0, 1)
    )
    actual = SparseLIFSimulator(graph).run(
        stimulus, duration_s=0.012, watched_indices=(0, 1)
    )
    tolerance = {"spike_time_s": 1e-12, "voltage_mv": 1e-10, "drive_mv": 1e-10}
    assert compare_traces(reference, actual, tolerance)["equivalent"]
    voltage_error = replace(actual, voltage_mv=actual.voltage_mv + 0.01)
    assert not compare_traces(reference, voltage_error, tolerance)["equivalent"]
    altered_readout = replace(
        actual.readout, spike_times_s=((), ()), neuron_spike_counts=(0, 0)
    )
    counts_error = replace(actual, readout=altered_readout)
    result = compare_traces(reference, counts_error, tolerance)
    assert not result["equivalent"]
    assert not result["spike_counts_equal"]
    assert result["maximum_spike_time_error_s"] is None
