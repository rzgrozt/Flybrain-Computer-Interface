from __future__ import annotations

import numpy as np
import pytest

from flybrain_interface.connectome_data.induced import InducedConnectome
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.runtime import RuntimeBackend, SparseLIFSimulator
from flybrain_interface.simulation.trace import ChunkRecording


def _delayed_graph() -> InducedConnectome:
    return InducedConnectome(
        original_indices=np.arange(3, dtype=np.int64),
        outgoing_indptr=np.array([0, 1, 2, 2], dtype=np.int64),
        target_indices=np.array([1, 2], dtype=np.int32),
        synapse_counts=np.array([250, 250], dtype=np.int32),
        presynaptic_signs=np.array([1, 1, -1], dtype=np.int8),
    )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_chunked_execution_matches_one_run_across_delayed_boundary(
    backend: RuntimeBackend,
) -> None:
    graph = _delayed_graph()
    watched = (0, 1, 2)
    stimulus = DeterministicSpikeInput(neuron_indices=(0,), times_s=(0.001,))
    continuous = SparseLIFSimulator(graph, backend=backend)
    expected = continuous.run(stimulus, duration_s=0.006, watched_indices=watched)

    chunked = SparseLIFSimulator(graph, backend=backend)
    recording = ChunkRecording(
        watched_indices=watched,
        include_neuron_counts=True,
        max_spike_events=100,
    )
    first = chunked.advance_chunk(stimulus, duration_s=0.002, recording=recording)
    second = chunked.advance_chunk(duration_s=0.004, recording=recording)

    assert first.start_step == 0
    assert first.end_step == second.start_step == 20
    assert second.end_step == chunked.current_step == 60
    assert chunked.current_time_s == pytest.approx(0.006)
    np.testing.assert_allclose(
        np.concatenate((first.sample_times_s, second.sample_times_s)),
        expected.sample_times_s,
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        np.concatenate((first.voltage_mv, second.voltage_mv)),
        expected.voltage_mv,
        atol=1e-10,
        rtol=0,
    )
    np.testing.assert_allclose(
        np.concatenate((first.synaptic_drive_mv, second.synaptic_drive_mv)),
        expected.synaptic_drive_mv,
        atol=1e-10,
        rtol=0,
    )
    assert first.neuron_spike_counts is not None
    assert second.neuron_spike_counts is not None
    np.testing.assert_array_equal(
        first.neuron_spike_counts + second.neuron_spike_counts,
        expected.readout.neuron_spike_counts,
    )
    np.testing.assert_array_equal(
        np.concatenate((first.spike_neuron_indices, second.spike_neuron_indices)),
        np.flatnonzero(expected.readout.neuron_spike_counts),
    )
    np.testing.assert_allclose(
        np.concatenate((first.spike_times_s, second.spike_times_s)),
        [times[0] for times in expected.readout.spike_times_s if times],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(chunked.voltage_mv, continuous.voltage_mv)
    np.testing.assert_allclose(chunked.synaptic_drive_mv, continuous.synaptic_drive_mv)
    np.testing.assert_array_equal(
        chunked.refractory_steps_left, continuous.refractory_steps_left
    )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_recording_options_do_not_change_state_and_cap_events(
    backend: RuntimeBackend,
) -> None:
    graph = _delayed_graph()
    stimulus = DeterministicSpikeInput(neuron_indices=(0, 1), times_s=(0.0, 0.0))
    unrecorded = SparseLIFSimulator(graph, backend=backend)
    summary = unrecorded.advance_chunk(stimulus, duration_s=0.006)
    recorded = SparseLIFSimulator(graph, backend=backend)
    detail = recorded.advance_chunk(
        stimulus,
        duration_s=0.006,
        recording=ChunkRecording(
            watched_indices=(0,),
            include_neuron_counts=True,
            max_spike_events=1,
        ),
    )

    assert summary.neuron_spike_counts is None
    assert summary.spike_neuron_indices.size == 0
    assert summary.spike_times_s.size == 0
    assert summary.dropped_spike_events == summary.total_spikes
    assert detail.neuron_spike_counts is not None
    assert detail.spike_neuron_indices.size <= 1
    assert detail.spike_times_s.size <= 1
    assert detail.dropped_spike_events == detail.total_spikes - 1
    np.testing.assert_allclose(recorded.voltage_mv, unrecorded.voltage_mv)
    np.testing.assert_allclose(recorded.synaptic_drive_mv, unrecorded.synaptic_drive_mv)
    np.testing.assert_array_equal(
        recorded.refractory_steps_left, unrecorded.refractory_steps_left
    )


def test_chunk_stimulus_times_are_relative_and_results_are_absolute() -> None:
    simulator = SparseLIFSimulator(_delayed_graph())
    simulator.advance_chunk(duration_s=0.002)
    result = simulator.advance_chunk(
        DeterministicSpikeInput(neuron_indices=(0,), times_s=(0.0,)),
        duration_s=0.002,
        recording=ChunkRecording(max_spike_events=10),
    )

    assert result.start_step == 20
    assert result.sample_times_s[0] == pytest.approx(0.002)
    assert result.spike_times_s[0] == pytest.approx(0.0021)


def test_advance_step_preserves_state_until_explicit_reset() -> None:
    simulator = SparseLIFSimulator(_delayed_graph())

    assert simulator.advance_step((0,), amplitude_mv=8.0).size == 0
    assert simulator.current_step == 1
    assert simulator.advance_step().tolist() == [0]
    assert simulator.current_step == 2
    simulator.reset()
    assert simulator.current_step == 0
    np.testing.assert_allclose(simulator.voltage_mv, simulator.config.resting_mv)


def test_pending_delayed_events_cross_chunk_boundaries_exactly_once() -> None:
    simulator = SparseLIFSimulator(_delayed_graph())
    recording = ChunkRecording(max_spike_events=10)

    first = simulator.advance_chunk(
        DeterministicSpikeInput(neuron_indices=(0,), times_s=(0.0,)),
        duration_s=0.001,
        recording=recording,
    )
    assert first.spike_neuron_indices.tolist() == [0]
    assert simulator.pending_delayed_events == 1

    second = simulator.advance_chunk(duration_s=0.001, recording=recording)
    assert second.total_spikes == 0
    assert simulator.pending_delayed_events == 0

    third = simulator.advance_chunk(duration_s=0.003, recording=recording)
    assert third.spike_neuron_indices.tolist() == [1]
    assert simulator.visited_edges == 1


def test_default_chunks_retain_no_detailed_history() -> None:
    simulator = SparseLIFSimulator(_delayed_graph())

    for _ in range(100):
        result = simulator.advance_chunk(duration_s=0.0001)
        assert result.neuron_spike_counts is None
        assert result.spike_neuron_indices.size == 0
        assert result.spike_times_s.size == 0
        assert result.voltage_mv.shape == (1, 0)
        assert result.synaptic_drive_mv.shape == (1, 0)

    assert simulator.current_step == 100
    assert not hasattr(simulator, "__dict__")
    assert len(simulator._delay_ring) == simulator.config.delay_steps + 1


def test_default_chunk_still_reports_configured_population_rates() -> None:
    simulator = SparseLIFSimulator(_delayed_graph(), populations={"motor": (0,)})
    result = simulator.advance_chunk(
        DeterministicSpikeInput(neuron_indices=(0,), times_s=(0.0,)),
        duration_s=0.006,
    )

    assert result.neuron_spike_counts is None
    assert result.population_rates_hz["motor"] == pytest.approx(1 / 0.006)
