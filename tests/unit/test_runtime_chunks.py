from __future__ import annotations

import numpy as np
import pytest

from flybrain_interface.connectome_data.induced import InducedConnectome
from flybrain_interface.sensory.drive import (
    DeterministicProjectedDriveInput,
    ProjectedDriveChannel,
)
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.config import ShiuLIFConfig
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
    assert "motor" in result.population_voltage_delta_mv
    assert "motor" in result.population_synaptic_drive_mv
    assert np.isfinite(result.population_voltage_delta_mv["motor"])
    assert np.isfinite(result.population_synaptic_drive_mv["motor"])


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_tonic_bias_is_stable_subthreshold_background(
    backend: RuntimeBackend,
) -> None:
    simulator = SparseLIFSimulator(
        _delayed_graph(),
        populations={"motor": (0,)},
        config=ShiuLIFConfig(tonic_bias_mv=1.0),
        backend=backend,
    )
    result = simulator.advance_chunk(duration_s=0.02)

    assert result.total_spikes == 0
    assert 0.6 < result.population_voltage_delta_mv["motor"] < 0.7


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_per_neuron_tonic_bias_only_changes_targeted_neuron(
    backend: RuntimeBackend,
) -> None:
    simulator = SparseLIFSimulator(
        _delayed_graph(),
        backend=backend,
        tonic_bias_overrides_mv={1: 2.0},
    )
    result = simulator.advance_chunk(
        duration_s=0.02,
        recording=ChunkRecording(watched_indices=(0, 1, 2)),
    )

    assert result.total_spikes == 0
    assert result.voltage_mv[-1, 0] == pytest.approx(simulator.config.resting_mv)
    assert result.voltage_mv[-1, 2] == pytest.approx(simulator.config.resting_mv)
    assert result.voltage_mv[-1, 1] > simulator.config.resting_mv + 1.2


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_per_neuron_threshold_only_changes_targeted_spike_gate(
    backend: RuntimeBackend,
) -> None:
    control = SparseLIFSimulator(_delayed_graph(), backend=backend)
    targeted = SparseLIFSimulator(
        _delayed_graph(),
        backend=backend,
        threshold_overrides_mv={1: -50.0},
    )
    stimulus = DeterministicSpikeInput(
        neuron_indices=(1,),
        times_s=(0.0,),
        amplitude_mv=3.0,
    )

    control_result = control.advance_chunk(
        stimulus,
        duration_s=0.0003,
        recording=ChunkRecording(include_neuron_counts=True),
    )
    targeted_result = targeted.advance_chunk(
        stimulus,
        duration_s=0.0003,
        recording=ChunkRecording(include_neuron_counts=True),
    )

    assert control_result.neuron_spike_counts is not None
    assert targeted_result.neuron_spike_counts is not None
    assert control_result.neuron_spike_counts[1] == 0
    assert targeted_result.neuron_spike_counts[1] == 1


def test_per_neuron_excitability_overrides_are_validated() -> None:
    graph = _delayed_graph()
    with pytest.raises(ValueError, match="outside network"):
        SparseLIFSimulator(graph, tonic_bias_overrides_mv={99: 1.0})
    with pytest.raises(ValueError, match="exceed resting_mv"):
        SparseLIFSimulator(graph, threshold_overrides_mv={1: -53.0})
    with pytest.raises(ValueError, match="zero-input equilibrium"):
        SparseLIFSimulator(
            graph,
            tonic_bias_overrides_mv={1: 4.0},
            threshold_overrides_mv={1: -49.0},
        )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_graded_relay_transmits_subthreshold_depolarization(
    backend: RuntimeBackend,
) -> None:
    control = SparseLIFSimulator(_delayed_graph(), backend=backend)
    graded = SparseLIFSimulator(
        _delayed_graph(),
        backend=backend,
        graded_relay_indices=(0,),
        graded_relay_gain=1.0,
        graded_relay_activation_scale_mv=7.0,
    )
    control.voltage_mv[0] = -49.0
    graded.voltage_mv[0] = -49.0

    control_result = control.advance_chunk(
        duration_s=0.0001,
        recording=ChunkRecording(watched_indices=(1,)),
    )
    graded_result = graded.advance_chunk(
        duration_s=0.0001,
        recording=ChunkRecording(watched_indices=(1,)),
    )

    assert control_result.synaptic_drive_mv[0, 0] == pytest.approx(0.0)
    assert graded_result.synaptic_drive_mv[0, 0] > 0.0
    assert graded_result.total_spikes == 0


def test_graded_relay_parameters_are_validated() -> None:
    graph = _delayed_graph()
    with pytest.raises(ValueError, match="outside network"):
        SparseLIFSimulator(graph, graded_relay_indices=(99,), graded_relay_gain=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        SparseLIFSimulator(graph, graded_relay_indices=(0,), graded_relay_gain=-1.0)
    with pytest.raises(ValueError, match="positive"):
        SparseLIFSimulator(
            graph,
            graded_relay_indices=(0,),
            graded_relay_gain=1.0,
            graded_relay_activation_scale_mv=0.0,
        )


def _projected_drive(
    amplitudes: list[float],
    *,
    target: int = 1,
    weight: float = 3.0,
    dt_ms: float = 0.1,
) -> DeterministicProjectedDriveInput:
    return DeterministicProjectedDriveInput(
        channels=(
            ProjectedDriveChannel(
                label="graded-test",
                target_indices=np.asarray([target], dtype=np.int64),
                signed_contact_weights=np.asarray([weight], dtype=np.float64),
                amplitudes_mv=np.asarray(amplitudes, dtype=np.float64),
                anatomical_source_count=1,
                anatomical_edge_count=1,
            ),
        ),
        dt_ms=dt_ms,
    )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_zero_projected_drive_preserves_baseline_state_exactly(
    backend: RuntimeBackend,
) -> None:
    baseline = SparseLIFSimulator(_delayed_graph(), backend=backend)
    driven = SparseLIFSimulator(_delayed_graph(), backend=backend)
    duration_s = 0.0004

    baseline_result = baseline.advance_chunk(
        duration_s=duration_s,
        recording=ChunkRecording(watched_indices=(0, 1, 2)),
    )
    driven_result = driven.advance_chunk(
        duration_s=duration_s,
        recording=ChunkRecording(watched_indices=(0, 1, 2)),
        projected_drive=_projected_drive([0.0, 0.0, 0.0, 0.0]),
    )

    np.testing.assert_array_equal(driven.voltage_mv, baseline.voltage_mv)
    np.testing.assert_array_equal(
        driven.synaptic_drive_mv, baseline.synaptic_drive_mv
    )
    np.testing.assert_array_equal(
        driven.refractory_steps_left, baseline.refractory_steps_left
    )
    np.testing.assert_array_equal(driven_result.voltage_mv, baseline_result.voltage_mv)
    np.testing.assert_array_equal(
        driven_result.synaptic_drive_mv, baseline_result.synaptic_drive_mv
    )
    assert driven.visited_edges == baseline.visited_edges == 0
    assert driven.projected_drive_target_updates == 0


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_projected_drive_enters_synaptic_state_without_source_spikes(
    backend: RuntimeBackend,
) -> None:
    simulator = SparseLIFSimulator(_delayed_graph(), backend=backend)
    result = simulator.advance_chunk(
        duration_s=0.0003,
        recording=ChunkRecording(watched_indices=(1,)),
        projected_drive=_projected_drive([2.0, 0.0, 0.0]),
    )

    assert result.total_spikes == 0
    assert result.synaptic_drive_mv[0, 0] == pytest.approx(6.0)
    assert result.voltage_mv[0, 0] == pytest.approx(simulator.config.resting_mv)
    assert result.voltage_mv[1, 0] > simulator.config.resting_mv
    assert simulator.visited_edges == 0
    assert simulator.projected_drive_target_updates == 1


def test_projected_drive_is_suppressed_while_target_is_refractory() -> None:
    simulator = SparseLIFSimulator(_delayed_graph(), backend="numpy")
    simulator.synaptic_drive_mv[1] = 3.0
    simulator.refractory_steps_left[1] = 1

    result = simulator.advance_chunk(
        duration_s=0.0001,
        recording=ChunkRecording(watched_indices=(1,)),
        projected_drive=_projected_drive([2.0]),
    )

    assert result.synaptic_drive_mv[0, 0] == pytest.approx(3.0)
    assert simulator.synaptic_drive_mv[1] == pytest.approx(3.0)


def test_projected_drive_must_match_chunk_grid_and_network_bounds() -> None:
    simulator = SparseLIFSimulator(_delayed_graph())

    with pytest.raises(ValueError, match="dt_ms"):
        simulator.advance_chunk(
            duration_s=0.0001,
            projected_drive=_projected_drive([1.0], dt_ms=0.2),
        )
    with pytest.raises(ValueError, match="trace length"):
        simulator.advance_chunk(
            duration_s=0.0002,
            projected_drive=_projected_drive([1.0]),
        )
    with pytest.raises(ValueError, match="outside network"):
        simulator.advance_chunk(
            duration_s=0.0001,
            projected_drive=_projected_drive([1.0], target=99),
        )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_clamped_neuron_ignores_external_and_projected_drive(
    backend: RuntimeBackend,
) -> None:
    simulator = SparseLIFSimulator(
        _delayed_graph(),
        backend=backend,
        clamped_indices=(1,),
    )
    stimulus = DeterministicSpikeInput(
        neuron_indices=(1,),
        times_s=(0.0,),
        amplitude_mv=100.0,
    )
    result = simulator.advance_chunk(
        stimulus,
        duration_s=0.0003,
        recording=ChunkRecording(
            watched_indices=(1,),
            include_neuron_counts=True,
        ),
        projected_drive=_projected_drive([10.0, 10.0, 10.0], target=1),
    )

    assert result.neuron_spike_counts is not None
    assert result.neuron_spike_counts[1] == 0
    np.testing.assert_allclose(
        result.voltage_mv[:, 0],
        simulator.config.resting_mv,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(result.synaptic_drive_mv[:, 0], 0.0, atol=0.0)
    assert simulator.voltage_mv[1] == simulator.config.resting_mv
    assert simulator.synaptic_drive_mv[1] == 0.0
    assert simulator.refractory_steps_left[1] == 0


def test_clamped_indices_are_validated() -> None:
    with pytest.raises(ValueError, match="unique"):
        SparseLIFSimulator(_delayed_graph(), clamped_indices=(1, 1))
    with pytest.raises(ValueError, match="outside network"):
        SparseLIFSimulator(_delayed_graph(), clamped_indices=(99,))
