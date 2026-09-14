from __future__ import annotations

from math import exp

import numpy as np
import pytest

from flybrain_interface.connectome_data.induced import InducedConnectome
from flybrain_interface.experiments.diagnose_subnormal import diagnose_subnormal_cost
from flybrain_interface.simulation.runtime import RuntimeBackend, SparseLIFSimulator


def _disconnected_graph(neuron_count: int = 2) -> InducedConnectome:
    return InducedConnectome(
        original_indices=np.arange(neuron_count, dtype=np.int64),
        outgoing_indptr=np.zeros(neuron_count + 1, dtype=np.int64),
        target_indices=np.empty(0, dtype=np.int32),
        synapse_counts=np.empty(0, dtype=np.int32),
        presynaptic_signs=np.ones(neuron_count, dtype=np.int8),
    )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_zero_policy_removes_small_signed_drive_without_changing_voltage(
    backend: RuntimeBackend,
) -> None:
    graph = _disconnected_graph()
    exact = SparseLIFSimulator(graph, backend=backend)
    zeroed = SparseLIFSimulator(graph, backend=backend, subnormal_drive_policy="zero")
    smallest_normal = np.finfo(np.float64).tiny
    signed = np.array([smallest_normal, -smallest_normal])
    exact.synaptic_drive_mv[:] = signed
    zeroed.synaptic_drive_mv[:] = signed

    exact.advance_step()
    zeroed.advance_step()

    expected = signed * exp(-exact.config.dt_ms / exact.config.synapse_tau_ms)
    np.testing.assert_array_equal(exact.synaptic_drive_mv, expected)
    np.testing.assert_array_equal(zeroed.synaptic_drive_mv, [0.0, 0.0])
    np.testing.assert_array_equal(zeroed.voltage_mv, exact.voltage_mv)
    assert not np.signbit(zeroed.synaptic_drive_mv).any()


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_zero_policy_preserves_smallest_normal_decay_result(
    backend: RuntimeBackend,
) -> None:
    simulator = SparseLIFSimulator(
        _disconnected_graph(1), backend=backend, subnormal_drive_policy="zero"
    )
    cutoff = simulator._minimum_preserved_drive_mv
    simulator.synaptic_drive_mv[0] = cutoff

    simulator.advance_step()

    assert simulator.synaptic_drive_mv[0] == np.finfo(np.float64).tiny


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_zero_policy_does_not_change_threshold_boundary(
    backend: RuntimeBackend,
) -> None:
    graph = _disconnected_graph(1)
    exact = SparseLIFSimulator(graph, backend=backend)
    zeroed = SparseLIFSimulator(graph, backend=backend, subnormal_drive_policy="zero")
    below_threshold = np.nextafter(exact.config.threshold_mv, -np.inf)
    for simulator in (exact, zeroed):
        simulator.voltage_mv[0] = below_threshold
        simulator.synaptic_drive_mv[0] = np.finfo(np.float64).tiny

    assert exact.advance_step().size == 0
    assert zeroed.advance_step().size == 0
    np.testing.assert_array_equal(zeroed.voltage_mv, exact.voltage_mv)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_zero_policy_is_state_continuous_across_chunks(
    backend: RuntimeBackend,
) -> None:
    graph = _disconnected_graph()
    continuous = SparseLIFSimulator(
        graph, backend=backend, subnormal_drive_policy="zero"
    )
    chunked = SparseLIFSimulator(graph, backend=backend, subnormal_drive_policy="zero")
    initial = np.array([1e-307, -1e-307])
    continuous.synaptic_drive_mv[:] = initial
    chunked.synaptic_drive_mv[:] = initial

    continuous.advance_chunk(duration_s=0.02)
    chunked.advance_chunk(duration_s=0.007)
    chunked.advance_chunk(duration_s=0.013)

    np.testing.assert_array_equal(chunked.voltage_mv, continuous.voltage_mv)
    np.testing.assert_array_equal(
        chunked.synaptic_drive_mv, continuous.synaptic_drive_mv
    )
    np.testing.assert_array_equal(
        chunked.refractory_steps_left, continuous.refractory_steps_left
    )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_zero_policy_changes_only_subnormal_drive_after_long_decay(
    backend: RuntimeBackend,
) -> None:
    graph = _disconnected_graph(1)
    exact = SparseLIFSimulator(graph, backend=backend)
    zeroed = SparseLIFSimulator(graph, backend=backend, subnormal_drive_policy="zero")
    exact.synaptic_drive_mv[0] = 27.5
    zeroed.synaptic_drive_mv[0] = 27.5

    exact_result = exact.advance_chunk(duration_s=3.7)
    zeroed_result = zeroed.advance_chunk(duration_s=3.7)

    assert exact_result.total_spikes == zeroed_result.total_spikes == 0
    assert 0.0 < exact.synaptic_drive_mv[0] < np.finfo(np.float64).tiny
    assert zeroed.synaptic_drive_mv[0] == 0.0
    assert (
        abs(exact.synaptic_drive_mv[0] - zeroed.synaptic_drive_mv[0])
        < np.finfo(np.float64).tiny
    )
    np.testing.assert_array_equal(zeroed.voltage_mv, exact.voltage_mv)


def test_subnormal_drive_policy_is_explicit() -> None:
    with pytest.raises(ValueError, match="unsupported subnormal drive policy"):
        SparseLIFSimulator(
            _disconnected_graph(),
            subnormal_drive_policy="unsupported",  # type: ignore[arg-type]
        )


def test_subnormal_diagnostic_covers_both_policies_and_value_classes() -> None:
    result = diagnose_subnormal_cost(neuron_count=2, steps=2, repeats=1)

    assert len(result["results"]) == 8
    assert {entry["policy"] for entry in result["results"]} == {
        "preserve",
        "zero",
    }
    assert result["zero_policy_cutoff_mv"] > result["smallest_normal_float64"]
