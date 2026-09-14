"""Validate the sparse LIF runtime against Brian2 on a real MaleCNS subgraph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from flybrain_interface.connectome_data.induced import strongest_outgoing_neighborhood
from flybrain_interface.connectome_data.runtime import MemoryMappedConnectome
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.reference import Brian2ReferenceSimulator
from flybrain_interface.simulation.runtime import SparseLIFSimulator
from flybrain_interface.simulation.trace import SimulationTrace


def validate_runtime(
    data_directory: Path,
    *,
    neuron_count: int = 64,
    duration_s: float = 0.012,
) -> dict[str, object]:
    graph = MemoryMappedConnectome.load(data_directory)
    induced = strongest_outgoing_neighborhood(graph, neuron_count=neuron_count)
    connectivity = induced.as_sparse_connectivity()
    stimulus = DeterministicSpikeInput(neuron_indices=(0,), times_s=(0.001,))
    watched = tuple(range(neuron_count))

    reference_started = perf_counter()
    reference = Brian2ReferenceSimulator(connectivity).trace(
        stimulus,
        duration_s=duration_s,
        watched_indices=watched,
    )
    reference_seconds = perf_counter() - reference_started

    runtime_simulator = SparseLIFSimulator(induced)
    runtime_started = perf_counter()
    runtime = runtime_simulator.run(
        stimulus,
        duration_s=duration_s,
        watched_indices=watched,
    )
    runtime_seconds = perf_counter() - runtime_started

    counts_equal = (
        runtime.readout.neuron_spike_counts == reference.readout.neuron_spike_counts
    )
    spike_time_error = _maximum_spike_time_error(reference, runtime)
    voltage_error = float(np.max(np.abs(runtime.voltage_mv - reference.voltage_mv)))
    drive_error = float(
        np.max(np.abs(runtime.synaptic_drive_mv - reference.synaptic_drive_mv))
    )
    equivalent = (
        counts_equal
        and spike_time_error <= 1e-12
        and voltage_error <= 1e-10
        and drive_error <= 1e-10
    )
    if not equivalent:
        raise RuntimeError(
            "sparse runtime diverged from the Brian2 reference: "
            f"counts_equal={counts_equal}, spike_time_error={spike_time_error}, "
            f"voltage_error={voltage_error}, drive_error={drive_error}"
        )

    full_runtime = SparseLIFSimulator(graph)
    full_started = perf_counter()
    full_runtime.run(
        DeterministicSpikeInput(neuron_indices=(), times_s=()),
        duration_s=0.001,
        watched_indices=(int(induced.original_indices[0]),),
    )
    full_runtime_seconds = perf_counter() - full_started

    seed_original_index = int(induced.original_indices[0])
    return {
        "scope": "induced real MaleCNS subgraph",
        "neurons": induced.neuron_count,
        "edges": induced.edge_count,
        "seed_neuron_index": seed_original_index,
        "seed_body_id": int(graph.catalog.body_ids([seed_original_index])[0]),
        "duration_s": duration_s,
        "dt_ms": runtime_simulator.config.dt_ms,
        "reference_seconds": reference_seconds,
        "runtime_seconds": runtime_seconds,
        "runtime_speedup": reference_seconds / runtime_seconds,
        "runtime_visited_edges": runtime_simulator.visited_edges,
        "full_network_smoke_steps": 10,
        "full_network_smoke_seconds": full_runtime_seconds,
        "full_network_seconds_per_step": full_runtime_seconds / 10,
        "total_spikes": sum(runtime.readout.neuron_spike_counts),
        "spike_counts_equal": counts_equal,
        "maximum_spike_time_error_s": spike_time_error,
        "maximum_voltage_error_mv": voltage_error,
        "maximum_synaptic_drive_error_mv": drive_error,
        "equivalent": equivalent,
    }


def _maximum_spike_time_error(
    reference: SimulationTrace, runtime: SimulationTrace
) -> float:
    reference_readout = reference.readout
    runtime_readout = runtime.readout
    maximum = 0.0
    for expected, actual in zip(
        reference_readout.spike_times_s,
        runtime_readout.spike_times_s,
        strict=True,
    ):
        if len(expected) != len(actual):
            return float("inf")
        if expected:
            maximum = max(
                maximum,
                float(np.max(np.abs(np.asarray(expected) - np.asarray(actual)))),
            )
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("data/processed/malecns-v1.0"),
    )
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--duration-ms", type=float, default=12.0)
    arguments = parser.parse_args()
    result = validate_runtime(
        arguments.data_directory,
        neuron_count=arguments.neurons,
        duration_s=arguments.duration_ms / 1000.0,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
