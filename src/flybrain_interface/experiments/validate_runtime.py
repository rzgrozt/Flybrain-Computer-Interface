"""Validate both sparse LIF backends on multiple real MaleCNS subgraphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np

from flybrain_interface.connectome_data.induced import (
    InducedConnectome,
    strong_outgoing_neighborhoods,
)
from flybrain_interface.connectome_data.runtime import MemoryMappedConnectome
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.reference import Brian2ReferenceSimulator
from flybrain_interface.simulation.runtime import RuntimeBackend, SparseLIFSimulator
from flybrain_interface.simulation.trace import SimulationTrace


def validate_runtime(
    data_directory: Path,
    *,
    neuron_count: int = 64,
    duration_s: float = 0.012,
    neighborhood_count: int = 3,
) -> dict[str, object]:
    graph = MemoryMappedConnectome.load(data_directory)
    neighborhoods = strong_outgoing_neighborhoods(
        graph,
        neuron_count=neuron_count,
        neighborhood_count=neighborhood_count,
    )
    watched = tuple(range(neuron_count))
    input_cases = (
        (
            "single-seed",
            DeterministicSpikeInput(neuron_indices=(0,), times_s=(0.001,)),
        ),
        (
            "late-pair",
            DeterministicSpikeInput(
                neuron_indices=(0, neuron_count - 1), times_s=(0.001, 0.006)
            ),
        ),
    )
    comparisons: list[dict[str, object]] = []

    for neighborhood_index, induced in enumerate(neighborhoods):
        connectivity = induced.as_sparse_connectivity()
        for case_name, stimulus in input_cases:
            reference_started = perf_counter()
            reference = Brian2ReferenceSimulator(connectivity).trace(
                stimulus,
                duration_s=duration_s,
                watched_indices=watched,
            )
            reference_seconds = perf_counter() - reference_started
            for backend in ("numpy", "numba"):
                comparisons.append(
                    _compare_runtime(
                        reference,
                        induced,
                        stimulus,
                        watched,
                        duration_s,
                        backend=backend,
                        neighborhood_index=neighborhood_index,
                        case_name=case_name,
                        reference_seconds=reference_seconds,
                    )
                )

    full_runtime = SparseLIFSimulator(graph)
    full_started = perf_counter()
    full_runtime.advance_chunk(duration_s=0.001)
    full_runtime_seconds = perf_counter() - full_started
    neighborhood_metadata = []
    for index, induced in enumerate(neighborhoods):
        seed_index = int(induced.original_indices[0])
        neighborhood_metadata.append(
            {
                "neighborhood": index,
                "neurons": induced.neuron_count,
                "edges": induced.edge_count,
                "seed_neuron_index": seed_index,
                "seed_body_id": int(graph.catalog.body_ids([seed_index])[0]),
            }
        )

    return {
        "scope": "multiple induced real MaleCNS subgraphs and input cases",
        "neighborhoods": neighborhood_count,
        "input_cases": len(input_cases),
        "backend_comparisons": len(comparisons),
        "neurons_per_neighborhood": neuron_count,
        "duration_s": duration_s,
        "dt_ms": full_runtime.config.dt_ms,
        "neighborhood_metadata": neighborhood_metadata,
        "comparisons": comparisons,
        "full_network_smoke_steps": 10,
        "full_network_smoke_seconds": full_runtime_seconds,
        "full_network_seconds_per_step": full_runtime_seconds / 10,
        "maximum_spike_time_error_s": max(
            cast(float, result["maximum_spike_time_error_s"]) for result in comparisons
        ),
        "maximum_voltage_error_mv": max(
            cast(float, result["maximum_voltage_error_mv"]) for result in comparisons
        ),
        "maximum_synaptic_drive_error_mv": max(
            cast(float, result["maximum_synaptic_drive_error_mv"])
            for result in comparisons
        ),
        "equivalent": all(bool(result["equivalent"]) for result in comparisons),
    }


def _compare_runtime(
    reference: SimulationTrace,
    induced: InducedConnectome,
    stimulus: DeterministicSpikeInput,
    watched: tuple[int, ...],
    duration_s: float,
    *,
    backend: RuntimeBackend,
    neighborhood_index: int,
    case_name: str,
    reference_seconds: float,
) -> dict[str, object]:
    runtime_simulator = SparseLIFSimulator(induced, backend=backend)
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
            f"neighborhood={neighborhood_index}, case={case_name}, backend={backend}, "
            f"counts_equal={counts_equal}, spike_time_error={spike_time_error}, "
            f"voltage_error={voltage_error}, drive_error={drive_error}"
        )
    return {
        "neighborhood": neighborhood_index,
        "input_case": case_name,
        "backend": backend,
        "reference_seconds": reference_seconds,
        "runtime_seconds": runtime_seconds,
        "runtime_speedup": reference_seconds / runtime_seconds,
        "runtime_visited_edges": runtime_simulator.visited_edges,
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
    parser.add_argument("--neighborhoods", type=int, default=3)
    parser.add_argument("--duration-ms", type=float, default=12.0)
    arguments = parser.parse_args()
    result = validate_runtime(
        arguments.data_directory,
        neuron_count=arguments.neurons,
        duration_s=arguments.duration_ms / 1000.0,
        neighborhood_count=arguments.neighborhoods,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
