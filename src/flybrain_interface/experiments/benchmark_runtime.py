"""Reproducible full-network MaleCNS runtime benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numba
import numpy as np

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.connectome_data.runtime import MemoryMappedConnectome
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.runtime import RuntimeBackend, SparseLIFSimulator

DEFAULT_DATA_DIRECTORY = Path("data/processed/malecns-v1.0")
DEFAULT_LOCK = Path("data/manifests/malecns-v1.0-normalized-v2.json")


def benchmark_full_network(
    data_directory: Path,
    lock_path: Path,
    *,
    repeats: int = 3,
    seed: int = 0,
    quiet_steps: tuple[int, ...] = (25, 100),
    active_steps: int = 50,
    sparse_input_count: int = 128,
    stress_input_count: int = 4096,
    backends: tuple[RuntimeBackend, ...] = ("numpy", "numba"),
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not quiet_steps or any(steps <= 0 for steps in quiet_steps):
        raise ValueError("quiet_steps must contain positive values")
    if active_steps <= 0:
        raise ValueError("active_steps must be positive")
    if not backends or len(set(backends)) != len(backends):
        raise ValueError("backends must be non-empty and unique")

    rss_before = _current_rss_bytes()
    loaded_at = perf_counter()
    graph = MemoryMappedConnectome.load(data_directory)
    load_seconds = perf_counter() - loaded_at
    rss_after_load = _current_rss_bytes()
    config = ShiuLIFConfig()
    rng = np.random.default_rng(seed)
    excitatory = np.flatnonzero(graph.presynaptic_signs == 1)
    stress_count = min(stress_input_count, excitatory.size)
    sparse_count = min(sparse_input_count, excitatory.size)
    selected = rng.choice(excitatory, size=stress_count, replace=False)

    initialization = {
        backend: _benchmark_initialization(graph, config, backend=backend)
        for backend in backends
    }
    scenarios: list[dict[str, Any]] = []
    for backend in backends:
        for steps in quiet_steps:
            scenarios.append(
                _benchmark_scenario(
                    graph,
                    config,
                    backend=backend,
                    name=f"quiet-{steps}-steps",
                    label="no external input",
                    steps=steps,
                    input_indices=np.empty(0, dtype=np.int64),
                    repeats=repeats,
                )
            )
        scenarios.append(
            _benchmark_scenario(
                graph,
                config,
                backend=backend,
                name="sparse-deterministic-input",
                label="engineering input: 128 excitatory neurons (or configured count)",
                steps=active_steps,
                input_indices=selected[:sparse_count],
                repeats=repeats,
            )
        )
        scenarios.append(
            _benchmark_scenario(
                graph,
                config,
                backend=backend,
                name="dense-engineering-stress",
                label=(
                    "non-physiological engineering stress test; not a biological "
                    "activity assumption"
                ),
                steps=active_steps,
                input_indices=selected,
                repeats=repeats,
            )
        )
    isolated_propagation = _benchmark_propagation(
        graph, selected[:sparse_count], repeats=repeats
    )

    return {
        "benchmark_schema_version": 2,
        "ratio_definition": "simulated_time_seconds / wall_time_seconds",
        "backends": list(backends),
        "backend_initialization": initialization,
        "host": _host_details(),
        "provenance": {
            "dataset_directory": str(graph.directory),
            "normalized_lock": str(lock_path.resolve()),
            "normalized_lock_sha256": file_sha256(lock_path),
            "sign_policy": graph.sign_policy.name,
            "neuron_count": graph.neuron_count,
            "edge_count": graph.edge_count,
            "mapped_edge_bytes": graph.mapped_edge_bytes,
            "lif_config": asdict(config),
            "random_seed": seed,
        },
        "phases": {
            "data_load_seconds": load_seconds,
            "rss_before_load_bytes": rss_before,
            "rss_after_load_bytes": rss_after_load,
            "rss_load_increase_bytes": max(0, rss_after_load - rss_before),
            "peak_process_rss_bytes": _peak_rss_bytes(),
            "isolated_synaptic_propagation": isolated_propagation,
        },
        "scenarios": scenarios,
        "limitations": [
            "Scenario execution includes threshold detection and delay handling but "
            "uses the default summary-only chunk recorder.",
            "First prepare time includes disk-cache loading or compilation and is "
            "process-dependent.",
            "Dense stress input is an engineering load case, not physiological data.",
        ],
    }


def _benchmark_scenario(
    graph: MemoryMappedConnectome,
    config: ShiuLIFConfig,
    *,
    backend: RuntimeBackend,
    name: str,
    label: str,
    steps: int,
    input_indices: np.ndarray[Any, Any],
    repeats: int,
) -> dict[str, Any]:
    duration_s = steps * config.dt_ms / 1000.0
    stimulus = DeterministicSpikeInput(
        neuron_indices=tuple(int(value) for value in input_indices),
        times_s=(0.0,) * int(input_indices.size),
    )
    construction_times: list[float] = []
    execution_times: list[float] = []
    spike_counts: list[int] = []
    visited_edges: list[int] = []
    state_update_times: list[float] = []
    propagation_times: list[float] = []
    recording_times: list[float] = []

    # One unreported warm-up controls page faults and allocator initialization.
    warmup = SparseLIFSimulator(graph, config=config, backend=backend)
    warmup.prepare()
    warmup.advance_chunk(stimulus, duration_s=duration_s)
    del warmup
    gc.collect()

    for _ in range(repeats):
        constructed_at = perf_counter()
        simulator = SparseLIFSimulator(graph, config=config, backend=backend)
        construction_times.append(perf_counter() - constructed_at)
        executed_at = perf_counter()
        result = simulator.advance_chunk(stimulus, duration_s=duration_s)
        execution_times.append(perf_counter() - executed_at)
        spike_counts.append(result.total_spikes)
        visited_edges.append(simulator.visited_edges)
        state_update_times.append(simulator.state_update_seconds)
        propagation_times.append(simulator.synaptic_propagation_seconds)
        recording_times.append(simulator.recording_seconds)
        del result, simulator
        gc.collect()

    execution_summary = _timing_summary(execution_times)
    median_wall = execution_summary["median_seconds"]
    return {
        "name": name,
        "backend": backend,
        "label": label,
        "steps": steps,
        "simulated_time_seconds": duration_s,
        "external_input_neurons": int(input_indices.size),
        "construction": _timing_summary(construction_times),
        "execution": execution_summary,
        "simulated_to_wall_ratio": duration_s / median_wall,
        "median_seconds_per_step": median_wall / steps,
        "spike_count_runs": spike_counts,
        "visited_edge_runs": visited_edges,
        "repeatable_counts": len(set(spike_counts)) == 1,
        "repeatable_visited_edges": len(set(visited_edges)) == 1,
        "phase_totals": {
            "state_update": _timing_summary(state_update_times),
            "synaptic_propagation": _timing_summary(propagation_times),
            "recording": _timing_summary(recording_times),
        },
        "recording": {
            "watchlist_neurons": 0,
            "maximum_spike_events": 0,
            "all_neuron_counts": False,
        },
    }


def _benchmark_initialization(
    graph: MemoryMappedConnectome,
    config: ShiuLIFConfig,
    *,
    backend: RuntimeBackend,
) -> dict[str, float | bool]:
    simulator = SparseLIFSimulator(graph, config=config, backend=backend)
    first_started = perf_counter()
    simulator.prepare()
    first_seconds = perf_counter() - first_started
    warm_started = perf_counter()
    simulator.prepare()
    warm_seconds = perf_counter() - warm_started
    return {
        "jit_enabled": backend == "numba",
        "first_prepare_seconds": first_seconds,
        "warm_prepare_seconds": warm_seconds,
    }


def _benchmark_propagation(
    graph: MemoryMappedConnectome,
    source_indices: np.ndarray[Any, Any],
    *,
    repeats: int,
) -> dict[str, Any]:
    destination = np.zeros(graph.neuron_count, dtype=np.float64)
    graph.accumulate_spikes(source_indices, destination)
    timings: list[float] = []
    visited = 0
    for _ in range(repeats):
        destination.fill(0.0)
        started = perf_counter()
        visited = graph.accumulate_spikes(source_indices, destination)
        timings.append(perf_counter() - started)
    return {
        "source_neurons": int(source_indices.size),
        "visited_edges": visited,
        "timing": _timing_summary(timings),
    }


def _timing_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("timing values cannot be empty")
    return {
        "repeats": len(values),
        "minimum_seconds": min(values),
        "median_seconds": statistics.median(values),
        "maximum_seconds": max(values),
    }


def _host_details() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "total_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
    }


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return "unavailable"
    for line in cpuinfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", maxsplit=1)[1].strip()
    return "unavailable"


def _current_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet-steps", type=int, nargs="+", default=[25, 100])
    parser.add_argument("--active-steps", type=int, default=50)
    parser.add_argument("--sparse-input-count", type=int, default=128)
    parser.add_argument("--stress-input-count", type=int, default=4096)
    parser.add_argument(
        "--backend", choices=("numpy", "numba"), nargs="+", default=["numpy", "numba"]
    )
    arguments = parser.parse_args()
    result = benchmark_full_network(
        arguments.data_directory,
        arguments.lock,
        repeats=arguments.repeats,
        seed=arguments.seed,
        quiet_steps=tuple(arguments.quiet_steps),
        active_steps=arguments.active_steps,
        sparse_input_count=arguments.sparse_input_count,
        stress_input_count=arguments.stress_input_count,
        backends=tuple(arguments.backend),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
