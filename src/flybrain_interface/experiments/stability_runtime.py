"""Long-duration, chunked stability benchmark for the full MaleCNS runtime."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.connectome_data.runtime import MemoryMappedConnectome
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.runtime import (
    SparseLIFSimulator,
    SubnormalDrivePolicy,
)

DEFAULT_DATA_DIRECTORY = Path("data/processed/malecns-v1.0")
DEFAULT_LOCK = Path("data/manifests/malecns-v1.0-normalized-v2.json")


@dataclass(frozen=True, slots=True)
class RunBudgets:
    """Hard per-run limits checked at every chunk boundary."""

    wall_seconds: float = 120.0
    rss_bytes: int = 2 * 1024**3
    spikes: int = 50_000_000
    visited_edges: int = 5_000_000_000

    def __post_init__(self) -> None:
        if self.wall_seconds <= 0:
            raise ValueError("wall_seconds must be positive")
        if self.rss_bytes <= 0 or self.spikes <= 0 or self.visited_edges <= 0:
            raise ValueError("memory and event budgets must be positive")


def benchmark_stability(
    data_directory: Path,
    lock_path: Path,
    *,
    durations_s: tuple[float, ...] = (0.1, 1.0, 10.0),
    chunk_duration_s: float = 0.01,
    repeats: int = 3,
    seed: int = 0,
    sparse_input_count: int = 128,
    stress_input_count: int = 4096,
    stress_duration_s: float = 0.1,
    subnormal_drive_policy: SubnormalDrivePolicy = "preserve",
    budgets: RunBudgets = RunBudgets(),
) -> dict[str, Any]:
    """Measure stateful Numba execution without retaining completed chunks."""

    _validate_settings(
        durations_s,
        chunk_duration_s,
        repeats,
        sparse_input_count,
        stress_input_count,
        stress_duration_s,
    )
    process_started = perf_counter()
    memory_before_load = _memory_snapshot()
    loaded_at = perf_counter()
    graph = MemoryMappedConnectome.load(data_directory)
    load_seconds = perf_counter() - loaded_at
    memory_after_load = _memory_snapshot()
    config = ShiuLIFConfig()
    populations = _sign_populations(graph)

    rng = np.random.default_rng(seed)
    excitatory = np.flatnonzero(graph.presynaptic_signs == 1)
    maximum_input_count = min(stress_input_count, int(excitatory.size))
    selected = rng.choice(excitatory, size=maximum_input_count, replace=False)
    sparse_selected = selected[: min(sparse_input_count, maximum_input_count)]

    constructed_at = perf_counter()
    prepared = SparseLIFSimulator(
        graph,
        populations=populations,
        config=config,
        backend="numba",
        subnormal_drive_policy=subnormal_drive_policy,
    )
    construction_seconds = perf_counter() - constructed_at
    prepared_at = perf_counter()
    prepared.prepare()
    compilation_warmup_seconds = perf_counter() - prepared_at
    memory_after_warmup = _memory_snapshot()
    del prepared

    scenarios: list[dict[str, Any]] = []
    for duration_s in durations_s:
        for scenario_name, label, inputs in (
            ("no-input", "no external input", np.empty(0, dtype=np.int64)),
            (
                "controlled-sparse-input",
                "one deterministic input event at t=0 to selected excitatory neurons",
                sparse_selected,
            ),
        ):
            scenarios.append(
                _run_scenario(
                    graph,
                    populations,
                    config,
                    name=scenario_name,
                    label=label,
                    duration_s=duration_s,
                    chunk_duration_s=chunk_duration_s,
                    input_indices=inputs,
                    repeats=repeats,
                    subnormal_drive_policy=subnormal_drive_policy,
                    budgets=budgets,
                )
            )
    scenarios.append(
        _run_scenario(
            graph,
            populations,
            config,
            name="bounded-denser-input-engineering-stress",
            label=(
                "bounded engineering stress test, not physiological activity; "
                "one deterministic input event at t=0"
            ),
            duration_s=stress_duration_s,
            chunk_duration_s=chunk_duration_s,
            input_indices=selected,
            repeats=repeats,
            subnormal_drive_policy=subnormal_drive_policy,
            budgets=budgets,
        )
    )

    all_runs = [run for item in scenarios for run in item["runs"]]
    return {
        "benchmark_schema_version": 1,
        "ratio_definition": "simulated_time_seconds / steady_state_execution_seconds",
        "status": (
            "complete"
            if all(run["status"] == "complete" for run in all_runs)
            else "budget-limited"
        ),
        "methodology": {
            "backend": "numba",
            "subnormal_drive_policy": subnormal_drive_policy,
            "durations_s": list(durations_s),
            "chunk_duration_s": chunk_duration_s,
            "repeats": repeats,
            "seed": seed,
            "sparse_input_count": int(sparse_selected.size),
            "stress_input_count": int(selected.size),
            "stress_duration_s": stress_duration_s,
            "recording": (
                "summary-only ChunkResult; population counts are reduced per chunk; "
                "no completed ChunkResult is retained by the simulator"
            ),
            "budgets_per_run": asdict(budgets),
        },
        "phases": {
            "data_load_seconds": load_seconds,
            "representative_construction_seconds": construction_seconds,
            "compilation_warmup_seconds": compilation_warmup_seconds,
            "process_elapsed_seconds": perf_counter() - process_started,
            "memory_before_load": memory_before_load,
            "memory_after_load": memory_after_load,
            "memory_after_warmup": memory_after_warmup,
        },
        "provenance": {
            "dataset_directory": str(graph.directory),
            "normalized_lock": str(lock_path.resolve()),
            "normalized_lock_sha256": file_sha256(lock_path),
            "neuron_count": graph.neuron_count,
            "edge_count": graph.edge_count,
            "mapped_edge_bytes": graph.mapped_edge_bytes,
            "sign_policy": graph.sign_policy.name,
            "sign_population_sizes": {
                name: len(indices) for name, indices in populations.items()
            },
            "lif_config": asdict(config),
            "git_revision": _git_revision(),
            "host": _host_details(),
        },
        "scenarios": scenarios,
        "limitations": [
            "Artificial input is a deterministic engineering probe, not a model "
            "of physiological sensory activity.",
            "Population labels report the configured fast-synaptic sign policy, "
            "not anatomical or functional brain regions.",
            "Linux RSS includes resident file-backed memory-mapped graph pages; "
            "anonymous RSS and file-backed RSS/PSS are reported separately.",
            "Process high-water RSS is cumulative across scenarios, while each "
            "run also reports boundary-sampled peak RSS.",
            "Finite-state scans and memory sampling are outside steady-state "
            "chunk execution timing but are included in total run wall time.",
        ],
    }


def _run_scenario(
    graph: MemoryMappedConnectome,
    populations: dict[str, tuple[int, ...]],
    config: ShiuLIFConfig,
    *,
    name: str,
    label: str,
    duration_s: float,
    chunk_duration_s: float,
    input_indices: np.ndarray[Any, Any],
    repeats: int,
    subnormal_drive_policy: SubnormalDrivePolicy,
    budgets: RunBudgets,
) -> dict[str, Any]:
    runs = [
        _run_once(
            graph,
            populations,
            config,
            duration_s=duration_s,
            chunk_duration_s=chunk_duration_s,
            input_indices=input_indices,
            repeat_index=repeat_index,
            subnormal_drive_policy=subnormal_drive_policy,
            budgets=budgets,
        )
        for repeat_index in range(repeats)
    ]
    completed = [run for run in runs if run["status"] == "complete"]
    return {
        "name": name,
        "label": label,
        "requested_simulated_time_seconds": duration_s,
        "external_input_neurons": int(input_indices.size),
        "runs": runs,
        "completed_run_summary": _completed_run_summary(completed),
    }


def _run_once(
    graph: MemoryMappedConnectome,
    populations: dict[str, tuple[int, ...]],
    config: ShiuLIFConfig,
    *,
    duration_s: float,
    chunk_duration_s: float,
    input_indices: np.ndarray[Any, Any],
    repeat_index: int,
    subnormal_drive_policy: SubnormalDrivePolicy,
    budgets: RunBudgets,
) -> dict[str, Any]:
    constructed_at = perf_counter()
    simulator = SparseLIFSimulator(
        graph,
        populations=populations,
        config=config,
        backend="numba",
        subnormal_drive_policy=subnormal_drive_policy,
    )
    construction_seconds = perf_counter() - constructed_at
    simulator.prepare()
    memory_samples = [_memory_sample(0.0)]
    chunk_seconds: list[float] = []
    chunk_spikes: list[int] = []
    chunk_visited_edges: list[int] = []
    chunk_pending_events: list[int] = []
    population_spikes = {name: 0 for name in populations}
    total_spikes = 0
    previous_visited = 0
    nonfinite_state = False
    budget_stop_reason: str | None = None
    run_started = perf_counter()
    chunk_count = round(duration_s / chunk_duration_s)
    empty_stimulus = DeterministicSpikeInput(neuron_indices=(), times_s=())
    initial_stimulus = DeterministicSpikeInput(
        neuron_indices=tuple(int(value) for value in input_indices),
        times_s=(0.0,) * int(input_indices.size),
    )

    for chunk_index in range(chunk_count):
        stimulus = initial_stimulus if chunk_index == 0 else empty_stimulus
        chunk_started = perf_counter()
        result = simulator.advance_chunk(stimulus, duration_s=chunk_duration_s)
        chunk_elapsed = perf_counter() - chunk_started
        chunk_seconds.append(chunk_elapsed)
        chunk_spikes.append(result.total_spikes)
        total_spikes += result.total_spikes
        visited_delta = simulator.visited_edges - previous_visited
        previous_visited = simulator.visited_edges
        chunk_visited_edges.append(visited_delta)
        chunk_pending_events.append(simulator.pending_delayed_events)
        for population, rate_hz in result.population_rates_hz.items():
            population_spikes[population] += round(
                rate_hz * len(populations[population]) * chunk_duration_s
            )

        nonfinite_state = not (
            np.isfinite(simulator.voltage_mv).all()
            and np.isfinite(simulator.synaptic_drive_mv).all()
        )
        sample = _memory_sample(simulator.current_time_s)
        memory_samples.append(sample)
        total_wall = perf_counter() - run_started
        budget_stop_reason = _budget_stop_reason(
            total_wall,
            int(sample["rss_bytes"]),
            total_spikes,
            simulator.visited_edges,
            nonfinite_state,
            budgets,
        )
        if budget_stop_reason is not None:
            break

    simulated_seconds = simulator.current_time_s
    execution_seconds = sum(chunk_seconds)
    return {
        "repeat_index": repeat_index,
        "status": "complete" if budget_stop_reason is None else "stopped",
        "stop_reason": budget_stop_reason,
        "construction_seconds": construction_seconds,
        "simulated_time_seconds": simulated_seconds,
        "steady_state_execution_seconds": execution_seconds,
        "total_run_wall_seconds": perf_counter() - run_started,
        "simulated_to_wall_ratio": simulated_seconds / execution_seconds,
        "chunk_latency_seconds": _distribution(chunk_seconds),
        "chunk_latency_samples_seconds": chunk_seconds,
        "total_spikes": total_spikes,
        "population_spikes": population_spikes,
        "chunk_spikes": chunk_spikes,
        "activity": _activity_summary(chunk_spikes, chunk_duration_s),
        "visited_edges": simulator.visited_edges,
        "chunk_visited_edges": chunk_visited_edges,
        "pending_delayed_events_at_end": simulator.pending_delayed_events,
        "maximum_pending_delayed_events": max(chunk_pending_events, default=0),
        "nonfinite_state": nonfinite_state,
        "voltage_range_mv_at_end": [
            float(np.min(simulator.voltage_mv)),
            float(np.max(simulator.voltage_mv)),
        ],
        "synaptic_drive_range_mv_at_end": [
            float(np.min(simulator.synaptic_drive_mv)),
            float(np.max(simulator.synaptic_drive_mv)),
        ],
        "phase_seconds": {
            "state_update": simulator.state_update_seconds,
            "synaptic_propagation": simulator.synaptic_propagation_seconds,
            "recording": simulator.recording_seconds,
        },
        "memory": _memory_summary(memory_samples),
        "memory_samples": memory_samples,
    }


def _validate_settings(
    durations_s: tuple[float, ...],
    chunk_duration_s: float,
    repeats: int,
    sparse_input_count: int,
    stress_input_count: int,
    stress_duration_s: float,
) -> None:
    if not durations_s or any(duration <= 0 for duration in durations_s):
        raise ValueError("durations_s must contain positive values")
    if chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if sparse_input_count < 0 or stress_input_count < sparse_input_count:
        raise ValueError("input counts must satisfy 0 <= sparse <= stress")
    for duration in (*durations_s, stress_duration_s):
        chunks = round(duration / chunk_duration_s)
        if chunks <= 0 or not np.isclose(chunks * chunk_duration_s, duration):
            raise ValueError("every duration must be a whole number of chunks")


def _sign_populations(graph: MemoryMappedConnectome) -> dict[str, tuple[int, ...]]:
    return {
        "inhibitory": tuple(
            int(value) for value in np.flatnonzero(graph.presynaptic_signs == -1)
        ),
        "separated_or_unknown": tuple(
            int(value) for value in np.flatnonzero(graph.presynaptic_signs == 0)
        ),
        "excitatory": tuple(
            int(value) for value in np.flatnonzero(graph.presynaptic_signs == 1)
        ),
    }


def _budget_stop_reason(
    wall_seconds: float,
    rss_bytes: int,
    spikes: int,
    visited_edges: int,
    nonfinite_state: bool,
    budgets: RunBudgets,
) -> str | None:
    if nonfinite_state:
        return "nonfinite-state"
    if wall_seconds > budgets.wall_seconds:
        return "wall-time-budget"
    if rss_bytes > budgets.rss_bytes:
        return "rss-budget"
    if spikes > budgets.spikes:
        return "spike-budget"
    if visited_edges > budgets.visited_edges:
        return "visited-edge-budget"
    return None


def _activity_summary(
    chunk_spikes: list[int], chunk_duration_s: float
) -> dict[str, Any]:
    active = [index for index, count in enumerate(chunk_spikes) if count]
    tail_chunks = max(1, len(chunk_spikes) // 10)
    return {
        "active_chunk_count": len(active),
        "peak_chunk_spikes": max(chunk_spikes, default=0),
        "last_active_time_seconds": (
            (active[-1] + 1) * chunk_duration_s if active else None
        ),
        "final_chunk_spikes": chunk_spikes[-1] if chunk_spikes else 0,
        "final_ten_percent_spikes": sum(chunk_spikes[-tail_chunks:]),
    }


def _completed_run_summary(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None
    return {
        "completed_repeats": len(runs),
        "execution_seconds": _distribution(
            [float(run["steady_state_execution_seconds"]) for run in runs]
        ),
        "simulated_to_wall_ratio": _distribution(
            [float(run["simulated_to_wall_ratio"]) for run in runs]
        ),
        "total_spike_runs": [int(run["total_spikes"]) for run in runs],
        "visited_edge_runs": [int(run["visited_edges"]) for run in runs],
        "repeatable_spikes": len({int(run["total_spikes"]) for run in runs}) == 1,
        "repeatable_visited_edges": len({int(run["visited_edges"]) for run in runs})
        == 1,
        "maximum_sampled_rss_bytes": max(
            int(run["memory"]["maximum_rss_bytes"]) for run in runs
        ),
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("distribution values cannot be empty")
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p95": _percentile(ordered, 0.95),
        "maximum": ordered[-1],
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _memory_sample(simulated_time_s: float) -> dict[str, float | int]:
    return {"simulated_time_s": simulated_time_s, **_memory_snapshot()}


def _memory_snapshot() -> dict[str, int]:
    status = _read_kib_fields(Path("/proc/self/status"))
    smaps = _read_kib_fields(Path("/proc/self/smaps_rollup"))
    return {
        "rss_bytes": status.get("VmRSS", 0),
        "peak_rss_bytes": status.get("VmHWM", 0),
        "anonymous_rss_bytes": status.get("RssAnon", 0),
        "file_rss_bytes": status.get("RssFile", 0),
        "file_pss_bytes": smaps.get("Pss_File", 0),
        "private_dirty_bytes": smaps.get("Private_Dirty", 0),
    }


def _read_kib_fields(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    fields: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":") and parts[1].isdigit():
            fields[parts[0][:-1]] = int(parts[1]) * 1024
    return fields


def _memory_summary(samples: list[dict[str, float | int]]) -> dict[str, int]:
    first = samples[0]
    last = samples[-1]
    return {
        "starting_rss_bytes": int(first["rss_bytes"]),
        "ending_rss_bytes": int(last["rss_bytes"]),
        "rss_growth_bytes": int(last["rss_bytes"]) - int(first["rss_bytes"]),
        "maximum_rss_bytes": max(int(sample["rss_bytes"]) for sample in samples),
        "process_peak_rss_bytes_at_end": int(last["peak_rss_bytes"]),
        "anonymous_rss_growth_bytes": int(last["anonymous_rss_bytes"])
        - int(first["anonymous_rss_bytes"]),
        "file_rss_growth_bytes": int(last["file_rss_bytes"])
        - int(first["file_rss_bytes"]),
        "file_pss_growth_bytes": int(last["file_pss_bytes"])
        - int(first["file_pss_bytes"]),
        "private_dirty_growth_bytes": int(last["private_dirty_bytes"])
        - int(first["private_dirty_bytes"]),
    }


def _host_details() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "total_memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "python": sys.version,
        "packages": {
            package: version(package)
            for package in (
                "flybrain-interface",
                "numpy",
                "numba",
                "brian2",
                "scipy",
            )
        },
    }


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    return "unavailable"


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--durations", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--chunk-duration", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparse-input-count", type=int, default=128)
    parser.add_argument("--stress-input-count", type=int, default=4096)
    parser.add_argument("--stress-duration", type=float, default=0.1)
    parser.add_argument(
        "--subnormal-drive-policy",
        choices=("preserve", "zero"),
        default="preserve",
        help="Preserve exact IEEE subnormals or explicitly zero them at onset.",
    )
    parser.add_argument("--max-wall-seconds", type=float, default=120.0)
    parser.add_argument("--max-rss-gib", type=float, default=2.0)
    parser.add_argument("--max-spikes", type=int, default=50_000_000)
    parser.add_argument("--max-visited-edges", type=int, default=5_000_000_000)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = benchmark_stability(
        arguments.data_directory,
        arguments.lock,
        durations_s=tuple(arguments.durations),
        chunk_duration_s=arguments.chunk_duration,
        repeats=arguments.repeats,
        seed=arguments.seed,
        sparse_input_count=arguments.sparse_input_count,
        stress_input_count=arguments.stress_input_count,
        stress_duration_s=arguments.stress_duration,
        subnormal_drive_policy=arguments.subnormal_drive_policy,
        budgets=RunBudgets(
            wall_seconds=arguments.max_wall_seconds,
            rss_bytes=round(arguments.max_rss_gib * 1024**3),
            spikes=arguments.max_spikes,
            visited_edges=arguments.max_visited_edges,
        ),
    )
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
