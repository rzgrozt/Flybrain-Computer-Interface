"""Controlled fixed-model sensory-to-descending characterization."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as parquet

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.connectome_data.runtime import MemoryMappedConnectome
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.runtime import SparseLIFSimulator
from flybrain_interface.simulation.trace import ChunkRecording, ChunkResult

DEFAULT_CONFIG = Path(__file__).parents[3] / "configs" / "sensory-descending-v1.json"


@dataclass(frozen=True, slots=True)
class ResolvedPopulation:
    key: str
    label: str
    query: dict[str, str]
    available_count: int
    neuron_indices: tuple[int, ...]
    body_ids: tuple[int, ...]
    sample_limit: int | None

    @property
    def selected_count(self) -> int:
        return len(self.neuron_indices)


@dataclass(frozen=True, slots=True)
class Condition:
    condition_id: str
    input_key: str | None
    amplitude_mv: float
    stimulus_duration_s: float
    repeat: int


@dataclass(frozen=True, slots=True)
class PopulationResponse:
    population: str
    neuron_count: int
    total_spikes: int
    total_rate_hz_per_neuron: float
    baseline_spikes: int
    baseline_rate_hz_per_neuron: float
    stimulus_spikes: int
    stimulus_rate_hz_per_neuron: float
    post_spikes: int
    post_rate_hz_per_neuron: float
    response_latency_s: float | None
    response_duration_s: float | None
    post_stimulus_persistence_s: float | None
    no_response: bool


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported sensory-descending config schema")
    return payload


def resolve_population(
    table: pa.Table, specification: dict[str, Any]
) -> ResolvedPopulation:
    query = dict(specification["query"])
    mask = np.ones(table.num_rows, dtype=np.bool_)
    for column, value in query.items():
        if column not in table.column_names:
            raise KeyError(f"unknown annotation column: {column}")
        values = np.asarray(table[column].to_pylist(), dtype=object)
        mask &= values == value
    matches = np.flatnonzero(mask)
    body_ids = np.asarray(table["body_id"].to_numpy(), dtype=np.int64)[matches]
    order = np.argsort(body_ids, kind="stable")
    ordered_indices = matches[order]
    ordered_body_ids = body_ids[order]
    sample_limit = specification.get("sample_limit")
    if sample_limit is not None:
        if not isinstance(sample_limit, int) or sample_limit <= 0:
            raise ValueError("sample_limit must be a positive integer or null")
        ordered_indices = ordered_indices[:sample_limit]
        ordered_body_ids = ordered_body_ids[:sample_limit]
    if not ordered_indices.size:
        raise ValueError(f"population query selected no neurons: {query}")
    neuron_column = np.asarray(table["neuron_index"].to_numpy(), dtype=np.int64)
    return ResolvedPopulation(
        key=str(specification["key"]),
        label=str(specification["label"]),
        query=query,
        available_count=int(matches.size),
        neuron_indices=tuple(int(value) for value in neuron_column[ordered_indices]),
        body_ids=tuple(int(value) for value in ordered_body_ids),
        sample_limit=sample_limit,
    )


def resolve_populations(
    table: pa.Table, config: dict[str, Any]
) -> tuple[tuple[ResolvedPopulation, ...], tuple[ResolvedPopulation, ...]]:
    inputs = tuple(resolve_population(table, item) for item in config["inputs"])
    outputs = tuple(resolve_population(table, item) for item in config["outputs"])
    return inputs, outputs


def build_conditions(config: dict[str, Any]) -> tuple[Condition, ...]:
    conditions = [
        Condition(
            condition_id=f"control-r{repeat}",
            input_key=None,
            amplitude_mv=0.0,
            stimulus_duration_s=0.0,
            repeat=repeat,
        )
        for repeat in range(int(config["repeats"]))
    ]
    for input_spec in config["inputs"]:
        for amplitude in config["amplitudes_mv"]:
            for duration in config["timing"]["stimulus_durations_s"]:
                for repeat in range(int(config["repeats"])):
                    key = str(input_spec["key"])
                    conditions.append(
                        Condition(
                            condition_id=(
                                f"{key}-a{float(amplitude):g}-"
                                f"d{float(duration) * 1000:g}ms-r{repeat}"
                            ),
                            input_key=key,
                            amplitude_mv=float(amplitude),
                            stimulus_duration_s=float(duration),
                            repeat=repeat,
                        )
                    )
    return tuple(conditions)


def make_stimulus(
    condition: Condition,
    inputs: dict[str, ResolvedPopulation],
    config: dict[str, Any],
    dt_ms: float,
) -> DeterministicSpikeInput:
    if condition.input_key is None:
        return DeterministicSpikeInput(neuron_indices=(), times_s=())
    population = inputs[condition.input_key]
    start_s = float(config["timing"]["baseline_s"])
    interval_s = float(config["timing"]["pulse_interval_ms"]) / 1000.0
    pulse_count = round(condition.stimulus_duration_s / interval_s)
    if not np.isclose(pulse_count * interval_s, condition.stimulus_duration_s):
        raise ValueError("stimulus duration must contain whole pulse intervals")
    dt_s = dt_ms / 1000.0
    times: list[float] = []
    indices: list[int] = []
    for pulse in range(pulse_count):
        pulse_time = start_s + pulse * interval_s
        if not np.isclose(pulse_time / dt_s, round(pulse_time / dt_s)):
            raise ValueError("stimulus pulse must align to the neural time grid")
        indices.extend(population.neuron_indices)
        times.extend([pulse_time] * population.selected_count)
    return DeterministicSpikeInput(
        neuron_indices=tuple(indices),
        times_s=tuple(times),
        amplitude_mv=condition.amplitude_mv,
    )


def response_metrics(
    result: ChunkResult,
    output: ResolvedPopulation,
    *,
    baseline_end_s: float,
    stimulus_end_s: float,
    total_s: float,
) -> PopulationResponse:
    members = np.asarray(output.neuron_indices, dtype=np.int64)
    selected = np.isin(result.spike_neuron_indices, members, assume_unique=False)
    times = result.spike_times_s[selected]
    baseline = times[times < baseline_end_s]
    stimulus = times[(times >= baseline_end_s) & (times < stimulus_end_s)]
    post = times[times >= stimulus_end_s]
    response = times[times >= baseline_end_s]
    neuron_count = output.selected_count
    baseline_duration = baseline_end_s
    stimulus_duration = stimulus_end_s - baseline_end_s
    post_duration = total_s - stimulus_end_s
    first = float(response[0]) if response.size else None
    last = float(response[-1]) if response.size else None
    return PopulationResponse(
        population=output.key,
        neuron_count=neuron_count,
        total_spikes=int(times.size),
        total_rate_hz_per_neuron=float(times.size) / (neuron_count * total_s),
        baseline_spikes=int(baseline.size),
        baseline_rate_hz_per_neuron=float(baseline.size)
        / (neuron_count * baseline_duration),
        stimulus_spikes=int(stimulus.size),
        stimulus_rate_hz_per_neuron=(
            float(stimulus.size) / (neuron_count * stimulus_duration)
            if stimulus_duration
            else 0.0
        ),
        post_spikes=int(post.size),
        post_rate_hz_per_neuron=float(post.size) / (neuron_count * post_duration),
        response_latency_s=(first - baseline_end_s if first is not None else None),
        response_duration_s=(
            last - first if first is not None and last is not None else None
        ),
        post_stimulus_persistence_s=(
            max(0.0, last - stimulus_end_s) if last is not None else None
        ),
        no_response=not bool(response.size),
    )


def run_characterization(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    graph = MemoryMappedConnectome.load(data_directory)
    inputs, outputs = resolve_populations(graph.catalog.table, config)
    output_mapping = {item.key: item.neuron_indices for item in outputs}
    simulator = SparseLIFSimulator(
        graph,
        populations=output_mapping,
        config=ShiuLIFConfig(),
        backend=str(config["backend"]),  # type: ignore[arg-type]
        subnormal_drive_policy=str(config["subnormal_drive_policy"]),  # type: ignore[arg-type]
    )
    simulator.prepare()
    conditions = build_conditions(config)
    input_map = {item.key: item for item in inputs}
    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    for condition in conditions:
        if time.perf_counter() - started > float(config["wall_time_budget_s"]):
            raise RuntimeError("wall-time budget exhausted before all conditions")
        simulator.reset()
        stimulus = make_stimulus(condition, input_map, config, simulator.config.dt_ms)
        run_started = time.perf_counter()
        result = simulator.advance_chunk(
            stimulus,
            duration_s=float(config["timing"]["total_s"]),
            recording=ChunkRecording(
                include_neuron_counts=True,
                max_spike_events=int(config["max_recorded_spike_events_per_run"]),
            ),
        )
        wall_s = time.perf_counter() - run_started
        if result.dropped_spike_events:
            raise RuntimeError("recorded spike-event budget exceeded")
        if result.total_spikes > int(config["max_total_network_spikes_per_run"]):
            raise RuntimeError("total network-spike budget exceeded")
        if simulator.visited_edges > int(config["max_visited_edges_per_run"]):
            raise RuntimeError(
                f"visited-edge budget exceeded in {condition.condition_id}: "
                f"{simulator.visited_edges}"
            )
        rss_bytes = int(psutil.Process().memory_info().rss)
        if rss_bytes > int(config["max_process_rss_bytes"]):
            raise RuntimeError(f"process RSS budget exceeded: {rss_bytes}")
        stimulus_end = (
            float(config["timing"]["baseline_s"]) + condition.stimulus_duration_s
        )
        responses = [
            response_metrics(
                result,
                output,
                baseline_end_s=float(config["timing"]["baseline_s"]),
                stimulus_end_s=stimulus_end,
                total_s=float(config["timing"]["total_s"]),
            )
            for output in outputs
        ]
        input_size = (
            input_map[condition.input_key].selected_count
            if condition.input_key is not None
            else 0
        )
        assert result.neuron_spike_counts is not None
        descending = outputs[0]
        descending_counts = result.neuron_spike_counts[
            np.asarray(descending.neuron_indices, dtype=np.int64)
        ]
        active = np.flatnonzero(descending_counts)
        runs.append(
            {
                "condition": asdict(condition),
                "input_selected_count": input_size,
                "pulse_count_per_input_neuron": (
                    len(stimulus.times_s) // input_size if input_size else 0
                ),
                "total_network_spikes": result.total_spikes,
                "visited_edges": simulator.visited_edges,
                "wall_seconds": wall_s,
                "process_rss_bytes": rss_bytes,
                "descending_pattern": {
                    "population": descending.key,
                    "active_neuron_count": int(active.size),
                    "body_spike_counts": [
                        {
                            "body_id": descending.body_ids[int(offset)],
                            "spike_count": int(descending_counts[int(offset)]),
                        }
                        for offset in active
                    ],
                },
                "responses": [
                    {
                        **asdict(response),
                        "spikes_per_stimulated_neuron": (
                            response.total_spikes / input_size if input_size else 0.0
                        ),
                    }
                    for response in responses
                ],
            }
        )
    _add_control_baseline_changes(runs)
    _verify_deterministic_repeats(runs)
    pattern_comparisons = compare_input_patterns(runs)
    manifest_directory = Path(__file__).parents[3] / "data" / "manifests"
    normalized_lock = manifest_directory / "malecns-v1.0-normalized-v2.json"
    source_lock = manifest_directory / "malecns-v1.0.json"
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "fixed-model engineering characterization; repeated deterministic runs "
            "test reproducibility and are not independent biological evidence"
        ),
        "config": config,
        "dataset": {
            "directory": str(graph.directory),
            "name": config["dataset"],
            "neuron_count": graph.neuron_count,
            "edge_count": graph.edge_count,
            "sign_policy": graph.sign_policy.name,
            "normalized_lock_sha256": file_sha256(normalized_lock),
            "source_lock_sha256": file_sha256(source_lock),
        },
        "config_sha256": file_sha256(config_path),
        "resolved_inputs": [_population_payload(item) for item in inputs],
        "resolved_outputs": [_population_payload(item) for item in outputs],
        "conditions": runs,
        "between_input_pattern_comparisons": pattern_comparisons,
        "software": {
            "python": platform.python_version(),
            "flybrain-interface": version("flybrain-interface"),
            "numpy": version("numpy"),
            "numba": version("numba"),
        },
        "total_wall_seconds": time.perf_counter() - started,
        "deterministic_repeats_equal": True,
    }


def _verify_deterministic_repeats(runs: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str | None, float, float], list[dict[str, Any]]] = {}
    for run in runs:
        condition = run["condition"]
        key = (
            condition["input_key"],
            condition["amplitude_mv"],
            condition["stimulus_duration_s"],
        )
        signature = {
            "total_network_spikes": run["total_network_spikes"],
            "visited_edges": run["visited_edges"],
            "responses": run["responses"],
            "descending_pattern": run["descending_pattern"],
        }
        groups.setdefault(key, []).append(signature)
    for key, signatures in groups.items():
        if any(item != signatures[0] for item in signatures[1:]):
            raise RuntimeError(f"deterministic repeats disagree for {key}")


def _add_control_baseline_changes(runs: list[dict[str, Any]]) -> None:
    controls = [run for run in runs if run["condition"]["input_key"] is None]
    if not controls:
        raise ValueError("at least one no-stimulus control is required")
    control_rates: dict[str, float] = {}
    for response in controls[0]["responses"]:
        control_rates[response["population"]] = response["baseline_rate_hz_per_neuron"]
    for run in runs:
        for response in run["responses"]:
            response["baseline_change_hz_per_neuron_vs_control"] = (
                response["baseline_rate_hz_per_neuron"]
                - control_rates[response["population"]]
            )


def compare_input_patterns(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repeat_zero = [
        run
        for run in runs
        if run["condition"]["repeat"] == 0 and run["condition"]["input_key"] is not None
    ]
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for run in repeat_zero:
        condition = run["condition"]
        grouped.setdefault(
            (condition["amplitude_mv"], condition["stimulus_duration_s"]), []
        ).append(run)
    comparisons: list[dict[str, Any]] = []
    for (amplitude, duration), group in sorted(grouped.items()):
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                left_counts = _pattern_counts(left)
                right_counts = _pattern_counts(right)
                union = sorted(set(left_counts) | set(right_counts))
                left_vector = np.asarray(
                    [left_counts.get(body, 0) for body in union], dtype=np.float64
                )
                right_vector = np.asarray(
                    [right_counts.get(body, 0) for body in union], dtype=np.float64
                )
                denominator = float(
                    np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
                )
                active_union = set(left_counts) | set(right_counts)
                active_intersection = set(left_counts) & set(right_counts)
                comparisons.append(
                    {
                        "amplitude_mv": amplitude,
                        "stimulus_duration_s": duration,
                        "left_input": left["condition"]["input_key"],
                        "right_input": right["condition"]["input_key"],
                        "cosine_similarity": (
                            float(np.dot(left_vector, right_vector) / denominator)
                            if denominator
                            else None
                        ),
                        "active_neuron_jaccard": (
                            len(active_intersection) / len(active_union)
                            if active_union
                            else None
                        ),
                        "undefined_reason": (
                            None
                            if denominator
                            else "one or both descending patterns had no spikes"
                        ),
                    }
                )
    return comparisons


def _pattern_counts(run: dict[str, Any]) -> dict[int, int]:
    return {
        int(item["body_id"]): int(item["spike_count"])
        for item in run["descending_pattern"]["body_spike_counts"]
    }


def _population_payload(population: ResolvedPopulation) -> dict[str, Any]:
    return {**asdict(population), "selected_count": population.selected_count}


def panel_presets(data_directory: Path) -> dict[str, Any]:
    config = load_config()
    table = parquet.read_table(data_directory / "neurons.parquet")
    inputs, outputs = resolve_populations(table, config)
    return {
        "name": config["name"],
        "inputs": [_population_payload(item) for item in inputs],
        "outputs": [_population_payload(item) for item in outputs],
        "caution": (
            "Engineering stimulation and observation assignments; labels describe "
            "annotations, not simulated or validated behavior."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_characterization(arguments.data_directory, arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)
    print(json.dumps(_summary(result), indent=2, sort_keys=True))


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for run in result["conditions"]:
        if run["condition"]["repeat"] != 0:
            continue
        rows.append(
            {
                "condition_id": run["condition"]["condition_id"],
                "total_network_spikes": run["total_network_spikes"],
                "responses": {
                    item["population"]: {
                        "spikes": item["total_spikes"],
                        "rate_hz_per_neuron": item["total_rate_hz_per_neuron"],
                        "latency_s": item["response_latency_s"],
                        "no_response": item["no_response"],
                    }
                    for item in run["responses"]
                },
            }
        )
    return {
        "experiment": result["experiment"],
        "deterministic_repeats_equal": result["deterministic_repeats_equal"],
        "total_wall_seconds": result["total_wall_seconds"],
        "conditions": rows,
    }


if __name__ == "__main__":
    main()
