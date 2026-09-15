"""Diagnose and characterize uniform visual input in the fixed MaleCNS model."""

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

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.connectome_data.runtime import MemoryMappedConnectome
from flybrain_interface.experiments.sensory_descending import (
    ResolvedPopulation,
    resolve_population,
)
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.sensory.vision import (
    UniformLuminanceConfig,
    UniformLuminanceEncoding,
    encode_uniform_luminance,
)
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.runtime import SparseLIFSimulator
from flybrain_interface.simulation.trace import ChunkRecording, ChunkResult

DEFAULT_CONFIG = (
    Path(__file__).parents[3] / "configs" / "visual-pathway-validation-v1.json"
)


@dataclass(frozen=True, slots=True)
class DirectTargetPopulation:
    key: str
    label: str
    type_name: str
    neuron_indices: tuple[int, ...]
    body_ids: tuple[int, ...]
    direct_edge_count: int
    direct_contact_count: int
    contacts_by_neuron: tuple[int, ...]

    @property
    def selected_count(self) -> int:
        return len(self.neuron_indices)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported visual-pathway config schema")
    return payload


def resolve_visual_input(table: pa.Table, config: dict[str, Any]) -> ResolvedPopulation:
    return resolve_population(
        table,
        {
            "key": "all_r1_r6",
            "label": "All reconstructed R1-R6 photoreceptors",
            "query": config["visual_input_query"],
            "sample_limit": None,
        },
    )


def reference_sample(
    visual_input: ResolvedPopulation, sample_limit: int
) -> ResolvedPopulation:
    if sample_limit <= 0 or sample_limit > visual_input.selected_count:
        raise ValueError("reference sample limit is outside the visual population")
    return ResolvedPopulation(
        key="reference_id_first_r1_r6",
        label="Earlier ascending-body-ID R1-R6 sample",
        query=visual_input.query,
        available_count=visual_input.available_count,
        neuron_indices=visual_input.neuron_indices[:sample_limit],
        body_ids=visual_input.body_ids[:sample_limit],
        sample_limit=sample_limit,
    )


def resolve_direct_targets(
    graph: MemoryMappedConnectome,
    source_indices: tuple[int, ...],
    type_name: str,
) -> DirectTargetPopulation:
    sources = np.asarray(source_indices, dtype=np.int64)
    target_types = np.asarray(graph.catalog.table["type"].to_pylist(), dtype=object)
    contact_by_target: dict[int, int] = {}
    edge_count = 0
    for source in sources:
        start = int(graph.outgoing_indptr[source])
        stop = int(graph.outgoing_indptr[source + 1])
        targets = graph.target_indices[start:stop]
        weights = graph.outgoing_synapse_counts[start:stop]
        selected = np.flatnonzero(target_types[targets] == type_name)
        edge_count += int(selected.size)
        for local in selected:
            target = int(targets[local])
            contact_by_target[target] = contact_by_target.get(target, 0) + int(
                weights[local]
            )
    if not contact_by_target:
        raise ValueError(f"R1-R6 has no direct targets annotated {type_name}")
    indices = np.fromiter(contact_by_target, dtype=np.int64)
    body_ids = graph.catalog.body_ids(indices.tolist())
    order = np.argsort(body_ids, kind="stable")
    ordered_indices = indices[order]
    ordered_body_ids = body_ids[order]
    return DirectTargetPopulation(
        key=f"direct_{type_name.lower()}",
        label=f"Direct R1-R6 targets: {type_name}",
        type_name=type_name,
        neuron_indices=tuple(int(value) for value in ordered_indices),
        body_ids=tuple(int(value) for value in ordered_body_ids),
        direct_edge_count=edge_count,
        direct_contact_count=sum(contact_by_target.values()),
        contacts_by_neuron=tuple(
            contact_by_target[int(index)] for index in ordered_indices
        ),
    )


def condition_frames(config: dict[str, Any], condition: str) -> np.ndarray:
    values = np.asarray(config["conditions"][condition], dtype=np.float64)
    if values.shape != (int(config["timing"]["frame_count"]),):
        raise ValueError(f"condition {condition} has the wrong frame count")
    return np.broadcast_to(values[:, None, None], (values.size, 2, 2)).copy()


def make_uniform_encoding(
    frames: np.ndarray,
    visual_input: ResolvedPopulation,
    config: dict[str, Any],
    neural_dt_ms: float,
) -> UniformLuminanceEncoding:
    return encode_uniform_luminance(
        frames,
        visual_input.neuron_indices,
        UniformLuminanceConfig(
            frame_rate_hz=float(config["timing"]["frame_rate_hz"]),
            max_pulses_per_frame=int(config["transduction"]["max_pulses_per_frame"]),
            pulse_amplitude_mv=float(config["transduction"]["pulse_amplitude_mv"]),
            neural_dt_ms=neural_dt_ms,
            start_s=float(config["timing"]["baseline_s"]),
        ),
    )


def make_reference_stimulus(
    population: ResolvedPopulation,
    config: dict[str, Any],
) -> DeterministicSpikeInput:
    specification = config["reference_condition"]
    start_s = float(config["timing"]["baseline_s"])
    interval_s = float(specification["pulse_interval_ms"]) / 1000.0
    pulse_count = round(float(specification["duration_s"]) / interval_s)
    indices: list[int] = []
    times: list[float] = []
    for pulse in range(pulse_count):
        indices.extend(population.neuron_indices)
        times.extend([start_s + pulse * interval_s] * population.selected_count)
    return DeterministicSpikeInput(
        neuron_indices=tuple(indices),
        times_s=tuple(times),
        amplitude_mv=float(specification["pulse_amplitude_mv"]),
    )


def run_validation(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    graph = MemoryMappedConnectome.load(data_directory)
    visual_input = resolve_visual_input(graph.catalog.table, config)
    reference = reference_sample(visual_input, int(config["reference_sample_limit"]))
    direct = tuple(
        resolve_direct_targets(graph, visual_input.neuron_indices, type_name)
        for type_name in config["direct_target_types"]
    )
    descending = resolve_population(
        graph.catalog.table,
        {
            "key": "all_descending",
            "label": "All anatomically annotated descending neurons",
            "query": config["descending_query"],
            "sample_limit": None,
        },
    )
    populations = {population.key: population.neuron_indices for population in direct}
    populations[descending.key] = descending.neuron_indices
    simulator = SparseLIFSimulator(
        graph,
        populations=populations,
        config=ShiuLIFConfig(),
        backend=str(config["backend"]),  # type: ignore[arg-type]
        subnormal_drive_policy=str(config["subnormal_drive_policy"]),  # type: ignore[arg-type]
    )
    simulator.prepare()
    watched = tuple(
        sorted({index for group in direct for index in group.neuron_indices})
    )
    watch_columns = {index: column for column, index in enumerate(watched)}
    conditions = ("reference_id_first", *config["conditions"].keys())
    runs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for condition in conditions:
        for repeat in range(int(config["repeats"])):
            if time.perf_counter() - started > float(config["wall_time_budget_s"]):
                raise RuntimeError("wall-time budget exhausted before all conditions")
            simulator.reset()
            encoding: UniformLuminanceEncoding | None = None
            input_population = visual_input
            if condition == "reference_id_first":
                input_population = reference
                stimulus = make_reference_stimulus(reference, config)
                stimulus_end_s = float(config["timing"]["baseline_s"]) + float(
                    config["reference_condition"]["duration_s"]
                )
            else:
                frames = condition_frames(config, condition)
                encoding = make_uniform_encoding(
                    frames, visual_input, config, simulator.config.dt_ms
                )
                stimulus = encoding.stimulus
                stimulus_end_s = float(config["timing"]["baseline_s"]) + (
                    int(config["timing"]["frame_count"])
                    / float(config["timing"]["frame_rate_hz"])
                )
            run_started = time.perf_counter()
            result = simulator.advance_chunk(
                stimulus,
                duration_s=float(config["timing"]["total_s"]),
                recording=ChunkRecording(
                    watched_indices=watched,
                    include_neuron_counts=True,
                    max_spike_events=int(config["max_recorded_spike_events_per_run"]),
                ),
            )
            wall_s = time.perf_counter() - run_started
            _enforce_budgets(result, simulator, config)
            assert result.neuron_spike_counts is not None
            direct_responses = [
                _direct_response(
                    result,
                    group,
                    watch_columns,
                    baseline_s=float(config["timing"]["baseline_s"]),
                    stimulus_end_s=stimulus_end_s,
                    threshold_mv=float(config["subthreshold_response_threshold_mv"]),
                    resting_mv=simulator.config.resting_mv,
                )
                for group in direct
            ]
            descending_counts = result.neuron_spike_counts[
                np.asarray(descending.neuron_indices, dtype=np.int64)
            ]
            input_counts = result.neuron_spike_counts[
                np.asarray(input_population.neuron_indices, dtype=np.int64)
            ]
            runs.append(
                {
                    "condition": condition,
                    "repeat": repeat,
                    "input_population": input_population.key,
                    "input_neuron_count": input_population.selected_count,
                    "scheduled_events": len(stimulus.times_s),
                    "input_spikes": int(input_counts.sum()),
                    "input_active_neuron_count": int(np.count_nonzero(input_counts)),
                    "total_network_spikes": result.total_spikes,
                    "visited_edges": simulator.visited_edges,
                    "wall_seconds": wall_s,
                    "process_rss_bytes": int(psutil.Process().memory_info().rss),
                    "encoding": _encoding_payload(encoding),
                    "direct_responses": direct_responses,
                    "descending_response": {
                        "neuron_count": descending.selected_count,
                        "total_spikes": int(descending_counts.sum()),
                        "active_neuron_count": int(np.count_nonzero(descending_counts)),
                    },
                }
            )
    _verify_repeats(runs)
    manifest_directory = Path(__file__).parents[3] / "data" / "manifests"
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "fixed point-neuron engineering characterization; artificial pulse-density "
            "transduction is not literal photoreceptor physiology"
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "dataset": {
            "directory": str(graph.directory),
            "name": config["dataset"],
            "neuron_count": graph.neuron_count,
            "edge_count": graph.edge_count,
            "sign_policy": graph.sign_policy.name,
            "normalized_lock_sha256": file_sha256(
                manifest_directory / "malecns-v1.0-normalized-v2.json"
            ),
            "source_lock_sha256": file_sha256(manifest_directory / "malecns-v1.0.json"),
        },
        "visual_input": _resolved_payload(visual_input),
        "reference_sample": _resolved_payload(reference),
        "direct_targets": [_direct_payload(group) for group in direct],
        "descending_output": _resolved_payload(descending),
        "anatomical_diagnosis": _anatomical_diagnosis(
            graph, visual_input, reference, direct
        ),
        "retinotopic_mapping_assessment": {
            "moving_bar_enabled": False,
            "reason": (
                "MaleCNS neuron annotations expose side but no R1-R6 column or "
                "receptive-field coordinate; the official supplemental column table "
                "lists L1, R7 and R8 IDs, not individual R1-R6 IDs"
            ),
            "soma_coordinates_used_as_retinal_coordinates": False,
        },
        "runs": runs,
        "deterministic_repeats_equal": True,
        "total_wall_seconds": time.perf_counter() - started,
        "software": {
            "python": platform.python_version(),
            "flybrain-interface": version("flybrain-interface"),
            "numpy": version("numpy"),
            "numba": version("numba"),
        },
    }


def _direct_response(
    result: ChunkResult,
    population: DirectTargetPopulation,
    watch_columns: dict[int, int],
    *,
    baseline_s: float,
    stimulus_end_s: float,
    threshold_mv: float,
    resting_mv: float,
) -> dict[str, Any]:
    columns = np.asarray(
        [watch_columns[index] for index in population.neuron_indices], dtype=np.int64
    )
    voltage = result.voltage_mv[:, columns]
    drive = result.synaptic_drive_mv[:, columns]
    hyperpolarization = np.maximum(0.0, resting_mv - voltage)
    peak_by_neuron = np.max(hyperpolarization, axis=0)
    responsive = peak_by_neuron >= threshold_mv
    active_by_time = np.any(hyperpolarization >= threshold_mv, axis=1)
    active_times = result.sample_times_s[active_by_time]
    assert result.neuron_spike_counts is not None
    spike_count = int(
        result.neuron_spike_counts[
            np.asarray(population.neuron_indices, dtype=np.int64)
        ].sum()
    )
    return {
        "population": population.key,
        "neuron_count": population.selected_count,
        "spikes": spike_count,
        "responsive_subthreshold_neuron_count": int(np.count_nonzero(responsive)),
        "subthreshold_threshold_mv": threshold_mv,
        "median_peak_hyperpolarization_mv": float(np.median(peak_by_neuron)),
        "max_peak_hyperpolarization_mv": float(np.max(peak_by_neuron)),
        "minimum_voltage_mv": float(np.min(voltage)),
        "minimum_synaptic_drive_mv": float(np.min(drive)),
        "subthreshold_latency_s": (
            float(active_times[0] - baseline_s) if active_times.size else None
        ),
        "subthreshold_persistence_s": (
            max(0.0, float(active_times[-1] - stimulus_end_s))
            if active_times.size
            else None
        ),
        "no_spike_response": spike_count == 0,
        "no_subthreshold_response": not bool(active_times.size),
    }


def _enforce_budgets(
    result: ChunkResult,
    simulator: SparseLIFSimulator,
    config: dict[str, Any],
) -> None:
    if result.dropped_spike_events:
        raise RuntimeError("recorded spike-event budget exceeded")
    if result.total_spikes > int(config["max_total_network_spikes_per_run"]):
        raise RuntimeError("total network-spike budget exceeded")
    if simulator.visited_edges > int(config["max_visited_edges_per_run"]):
        raise RuntimeError("visited-edge budget exceeded")
    rss = int(psutil.Process().memory_info().rss)
    if rss > int(config["max_process_rss_bytes"]):
        raise RuntimeError(f"process RSS budget exceeded: {rss}")


def _verify_repeats(runs: list[dict[str, Any]]) -> None:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        signature = {
            key: value
            for key, value in run.items()
            if key not in {"repeat", "wall_seconds", "process_rss_bytes"}
        }
        by_condition.setdefault(str(run["condition"]), []).append(signature)
    for condition, signatures in by_condition.items():
        if any(value != signatures[0] for value in signatures[1:]):
            raise RuntimeError(f"deterministic repeats disagree for {condition}")


def _anatomical_diagnosis(
    graph: MemoryMappedConnectome,
    visual_input: ResolvedPopulation,
    reference: ResolvedPopulation,
    direct: tuple[DirectTargetPopulation, ...],
) -> dict[str, Any]:
    table = graph.catalog.table
    visual_rows = table.take(pa.array(visual_input.neuron_indices))
    side_counts: dict[str, int] = {}
    for side in visual_rows["root_side"].to_pylist():
        key = "missing" if side is None else str(side)
        side_counts[key] = side_counts.get(key, 0) + 1
    return {
        "all_r1_r6_transmitters": sorted(
            {str(graph.transmitters[index]) for index in visual_input.neuron_indices}
        ),
        "all_r1_r6_signs": sorted(
            {
                int(graph.presynaptic_signs[index])
                for index in visual_input.neuron_indices
            }
        ),
        "root_side_counts": side_counts,
        "available_annotation_fields": list(table.column_names),
        "r1_r6_instance_values": sorted(
            {str(value) for value in visual_rows["instance"].to_pylist()}
        ),
        "reference_outgoing": _outgoing_summary(graph, reference.neuron_indices),
        "all_r1_r6_outgoing": _outgoing_summary(graph, visual_input.neuron_indices),
        "direct_target_summary": [
            {
                "population": group.key,
                "neuron_count": group.selected_count,
                "edge_count": group.direct_edge_count,
                "contact_count": group.direct_contact_count,
            }
            for group in direct
        ],
    }


def _outgoing_summary(
    graph: MemoryMappedConnectome, sources: tuple[int, ...]
) -> dict[str, int]:
    edge_count = 0
    contacts = 0
    for source in sources:
        start = int(graph.outgoing_indptr[source])
        stop = int(graph.outgoing_indptr[source + 1])
        edge_count += stop - start
        contacts += int(graph.outgoing_synapse_counts[start:stop].sum())
    return {"edge_count": edge_count, "contact_count": contacts}


def _encoding_payload(
    encoding: UniformLuminanceEncoding | None,
) -> dict[str, Any] | None:
    if encoding is None:
        return None
    return {
        "frame_timestamps_s": encoding.frame_timestamps_s,
        "mean_luminance": encoding.mean_luminance,
        "pulse_counts_per_frame": encoding.pulse_counts_per_frame,
        "frame_sha256": encoding.frame_sha256,
        "quantization": encoding.quantization,
        "emitted_event_timestamps_s": sorted(set(encoding.stimulus.times_s)),
    }


def _resolved_payload(population: ResolvedPopulation) -> dict[str, Any]:
    return {**asdict(population), "selected_count": population.selected_count}


def _direct_payload(population: DirectTargetPopulation) -> dict[str, Any]:
    return {**asdict(population), "selected_count": population.selected_count}


def visual_panel_preset(data_directory: Path) -> dict[str, Any]:
    config = load_config()
    graph = MemoryMappedConnectome.load(data_directory)
    visual_input = resolve_visual_input(graph.catalog.table, config)
    direct = tuple(
        resolve_direct_targets(graph, visual_input.neuron_indices, type_name)
        for type_name in config["direct_target_types"]
    )
    strongest: list[int] = []
    watch: list[int] = []
    for group in direct:
        ranked = sorted(
            zip(group.contacts_by_neuron, group.neuron_indices, strict=True),
            key=lambda item: (-item[0], item[1]),
        )
        strongest.extend(index for _, index in ranked[:200])
        watch.append(ranked[0][1])
    return {
        "key": "uniform_r1_r6",
        "label": "Uniform field · all R1-R6",
        "input": _resolved_payload(visual_input),
        "outputs": [
            {
                "key": group.key,
                "label": f"Direct R1-R6 {group.type_name}",
                "neuron_indices": group.neuron_indices,
                "body_ids": group.body_ids,
                "selected_count": group.selected_count,
            }
            for group in direct
        ],
        "watch_indices": watch,
        "display_output_indices": sorted(set(strongest)),
        "visualization_indices": sorted(
            set(visual_input.neuron_indices) | set(strongest)
        ),
        "interactive_protocol": {
            "duration_s": float(config["timing"]["total_s"]),
            "start_s": float(config["timing"]["baseline_s"]),
            "stop_s": float(config["timing"]["baseline_s"])
            + int(config["timing"]["frame_count"])
            / float(config["timing"]["frame_rate_hz"]),
            "interval_ms": 4.2,
            "amplitude_mv": float(config["transduction"]["pulse_amplitude_mv"]),
        },
        "caution": (
            "Interactive regular pulses preview the uniform-field engineering proxy; "
            "the offline visual-pathway experiment is the canonical frame-clock run. "
            "R1-R6 soma markers are generally unavailable and no behavior is implied."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_validation(arguments.data_directory, arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)
    print(json.dumps(_summary(result), indent=2, sort_keys=True))


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": result["experiment"],
        "deterministic_repeats_equal": result["deterministic_repeats_equal"],
        "total_wall_seconds": result["total_wall_seconds"],
        "runs": [run for run in result["runs"] if run["repeat"] == 0],
    }


if __name__ == "__main__":
    main()
