"""Compare the point-LIF visual baseline with a graded early-vision reference."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import psutil
import pyarrow as pa

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.connectome_data.runtime import (
    MemoryMappedConnectome,
    SignedSourceProjection,
)
from flybrain_interface.experiments.sensory_descending import resolve_population
from flybrain_interface.experiments.visual_pathway import (
    DirectTargetPopulation,
    resolve_direct_targets,
    resolve_visual_input,
)
from flybrain_interface.sensory.drive import (
    DeterministicProjectedDriveInput,
    ProjectedDriveChannel,
)
from flybrain_interface.sensory.vision import (
    GradedEarlyVisionConfig,
    GradedEarlyVisionTrace,
    UniformLuminanceConfig,
    encode_uniform_luminance,
    simulate_graded_early_vision,
)
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.runtime import (
    RuntimeBackend,
    SparseLIFSimulator,
    SubnormalDrivePolicy,
)
from flybrain_interface.simulation.trace import ChunkRecording, ChunkResult

DEFAULT_CONFIG = Path(__file__).parents[3] / "configs" / "early-vision-v1.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported early-vision config schema")
    return payload


def condition_frames(config: dict[str, Any], condition: str) -> np.ndarray:
    specification = config["conditions"][condition]
    values = np.asarray(specification["frames"], dtype=np.float64)
    expected = int(config["timing"]["frame_count"])
    if values.shape != (expected,):
        raise ValueError(f"condition {condition} must contain {expected} frames")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("condition luminance must be normalized to [0, 1]")
    return np.broadcast_to(values[:, None, None], (values.size, 2, 2)).copy()


def graded_config(config: dict[str, Any]) -> GradedEarlyVisionConfig:
    timing = config["timing"]
    model = config["graded_model"]
    return GradedEarlyVisionConfig(
        frame_rate_hz=float(timing["frame_rate_hz"]),
        neural_dt_ms=float(timing["neural_dt_ms"]),
        background_luminance=float(model["background_luminance"]),
        tonic_release_fraction=float(model["tonic_release_fraction"]),
        photoreceptor_tau_ms=float(model["photoreceptor_tau_ms"]),
        transient_fast_tau_ms=float(model["transient_fast_tau_ms"]),
        transient_slow_tau_ms=float(model["transient_slow_tau_ms"]),
        l3_tau_ms=float(model["l3_tau_ms"]),
        l1_gain_mv=float(model["l1_gain_mv"]),
        l2_gain_mv=float(model["l2_gain_mv"]),
        l2_slow_weight=float(model["l2_slow_weight"]),
        l3_gain_mv=float(model["l3_gain_mv"]),
    )




def make_projected_lamina_drive(
    trace: GradedEarlyVisionTrace,
    direct: tuple[DirectTargetPopulation, ...],
    projections: dict[str, SignedSourceProjection],
    config: dict[str, Any],
    lif_config: ShiuLIFConfig,
) -> DeterministicProjectedDriveInput:
    """Convert graded L1/L2/L3 deltas into signed MaleCNS projected drive."""

    specification = config["projected_drive"]
    equivalent_rate_hz = float(specification["equivalent_rate_hz"])
    activity_clip = float(specification["activity_clip"])
    zero_epsilon = float(specification["activity_zero_epsilon"])
    delay_ms = float(specification["axonal_delay_ms"])
    if equivalent_rate_hz < 0.0 or activity_clip <= 0.0 or zero_epsilon < 0.0:
        raise ValueError("invalid projected-drive scaling parameters")
    exact_delay_steps = delay_ms / lif_config.dt_ms
    delay_steps = round(exact_delay_steps)
    if not np.isclose(exact_delay_steps, delay_steps, atol=1e-12, rtol=0.0):
        raise ValueError("projected-drive axonal delay must align to simulator dt")

    signal_by_type = {
        "L1": trace.l1_delta_mv,
        "L2": trace.l2_delta_mv,
        "L3": trace.l3_delta_mv,
    }
    gain_by_type = {
        "L1": float(config["graded_model"]["l1_gain_mv"]),
        "L2": float(config["graded_model"]["l2_gain_mv"]),
        "L3": float(config["graded_model"]["l3_gain_mv"]),
    }
    amplitude_scale_mv = (
        lif_config.synapse_scale_mv
        * equivalent_rate_hz
        * (lif_config.dt_ms / 1000.0)
    )

    channels: list[ProjectedDriveChannel] = []
    for group in direct:
        signal = np.asarray(signal_by_type[group.type_name], dtype=np.float64)
        activity = np.clip(
            signal / gain_by_type[group.type_name],
            -activity_clip,
            activity_clip,
        )
        activity[np.abs(activity) <= zero_epsilon] = 0.0
        amplitudes = activity * amplitude_scale_mv
        if delay_steps:
            shifted = np.zeros_like(amplitudes)
            if delay_steps < amplitudes.size:
                shifted[delay_steps:] = amplitudes[:-delay_steps]
            amplitudes = shifted

        projection = projections[group.key]
        channels.append(
            ProjectedDriveChannel(
                label=group.key,
                target_indices=projection.target_indices,
                signed_contact_weights=projection.signed_contact_weights,
                amplitudes_mv=amplitudes,
                anatomical_source_count=projection.source_count,
                anatomical_edge_count=projection.anatomical_edge_count,
            )
        )

    return DeterministicProjectedDriveInput(
        channels=tuple(channels),
        dt_ms=lif_config.dt_ms,
    )


def run_comparison(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    graph = MemoryMappedConnectome.load(data_directory)
    visual_input = resolve_visual_input(graph.catalog.table, config)
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
    projections = {
        group.key: graph.project_sources(group.neuron_indices) for group in direct
    }

    lif = config["point_lif_baseline"]
    populations = {group.key: group.neuron_indices for group in direct}
    populations[descending.key] = descending.neuron_indices
    simulator = SparseLIFSimulator(
        graph,
        populations=populations,
        config=ShiuLIFConfig(dt_ms=float(config["timing"]["neural_dt_ms"])),
        backend=cast(RuntimeBackend, str(lif["backend"])),
        subnormal_drive_policy=cast(
            SubnormalDrivePolicy, str(lif["subnormal_drive_policy"])
        ),
    )
    simulator.prepare()

    watch_per_type = int(lif["watch_per_type"])
    watched_by_group = {
        group.key: _strongest_watch(group, watch_per_type) for group in direct
    }
    watched = tuple(
        sorted({index for indices in watched_by_group.values() for index in indices})
    )
    watch_columns = {index: column for column, index in enumerate(watched)}
    duration_s = int(config["timing"]["frame_count"]) / float(
        config["timing"]["frame_rate_hz"]
    )
    graded_model = graded_config(config)

    runs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for condition, specification in config["conditions"].items():
        frames = condition_frames(config, condition)
        stimulus_start_s, stimulus_stop_s = _stimulus_window_s(
            specification, float(config["timing"]["frame_rate_hz"])
        )

        graded_trace = simulate_graded_early_vision(frames, graded_model)
        graded_summary = _graded_summary(
            graded_trace, stimulus_start_s, stimulus_stop_s
        )
        projected_drive = make_projected_lamina_drive(
            graded_trace,
            direct,
            projections,
            config,
            simulator.config,
        )

        encoding = encode_uniform_luminance(
            frames,
            visual_input.neuron_indices,
            UniformLuminanceConfig(
                frame_rate_hz=float(config["timing"]["frame_rate_hz"]),
                max_pulses_per_frame=int(lif["max_pulses_per_frame"]),
                pulse_amplitude_mv=float(lif["pulse_amplitude_mv"]),
                neural_dt_ms=simulator.config.dt_ms,
            ),
        )
        simulator.reset()
        run_started = time.perf_counter()
        point_result = simulator.advance_chunk(
            encoding.stimulus,
            duration_s=duration_s,
            recording=ChunkRecording(
                watched_indices=watched,
                include_neuron_counts=True,
                max_spike_events=0,
            ),
        )
        wall_seconds = time.perf_counter() - run_started
        assert point_result.neuron_spike_counts is not None
        input_indices = np.asarray(visual_input.neuron_indices, dtype=np.int64)
        point_summary = {
            group.key: _point_lif_summary(
                point_result,
                group,
                watched_by_group[group.key],
                watch_columns,
                resting_mv=simulator.config.resting_mv,
                stimulus_start_s=stimulus_start_s,
                stimulus_stop_s=stimulus_stop_s,
            )
            for group in direct
        }
        descending_indices = np.asarray(descending.neuron_indices, dtype=np.int64)
        point_descending_counts = point_result.neuron_spike_counts[descending_indices]
        point_visited_edges = simulator.visited_edges

        simulator.reset()
        hybrid_started = time.perf_counter()
        hybrid_result = simulator.advance_chunk(
            duration_s=duration_s,
            recording=ChunkRecording(
                include_neuron_counts=True,
                max_spike_events=0,
            ),
            projected_drive=projected_drive,
        )
        hybrid_wall_seconds = time.perf_counter() - hybrid_started
        assert hybrid_result.neuron_spike_counts is not None
        hybrid_descending_counts = hybrid_result.neuron_spike_counts[
            descending_indices
        ]
        hybrid_channels = [
            {
                "label": channel.label,
                "source_count": channel.anatomical_source_count,
                "anatomical_edge_count": channel.anatomical_edge_count,
                "unique_target_count": int(channel.target_indices.size),
                "nonzero_step_count": int(
                    np.count_nonzero(channel.amplitudes_mv)
                ),
                "peak_abs_amplitude_mv_per_contact_step": float(
                    np.max(np.abs(channel.amplitudes_mv))
                ),
            }
            for channel in projected_drive.channels
        ]

        runs.append(
            {
                "condition": condition,
                "stimulus_start_s": stimulus_start_s,
                "stimulus_stop_s": stimulus_stop_s,
                "frame_sha256": graded_trace.frame_sha256,
                "mean_luminance": graded_trace.mean_luminance,
                "graded_reference": graded_summary,
                "point_lif_baseline": {
                    "scheduled_events": len(encoding.stimulus.times_s),
                    "input_spikes": int(
                        point_result.neuron_spike_counts[input_indices].sum()
                    ),
                    "total_network_spikes": point_result.total_spikes,
                    "visited_edges": point_visited_edges,
                    "wall_seconds": wall_seconds,
                    "population_responses": point_summary,
                    "descending_response": {
                        "total_spikes": int(point_descending_counts.sum()),
                        "active_neuron_count": int(
                            np.count_nonzero(point_descending_counts)
                        ),
                        "population_rate_hz": point_result.population_rates_hz[
                            descending.key
                        ],
                    },
                },
                "graded_projected_runtime": {
                    "total_network_spikes": hybrid_result.total_spikes,
                    "recurrent_visited_edges": simulator.visited_edges,
                    "projected_drive_target_updates": (
                        simulator.projected_drive_target_updates
                    ),
                    "wall_seconds": hybrid_wall_seconds,
                    "channels": hybrid_channels,
                    "source_population_rates_hz": {
                        group.key: hybrid_result.population_rates_hz[group.key]
                        for group in direct
                    },
                    "active_annotations": _active_annotation_summary(
                        graph.catalog.table,
                        hybrid_result.neuron_spike_counts,
                    ),
                    "descending_response": {
                        "total_spikes": int(hybrid_descending_counts.sum()),
                        "active_neuron_count": int(
                            np.count_nonzero(hybrid_descending_counts)
                        ),
                        "population_rate_hz": hybrid_result.population_rates_hz[
                            descending.key
                        ],
                    },
                },
            }
        )

    gates = evaluate_qualitative_gates(runs, config)
    manifest_directory = Path(__file__).parents[3] / "data" / "manifests"
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Three-way engineering comparison: unchanged point-LIF visual baseline, "
            "a qualitative graded retina-lamina reference, and an optional projected "
            "graded runtime that sends differential L1/L2/L3 activity through signed "
            "MaleCNS outgoing contacts. The projected-drive gain is an engineering "
            "parameter, not a fitted physiological calibration."
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
        },
        "visual_input": {
            "neuron_count": visual_input.selected_count,
            "body_ids": visual_input.body_ids,
        },
        "direct_targets": [
            {
                "key": group.key,
                "type": group.type_name,
                "neuron_count": group.selected_count,
                "direct_edge_count": group.direct_edge_count,
                "direct_contact_count": group.direct_contact_count,
                "projected_outgoing_edge_count": projections[
                    group.key
                ].anatomical_edge_count,
                "projected_unique_target_count": int(
                    projections[group.key].target_indices.size
                ),
                "watched_representatives": watched_by_group[group.key],
            }
            for group in direct
        ],
        "graded_model": asdict(graded_model),
        "qualitative_gates": gates,
        "runs": runs,
        "total_wall_seconds": time.perf_counter() - started,
        "process_rss_bytes": int(psutil.Process().memory_info().rss),
        "software": {
            "python": platform.python_version(),
            "flybrain-interface": version("flybrain-interface"),
            "numpy": version("numpy"),
            "numba": version("numba"),
        },
    }




def _active_annotation_summary(
    table: pa.Table,
    spike_counts: np.ndarray,
    *,
    top_n: int = 12,
) -> dict[str, Any]:
    active_indices = np.flatnonzero(spike_counts > 0).astype(np.int64, copy=False)
    if active_indices.size == 0:
        return {
            "active_neuron_count": 0,
            "top_neurons": [],
            "by_superclass": [],
            "by_class": [],
            "by_type": [],
        }

    active_counts = spike_counts[active_indices]
    order = np.argsort(-active_counts, kind="stable")
    ranked = active_indices[order[:top_n]]
    ranked_counts = spike_counts[ranked]
    ranked_rows = table.take(pa.array(ranked))
    top_neurons = []
    for row, count in enumerate(ranked_counts):
        top_neurons.append(
            {
                "neuron_index": int(ranked[row]),
                "body_id": int(ranked_rows["body_id"][row].as_py()),
                "spikes": int(count),
                "superclass": ranked_rows["superclass"][row].as_py(),
                "class": ranked_rows["class"][row].as_py(),
                "type": ranked_rows["type"][row].as_py(),
            }
        )

    def aggregate(column: str) -> list[dict[str, Any]]:
        values = table[column].take(pa.array(active_indices)).to_pylist()
        totals: dict[str, list[int]] = {}
        for value, count in zip(values, active_counts, strict=True):
            label = "<missing>" if value is None else str(value)
            bucket = totals.setdefault(label, [0, 0])
            bucket[0] += 1
            bucket[1] += int(count)
        ranked_items = sorted(
            totals.items(),
            key=lambda item: (-item[1][1], -item[1][0], item[0]),
        )
        return [
            {
                "label": label,
                "active_neuron_count": values[0],
                "total_spikes": values[1],
            }
            for label, values in ranked_items[:top_n]
        ]

    return {
        "active_neuron_count": int(active_indices.size),
        "top_neurons": top_neurons,
        "by_superclass": aggregate("superclass"),
        "by_class": aggregate("class"),
        "by_type": aggregate("type"),
    }


def evaluate_qualitative_gates(
    runs: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    by_condition = {str(run["condition"]): run["graded_reference"] for run in runs}
    required = {"baseline", "light_step", "dark_step", "light_flash", "dark_flash"}
    missing = required - by_condition.keys()
    if missing:
        raise ValueError(f"missing qualitative-gate conditions: {sorted(missing)}")

    thresholds = config["qualitative_gates"]
    baseline_limit = float(thresholds["baseline_absolute_delta_mv_max"])
    transient_limit = float(thresholds["transient_late_to_peak_ratio_max"])
    l3_minimum = float(thresholds["l3_late_to_peak_ratio_min"])

    baseline = by_condition["baseline"]
    light = by_condition["light_step"]
    dark = by_condition["dark_step"]
    light_flash = by_condition["light_flash"]

    checks = {
        "tonic_background_is_adapted": max(
            baseline[name]["absolute_peak_mv"] for name in ("l1", "l2", "l3")
        )
        <= baseline_limit,
        "light_increment_hyperpolarizes_l1_l2_l3": all(
            light[name]["stimulus_min_mv"] < 0.0 for name in ("l1", "l2", "l3")
        ),
        "light_decrement_depolarizes_l1_l2_l3": all(
            dark[name]["stimulus_max_mv"] > 0.0 for name in ("l1", "l2", "l3")
        ),
        "l1_is_transient": _late_to_peak_ratio(light["l1"]) <= transient_limit,
        "l2_is_transient": _late_to_peak_ratio(light["l2"]) <= transient_limit,
        "l3_is_sustained": _late_to_peak_ratio(light["l3"]) >= l3_minimum,
        "l2_has_post_light_rebound": (
            light_flash["l2"]["post_stimulus_max_mv"] > 0.0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": dict(thresholds),
    }


def _graded_summary(
    trace: GradedEarlyVisionTrace,
    stimulus_start_s: float,
    stimulus_stop_s: float,
) -> dict[str, Any]:
    return {
        "baseline_release": trace.baseline_release,
        "release_min": float(np.min(trace.histamine_release)),
        "release_max": float(np.max(trace.histamine_release)),
        "l1": _signal_metrics(
            trace.l1_delta_mv,
            trace.sample_times_s,
            stimulus_start_s,
            stimulus_stop_s,
        ),
        "l2": _signal_metrics(
            trace.l2_delta_mv,
            trace.sample_times_s,
            stimulus_start_s,
            stimulus_stop_s,
        ),
        "l3": _signal_metrics(
            trace.l3_delta_mv,
            trace.sample_times_s,
            stimulus_start_s,
            stimulus_stop_s,
        ),
    }


def _point_lif_summary(
    result: ChunkResult,
    group: DirectTargetPopulation,
    watched_indices: tuple[int, ...],
    watch_columns: dict[int, int],
    *,
    resting_mv: float,
    stimulus_start_s: float,
    stimulus_stop_s: float,
) -> dict[str, Any]:
    columns = np.asarray(
        [watch_columns[index] for index in watched_indices], dtype=np.int64
    )
    delta = result.voltage_mv[:, columns] - resting_mv
    representative_signal = np.median(delta, axis=1)
    return {
        "population_neuron_count": group.selected_count,
        "watched_representative_count": len(watched_indices),
        "population_rate_hz": result.population_rates_hz[group.key],
        **_signal_metrics(
            representative_signal,
            result.sample_times_s,
            stimulus_start_s,
            stimulus_stop_s,
        ),
    }


def _signal_metrics(
    signal: np.ndarray,
    sample_times_s: np.ndarray,
    stimulus_start_s: float,
    stimulus_stop_s: float,
) -> dict[str, float]:
    stimulus = (sample_times_s >= stimulus_start_s) & (
        sample_times_s < stimulus_stop_s
    )
    late_start_s = stimulus_start_s + 0.75 * (
        stimulus_stop_s - stimulus_start_s
    )
    late = (sample_times_s >= late_start_s) & (sample_times_s < stimulus_stop_s)
    post = sample_times_s >= stimulus_stop_s
    if not np.any(stimulus) or not np.any(late):
        raise ValueError("stimulus window has no neural-grid samples")
    stimulus_values = signal[stimulus]
    late_values = signal[late]
    post_values = signal[post]
    return {
        "absolute_peak_mv": float(np.max(np.abs(signal))),
        "stimulus_min_mv": float(np.min(stimulus_values)),
        "stimulus_max_mv": float(np.max(stimulus_values)),
        "stimulus_peak_abs_mv": float(np.max(np.abs(stimulus_values))),
        "late_stimulus_mean_mv": float(np.mean(late_values)),
        "post_stimulus_min_mv": (
            float(np.min(post_values)) if post_values.size else 0.0
        ),
        "post_stimulus_max_mv": (
            float(np.max(post_values)) if post_values.size else 0.0
        ),
    }


def _late_to_peak_ratio(metrics: dict[str, Any]) -> float:
    peak = float(metrics["stimulus_peak_abs_mv"])
    if peak == 0.0:
        return 0.0
    return abs(float(metrics["late_stimulus_mean_mv"])) / peak


def _strongest_watch(
    group: DirectTargetPopulation, limit: int
) -> tuple[int, ...]:
    if limit <= 0:
        raise ValueError("watch_per_type must be positive")
    ranked = sorted(
        zip(group.neuron_indices, group.contacts_by_neuron, strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(index for index, _ in ranked[:limit])


def _stimulus_window_s(
    specification: dict[str, Any], frame_rate_hz: float
) -> tuple[float, float]:
    start = int(specification["stimulus_start_frame"])
    stop = int(specification["stimulus_stop_frame"])
    if start < 0 or stop <= start or stop > len(specification["frames"]):
        raise ValueError("invalid stimulus frame window")
    return start / frame_rate_hz, stop / frame_rate_hz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare point-LIF and graded early-vision representations"
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("data/processed/malecns-v1.0"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    result = run_comparison(arguments.data_directory, arguments.config)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "qualitative_gates_passed": result["qualitative_gates"]["passed"],
                "total_wall_seconds": result["total_wall_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
