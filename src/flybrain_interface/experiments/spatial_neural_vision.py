"""Propagate spatial graded R1-R6 screen input through the full MaleCNS graph."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.connectome_data.optic_columns import (
    OpticColumnRecord,
    load_optic_columns,
)
from flybrain_interface.connectome_data.runtime import (
    MemoryMappedConnectome,
    SignedSourceProjection,
)
from flybrain_interface.experiments.retinotopy import (
    R1R6ColumnAssignment,
    infer_r1r6_columns,
)
from flybrain_interface.experiments.retinotopy import (
    load_config as load_retinotopy_config,
)
from flybrain_interface.sensory.drive import (
    DeterministicProjectedDriveInput,
    ProjectedDriveChannel,
)
from flybrain_interface.sensory.vision import (
    GradedEarlyVisionConfig,
    RetinotopicScreenEncoding,
    VirtualScreenConfig,
    encode_screen_sequence,
    load_measured_column_directions,
    project_directions_to_screen,
    simulate_graded_photoreceptor_channels,
)
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.runtime import (
    RuntimeBackend,
    SparseLIFSimulator,
    SubnormalDrivePolicy,
)
from flybrain_interface.simulation.trace import ChunkRecording

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-neural-vision-v1.json"
Axis = Literal["horizontal", "vertical"]


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported spatial-neural config schema")
    return payload


def run_validation(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    retinotopy_config = load_retinotopy_config(
        ROOT / str(config["retinotopy_config"])
    )
    workbook_path = ROOT / str(config["optic_workbook"])
    directions_path = ROOT / str(config["measured_directions"])
    eye_map_manifest_path = ROOT / str(config["eye_map_manifest"])
    eye_map_lock = _verify_measured_directions(
        directions_path,
        eye_map_manifest_path,
    )

    graph = MemoryMappedConnectome.load(data_directory)
    columns = load_optic_columns(workbook_path)
    assignments = infer_r1r6_columns(graph, columns, retinotopy_config)
    directions = load_measured_column_directions(directions_path)

    screen_spec = config["screen"]
    screen = VirtualScreenConfig(
        horizontal_fov_deg=float(screen_spec["horizontal_fov_deg"]),
        vertical_fov_deg=float(screen_spec["vertical_fov_deg"]),
    )
    visible_projection = {
        projection.column: projection
        for projection in project_directions_to_screen(directions, screen)
        if projection.visible
    }
    source_groups = _source_groups(assignments, config, set(visible_projection))
    _validate_r1r6_signs(graph, source_groups)

    official_by_name = {column.column: column for column in columns}
    body_ids = np.asarray(
        graph.catalog.table["body_id"].to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    body_to_index = {int(body_id): index for index, body_id in enumerate(body_ids)}
    l1_by_column = _resolve_l1_indices(
        source_groups,
        official_by_name,
        body_to_index,
    )
    source_columns = tuple(sorted(source_groups))
    watch_indices = tuple(l1_by_column[column] for column in source_columns)
    watch_column = {
        neuron_index: column
        for column, neuron_index in l1_by_column.items()
    }
    column_watch_position = {
        column: watch_indices.index(l1_by_column[column])
        for column in source_columns
    }
    projections = {
        column: graph.project_sources(source_groups[column])
        for column in source_columns
    }

    timing = config["timing"]
    lif_config = ShiuLIFConfig(dt_ms=float(timing["neural_dt_ms"]))
    runtime_spec = config["runtime"]
    simulator = SparseLIFSimulator(
        graph,
        config=lif_config,
        backend=cast(RuntimeBackend, str(runtime_spec["backend"])),
        subnormal_drive_policy=cast(
            SubnormalDrivePolicy,
            str(runtime_spec["subnormal_drive_policy"]),
        ),
    )
    simulator.prepare()

    photo_spec = config["graded_photoreceptor"]
    photo_config = GradedEarlyVisionConfig(
        frame_rate_hz=float(timing["frame_rate_hz"]),
        neural_dt_ms=lif_config.dt_ms,
        background_luminance=float(photo_spec["background_luminance"]),
        tonic_release_fraction=float(photo_spec["tonic_release_fraction"]),
        photoreceptor_tau_ms=float(photo_spec["photoreceptor_tau_ms"]),
    )

    conditions = _conditions(config)
    duration_s = _total_frames(config) / float(timing["frame_rate_hz"])
    runs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for condition in conditions:
        frames = _condition_frames(condition, config)
        encoding = encode_screen_sequence(
            frames,
            directions,
            screen,
            frame_rate_hz=float(timing["frame_rate_hz"]),
        )
        intensities = _source_column_intensities(
            encoding,
            source_columns,
            background=float(screen_spec["background_intensity"]),
        )
        photo_trace = simulate_graded_photoreceptor_channels(
            intensities,
            photo_config,
        )
        projected_drive = _make_r1r6_drive(
            photo_trace.histamine_release_delta,
            source_columns,
            source_groups,
            projections,
            config,
            lif_config,
        )

        simulator.reset()
        run_started = time.perf_counter()
        result = simulator.advance_chunk(
            duration_s=duration_s,
            recording=ChunkRecording(
                watched_indices=watch_indices,
                include_neuron_counts=True,
                max_spike_events=0,
            ),
            projected_drive=projected_drive,
        )
        wall_seconds = time.perf_counter() - run_started
        summary = _condition_summary(
            condition,
            result.sample_times_s,
            result.voltage_mv,
            result.synaptic_drive_mv,
            encoding,
            intensities,
            source_columns,
            source_groups,
            visible_projection,
            column_watch_position,
            config,
            simulator.config.resting_mv,
        )
        assert result.neuron_spike_counts is not None
        l1_counts = result.neuron_spike_counts[
            np.asarray(watch_indices, dtype=np.int64)
        ]
        summary.update(
            {
                "total_network_spikes": result.total_spikes,
                "watched_l1_spikes": int(l1_counts.sum()),
                "active_spiking_l1_count": int(np.count_nonzero(l1_counts)),
                "recurrent_visited_edges": simulator.visited_edges,
                "projected_drive_target_updates": (
                    simulator.projected_drive_target_updates
                ),
                "wall_seconds": wall_seconds,
                "release_delta_peak_abs": float(
                    np.max(np.abs(photo_trace.histamine_release_delta))
                ),
                "projected_channel_count": (
                    0 if projected_drive is None else len(projected_drive.channels)
                ),
            }
        )
        runs.append(summary)

    gates = _evaluate_gates(runs, config)
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Spatial screen luminance is sampled at measured viewing directions, "
            "converted to differential graded R1-R6 histamine release, and propagated "
            "through the real signed R1-R6 MaleCNS outgoing contacts. This is a "
            "neural-engineering validation, not a behavioral result."
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "retinotopy_config_sha256": file_sha256(
            ROOT / str(config["retinotopy_config"])
        ),
        "eye_map_lock": eye_map_lock,
        "dataset": {
            "directory": str(graph.directory),
            "neuron_count": graph.neuron_count,
            "edge_count": graph.edge_count,
            "sign_policy": graph.sign_policy.name,
        },
        "source": {
            "visible_source_column_count": len(source_columns),
            "visible_source_r1r6_count": sum(
                len(indices) for indices in source_groups.values()
            ),
            "all_source_signs": sorted(
                {
                    int(graph.presynaptic_signs[index])
                    for indices in source_groups.values()
                    for index in indices
                }
            ),
            "watched_l1_count": len(watch_indices),
            "watch_body_ids": {
                watch_column[index]: int(body_ids[index])
                for index in watch_indices
            },
        },
        "runs": runs,
        "gates": gates,
        "total_wall_seconds": time.perf_counter() - started,
    }




def _verify_measured_directions(
    path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    derived = manifest.get("derived_output")
    if not isinstance(derived, dict):
        raise ValueError("eye-map manifest is missing derived_output lock")
    digest = file_sha256(path)
    expected = str(derived["sha256"])
    if digest != expected:
        raise ValueError("measured-direction CSV SHA-256 mismatch")
    return {
        "path": str(path),
        "record_count": int(derived["record_count"]),
        "sha256": digest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }


def _source_groups(
    assignments: tuple[R1R6ColumnAssignment, ...],
    config: dict[str, Any],
    visible_columns: set[str],
) -> dict[str, tuple[int, ...]]:
    allowed = {str(value) for value in config["neuron_assignment_statuses"]}
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for assignment in assignments:
        if (
            assignment.status in allowed
            and assignment.assigned_column is not None
            and assignment.assigned_column in visible_columns
        ):
            grouped[assignment.assigned_column].append(assignment.neuron_index)
    return {
        column: tuple(sorted(indices))
        for column, indices in grouped.items()
        if indices
    }


def _validate_r1r6_signs(
    graph: MemoryMappedConnectome,
    source_groups: dict[str, tuple[int, ...]],
) -> None:
    if not source_groups:
        raise RuntimeError("spatial screen has no mapped R1-R6 source groups")
    signs = {
        int(graph.presynaptic_signs[index])
        for indices in source_groups.values()
        for index in indices
    }
    if signs != {-1}:
        raise RuntimeError(
            f"spatial R1-R6 sources must be inhibitory under this policy: {signs}"
        )


def _resolve_l1_indices(
    source_groups: dict[str, tuple[int, ...]],
    official_by_name: dict[str, OpticColumnRecord],
    body_to_index: dict[int, int],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for column in source_groups:
        record = official_by_name[column]
        if record.l1_body_id is None:
            raise RuntimeError(f"source column has no official L1: {column}")
        index = body_to_index.get(record.l1_body_id)
        if index is None:
            raise RuntimeError(f"official L1 is absent from graph: {record.l1_body_id}")
        result[column] = index
    return result


def _source_column_intensities(
    encoding: RetinotopicScreenEncoding,
    source_columns: tuple[str, ...],
    *,
    background: float,
) -> np.ndarray:
    lookup = {column: index for index, column in enumerate(encoding.columns)}
    intensity = np.full(
        (len(encoding.frame_timestamps_s), len(source_columns)),
        background,
        dtype=np.float64,
    )
    for destination, column in enumerate(source_columns):
        source = lookup.get(column)
        if source is not None:
            intensity[:, destination] = encoding.intensity_by_frame[:, source]
    return intensity


def _make_r1r6_drive(
    release_delta: np.ndarray,
    source_columns: tuple[str, ...],
    source_groups: dict[str, tuple[int, ...]],
    projections: dict[str, SignedSourceProjection],
    config: dict[str, Any],
    lif_config: ShiuLIFConfig,
) -> DeterministicProjectedDriveInput | None:
    specification = config["projected_drive"]
    photo = config["graded_photoreceptor"]
    background = float(photo["background_luminance"])
    tonic = float(photo["tonic_release_fraction"])
    release_scale = (1.0 - tonic) * max(background, 1.0 - background)
    if release_scale <= 0.0:
        raise ValueError("graded release normalization has zero dynamic range")

    activity_clip = float(specification["activity_clip"])
    zero_epsilon = float(specification["activity_zero_epsilon"])
    equivalent_rate_hz = float(specification["equivalent_rate_hz"])
    delay_ms = float(specification["axonal_delay_ms"])
    exact_delay_steps = delay_ms / lif_config.dt_ms
    delay_steps = round(exact_delay_steps)
    if not np.isclose(exact_delay_steps, delay_steps, atol=1e-12, rtol=0.0):
        raise ValueError("spatial drive delay must align to simulator dt")
    amplitude_scale = (
        lif_config.synapse_scale_mv
        * equivalent_rate_hz
        * (lif_config.dt_ms / 1000.0)
    )

    channels: list[ProjectedDriveChannel] = []
    for column_index, column in enumerate(source_columns):
        activity = np.clip(
            release_delta[:, column_index] / release_scale,
            -activity_clip,
            activity_clip,
        ).copy()
        activity[np.abs(activity) <= zero_epsilon] = 0.0
        amplitudes = activity * amplitude_scale
        if delay_steps:
            shifted = np.zeros_like(amplitudes)
            if delay_steps < amplitudes.size:
                shifted[delay_steps:] = amplitudes[:-delay_steps]
            amplitudes = shifted
        if not np.any(amplitudes):
            continue
        projection = projections[column]
        channels.append(
            ProjectedDriveChannel(
                label=f"r1r6:{column}",
                target_indices=projection.target_indices,
                signed_contact_weights=projection.signed_contact_weights,
                amplitudes_mv=amplitudes,
                anatomical_source_count=len(source_groups[column]),
                anatomical_edge_count=projection.anatomical_edge_count,
            )
        )

    if not channels:
        return None
    return DeterministicProjectedDriveInput(
        channels=tuple(channels),
        dt_ms=lif_config.dt_ms,
    )


def _conditions(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    positions = tuple(float(value) for value in config["screen"]["positions"])
    result: list[dict[str, Any]] = [
        {"name": "baseline", "kind": "baseline", "axis": None, "position": None}
    ]
    for axis in ("horizontal", "vertical"):
        result.extend(
            {
                "name": f"dark_{axis}_{position:.3f}",
                "kind": "dark",
                "axis": axis,
                "position": position,
            }
            for position in positions
        )
    center = min(positions, key=lambda value: abs(value - 0.5))
    result.append(
        {
            "name": "bright_horizontal_center",
            "kind": "bright",
            "axis": "horizontal",
            "position": center,
        }
    )
    return tuple(result)


def _total_frames(config: dict[str, Any]) -> int:
    timing = config["timing"]
    return (
        int(timing["baseline_frames"])
        + int(timing["stimulus_frames"])
        + int(timing["recovery_frames"])
    )


def _condition_frames(
    condition: dict[str, Any],
    config: dict[str, Any],
) -> np.ndarray:
    screen = config["screen"]
    timing = config["timing"]
    width = int(screen["width_px"])
    height = int(screen["height_px"])
    background = float(screen["background_intensity"])
    frames = np.full(
        (_total_frames(config), height, width),
        background,
        dtype=np.float64,
    )
    if condition["kind"] == "baseline":
        return frames

    intensity = (
        float(screen["dark_bar_intensity"])
        if condition["kind"] == "dark"
        else float(screen["bright_bar_intensity"])
    )
    bar_width = float(screen["bar_width_fraction"])
    position = float(condition["position"])
    axis = cast(Axis, condition["axis"])
    extent = width if axis == "horizontal" else height
    low = max(0, int(round((position - bar_width / 2.0) * (extent - 1))))
    high = min(
        extent,
        int(round((position + bar_width / 2.0) * (extent - 1))) + 1,
    )
    start = int(timing["baseline_frames"])
    stop = start + int(timing["stimulus_frames"])
    if axis == "horizontal":
        frames[start:stop, :, low:high] = intensity
    else:
        frames[start:stop, low:high, :] = intensity
    return frames


def _condition_summary(
    condition: dict[str, Any],
    sample_times_s: np.ndarray,
    voltage_mv: np.ndarray,
    drive_mv: np.ndarray,
    encoding: RetinotopicScreenEncoding,
    intensities: np.ndarray,
    source_columns: tuple[str, ...],
    source_groups: dict[str, tuple[int, ...]],
    visible_projection: dict[str, Any],
    column_watch_position: dict[str, int],
    config: dict[str, Any],
    resting_mv: float,
) -> dict[str, Any]:
    timing = config["timing"]
    frame_rate = float(timing["frame_rate_hz"])
    stimulus_start_s = int(timing["baseline_frames"]) / frame_rate
    stimulus_stop_s = (
        int(timing["baseline_frames"]) + int(timing["stimulus_frames"])
    ) / frame_rate
    window = (sample_times_s >= stimulus_start_s) & (
        sample_times_s < stimulus_stop_s
    )
    if not np.any(window):
        raise RuntimeError("spatial neural stimulus window has no samples")

    peak_positive = np.max(drive_mv[window], axis=0)
    peak_negative = np.min(drive_mv[window], axis=0)
    peak_voltage = np.max(voltage_mv[window] - resting_mv, axis=0)
    min_voltage = np.min(voltage_mv[window] - resting_mv, axis=0)
    baseline_abs_drive = float(np.max(np.abs(drive_mv)))

    start_frame = int(timing["baseline_frames"])
    stop_frame = start_frame + int(timing["stimulus_frames"])
    mean_stimulus = np.mean(intensities[start_frame:stop_frame], axis=0)
    background = float(config["screen"]["background_intensity"])
    if condition["kind"] == "dark":
        active = mean_stimulus < background - 1e-9
        sign_fraction = _fraction(
            [
                peak_positive[column_watch_position[column]] > 0.0
                for column, is_active in zip(source_columns, active, strict=True)
                if is_active
            ]
        )
    elif condition["kind"] == "bright":
        active = mean_stimulus > background + 1e-9
        sign_fraction = _fraction(
            [
                peak_negative[column_watch_position[column]] < 0.0
                for column, is_active in zip(source_columns, active, strict=True)
                if is_active
            ]
        )
    else:
        active = np.zeros(len(source_columns), dtype=np.bool_)
        sign_fraction = None

    centroid = None
    active_r1r6_count = int(
        sum(
            len(source_groups[column])
            for column, is_active in zip(source_columns, active, strict=True)
            if is_active
        )
    )
    if condition["kind"] == "dark":
        axis = cast(Axis, condition["axis"])
        weights = np.maximum(0.0, peak_positive)
        coordinates = np.asarray(
            [
                (
                    visible_projection[column].u
                    if axis == "horizontal"
                    else visible_projection[column].v
                )
                for column in source_columns
            ],
            dtype=np.float64,
        )
        if float(np.sum(weights)) > 0.0:
            centroid = float(np.sum(coordinates * weights) / np.sum(weights))

    return {
        "condition": condition["name"],
        "kind": condition["kind"],
        "axis": condition["axis"],
        "position": condition["position"],
        "frame_sha256": encoding.frame_sha256,
        "active_source_column_count": int(np.count_nonzero(active)),
        "active_source_r1r6_count": active_r1r6_count,
        "active_l1_expected_sign_fraction": sign_fraction,
        "l1_drive_centroid": centroid,
        "max_abs_l1_drive_mv": baseline_abs_drive,
        "max_l1_positive_drive_mv": float(np.max(peak_positive)),
        "min_l1_negative_drive_mv": float(np.min(peak_negative)),
        "max_l1_voltage_delta_mv": float(np.max(peak_voltage)),
        "min_l1_voltage_delta_mv": float(np.min(min_voltage)),
    }


def _evaluate_gates(
    runs: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    baseline = next(run for run in runs if run["kind"] == "baseline")
    bright = next(run for run in runs if run["kind"] == "bright")
    dark = [run for run in runs if run["kind"] == "dark"]
    horizontal = sorted(
        (run for run in dark if run["axis"] == "horizontal"),
        key=lambda run: float(run["position"]),
    )
    vertical = sorted(
        (run for run in dark if run["axis"] == "vertical"),
        key=lambda run: float(run["position"]),
    )

    horizontal_correlation = _centroid_correlation(horizontal)
    vertical_correlation = _centroid_correlation(vertical)
    checks = {
        "baseline_quiet": baseline["max_abs_l1_drive_mv"]
        <= float(gates["baseline_max_abs_l1_drive_mv"]),
        "dark_l1_sign": min(
            float(run["active_l1_expected_sign_fraction"]) for run in dark
        )
        >= float(gates["minimum_dark_active_l1_positive_fraction"]),
        "bright_l1_sign": float(bright["active_l1_expected_sign_fraction"])
        >= float(gates["minimum_bright_active_l1_negative_fraction"]),
        "horizontal_drive_centroid": horizontal_correlation
        >= float(gates["minimum_horizontal_drive_centroid_correlation"]),
        "vertical_drive_centroid": vertical_correlation
        >= float(gates["minimum_vertical_drive_centroid_correlation"]),
        "network_spike_budget": max(int(run["total_network_spikes"]) for run in runs)
        <= int(gates["maximum_network_spikes_per_run"]),
    }
    if bool(gates["require_monotonic_drive_centroid"]):
        checks["horizontal_drive_monotonic"] = _strictly_increasing(
            [float(run["l1_drive_centroid"]) for run in horizontal]
        )
        checks["vertical_drive_monotonic"] = _strictly_increasing(
            [float(run["l1_drive_centroid"]) for run in vertical]
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "horizontal_drive_centroid_correlation": horizontal_correlation,
        "vertical_drive_centroid_correlation": vertical_correlation,
        "thresholds": gates,
    }


def _centroid_correlation(runs: list[dict[str, Any]]) -> float:
    expected = np.asarray([float(run["position"]) for run in runs])
    actual = np.asarray([float(run["l1_drive_centroid"]) for run in runs])
    value = float(np.corrcoef(expected, actual)[0, 1])
    if not np.isfinite(value):
        raise RuntimeError("spatial neural centroid correlation is non-finite")
    return value


def _strictly_increasing(values: list[float]) -> bool:
    return all(after > before for before, after in zip(values, values[1:]))


def _fraction(values: list[bool]) -> float:
    if not values:
        raise RuntimeError("spatial condition has no active mapped R1-R6 columns")
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("data/processed/malecns-v1.0"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    result = run_validation(arguments.data_directory, arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "source": result["source"],
                "gates": result["gates"],
                "runs": result["runs"],
                "total_wall_seconds": result["total_wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
