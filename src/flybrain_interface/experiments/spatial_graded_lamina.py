"""Compare spatial graded-lamina hybrid against point-LIF early visual layers."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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
    infer_r1r6_columns,
)
from flybrain_interface.experiments.retinotopy import (
    load_config as load_retinotopy_config,
)
from flybrain_interface.experiments.spatial_neural_vision import (
    _condition_frames,
    _conditions,
    _source_column_intensities,
    _source_groups,
    _total_frames,
    _verify_measured_directions,
)
from flybrain_interface.sensory.drive import (
    DeterministicProjectedDriveInput,
    ProjectedDriveChannel,
)
from flybrain_interface.sensory.vision import (
    GradedEarlyVisionChannelTrace,
    GradedEarlyVisionConfig,
    VirtualScreenConfig,
    encode_screen_sequence,
    load_measured_column_directions,
    project_directions_to_screen,
    simulate_graded_early_vision_channels,
)
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.runtime import (
    RuntimeBackend,
    SparseLIFSimulator,
    SubnormalDrivePolicy,
)
from flybrain_interface.simulation.trace import ChunkRecording

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-graded-lamina-v1.json"


@dataclass(frozen=True, slots=True)
class CartridgeSources:
    column: str
    r1r6_indices: tuple[int, ...]
    l1_index: int
    l2_index: int | None
    l3_index: int | None

    def source(self, type_name: str) -> int | None:
        if type_name == "L1":
            return self.l1_index
        if type_name == "L2":
            return self.l2_index
        if type_name == "L3":
            return self.l3_index
        raise KeyError(type_name)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported spatial graded-lamina config schema")
    return payload


def run_validation(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    retinotopy_config_path = ROOT / str(config["retinotopy_config"])
    workbook_path = ROOT / str(config["optic_workbook"])
    directions_path = ROOT / str(config["measured_directions"])
    eye_map_manifest_path = ROOT / str(config["eye_map_manifest"])
    eye_map_lock = _verify_measured_directions(
        directions_path,
        eye_map_manifest_path,
    )

    graph = MemoryMappedConnectome.load(data_directory)
    columns = load_optic_columns(workbook_path)
    assignments = infer_r1r6_columns(
        graph,
        columns,
        load_retinotopy_config(retinotopy_config_path),
    )
    directions = load_measured_column_directions(directions_path)
    screen_spec = config["screen"]
    screen = VirtualScreenConfig(
        horizontal_fov_deg=float(screen_spec["horizontal_fov_deg"]),
        vertical_fov_deg=float(screen_spec["vertical_fov_deg"]),
    )
    visible_columns = {
        projection.column
        for projection in project_directions_to_screen(directions, screen)
        if projection.visible
    }
    source_groups = _source_groups(assignments, config, visible_columns)
    cartridges = resolve_cartridge_sources(
        graph,
        source_groups,
        columns,
    )
    source_columns = tuple(cartridge.column for cartridge in cartridges)
    _validate_lamina_signs(graph, cartridges)

    clamped = tuple(
        sorted(
            {
                index
                for cartridge in cartridges
                for index in (
                    *cartridge.r1r6_indices,
                    cartridge.l1_index,
                    cartridge.l2_index,
                    cartridge.l3_index,
                )
                if index is not None
            }
        )
    )
    projections = _lamina_projections(graph, cartridges)
    observation = config["observation"]
    downstream_watch = _top_downstream_targets(
        projections,
        set(clamped),
        limit=int(observation["top_downstream_targets"]),
    )

    timing = config["timing"]
    lif_config = ShiuLIFConfig(dt_ms=float(timing["neural_dt_ms"]))
    runtime = config["runtime"]
    simulator = SparseLIFSimulator(
        graph,
        clamped_indices=clamped,
        config=lif_config,
        backend=cast(RuntimeBackend, str(runtime["backend"])),
        subnormal_drive_policy=cast(
            SubnormalDrivePolicy,
            str(runtime["subnormal_drive_policy"]),
        ),
    )
    simulator.prepare()

    graded_spec = config["graded_model"]
    graded_config = GradedEarlyVisionConfig(
        frame_rate_hz=float(timing["frame_rate_hz"]),
        neural_dt_ms=lif_config.dt_ms,
        background_luminance=float(graded_spec["background_luminance"]),
        tonic_release_fraction=float(graded_spec["tonic_release_fraction"]),
        photoreceptor_tau_ms=float(graded_spec["photoreceptor_tau_ms"]),
        transient_fast_tau_ms=float(graded_spec["transient_fast_tau_ms"]),
        transient_slow_tau_ms=float(graded_spec["transient_slow_tau_ms"]),
        l3_tau_ms=float(graded_spec["l3_tau_ms"]),
        l1_gain_mv=float(graded_spec["l1_gain_mv"]),
        l2_gain_mv=float(graded_spec["l2_gain_mv"]),
        l2_slow_weight=float(graded_spec["l2_slow_weight"]),
        l3_gain_mv=float(graded_spec["l3_gain_mv"]),
    )

    duration_s = _total_frames(config) / float(timing["frame_rate_hz"])
    runs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for condition in _conditions(config):
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
        graded = simulate_graded_early_vision_channels(
            intensities,
            graded_config,
        )
        projected = _make_lamina_drive(
            graded,
            cartridges,
            projections,
            config,
            lif_config,
        )

        simulator.reset()
        run_started = time.perf_counter()
        result = simulator.advance_chunk(
            duration_s=duration_s,
            recording=ChunkRecording(
                watched_indices=downstream_watch,
                include_neuron_counts=True,
                max_spike_events=0,
            ),
            projected_drive=projected,
        )
        wall_seconds = time.perf_counter() - run_started
        assert result.neuron_spike_counts is not None
        counts = result.neuron_spike_counts
        clamped_counts = counts[np.asarray(clamped, dtype=np.int64)]
        clamped_voltage_delta = float(
            np.max(
                np.abs(
                    simulator.voltage_mv[np.asarray(clamped, dtype=np.int64)]
                    - simulator.config.resting_mv
                )
            )
        )
        runs.append(
            {
                "condition": condition["name"],
                "kind": condition["kind"],
                "axis": condition["axis"],
                "position": condition["position"],
                "frame_sha256": encoding.frame_sha256,
                "projected_channel_count": (
                    0 if projected is None else len(projected.channels)
                ),
                "total_network_spikes": result.total_spikes,
                "clamped_visual_spikes": int(clamped_counts.sum()),
                "active_clamped_visual_count": int(
                    np.count_nonzero(clamped_counts)
                ),
                "clamped_visual_abs_voltage_delta_mv": clamped_voltage_delta,
                "recurrent_visited_edges": simulator.visited_edges,
                "projected_drive_target_updates": (
                    simulator.projected_drive_target_updates
                ),
                "active_annotations": _active_annotation_summary(
                    graph,
                    counts,
                ),
                "downstream_subthreshold": _subthreshold_summary(
                    graph,
                    downstream_watch,
                    result.voltage_mv,
                    result.synaptic_drive_mv,
                    simulator.config.resting_mv,
                    threshold_mv=float(
                        observation["subthreshold_threshold_mv"]
                    ),
                ),
                "wall_seconds": wall_seconds,
            }
        )

    gates = _evaluate_gates(runs, config)
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Spatial screen input is converted to graded R1-R6/L1/L2/L3 states. "
            "Those externally managed early-vision neurons are clamped in the point-"
            "LIF runtime while their graded differential outputs are projected through "
            "their real signed MaleCNS outgoing contacts."
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "retinotopy_config_sha256": file_sha256(retinotopy_config_path),
        "eye_map_lock": eye_map_lock,
        "dataset": {
            "directory": str(graph.directory),
            "neuron_count": graph.neuron_count,
            "edge_count": graph.edge_count,
            "sign_policy": graph.sign_policy.name,
        },
        "frontend": {
            "visible_column_count": len(cartridges),
            "visible_r1r6_count": sum(
                len(cartridge.r1r6_indices) for cartridge in cartridges
            ),
            "l1_source_count": sum(
                cartridge.l1_index is not None for cartridge in cartridges
            ),
            "l2_source_count": sum(
                cartridge.l2_index is not None for cartridge in cartridges
            ),
            "l3_source_count": sum(
                cartridge.l3_index is not None for cartridge in cartridges
            ),
            "clamped_neuron_count": len(clamped),
            "downstream_watch_count": len(downstream_watch),
            "source_signs": _source_sign_summary(graph, cartridges),
        },
        "runs": runs,
        "gates": gates,
        "total_wall_seconds": time.perf_counter() - started,
    }


def resolve_cartridge_sources(
    graph: MemoryMappedConnectome,
    source_groups: dict[str, tuple[int, ...]],
    columns: tuple[OpticColumnRecord, ...],
) -> tuple[CartridgeSources, ...]:
    target_types = np.asarray(graph.catalog.table["type"].to_pylist(), dtype=object)
    body_ids = np.asarray(
        graph.catalog.table["body_id"].to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    body_to_index = {int(body_id): index for index, body_id in enumerate(body_ids)}
    official = {column.column: column for column in columns}

    resolved: list[CartridgeSources] = []
    for column in sorted(source_groups):
        candidates: dict[str, dict[int, int]] = {
            "L1": defaultdict(int),
            "L2": defaultdict(int),
            "L3": defaultdict(int),
        }
        for source in source_groups[column]:
            start = int(graph.outgoing_indptr[source])
            stop = int(graph.outgoing_indptr[source + 1])
            for target, weight in zip(
                graph.target_indices[start:stop],
                graph.outgoing_synapse_counts[start:stop],
                strict=True,
            ):
                target_index = int(target)
                type_name = target_types[target_index]
                if type_name in candidates:
                    candidates[str(type_name)][target_index] += int(weight)

        selected: dict[str, int | None] = {}
        for type_name, contacts in candidates.items():
            if len(contacts) > 1:
                raise RuntimeError(
                    f"{column} has competing direct {type_name} targets: "
                    f"{len(contacts)}"
                )
            selected[type_name] = next(iter(contacts), None)

        l1_index = selected["L1"]
        if l1_index is None:
            raise RuntimeError(f"{column} has no direct L1 target")
        official_l1 = official[column].l1_body_id
        if official_l1 is None:
            raise RuntimeError(f"{column} has no official L1 assignment")
        official_index = body_to_index.get(official_l1)
        if l1_index != official_index:
            raise RuntimeError(
                f"{column} direct L1 does not match official L1 assignment"
            )

        resolved.append(
            CartridgeSources(
                column=column,
                r1r6_indices=source_groups[column],
                l1_index=l1_index,
                l2_index=selected["L2"],
                l3_index=selected["L3"],
            )
        )
    return tuple(resolved)


def _validate_lamina_signs(
    graph: MemoryMappedConnectome,
    cartridges: tuple[CartridgeSources, ...],
) -> None:
    expected = {"L1": {-1}, "L2": {1}, "L3": {1}}
    actual = _source_sign_summary(graph, cartridges)
    for type_name, expected_signs in expected.items():
        signs = set(actual[type_name])
        if signs != expected_signs:
            raise RuntimeError(
                f"{type_name} source signs differ from expected policy: {signs}"
            )


def _source_sign_summary(
    graph: MemoryMappedConnectome,
    cartridges: tuple[CartridgeSources, ...],
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for type_name in ("L1", "L2", "L3"):
        indices = [
            source
            for cartridge in cartridges
            if (source := cartridge.source(type_name)) is not None
        ]
        result[type_name] = sorted(
            {int(graph.presynaptic_signs[index]) for index in indices}
        )
    return result


def _lamina_projections(
    graph: MemoryMappedConnectome,
    cartridges: tuple[CartridgeSources, ...],
) -> dict[tuple[str, str], SignedSourceProjection]:
    projections: dict[tuple[str, str], SignedSourceProjection] = {}
    for cartridge in cartridges:
        for type_name in ("L1", "L2", "L3"):
            source = cartridge.source(type_name)
            if source is None:
                continue
            projections[(type_name, cartridge.column)] = graph.project_sources(
                (source,)
            )
    return projections


def _make_lamina_drive(
    graded: GradedEarlyVisionChannelTrace,
    cartridges: tuple[CartridgeSources, ...],
    projections: dict[tuple[str, str], SignedSourceProjection],
    config: dict[str, Any],
    lif_config: ShiuLIFConfig,
) -> DeterministicProjectedDriveInput | None:
    specification = config["projected_drive"]
    model = config["graded_model"]
    signals = {
        "L1": graded.l1_delta_mv,
        "L2": graded.l2_delta_mv,
        "L3": graded.l3_delta_mv,
    }
    gains = {
        "L1": float(model["l1_gain_mv"]),
        "L2": float(model["l2_gain_mv"]),
        "L3": float(model["l3_gain_mv"]),
    }
    equivalent_rate_hz = float(specification["equivalent_rate_hz"])
    activity_clip = float(specification["activity_clip"])
    zero_epsilon = float(specification["activity_zero_epsilon"])
    delay_ms = float(specification["axonal_delay_ms"])
    delay_exact = delay_ms / lif_config.dt_ms
    delay_steps = round(delay_exact)
    if not np.isclose(delay_exact, delay_steps, atol=1e-12, rtol=0.0):
        raise ValueError("graded-lamina delay must align to simulator dt")
    amplitude_scale = (
        lif_config.synapse_scale_mv
        * equivalent_rate_hz
        * (lif_config.dt_ms / 1000.0)
    )

    channels: list[ProjectedDriveChannel] = []
    for column_index, cartridge in enumerate(cartridges):
        for type_name in ("L1", "L2", "L3"):
            source = cartridge.source(type_name)
            if source is None:
                continue
            activity = np.clip(
                signals[type_name][:, column_index] / gains[type_name],
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
            projection = projections[(type_name, cartridge.column)]
            channels.append(
                ProjectedDriveChannel(
                    label=f"{type_name.lower()}:{cartridge.column}",
                    target_indices=projection.target_indices,
                    signed_contact_weights=projection.signed_contact_weights,
                    amplitudes_mv=amplitudes,
                    anatomical_source_count=1,
                    anatomical_edge_count=projection.anatomical_edge_count,
                )
            )
    if not channels:
        return None
    return DeterministicProjectedDriveInput(
        channels=tuple(channels),
        dt_ms=lif_config.dt_ms,
    )




def _top_downstream_targets(
    projections: dict[tuple[str, str], SignedSourceProjection],
    excluded: set[int],
    *,
    limit: int,
) -> tuple[int, ...]:
    if limit <= 0:
        raise ValueError("top_downstream_targets must be positive")
    scores: defaultdict[int, float] = defaultdict(float)
    for projection in projections.values():
        for target, weight in zip(
            projection.target_indices,
            projection.signed_contact_weights,
            strict=True,
        ):
            target_index = int(target)
            if target_index in excluded:
                continue
            scores[target_index] += abs(float(weight))
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return tuple(index for index, _ in ranked[:limit])


def _subthreshold_summary(
    graph: MemoryMappedConnectome,
    watched_indices: tuple[int, ...],
    voltage_mv: np.ndarray,
    drive_mv: np.ndarray,
    resting_mv: float,
    *,
    threshold_mv: float,
    top_n: int = 10,
) -> dict[str, Any]:
    if threshold_mv <= 0.0:
        raise ValueError("subthreshold threshold must be positive")
    if not watched_indices:
        return {
            "watched_count": 0,
            "responsive_count": 0,
            "threshold_mv": threshold_mv,
            "max_abs_voltage_delta_mv": 0.0,
            "max_abs_drive_mv": 0.0,
            "top_neurons": [],
        }
    delta = voltage_mv - resting_mv
    peak_voltage = np.max(np.abs(delta), axis=0)
    peak_drive = np.max(np.abs(drive_mv), axis=0)
    responsive = peak_voltage >= threshold_mv
    order = np.argsort(-peak_voltage, kind="stable")[:top_n]
    table = graph.catalog.table
    return {
        "watched_count": len(watched_indices),
        "responsive_count": int(np.count_nonzero(responsive)),
        "threshold_mv": threshold_mv,
        "max_abs_voltage_delta_mv": float(np.max(peak_voltage)),
        "median_peak_abs_voltage_delta_mv": float(np.median(peak_voltage)),
        "max_abs_drive_mv": float(np.max(peak_drive)),
        "max_positive_voltage_delta_mv": float(np.max(delta)),
        "min_negative_voltage_delta_mv": float(np.min(delta)),
        "top_neurons": [
            {
                "neuron_index": watched_indices[int(position)],
                "body_id": int(
                    table["body_id"][watched_indices[int(position)]].as_py()
                ),
                "type": table["type"][watched_indices[int(position)]].as_py(),
                "superclass": table["superclass"][
                    watched_indices[int(position)]
                ].as_py(),
                "peak_abs_voltage_delta_mv": float(peak_voltage[int(position)]),
                "peak_abs_drive_mv": float(peak_drive[int(position)]),
            }
            for position in order
        ],
    }


def _active_annotation_summary(
    graph: MemoryMappedConnectome,
    counts: np.ndarray,
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    active = np.flatnonzero(counts > 0)
    if not active.size:
        return {
            "active_neuron_count": 0,
            "by_superclass": [],
            "by_type": [],
            "top_neurons": [],
        }
    table = graph.catalog.table
    superclass_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    for index in active:
        spikes = int(counts[index])
        superclass = table["superclass"][int(index)].as_py()
        type_name = table["type"][int(index)].as_py()
        superclass_counts["<missing>" if superclass is None else str(superclass)] += (
            spikes
        )
        type_counts["<missing>" if type_name is None else str(type_name)] += spikes

    ranked = active[np.argsort(-counts[active], kind="stable")[:top_n]]
    return {
        "active_neuron_count": int(active.size),
        "by_superclass": [
            {"label": label, "total_spikes": spikes}
            for label, spikes in superclass_counts.most_common(top_n)
        ],
        "by_type": [
            {"label": label, "total_spikes": spikes}
            for label, spikes in type_counts.most_common(top_n)
        ],
        "top_neurons": [
            {
                "neuron_index": int(index),
                "body_id": int(table["body_id"][int(index)].as_py()),
                "type": table["type"][int(index)].as_py(),
                "superclass": table["superclass"][int(index)].as_py(),
                "spikes": int(counts[index]),
            }
            for index in ranked
        ],
    }


def _evaluate_gates(
    runs: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    specification = config["gates"]
    baseline = next(run for run in runs if run["kind"] == "baseline")
    stimuli = [run for run in runs if run["kind"] != "baseline"]
    dark = [run for run in runs if run["kind"] == "dark"]
    checks = {
        "baseline_quiet": int(baseline["total_network_spikes"])
        <= int(specification["baseline_max_network_spikes"]),
        "clamped_visual_spikes_zero": max(
            int(run["clamped_visual_spikes"]) for run in runs
        )
        <= int(specification["maximum_clamped_visual_spikes"]),
        "clamped_visual_voltage_zero": max(
            float(run["clamped_visual_abs_voltage_delta_mv"]) for run in runs
        )
        <= float(specification["maximum_clamped_visual_abs_voltage_delta_mv"]),
        "network_spike_budget": max(
            int(run["total_network_spikes"]) for run in runs
        )
        <= int(specification["maximum_network_spikes_per_run"]),
        "dark_reaches_downstream": sum(
            int(run["total_network_spikes"]) > 0 for run in dark
        )
        >= int(specification["minimum_dark_conditions_with_downstream_spikes"]),
    }
    if bool(specification["require_all_stimulus_conditions_projected"]):
        checks["all_stimuli_projected"] = all(
            int(run["projected_channel_count"]) > 0 for run in stimuli
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": specification,
    }


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
                "frontend": result["frontend"],
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
