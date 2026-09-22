"""Characterize continuous MaleCNS motor-candidate state under spatial vision."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.connectome_data.optic_columns import load_optic_columns
from flybrain_interface.connectome_data.runtime import MemoryMappedConnectome
from flybrain_interface.experiments.retinotopy import infer_r1r6_columns
from flybrain_interface.experiments.retinotopy import (
    load_config as load_retinotopy_config,
)
from flybrain_interface.experiments.spatial_graded_lamina import (
    _lamina_projections,
    _validate_lamina_signs,
    resolve_cartridge_sources,
)
from flybrain_interface.experiments.spatial_graded_lamina import (
    load_config as load_base_config,
)
from flybrain_interface.experiments.spatial_neural_vision import (
    _source_groups,
    _verify_measured_directions,
)
from flybrain_interface.experiments.spatial_subthreshold_readout import (
    _graded_config,
    _run_position,
)
from flybrain_interface.sensory.vision import (
    VirtualScreenConfig,
    load_measured_column_directions,
    project_directions_to_screen,
)
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.runtime import (
    RuntimeBackend,
    SparseLIFSimulator,
    SubnormalDrivePolicy,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-candidate-readout-v1.json"
Axis = Literal["horizontal", "vertical"]


@dataclass(frozen=True, slots=True)
class MotorPopulation:
    key: str
    type_name: str
    soma_side: str | None
    neuron_indices: tuple[int, ...]
    body_ids: tuple[int, ...]
    instances: tuple[str, ...]


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported spatial motor-candidate config schema")
    return payload


def resolve_motor_population(
    table: pa.Table,
    specification: dict[str, Any],
) -> MotorPopulation:
    type_name = str(specification["type"])
    soma_side = specification.get("soma_side")
    if soma_side is not None:
        soma_side = str(soma_side)
        if soma_side not in {"L", "R"}:
            raise ValueError("motor population soma_side must be L, R, or null")

    superclasses = table["superclass"].to_pylist()
    types = table["type"].to_pylist()
    sides = table["soma_side"].to_pylist()
    neuron_indices = table["neuron_index"].to_pylist()
    body_ids = table["body_id"].to_pylist()
    instances = table["instance"].to_pylist()

    matched = [
        index
        for index, (superclass, candidate_type, candidate_side) in enumerate(
            zip(superclasses, types, sides, strict=True)
        )
        if superclass == "descending_neuron"
        and candidate_type == type_name
        and (soma_side is None or candidate_side == soma_side)
    ]
    if not matched:
        raise ValueError(
            f"motor population selected no neurons: {type_name=} {soma_side=}"
        )
    ordered = sorted(matched, key=lambda index: int(body_ids[index]))
    return MotorPopulation(
        key=str(specification["key"]),
        type_name=type_name,
        soma_side=soma_side,
        neuron_indices=tuple(int(neuron_indices[index]) for index in ordered),
        body_ids=tuple(int(body_ids[index]) for index in ordered),
        instances=tuple(str(instances[index]) for index in ordered),
    )


def resolve_motor_populations(
    table: pa.Table,
    config: dict[str, Any],
) -> tuple[MotorPopulation, ...]:
    populations = tuple(
        resolve_motor_population(table, specification)
        for specification in config["motor_populations"]
    )
    keys = [population.key for population in populations]
    if len(set(keys)) != len(keys):
        raise ValueError("motor population keys must be unique")
    return populations


def _population_metrics(
    sample: dict[str, Any],
    population: MotorPopulation,
    watch_position: dict[int, int],
) -> dict[str, Any]:
    positions = np.asarray(
        [watch_position[index] for index in population.neuron_indices],
        dtype=np.int64,
    )
    voltage = np.asarray(sample["voltage"], dtype=np.float64)[positions]
    drive = np.asarray(sample["drive"], dtype=np.float64)[positions]
    spike = np.asarray(sample["spike"], dtype=np.float64)[positions]
    return {
        "member_count": len(population.neuron_indices),
        "mean_signed_peak_voltage_delta_mv": float(np.mean(voltage)),
        "mean_abs_peak_voltage_delta_mv": float(np.mean(np.abs(voltage))),
        "max_abs_peak_voltage_delta_mv": float(np.max(np.abs(voltage))),
        "mean_signed_peak_drive_mv": float(np.mean(drive)),
        "mean_abs_peak_drive_mv": float(np.mean(np.abs(drive))),
        "max_abs_peak_drive_mv": float(np.max(np.abs(drive))),
        "spike_count": int(np.sum(spike)),
        "members": [
            {
                "neuron_index": int(neuron_index),
                "body_id": int(body_id),
                "instance": instance,
                "signed_peak_voltage_delta_mv": float(member_voltage),
                "signed_peak_drive_mv": float(member_drive),
                "spike_count": int(member_spike),
            }
            for (
                neuron_index,
                body_id,
                instance,
                member_voltage,
                member_drive,
                member_spike,
            ) in zip(
                population.neuron_indices,
                population.body_ids,
                population.instances,
                voltage,
                drive,
                spike,
                strict=True,
            )
        ],
    }


def _channel_state(populations: dict[str, dict[str, Any]]) -> dict[str, float]:
    voltage = "mean_signed_peak_voltage_delta_mv"
    drive = "mean_signed_peak_drive_mv"
    return {
        "steering_voltage_right_minus_left_mv": (
            float(populations["steer_right"][voltage])
            - float(populations["steer_left"][voltage])
        ),
        "steering_drive_right_minus_left_mv": (
            float(populations["steer_right"][drive])
            - float(populations["steer_left"][drive])
        ),
        "secondary_steering_voltage_right_minus_left_mv": (
            float(populations["steer_right_secondary"][voltage])
            - float(populations["steer_left_secondary"][voltage])
        ),
        "vertical_voltage_backward_minus_forward_mv": (
            float(populations["backward"][voltage])
            - float(populations["forward"][voltage])
        ),
        "vertical_drive_backward_minus_forward_mv": (
            float(populations["backward"][drive])
            - float(populations["forward"][drive])
        ),
    }


def _shortest_upstream_hop(
    graph: MemoryMappedConnectome,
    target_index: int,
    source_indices: set[int],
    *,
    max_hops: int = 8,
) -> int | None:
    """Return the shortest directed source-to-target hop count."""

    frontier = {target_index}
    visited = set(frontier)
    for hop in range(1, max_hops + 1):
        upstream: set[int] = set()
        for neuron_index in frontier:
            upstream.update(
                int(value)
                for value in graph.incoming(neuron_index).source_indices
            )
        if upstream & source_indices:
            return hop
        frontier = upstream - visited
        if not frontier:
            return None
        visited.update(frontier)
    return None


def _effective_forward_distances(
    graph: MemoryMappedConnectome,
    sources: set[int],
    *,
    max_hops: int,
) -> dict[int, int]:
    distances = {int(source): 0 for source in sources}
    frontier = set(distances)
    for hop in range(1, max_hops + 1):
        next_frontier: set[int] = set()
        for source in frontier:
            if int(graph.presynaptic_signs[source]) == 0:
                continue
            for target in graph.outgoing(source).target_indices:
                target_index = int(target)
                if target_index not in distances:
                    distances[target_index] = hop
                    next_frontier.add(target_index)
        if not next_frontier:
            break
        frontier = next_frontier
    return distances


def _shortest_effective_layers(
    graph: MemoryMappedConnectome,
    forward: dict[int, int],
    target_index: int,
    *,
    max_hops: int,
) -> tuple[tuple[int, ...], ...] | None:
    distance = forward.get(target_index)
    if distance is None or distance > max_hops:
        return None
    reverse = {target_index: 0}
    frontier = {target_index}
    for hop in range(1, distance + 1):
        next_frontier: set[int] = set()
        for target in frontier:
            for source in graph.incoming(target).source_indices:
                source_index = int(source)
                if int(graph.presynaptic_signs[source_index]) == 0:
                    continue
                if source_index not in reverse:
                    reverse[source_index] = hop
                    next_frontier.add(source_index)
        frontier = next_frontier
    layers = tuple(
        tuple(
            sorted(
                neuron
                for neuron, source_distance in forward.items()
                if source_distance == hop
                and reverse.get(neuron) == distance - hop
            )
        )
        for hop in range(distance + 1)
    )
    return layers if all(layers) else None


def _correlation(
    positions: tuple[float, ...],
    values: list[float],
) -> float | None:
    x = np.asarray(positions, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if float(np.std(y)) <= 1e-15:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run_characterization(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    base_config_path = ROOT / str(config["base_config"])
    base = load_base_config(base_config_path)
    positions = tuple(float(value) for value in config["positions"])
    if len(positions) < 3 or tuple(sorted(positions)) != positions:
        raise ValueError("motor characterization positions must be sorted")

    graph = MemoryMappedConnectome.load(data_directory)
    motor_populations = resolve_motor_populations(graph.catalog.table, config)
    motor_by_key = {population.key: population for population in motor_populations}
    required_keys = {
        "steer_left",
        "steer_right",
        "steer_left_secondary",
        "steer_right_secondary",
        "forward",
        "backward",
    }
    if set(motor_by_key) != required_keys:
        raise ValueError("motor population configuration has unexpected keys")
    watch_indices = tuple(
        sorted(
            {
                index
                for population in motor_populations
                for index in population.neuron_indices
            }
        )
    )
    watch_position = {
        neuron_index: position
        for position, neuron_index in enumerate(watch_indices)
    }

    retinotopy_config_path = ROOT / str(base["retinotopy_config"])
    workbook_path = ROOT / str(base["optic_workbook"])
    directions_path = ROOT / str(base["measured_directions"])
    eye_map_manifest_path = ROOT / str(base["eye_map_manifest"])
    eye_map_lock = _verify_measured_directions(
        directions_path,
        eye_map_manifest_path,
    )
    columns = load_optic_columns(workbook_path)
    assignments = infer_r1r6_columns(
        graph,
        columns,
        load_retinotopy_config(retinotopy_config_path),
    )
    directions = load_measured_column_directions(directions_path)
    screen_spec = base["screen"]
    screen = VirtualScreenConfig(
        horizontal_fov_deg=float(screen_spec["horizontal_fov_deg"]),
        vertical_fov_deg=float(screen_spec["vertical_fov_deg"]),
    )
    visible_columns = {
        projection.column
        for projection in project_directions_to_screen(directions, screen)
        if projection.visible
    }
    source_groups = _source_groups(assignments, base, visible_columns)
    cartridges = resolve_cartridge_sources(graph, source_groups, columns)
    _validate_lamina_signs(graph, cartridges)
    source_columns = tuple(cartridge.column for cartridge in cartridges)
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
    if set(watch_indices) & set(clamped):
        raise RuntimeError("motor candidates overlap clamped early-vision neurons")
    projections = _lamina_projections(graph, cartridges)
    lamina_sources = {
        source
        for cartridge in cartridges
        for source in (
            cartridge.l1_index,
            cartridge.l2_index,
            cartridge.l3_index,
        )
        if source is not None
    }
    graded_relay_spec = config.get("graded_relay")
    graded_relay_indices: tuple[int, ...] = ()
    graded_relay_gain = 0.0
    graded_relay_activation_scale_mv = 7.0
    if graded_relay_spec is not None:
        maximum_path_hops = int(graded_relay_spec.get("maximum_path_hops", 3))
        allowed_superclasses = {
            str(value) for value in graded_relay_spec["superclasses"]
        }
        excluded_types = {
            str(value) for value in graded_relay_spec.get("exclude_types", [])
        }
        forward = _effective_forward_distances(
            graph,
            set(lamina_sources),
            max_hops=maximum_path_hops,
        )
        relay_set: set[int] = set()
        for population in motor_populations:
            for neuron_index in population.neuron_indices:
                layers = _shortest_effective_layers(
                    graph,
                    forward,
                    neuron_index,
                    max_hops=maximum_path_hops,
                )
                if layers is None:
                    continue
                for layer in layers[1:-1]:
                    for relay_index in layer:
                        superclass = graph.catalog.table["superclass"][
                            relay_index
                        ].as_py()
                        type_name = graph.catalog.table["type"][relay_index].as_py()
                        if (
                            superclass in allowed_superclasses
                            and str(type_name) not in excluded_types
                        ):
                            relay_set.add(relay_index)
        graded_relay_indices = tuple(sorted(relay_set))
        graded_relay_gain = float(graded_relay_spec["gain"])
        graded_relay_activation_scale_mv = float(
            graded_relay_spec.get("activation_scale_mv", 7.0)
        )

    motor_population_hops = (
        {
            population.key: [
                {
                    "neuron_index": neuron_index,
                    "body_id": body_id,
                    "instance": instance,
                    "shortest_lamina_hops": _shortest_upstream_hop(
                        graph,
                        neuron_index,
                        lamina_sources,
                    ),
                }
                for neuron_index, body_id, instance in zip(
                    population.neuron_indices,
                    population.body_ids,
                    population.instances,
                    strict=True,
                )
            ]
            for population in motor_populations
        }
        if bool(config.get("compute_hops", True))
        else None
    )

    timing = base["timing"]
    lif_config = ShiuLIFConfig(
        dt_ms=float(timing["neural_dt_ms"]),
        tonic_bias_mv=float(config.get("tonic_bias_mv", 0.0)),
    )
    runtime = base["runtime"]
    population_mapping = {
        population.key: population.neuron_indices
        for population in motor_populations
    }
    simulator = SparseLIFSimulator(
        graph,
        populations=population_mapping,
        clamped_indices=clamped,
        config=lif_config,
        backend=cast(RuntimeBackend, str(runtime["backend"])),
        subnormal_drive_policy=cast(
            SubnormalDrivePolicy,
            str(runtime["subnormal_drive_policy"]),
        ),
        graded_relay_indices=graded_relay_indices,
        graded_relay_gain=graded_relay_gain,
        graded_relay_activation_scale_mv=graded_relay_activation_scale_mv,
    )
    simulator.prepare()
    graded_config = _graded_config(base, lif_config)
    duration_s = (
        int(timing["baseline_frames"])
        + int(timing["stimulus_frames"])
        + int(timing["recovery_frames"])
    ) / float(timing["frame_rate_hz"])

    started = time.perf_counter()
    axes: dict[str, Any] = {}
    for axis_value in config["axes"]:
        axis = cast(Axis, str(axis_value))
        runs: list[dict[str, Any]] = []
        for position in positions:
            sample = _run_position(
                axis,
                position,
                graph,
                simulator,
                base,
                directions,
                screen,
                source_columns,
                source_groups,
                cartridges,
                projections,
                watch_indices,
                graded_config,
                lif_config,
                duration_s,
            )
            population_metrics = {
                population.key: _population_metrics(
                    sample,
                    population,
                    watch_position,
                )
                for population in motor_populations
            }
            runs.append(
                {
                    "position": position,
                    "populations": population_metrics,
                    "channels": _channel_state(population_metrics),
                    "total_network_spikes": int(sample["total_network_spikes"]),
                }
            )
        axes[axis] = {
            "runs": runs,
            "steering_voltage_position_correlation": _correlation(
                positions,
                [
                    float(
                        run["channels"][
                            "steering_voltage_right_minus_left_mv"
                        ]
                    )
                    for run in runs
                ],
            ),
            "vertical_voltage_position_correlation": _correlation(
                positions,
                [
                    float(
                        run["channels"][
                            "vertical_voltage_backward_minus_forward_mv"
                        ]
                    )
                    for run in runs
                ],
            ),
        }

    gates = _evaluate_gates(axes, config)
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Known descending motor-candidate populations are observed under the "
            "unchanged canonical spatial graded-vision frontend. This experiment "
            "tests signal availability only; it does not claim that untrained visual "
            "responses already implement a cursor policy."
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "base_config_sha256": file_sha256(base_config_path),
        "eye_map_lock": eye_map_lock,
        "dataset": {
            "directory": str(graph.directory),
            "neuron_count": graph.neuron_count,
            "edge_count": graph.edge_count,
            "sign_policy": graph.sign_policy.name,
        },
        "motor_populations": [asdict(population) for population in motor_populations],
        "motor_population_hops": motor_population_hops,
        "watch_count": len(watch_indices),
        "graded_relay": {
            "enabled": bool(graded_relay_indices),
            "neuron_count": len(graded_relay_indices),
            "gain": graded_relay_gain,
            "activation_scale_mv": graded_relay_activation_scale_mv,
            "excluded_types": sorted(excluded_types) if graded_relay_spec else [],
        },
        "axes": axes,
        "gates": gates,
        "total_wall_seconds": time.perf_counter() - started,
    }


def _evaluate_gates(
    axes: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    specification = config["gates"]
    threshold = float(specification["minimum_peak_abs_voltage_delta_mv"])
    checks: dict[str, bool] = {}
    for key in specification["required_voltage_populations"]:
        maximum = max(
            float(run["populations"][key]["max_abs_peak_voltage_delta_mv"])
            for axis in axes.values()
            for run in axis["runs"]
        )
        checks[f"{key}_voltage_reaches_threshold"] = maximum >= threshold
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

    result = run_characterization(arguments.data_directory, arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "gates_passed": result["gates"]["passed"],
                "gates": result["gates"]["checks"],
                "horizontal_steering_correlation": result["axes"]["horizontal"][
                    "steering_voltage_position_correlation"
                ],
                "vertical_forward_backward_correlation": result["axes"]["vertical"][
                    "vertical_voltage_position_correlation"
                ],
                "total_wall_seconds": result["total_wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
