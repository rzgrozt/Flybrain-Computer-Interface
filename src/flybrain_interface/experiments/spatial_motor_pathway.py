"""Localize the spatial-vision propagation break along real motor pathways."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.connectome_data.optic_columns import load_optic_columns
from flybrain_interface.connectome_data.runtime import (
    IncomingConnections,
    MemoryMappedConnectome,
    OutgoingConnections,
)
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
from flybrain_interface.experiments.spatial_motor_readout import (
    MotorPopulation,
    resolve_motor_populations,
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
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-pathway-localization-v1.json"
Axis = Literal["horizontal", "vertical"]


class DirectedFastGraph(Protocol):
    neuron_count: int
    presynaptic_signs: np.ndarray

    def outgoing(self, source_index: int) -> OutgoingConnections: ...

    def incoming(self, target_index: int) -> IncomingConnections: ...


@dataclass(frozen=True, slots=True)
class PathEdge:
    source: int
    target: int
    synapse_count: int
    sign: int


@dataclass(frozen=True, slots=True)
class PathwayLayers:
    target_index: int
    distance: int
    layers: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[PathEdge, ...], ...]


@dataclass(frozen=True, slots=True)
class RankedPath:
    neuron_indices: tuple[int, ...]
    synapse_counts: tuple[int, ...]
    log_contact_score: float


@dataclass(frozen=True, slots=True)
class VisualSetup:
    graph: MemoryMappedConnectome
    base: dict[str, Any]
    directions: tuple[Any, ...]
    screen: VirtualScreenConfig
    source_columns: tuple[str, ...]
    source_groups: dict[str, tuple[int, ...]]
    cartridges: tuple[Any, ...]
    projections: dict[Any, Any]
    clamped: tuple[int, ...]
    lamina_sources: frozenset[int]
    lif_config: ShiuLIFConfig
    graded_config: Any
    duration_s: float
    eye_map_lock: dict[str, Any]


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported spatial motor-pathway config schema")
    return payload


def effective_forward_distances(
    graph: DirectedFastGraph,
    sources: set[int],
    *,
    max_hops: int,
) -> dict[int, int]:
    if max_hops <= 0:
        raise ValueError("max_hops must be positive")
    if not sources:
        raise ValueError("sources cannot be empty")
    distances = {int(source): 0 for source in sources}
    frontier = set(distances)
    for hop in range(1, max_hops + 1):
        next_frontier: set[int] = set()
        for source in frontier:
            if int(graph.presynaptic_signs[source]) == 0:
                continue
            outgoing = graph.outgoing(source)
            for target in outgoing.target_indices:
                target_index = int(target)
                if target_index not in distances:
                    distances[target_index] = hop
                    next_frontier.add(target_index)
        if not next_frontier:
            break
        frontier = next_frontier
    return distances


def effective_reverse_distances(
    graph: DirectedFastGraph,
    target_index: int,
    *,
    max_hops: int,
) -> dict[int, int]:
    if max_hops <= 0:
        raise ValueError("max_hops must be positive")
    distances = {target_index: 0}
    frontier = {target_index}
    for hop in range(1, max_hops + 1):
        next_frontier: set[int] = set()
        for target in frontier:
            incoming = graph.incoming(target)
            for source in incoming.source_indices:
                source_index = int(source)
                if int(graph.presynaptic_signs[source_index]) == 0:
                    continue
                if source_index not in distances:
                    distances[source_index] = hop
                    next_frontier.add(source_index)
        if not next_frontier:
            break
        frontier = next_frontier
    return distances


def shortest_effective_pathway(
    graph: DirectedFastGraph,
    sources: set[int],
    target_index: int,
    *,
    max_hops: int,
    forward_distances: dict[int, int] | None = None,
) -> PathwayLayers | None:
    forward = (
        effective_forward_distances(graph, sources, max_hops=max_hops)
        if forward_distances is None
        else forward_distances
    )
    distance = forward.get(target_index)
    if distance is None:
        return None
    reverse = effective_reverse_distances(
        graph,
        target_index,
        max_hops=distance,
    )
    layers: list[tuple[int, ...]] = []
    for hop in range(distance + 1):
        layer = tuple(
            sorted(
                neuron
                for neuron, source_distance in forward.items()
                if source_distance == hop
                and reverse.get(neuron) == distance - hop
            )
        )
        if not layer:
            raise RuntimeError(
                f"shortest-path reconstruction produced empty layer {hop}"
            )
        layers.append(layer)

    edges: list[tuple[PathEdge, ...]] = []
    for hop in range(distance):
        next_layer = set(layers[hop + 1])
        layer_edges: list[PathEdge] = []
        for source in layers[hop]:
            if int(graph.presynaptic_signs[source]) == 0:
                continue
            outgoing = graph.outgoing(source)
            for target, count in zip(
                outgoing.target_indices,
                outgoing.synapse_counts,
                strict=True,
            ):
                target_index_value = int(target)
                if target_index_value in next_layer:
                    layer_edges.append(
                        PathEdge(
                            source=source,
                            target=target_index_value,
                            synapse_count=int(count),
                            sign=int(graph.presynaptic_signs[source]),
                        )
                    )
        if not layer_edges:
            raise RuntimeError(
                f"shortest-path reconstruction produced no edges at hop {hop + 1}"
            )
        edges.append(
            tuple(
                sorted(
                    layer_edges,
                    key=lambda edge: (
                        edge.source,
                        edge.target,
                        -edge.synapse_count,
                    ),
                )
            )
        )
    return PathwayLayers(
        target_index=target_index,
        distance=distance,
        layers=tuple(layers),
        edges=tuple(edges),
    )


def top_ranked_paths(
    pathway: PathwayLayers,
    *,
    limit: int,
) -> tuple[RankedPath, ...]:
    if limit <= 0:
        raise ValueError("path limit must be positive")
    partial: dict[int, list[tuple[float, tuple[int, ...], tuple[int, ...]]]] = {
        source: [(0.0, (source,), ())] for source in pathway.layers[0]
    }
    for layer_edges in pathway.edges:
        by_target: dict[
            int,
            list[tuple[float, tuple[int, ...], tuple[int, ...]]],
        ] = {}
        for edge in layer_edges:
            for score, nodes, contacts in partial.get(edge.source, []):
                candidate = (
                    score + math.log1p(edge.synapse_count),
                    nodes + (edge.target,),
                    contacts + (edge.synapse_count,),
                )
                heap = by_target.setdefault(edge.target, [])
                heapq.heappush(heap, candidate)
                if len(heap) > limit:
                    heapq.heappop(heap)
        partial = {
            target: sorted(values, reverse=True)
            for target, values in by_target.items()
        }
    final = partial.get(pathway.target_index, [])
    ranked = sorted(final, reverse=True)[:limit]
    return tuple(
        RankedPath(
            neuron_indices=nodes,
            synapse_counts=contacts,
            log_contact_score=score,
        )
        for score, nodes, contacts in ranked
    )


def _annotation(graph: MemoryMappedConnectome, neuron_index: int) -> dict[str, Any]:
    table = graph.catalog.table
    return {
        "neuron_index": neuron_index,
        "body_id": int(table["body_id"][neuron_index].as_py()),
        "type": table["type"][neuron_index].as_py(),
        "instance": table["instance"][neuron_index].as_py(),
        "superclass": table["superclass"][neuron_index].as_py(),
        "class": table["class"][neuron_index].as_py(),
        "subclass": table["subclass"][neuron_index].as_py(),
        "soma_side": table["soma_side"][neuron_index].as_py(),
        "transmitter": graph.transmitters[neuron_index],
        "fast_sign": int(graph.presynaptic_signs[neuron_index]),
    }


def _setup_visual_system(
    data_directory: Path,
    base_config_path: Path,
) -> VisualSetup:
    base = load_base_config(base_config_path)
    graph = MemoryMappedConnectome.load(data_directory)
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
    lamina_sources = frozenset(
        source
        for cartridge in cartridges
        for source in (
            cartridge.l1_index,
            cartridge.l2_index,
            cartridge.l3_index,
        )
        if source is not None
    )
    projections = _lamina_projections(graph, cartridges)
    timing = base["timing"]
    lif_config = ShiuLIFConfig(dt_ms=float(timing["neural_dt_ms"]))
    graded_config = _graded_config(base, lif_config)
    duration_s = (
        int(timing["baseline_frames"])
        + int(timing["stimulus_frames"])
        + int(timing["recovery_frames"])
    ) / float(timing["frame_rate_hz"])
    return VisualSetup(
        graph=graph,
        base=base,
        directions=directions,
        screen=screen,
        source_columns=source_columns,
        source_groups=source_groups,
        cartridges=cartridges,
        projections=projections,
        clamped=clamped,
        lamina_sources=lamina_sources,
        lif_config=lif_config,
        graded_config=graded_config,
        duration_s=duration_s,
        eye_map_lock=eye_map_lock,
    )


def _motor_targets(
    graph: MemoryMappedConnectome,
    motor_config_path: Path,
) -> tuple[MotorPopulation, ...]:
    payload = json.loads(motor_config_path.read_text(encoding="utf-8"))
    return resolve_motor_populations(graph.catalog.table, payload)


def _pathway_records(
    graph: MemoryMappedConnectome,
    motor_populations: tuple[MotorPopulation, ...],
    lamina_sources: frozenset[int],
    *,
    max_hops: int,
    top_paths: int,
) -> tuple[list[dict[str, Any]], dict[int, PathwayLayers]]:
    records: list[dict[str, Any]] = []
    by_target: dict[int, PathwayLayers] = {}
    forward = effective_forward_distances(
        graph,
        set(lamina_sources),
        max_hops=max_hops,
    )
    for population in motor_populations:
        for neuron_index, body_id, instance in zip(
            population.neuron_indices,
            population.body_ids,
            population.instances,
            strict=True,
        ):
            pathway = shortest_effective_pathway(
                graph,
                set(lamina_sources),
                neuron_index,
                max_hops=max_hops,
                forward_distances=forward,
            )
            if pathway is None:
                records.append(
                    {
                        "motor_population": population.key,
                        "target_neuron_index": neuron_index,
                        "target_body_id": body_id,
                        "target_instance": instance,
                        "reachable": False,
                        "distance": None,
                        "layers": [],
                        "edges": [],
                        "top_paths": [],
                    }
                )
                continue
            by_target[neuron_index] = pathway
            ranked = top_ranked_paths(pathway, limit=top_paths)
            records.append(
                {
                    "motor_population": population.key,
                    "target_neuron_index": neuron_index,
                    "target_body_id": body_id,
                    "target_instance": instance,
                    "reachable": True,
                    "distance": pathway.distance,
                    "layers": [
                        {
                            "hop": hop,
                            "neuron_count": len(layer),
                            "neurons": [
                                _annotation(graph, neuron)
                                for neuron in layer
                            ],
                        }
                        for hop, layer in enumerate(pathway.layers)
                    ],
                    "edges": [
                        {
                            "hop": hop + 1,
                            "edge_count": len(layer_edges),
                            "total_contacts": sum(
                                edge.synapse_count for edge in layer_edges
                            ),
                            "edges": [
                                asdict(edge)
                                for edge in sorted(
                                    layer_edges,
                                    key=lambda edge: -edge.synapse_count,
                                )[:50]
                            ],
                        }
                        for hop, layer_edges in enumerate(pathway.edges)
                    ],
                    "top_paths": [
                        {
                            "log_contact_score": path.log_contact_score,
                            "synapse_counts": path.synapse_counts,
                            "neurons": [
                                _annotation(graph, neuron)
                                for neuron in path.neuron_indices
                            ],
                        }
                        for path in ranked
                    ],
                }
            )
    return records, by_target


def _watchlist(
    pathways: dict[int, PathwayLayers],
    lamina_sources: frozenset[int],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                neuron
                for pathway in pathways.values()
                for layer in pathway.layers[1:]
                for neuron in layer
                if neuron not in lamina_sources
            }
        )
    )


def _layer_response(
    layer: tuple[int, ...],
    samples: list[dict[str, Any]],
    watch_position: dict[int, int],
    *,
    threshold_mv: float,
) -> dict[str, Any]:
    positions = np.asarray(
        [watch_position[index] for index in layer],
        dtype=np.int64,
    )
    voltage_matrix = np.vstack(
        [
            np.asarray(sample["voltage"], dtype=np.float64)[positions]
            for sample in samples
        ]
    )
    drive_matrix = np.vstack(
        [
            np.asarray(sample["drive"], dtype=np.float64)[positions]
            for sample in samples
        ]
    )
    spike_matrix = np.vstack(
        [
            np.asarray(sample["spike"], dtype=np.float64)[positions]
            for sample in samples
        ]
    )
    peak_voltage_by_neuron = np.max(np.abs(voltage_matrix), axis=0)
    peak_drive_by_neuron = np.max(np.abs(drive_matrix), axis=0)
    spike_by_neuron = np.sum(spike_matrix, axis=0)
    return {
        "neuron_count": len(layer),
        "max_abs_voltage_delta_mv": float(np.max(peak_voltage_by_neuron)),
        "median_peak_abs_voltage_delta_mv": float(
            np.median(peak_voltage_by_neuron)
        ),
        "responsive_voltage_neuron_count": int(
            np.count_nonzero(peak_voltage_by_neuron >= threshold_mv)
        ),
        "nonzero_voltage_neuron_count": int(
            np.count_nonzero(peak_voltage_by_neuron > 1e-12)
        ),
        "max_abs_drive_mv": float(np.max(peak_drive_by_neuron)),
        "responsive_drive_neuron_count": int(
            np.count_nonzero(peak_drive_by_neuron >= threshold_mv)
        ),
        "nonzero_drive_neuron_count": int(
            np.count_nonzero(peak_drive_by_neuron > 1e-12)
        ),
        "spiking_neuron_count": int(np.count_nonzero(spike_by_neuron)),
        "total_spikes": int(np.sum(spike_by_neuron)),
    }


def _response_records(
    graph: MemoryMappedConnectome,
    anatomy_records: list[dict[str, Any]],
    pathways: dict[int, PathwayLayers],
    samples_by_axis: dict[str, list[dict[str, Any]]],
    watch_position: dict[int, int],
    *,
    threshold_mv: float,
    top_n: int,
) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    record_by_target = {
        int(record["target_neuron_index"]): record
        for record in anatomy_records
        if bool(record["reachable"])
    }
    for target_index, pathway in pathways.items():
        axis_records: dict[str, Any] = {}
        for axis, samples in samples_by_axis.items():
            layer_records = []
            for hop, layer in enumerate(pathway.layers[1:], start=1):
                metrics = _layer_response(
                    layer,
                    samples,
                    watch_position,
                    threshold_mv=threshold_mv,
                )
                layer_positions = np.asarray(
                    [watch_position[index] for index in layer],
                    dtype=np.int64,
                )
                peak_by_neuron = np.max(
                    np.vstack(
                        [
                            np.abs(
                                np.asarray(sample["voltage"], dtype=np.float64)[
                                    layer_positions
                                ]
                            )
                            for sample in samples
                        ]
                    ),
                    axis=0,
                )
                order = np.argsort(-peak_by_neuron, kind="stable")[:top_n]
                metrics["hop"] = hop
                metrics["top_voltage_neurons"] = [
                    {
                        **_annotation(graph, layer[int(position)]),
                        "peak_abs_voltage_delta_mv": float(
                            peak_by_neuron[int(position)]
                        ),
                    }
                    for position in order
                ]
                layer_records.append(metrics)
            first_zero_drive_hop = next(
                (
                    int(layer["hop"])
                    for layer in layer_records
                    if int(layer["nonzero_drive_neuron_count"]) == 0
                ),
                None,
            )
            last_nonzero_drive_hop = max(
                (
                    int(layer["hop"])
                    for layer in layer_records
                    if int(layer["nonzero_drive_neuron_count"]) > 0
                ),
                default=0,
            )
            first_spike_hop = next(
                (
                    int(layer["hop"])
                    for layer in layer_records
                    if int(layer["spiking_neuron_count"]) > 0
                ),
                None,
            )
            axis_records[axis] = {
                "layers": layer_records,
                "last_nonzero_drive_hop": last_nonzero_drive_hop,
                "first_zero_drive_hop": first_zero_drive_hop,
                "first_spike_hop": first_spike_hop,
            }
        source = record_by_target[target_index]
        responses.append(
            {
                "motor_population": source["motor_population"],
                "target_neuron_index": target_index,
                "target_body_id": source["target_body_id"],
                "target_instance": source["target_instance"],
                "distance": pathway.distance,
                "axes": axis_records,
            }
        )
    return responses


def _evaluate_gates(
    anatomy_records: list[dict[str, Any]],
    responses: list[dict[str, Any]] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    specification = config["gates"]
    checks: dict[str, bool] = {}
    if bool(specification["require_all_motor_targets_reachable"]):
        checks["all_motor_targets_reachable"] = all(
            bool(record["reachable"]) for record in anatomy_records
        )
    if (
        responses is not None
        and bool(specification["require_at_least_one_pathway_hop1_response"])
    ):
        threshold = float(config["response_threshold_mv"])
        checks["at_least_one_pathway_hop1_response"] = any(
            float(axis["layers"][0]["max_abs_voltage_delta_mv"]) >= threshold
            for response in responses
            for axis in response["axes"].values()
            if axis["layers"]
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": specification,
    }


def run_localization(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
    *,
    anatomy_only: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    base_config_path = ROOT / str(config["base_config"])
    motor_config_path = ROOT / str(config["motor_config"])
    setup = _setup_visual_system(data_directory, base_config_path)
    motor_populations = _motor_targets(setup.graph, motor_config_path)
    anatomy_records, pathways = _pathway_records(
        setup.graph,
        motor_populations,
        setup.lamina_sources,
        max_hops=int(config["maximum_path_hops"]),
        top_paths=int(config["top_paths_per_target"]),
    )
    watch_indices = _watchlist(pathways, setup.lamina_sources)
    anatomy_summary = {
        "lamina_source_count": len(setup.lamina_sources),
        "reachable_target_count": len(pathways),
        "motor_target_count": sum(
            len(population.neuron_indices)
            for population in motor_populations
        ),
        "pathway_watch_count": len(watch_indices),
        "hop_layer_union_counts": {
            str(hop): len(
                {
                    neuron
                    for pathway in pathways.values()
                    if pathway.distance >= hop
                    for neuron in pathway.layers[hop]
                }
            )
            for hop in range(
                1,
                int(config["maximum_path_hops"]) + 1,
            )
        },
    }
    if anatomy_only:
        gates = _evaluate_gates(anatomy_records, None, config)
        return {
            "schema_version": 1,
            "experiment": config["name"],
            "mode": "anatomy-only",
            "config": config,
            "config_sha256": file_sha256(config_path),
            "base_config_sha256": file_sha256(base_config_path),
            "motor_config_sha256": file_sha256(motor_config_path),
            "eye_map_lock": setup.eye_map_lock,
            "dataset": {
                "directory": str(setup.graph.directory),
                "neuron_count": setup.graph.neuron_count,
                "edge_count": setup.graph.edge_count,
                "sign_policy": setup.graph.sign_policy.name,
            },
            "anatomy_summary": anatomy_summary,
            "pathways": anatomy_records,
            "gates": gates,
        }

    if not watch_indices:
        raise RuntimeError("no motor-pathway neurons available for recording")
    maximum_watch_neurons = int(config["maximum_watch_neurons"])
    if maximum_watch_neurons <= 0:
        raise ValueError("maximum_watch_neurons must be positive")
    if len(watch_indices) > maximum_watch_neurons:
        raise RuntimeError(
            "motor pathway watchlist "
            f"{len(watch_indices)} exceeds {maximum_watch_neurons}-neuron bound"
        )

    populations = {
        population.key: population.neuron_indices
        for population in motor_populations
    }
    runtime = setup.base["runtime"]
    simulator = SparseLIFSimulator(
        setup.graph,
        populations=populations,
        clamped_indices=setup.clamped,
        config=setup.lif_config,
        backend=cast(RuntimeBackend, str(runtime["backend"])),
        subnormal_drive_policy=cast(
            SubnormalDrivePolicy,
            str(runtime["subnormal_drive_policy"]),
        ),
    )
    simulator.prepare()
    positions = tuple(float(value) for value in config["positions"])
    axes = tuple(cast(Axis, str(value)) for value in config["axes"])
    started = time.perf_counter()
    samples_by_axis: dict[str, list[dict[str, Any]]] = {}
    for axis in axes:
        samples_by_axis[axis] = [
            _run_position(
                axis,
                position,
                setup.graph,
                simulator,
                setup.base,
                setup.directions,
                setup.screen,
                setup.source_columns,
                setup.source_groups,
                setup.cartridges,
                setup.projections,
                watch_indices,
                setup.graded_config,
                setup.lif_config,
                setup.duration_s,
            )
            for position in positions
        ]
    watch_position = {
        neuron_index: position
        for position, neuron_index in enumerate(watch_indices)
    }
    responses = _response_records(
        setup.graph,
        anatomy_records,
        pathways,
        samples_by_axis,
        watch_position,
        threshold_mv=float(config["response_threshold_mv"]),
        top_n=int(config["top_neurons_per_layer"]),
    )
    gates = _evaluate_gates(anatomy_records, responses, config)
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "mode": "full",
        "interpretation": (
            "Shortest effective fast-chemical paths are reconstructed from the "
            "spatial L1/L2/L3 source set to anatomically selected motor DNs. "
            "Canonical visual stimulation is then recorded on every intermediate "
            "path layer to localize the first loss of synaptic drive, voltage and "
            "spiking without changing gain, threshold or tonic bias."
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "base_config_sha256": file_sha256(base_config_path),
        "motor_config_sha256": file_sha256(motor_config_path),
        "eye_map_lock": setup.eye_map_lock,
        "dataset": {
            "directory": str(setup.graph.directory),
            "neuron_count": setup.graph.neuron_count,
            "edge_count": setup.graph.edge_count,
            "sign_policy": setup.graph.sign_policy.name,
        },
        "anatomy_summary": anatomy_summary,
        "pathways": anatomy_records,
        "responses": responses,
        "network_spike_counts": {
            axis: [
                int(sample["total_network_spikes"])
                for sample in samples
            ]
            for axis, samples in samples_by_axis.items()
        },
        "gates": gates,
        "total_wall_seconds": time.perf_counter() - started,
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
    parser.add_argument("--anatomy-only", action="store_true")
    arguments = parser.parse_args()

    result = run_localization(
        arguments.data_directory,
        arguments.config,
        anatomy_only=arguments.anatomy_only,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)
    summary = {
        "output": str(arguments.output),
        "mode": result["mode"],
        "anatomy_summary": result["anatomy_summary"],
        "gates": result["gates"],
    }
    if result["mode"] == "full":
        summary["total_wall_seconds"] = result["total_wall_seconds"]
        summary["network_spike_counts"] = result["network_spike_counts"]
        summary["breaks"] = [
            {
                "target_instance": response["target_instance"],
                "distance": response["distance"],
                "horizontal_first_zero_drive_hop": response["axes"][
                    "horizontal"
                ]["first_zero_drive_hop"],
                "vertical_first_zero_drive_hop": response["axes"][
                    "vertical"
                ]["first_zero_drive_hop"],
            }
            for response in result["responses"]
        ]
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
