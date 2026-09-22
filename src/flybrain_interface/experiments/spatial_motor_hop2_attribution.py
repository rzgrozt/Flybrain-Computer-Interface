"""Attribute hop-2 spatial geometry to anatomical relay groups."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.experiments.spatial_motor_hop_geometry import (
    _graded_relays,
    _layer_decodability,
)
from flybrain_interface.experiments.spatial_motor_pathway import (
    PathwayLayers,
    _motor_targets,
    _pathway_records,
    _setup_visual_system,
)
from flybrain_interface.experiments.spatial_subthreshold_readout import _run_position
from flybrain_interface.simulation.runtime import (
    RuntimeBackend,
    SparseLIFSimulator,
    SubnormalDrivePolicy,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-hop2-attribution-v1.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported hop2 attribution schema")
    return payload


def _group_layer(
    graph: Any,
    layer: tuple[int, ...],
    field: str,
    *,
    minimum_size: int,
) -> dict[str, tuple[int, ...]]:
    if minimum_size <= 0:
        raise ValueError("minimum group size must be positive")
    table = graph.catalog.table
    grouped: dict[str, list[int]] = defaultdict(list)
    for neuron in layer:
        value = table[field][neuron].as_py()
        label = "<none>" if value is None else str(value)
        grouped[label].append(neuron)
    return {
        label: tuple(indices)
        for label, indices in sorted(grouped.items())
        if len(indices) >= minimum_size
    }


def _group_status(result: dict[str, Any]) -> str:
    if bool(result["passes_generalization"]):
        return "preserve"
    sides = [
        float(feature["test"]["side_accuracy"])
        for feature in result["features"].values()
    ]
    if sides and max(sides) == 0.0:
        return "wrong-side"
    return "mixed"


def _summarize_groups(
    axes: dict[str, Any],
    group_fields: tuple[str, ...],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for field in group_fields:
        accumulator: dict[str, dict[str, Any]] = {}
        for axis_name, axis in axes.items():
            for target in axis["targets"]:
                for group in target["groups"][field]:
                    label = str(group["label"])
                    row = accumulator.setdefault(
                        label,
                        {
                            "evaluations": 0,
                            "preserve": 0,
                            "wrong_side": 0,
                            "mixed": 0,
                            "best_mae_sum": 0.0,
                            "axes": set(),
                            "targets": set(),
                        },
                    )
                    row["evaluations"] += 1
                    status = str(group["status"])
                    if status == "preserve":
                        row["preserve"] += 1
                    elif status == "wrong-side":
                        row["wrong_side"] += 1
                    else:
                        row["mixed"] += 1
                    row["best_mae_sum"] += float(group["best_test_mae"])
                    row["axes"].add(axis_name)
                    row["targets"].add(target["target_instance"])
        summary[field] = [
            {
                "label": label,
                "evaluations": int(row["evaluations"]),
                "preserve": int(row["preserve"]),
                "preserve_fraction": (
                    float(row["preserve"]) / float(row["evaluations"])
                ),
                "wrong_side": int(row["wrong_side"]),
                "mixed": int(row["mixed"]),
                "mean_best_test_mae": (
                    float(row["best_mae_sum"]) / float(row["evaluations"])
                ),
                "axes": sorted(row["axes"]),
                "targets": sorted(row["targets"]),
            }
            for label, row in sorted(
                accumulator.items(),
                key=lambda item: (
                    -int(item[1]["preserve"]),
                    float(item[1]["best_mae_sum"])
                    / float(item[1]["evaluations"]),
                    item[0],
                ),
            )
        ]
    return summary


def run_attribution(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    geometry_config_path = ROOT / str(config["hop_geometry_config"])
    geometry_config = json.loads(
        geometry_config_path.read_text(encoding="utf-8")
    )
    pathway_config_path = ROOT / str(geometry_config["pathway_config"])
    pathway_config = json.loads(
        pathway_config_path.read_text(encoding="utf-8")
    )
    base_config_path = ROOT / str(pathway_config["base_config"])
    motor_config_path = ROOT / str(pathway_config["motor_config"])

    train_positions = tuple(
        float(value) for value in geometry_config["train_positions"]
    )
    test_positions = tuple(
        float(value) for value in geometry_config["test_positions"]
    )
    all_positions = (*train_positions, *test_positions)
    target_populations = {
        str(value) for value in config["target_populations"]
    }
    group_fields = tuple(str(value) for value in config["group_fields"])
    minimum_size = int(config["minimum_group_neurons"])

    setup = _setup_visual_system(data_directory, base_config_path)
    motor_populations = _motor_targets(setup.graph, motor_config_path)
    pathway_records, pathways = _pathway_records(
        setup.graph,
        motor_populations,
        setup.lamina_sources,
        max_hops=int(pathway_config["maximum_path_hops"]),
        top_paths=int(pathway_config["top_paths_per_target"]),
    )
    record_by_target = {
        int(record["target_neuron_index"]): record
        for record in pathway_records
        if bool(record["reachable"])
    }
    focus: dict[int, PathwayLayers] = {
        target: pathway
        for target, pathway in pathways.items()
        if str(record_by_target[target]["motor_population"])
        in target_populations
        and pathway.distance >= 2
    }
    if not focus:
        raise RuntimeError("no hop-2 steering pathways resolved")

    hop2_watch = tuple(
        sorted(
            {
                neuron
                for pathway in focus.values()
                for neuron in pathway.layers[2]
            }
        )
    )
    watch_position = {
        neuron: position for position, neuron in enumerate(hop2_watch)
    }

    relay_spec = geometry_config["graded_relay"]
    graded_indices = _graded_relays(
        setup,
        pathways,
        {str(value) for value in relay_spec["superclasses"]},
    )
    runtime = setup.base["runtime"]
    simulator = SparseLIFSimulator(
        setup.graph,
        clamped_indices=setup.clamped,
        config=setup.lif_config,
        backend=cast(RuntimeBackend, str(runtime["backend"])),
        subnormal_drive_policy=cast(
            SubnormalDrivePolicy,
            str(runtime["subnormal_drive_policy"]),
        ),
        graded_relay_indices=graded_indices,
        graded_relay_gain=float(relay_spec["gain"]),
        graded_relay_activation_scale_mv=float(
            relay_spec["activation_scale_mv"]
        ),
    )
    simulator.prepare()

    started = time.perf_counter()
    axes: dict[str, Any] = {}
    for axis in pathway_config["axes"]:
        samples = [
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
                hop2_watch,
                setup.graded_config,
                setup.lif_config,
                setup.duration_s,
            )
            for position in all_positions
        ]
        target_results: list[dict[str, Any]] = []
        for target_index, pathway in focus.items():
            record = record_by_target[target_index]
            hop2 = pathway.layers[2]
            whole = _layer_decodability(
                hop2,
                samples,
                watch_position,
                train_positions,
                test_positions,
                geometry_config,
            )
            grouped_results: dict[str, list[dict[str, Any]]] = {}
            for field in group_fields:
                grouped_results[field] = []
                for label, neurons in _group_layer(
                    setup.graph,
                    hop2,
                    field,
                    minimum_size=minimum_size,
                ).items():
                    decoded = _layer_decodability(
                        neurons,
                        samples,
                        watch_position,
                        train_positions,
                        test_positions,
                        geometry_config,
                    )
                    grouped_results[field].append(
                        {
                            "label": label,
                            "neuron_count": len(neurons),
                            "status": _group_status(decoded),
                            **decoded,
                        }
                    )
                grouped_results[field].sort(
                    key=lambda row: (
                        0 if row["status"] == "preserve" else 1,
                        float(row["best_test_mae"]),
                        -int(row["neuron_count"]),
                        str(row["label"]),
                    )
                )
            target_results.append(
                {
                    "motor_population": record["motor_population"],
                    "target_instance": record["target_instance"],
                    "target_neuron_index": target_index,
                    "hop2_neuron_count": len(hop2),
                    "whole_layer": whole,
                    "groups": grouped_results,
                }
            )
        axes[str(axis)] = {
            "samples": [
                {
                    "position": float(sample["position"]),
                    "total_network_spikes": int(sample["total_network_spikes"]),
                }
                for sample in samples
            ],
            "targets": target_results,
        }

    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Hop-2 shortest-path relay layers feeding DNa02/DNg13 are partitioned "
            "by anatomical annotations. Each group is independently tested for "
            "held-out spatial decodability under the unchanged gain=0.25 graded "
            "network. This is attribution only; no causal relay intervention is "
            "selected from the held-out results."
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "hop_geometry_config_sha256": file_sha256(geometry_config_path),
        "graded_relay_neuron_count": len(graded_indices),
        "hop2_watch_count": len(hop2_watch),
        "axes": axes,
        "group_summary": _summarize_groups(axes, group_fields),
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
    args = parser.parse_args()

    result = run_attribution(args.data_directory, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "hop2_watch_count": result["hop2_watch_count"],
                "group_summary": result["group_summary"],
                "total_wall_seconds": result["total_wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
