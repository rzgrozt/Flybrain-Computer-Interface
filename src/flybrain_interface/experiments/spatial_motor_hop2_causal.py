"""Causal hop-2 relay subset test for steering pathway geometry."""

from __future__ import annotations

import argparse
import json
import time
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
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-hop2-causal-v1.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported hop2 causal schema")
    return payload


def _typed_hop2_indices(
    graph: Any,
    pathways: dict[int, PathwayLayers],
    target_records: dict[int, dict[str, Any]],
    target_populations: set[str],
    type_names: set[str],
) -> tuple[int, ...]:
    table = graph.catalog.table
    return tuple(
        sorted(
            {
                neuron
                for target, pathway in pathways.items()
                if target_records[target]["motor_population"] in target_populations
                and pathway.distance >= 2
                for neuron in pathway.layers[2]
                if str(table["type"][neuron].as_py()) in type_names
            }
        )
    )


def _variant_indices(
    baseline: set[int],
    *,
    remove_indices: set[int],
    add_indices: set[int],
    variant: str,
) -> tuple[int, ...]:
    result = set(baseline)
    if variant in {"remove_disruptive_visual_projection", "combined"}:
        result -= remove_indices
    if variant in {"add_preserving_cb_intrinsic", "combined"}:
        result |= add_indices
    return tuple(sorted(result))


def _variant_summary(targets: list[dict[str, Any]]) -> dict[str, Any]:
    hop2_pass = sum(
        1 for target in targets if target["hop2"]["passes_generalization"]
    )
    target_pass = sum(
        1 for target in targets if target["target"]["passes_generalization"]
    )
    hop2_mae = sum(float(target["hop2"]["best_test_mae"]) for target in targets)
    target_mae = sum(float(target["target"]["best_test_mae"]) for target in targets)
    count = len(targets)
    return {
        "target_count": count,
        "hop2_pass_count": hop2_pass,
        "target_pass_count": target_pass,
        "mean_hop2_best_test_mae": hop2_mae / count,
        "mean_target_best_test_mae": target_mae / count,
    }


def run_causal_test(
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

    setup = _setup_visual_system(data_directory, base_config_path)
    motor_populations = _motor_targets(setup.graph, motor_config_path)
    pathway_records, pathways = _pathway_records(
        setup.graph,
        motor_populations,
        setup.lamina_sources,
        max_hops=int(pathway_config["maximum_path_hops"]),
        top_paths=int(pathway_config["top_paths_per_target"]),
    )
    target_records = {
        int(record["target_neuron_index"]): record
        for record in pathway_records
        if bool(record["reachable"])
    }
    focus = {
        target: pathway
        for target, pathway in pathways.items()
        if target_records[target]["motor_population"] in target_populations
        and pathway.distance >= 2
    }
    if not focus:
        raise RuntimeError("no steering pathways resolved for causal test")

    relay_spec = geometry_config["graded_relay"]
    baseline_indices = set(
        _graded_relays(
            setup,
            pathways,
            {str(value) for value in relay_spec["superclasses"]},
        )
    )
    remove_indices = set(
        _typed_hop2_indices(
            setup.graph,
            pathways,
            target_records,
            target_populations,
            {str(value) for value in config["remove_hop2_types"]},
        )
    )
    add_indices = set(
        _typed_hop2_indices(
            setup.graph,
            pathways,
            target_records,
            target_populations,
            {str(value) for value in config["add_hop2_types"]},
        )
    )

    watch_indices = tuple(
        sorted(
            {
                neuron
                for pathway in focus.values()
                for neuron in (*pathway.layers[2], pathway.target_index)
            }
        )
    )
    watch_position = {
        neuron: position for position, neuron in enumerate(watch_indices)
    }
    runtime = setup.base["runtime"]

    started = time.perf_counter()
    variants: list[dict[str, Any]] = []
    for variant in config["variants"]:
        graded_indices = _variant_indices(
            baseline_indices,
            remove_indices=remove_indices,
            add_indices=add_indices,
            variant=str(variant),
        )
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

        axes: dict[str, Any] = {}
        for axis in config["axes"]:
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
                    watch_indices,
                    setup.graded_config,
                    setup.lif_config,
                    setup.duration_s,
                )
                for position in all_positions
            ]
            target_results: list[dict[str, Any]] = []
            for target_index, pathway in focus.items():
                hop2 = _layer_decodability(
                    pathway.layers[2],
                    samples,
                    watch_position,
                    train_positions,
                    test_positions,
                    geometry_config,
                )
                target = _layer_decodability(
                    (target_index,),
                    samples,
                    watch_position,
                    train_positions,
                    test_positions,
                    geometry_config,
                )
                record = target_records[target_index]
                target_results.append(
                    {
                        "motor_population": record["motor_population"],
                        "target_instance": record["target_instance"],
                        "hop2": hop2,
                        "target": target,
                    }
                )
            axes[str(axis)] = {
                "targets": target_results,
                "summary": _variant_summary(target_results),
                "network_spike_counts": [
                    int(sample["total_network_spikes"]) for sample in samples
                ],
            }

        variants.append(
            {
                "variant": str(variant),
                "graded_relay_count": len(graded_indices),
                "removed_hop2_count": (
                    len(baseline_indices & remove_indices)
                    if variant
                    in {"remove_disruptive_visual_projection", "combined"}
                    else 0
                ),
                "added_hop2_count": (
                    len(add_indices - baseline_indices)
                    if variant in {"add_preserving_cb_intrinsic", "combined"}
                    else 0
                ),
                "axes": axes,
            }
        )

    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "The baseline graded relay model is compared against exact hop-2 type "
            "interventions selected from the prior attribution experiment. The visual "
            "frontend, connectome, gain and all non-selected relays remain unchanged."
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "hop_geometry_config_sha256": file_sha256(geometry_config_path),
        "baseline_graded_relay_count": len(baseline_indices),
        "candidate_remove_count": len(remove_indices),
        "candidate_add_count": len(add_indices),
        "variants": variants,
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

    result = run_causal_test(args.data_directory, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "variants": [
                    {
                        "variant": row["variant"],
                        "axes": {
                            axis: value["summary"]
                            for axis, value in row["axes"].items()
                        },
                    }
                    for row in result["variants"]
                ],
                "total_wall_seconds": result["total_wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
