"""Analyze signed hop-2 to descending-neuron edge structure for steering targets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.experiments.spatial_motor_pathway import (
    PathEdge,
    _motor_targets,
    _pathway_records,
    _setup_visual_system,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-hop2-dn-edge-attribution-v1.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported hop2-dn edge attribution schema")
    return payload


def _edge_group_summary(
    graph: Any,
    edges: tuple[PathEdge, ...],
    preserving_types: set[str],
    unstable_types: set[str],
) -> dict[str, Any]:
    table = graph.catalog.table
    by_type: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "edge_count": 0.0,
            "contacts": 0.0,
            "signed_contacts": 0.0,
            "excitatory_contacts": 0.0,
            "inhibitory_contacts": 0.0,
        }
    )
    categories = {
        "preserving": {
            "edge_count": 0,
            "contacts": 0,
            "signed_contacts": 0,
            "excitatory_contacts": 0,
            "inhibitory_contacts": 0,
        },
        "unstable": {
            "edge_count": 0,
            "contacts": 0,
            "signed_contacts": 0,
            "excitatory_contacts": 0,
            "inhibitory_contacts": 0,
        },
        "other": {
            "edge_count": 0,
            "contacts": 0,
            "signed_contacts": 0,
            "excitatory_contacts": 0,
            "inhibitory_contacts": 0,
        },
    }
    total_contacts = 0
    total_signed_contacts = 0
    for edge in edges:
        source_type_value = table["type"][edge.source].as_py()
        source_type = "<none>" if source_type_value is None else str(source_type_value)
        contacts = int(edge.synapse_count)
        signed = int(edge.sign) * contacts
        total_contacts += contacts
        total_signed_contacts += signed
        row = by_type[source_type]
        row["edge_count"] += 1
        row["contacts"] += contacts
        row["signed_contacts"] += signed
        if edge.sign > 0:
            row["excitatory_contacts"] += contacts
        elif edge.sign < 0:
            row["inhibitory_contacts"] += contacts

        if source_type in preserving_types:
            category = "preserving"
        elif source_type in unstable_types:
            category = "unstable"
        else:
            category = "other"
        cat = categories[category]
        cat["edge_count"] += 1
        cat["contacts"] += contacts
        cat["signed_contacts"] += signed
        if edge.sign > 0:
            cat["excitatory_contacts"] += contacts
        elif edge.sign < 0:
            cat["inhibitory_contacts"] += contacts

    type_rows = [
        {
            "type": type_name,
            **{
                key: int(value) if key != "signed_contacts" else int(value)
                for key, value in values.items()
            },
        }
        for type_name, values in by_type.items()
    ]
    type_rows.sort(
        key=lambda row: (
            -int(row["contacts"]),
            row["type"],
        )
    )
    return {
        "edge_count": len(edges),
        "total_contacts": total_contacts,
        "signed_contacts": total_signed_contacts,
        "categories": categories,
        "types": type_rows,
    }


def _paired_asymmetry(targets: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for target in targets:
        instance = str(target["target_instance"])
        side = (
            "L"
            if instance.endswith("_L")
            else "R"
            if instance.endswith("_R")
            else None
        )
        if side is None:
            continue
        family = instance.rsplit("_", 1)[0]
        by_family[family][side] = target

    comparisons: list[dict[str, Any]] = []
    for family, sides in sorted(by_family.items()):
        if set(sides) != {"L", "R"}:
            continue
        left = sides["L"]["edge_summary"]["categories"]
        right = sides["R"]["edge_summary"]["categories"]
        category_rows = {}
        for category in ("preserving", "unstable", "other"):
            category_rows[category] = {
                "left_contacts": int(left[category]["contacts"]),
                "right_contacts": int(right[category]["contacts"]),
                "right_minus_left_contacts": int(
                    right[category]["contacts"] - left[category]["contacts"]
                ),
                "left_signed_contacts": int(left[category]["signed_contacts"]),
                "right_signed_contacts": int(right[category]["signed_contacts"]),
                "right_minus_left_signed_contacts": int(
                    right[category]["signed_contacts"]
                    - left[category]["signed_contacts"]
                ),
            }
        comparisons.append(
            {
                "family": family,
                "left_target": sides["L"]["target_instance"],
                "right_target": sides["R"]["target_instance"],
                "categories": category_rows,
            }
        )
    return {"paired_targets": comparisons}


def run_analysis(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    pathway_config_path = ROOT / str(config["pathway_config"])
    pathway_config = json.loads(pathway_config_path.read_text(encoding="utf-8"))
    base_config_path = ROOT / str(pathway_config["base_config"])
    motor_config_path = ROOT / str(pathway_config["motor_config"])

    setup = _setup_visual_system(data_directory, base_config_path)
    motor_populations = _motor_targets(setup.graph, motor_config_path)
    anatomy_records, pathways = _pathway_records(
        setup.graph,
        motor_populations,
        setup.lamina_sources,
        max_hops=int(pathway_config["maximum_path_hops"]),
        top_paths=int(pathway_config["top_paths_per_target"]),
    )
    records_by_target = {
        int(record["target_neuron_index"]): record
        for record in anatomy_records
        if bool(record["reachable"])
    }
    target_populations = {
        str(value) for value in config["target_populations"]
    }
    preserving_types = {str(value) for value in config["preserving_types"]}
    unstable_types = {str(value) for value in config["unstable_types"]}

    targets: list[dict[str, Any]] = []
    for target_index, pathway in pathways.items():
        record = records_by_target[target_index]
        if str(record["motor_population"]) not in target_populations:
            continue
        if pathway.distance < 2:
            continue
        final_edges = pathway.edges[pathway.distance - 1]
        targets.append(
            {
                "motor_population": record["motor_population"],
                "target_instance": record["target_instance"],
                "target_body_id": record["target_body_id"],
                "target_neuron_index": target_index,
                "distance": pathway.distance,
                "hop2_neuron_count": len(pathway.layers[-2]),
                "edge_summary": _edge_group_summary(
                    setup.graph,
                    final_edges,
                    preserving_types,
                    unstable_types,
                ),
            }
        )
    targets.sort(key=lambda row: str(row["target_instance"]))

    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "The final shortest-path edge layer from the penultimate relay layer into "
            "DNa02/DNg13 is summarized by signed contact count and anatomical source "
            "type. Preserving and unstable type labels come from the prior hop-2 "
            "attribution experiment; this analysis itself is purely anatomical."
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "pathway_config_sha256": file_sha256(pathway_config_path),
        "targets": targets,
        "asymmetry": _paired_asymmetry(targets),
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

    result = run_analysis(args.data_directory, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "asymmetry": result["asymmetry"],
                "targets": [
                    {
                        "target_instance": row["target_instance"],
                        "categories": row["edge_summary"]["categories"],
                    }
                    for row in result["targets"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
