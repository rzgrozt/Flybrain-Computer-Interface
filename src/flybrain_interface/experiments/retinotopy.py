"""Infer R1-R6 retinotopic columns from official MaleCNS L1 assignments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from flybrain_interface.connectome_data.manifest import file_sha256
from flybrain_interface.connectome_data.optic_columns import (
    OpticColumnRecord,
    load_optic_columns,
)
from flybrain_interface.connectome_data.runtime import MemoryMappedConnectome
from flybrain_interface.experiments.sensory_descending import resolve_population

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "retinotopy-v1.json"
DEFAULT_WORKBOOK = ROOT / "data" / "raw" / "optic-column-type-assignments-v1.0.xlsx"

AssignmentStatus = Literal[
    "unmapped",
    "unique_low",
    "unique_high",
    "dominant",
    "ambiguous",
]


@dataclass(frozen=True, slots=True)
class ColumnCandidate:
    column: str
    medulla_side: str
    grid_row: int
    grid_column: int
    l1_body_id: int
    contacts: int


@dataclass(frozen=True, slots=True)
class R1R6ColumnAssignment:
    neuron_index: int
    body_id: int
    root_side: str
    status: AssignmentStatus
    unmapped_reason: str | None
    candidate_count: int
    candidates: tuple[ColumnCandidate, ...]
    best_column: str | None
    assigned_column: str | None
    best_l1_body_id: int | None
    best_contacts: int
    second_contacts: int
    total_candidate_contacts: int
    best_fraction: float | None
    margin_fraction: float | None
    candidate_side_consistent: bool | None


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported retinotopy config schema")
    return payload


def verify_official_workbook(
    workbook_path: Path,
    source_manifest_path: Path,
) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported optic-column source-manifest schema")
    if not workbook_path.is_file():
        raise FileNotFoundError(workbook_path)
    size = workbook_path.stat().st_size
    expected_size = int(manifest["size_bytes"])
    if size != expected_size:
        raise ValueError(
            f"optic-column workbook size mismatch: {size} != {expected_size}"
        )
    digest = file_sha256(workbook_path)
    expected_digest = str(manifest["sha256"])
    if digest != expected_digest:
        raise ValueError("optic-column workbook SHA-256 mismatch")
    return {
        "filename": manifest["filename"],
        "source_url": manifest["source_url"],
        "size_bytes": size,
        "sha256": digest,
        "manifest_path": str(source_manifest_path),
        "manifest_sha256": file_sha256(source_manifest_path),
    }


def infer_r1r6_columns(
    graph: MemoryMappedConnectome,
    columns: tuple[OpticColumnRecord, ...],
    config: dict[str, Any],
) -> tuple[R1R6ColumnAssignment, ...]:
    visual = resolve_population(
        graph.catalog.table,
        {
            "key": "all_r1_r6",
            "label": "All reconstructed R1-R6 photoreceptors",
            "query": config["visual_input_query"],
            "sample_limit": None,
        },
    )
    table = graph.catalog.table
    body_ids = np.asarray(
        table["body_id"].to_numpy(zero_copy_only=False), dtype=np.int64
    )
    body_to_index = {int(body_id): index for index, body_id in enumerate(body_ids)}
    target_types = np.asarray(table["type"].to_pylist(), dtype=object)
    root_sides = table["root_side"].to_pylist()

    official_l1_by_index: dict[int, OpticColumnRecord] = {}
    for column in columns:
        if column.l1_body_id is None:
            continue
        index = body_to_index.get(column.l1_body_id)
        if index is not None:
            official_l1_by_index[index] = column

    thresholds = config["confidence"]
    unique_min_contacts = int(thresholds["unique_min_contacts"])
    dominant_fraction = float(thresholds["dominant_min_best_fraction"])
    dominant_margin = float(thresholds["dominant_min_margin_fraction"])
    if unique_min_contacts <= 0:
        raise ValueError("unique_min_contacts must be positive")
    if not 0.0 <= dominant_fraction <= 1.0:
        raise ValueError("dominant_min_best_fraction must be in [0, 1]")
    if not 0.0 <= dominant_margin <= 1.0:
        raise ValueError("dominant_min_margin_fraction must be in [0, 1]")

    assignments: list[R1R6ColumnAssignment] = []
    for source in visual.neuron_indices:
        start = int(graph.outgoing_indptr[source])
        stop = int(graph.outgoing_indptr[source + 1])
        targets = graph.target_indices[start:stop]
        weights = graph.outgoing_synapse_counts[start:stop]

        any_l1 = False
        contact_by_column: dict[str, tuple[OpticColumnRecord, int]] = {}
        for target, weight in zip(targets, weights, strict=True):
            target_index = int(target)
            if target_types[target_index] == "L1":
                any_l1 = True
            matched_column = official_l1_by_index.get(target_index)
            if matched_column is None:
                continue
            previous = contact_by_column.get(matched_column.column)
            contacts = int(weight) + (0 if previous is None else previous[1])
            contact_by_column[matched_column.column] = (matched_column, contacts)

        candidates = tuple(
            ColumnCandidate(
                column=candidate_column.column,
                medulla_side=candidate_column.medulla_side,
                grid_row=candidate_column.grid_row,
                grid_column=candidate_column.grid_column,
                l1_body_id=cast(int, candidate_column.l1_body_id),
                contacts=contacts,
            )
            for candidate_column, contacts in sorted(
                contact_by_column.values(),
                key=lambda item: (-item[1], item[0].column),
            )
        )
        root_side_value = root_sides[source]
        root_side = "" if root_side_value is None else str(root_side_value)
        assignment = _classify_assignment(
            neuron_index=source,
            body_id=int(body_ids[source]),
            root_side=root_side,
            candidates=candidates,
            any_l1=any_l1,
            unique_min_contacts=unique_min_contacts,
            dominant_fraction=dominant_fraction,
            dominant_margin=dominant_margin,
        )
        assignments.append(assignment)

    assignments.sort(key=lambda value: value.body_id)
    return tuple(assignments)


def summarize_mapping(
    assignments: tuple[R1R6ColumnAssignment, ...],
    columns: tuple[OpticColumnRecord, ...],
    config: dict[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(assignment.status for assignment in assignments)
    assigned = [
        assignment
        for assignment in assignments
        if assignment.assigned_column is not None
    ]
    occupancy = Counter(
        assignment.assigned_column
        for assignment in assigned
        if assignment.assigned_column is not None
    )
    occupancy_histogram = Counter(occupancy.values())
    candidate_histogram = Counter(
        assignment.candidate_count for assignment in assignments
    )
    expected = int(config["validation"]["expected_r1_r6_per_complete_column"])
    side_mismatch_count = sum(
        1
        for assignment in assignments
        if assignment.candidate_side_consistent is False
    )
    if config["validation"]["require_side_consistency"] and side_mismatch_count:
        raise RuntimeError(
            f"retinotopic mapping produced {side_mismatch_count} side mismatches"
        )

    official_by_side = Counter(column.medulla_side for column in columns)
    assigned_by_side = Counter(
        assignment.root_side for assignment in assigned if assignment.root_side
    )
    mapped_contacts = np.asarray(
        [
            assignment.best_contacts
            for assignment in assignments
            if assignment.best_contacts
        ],
        dtype=np.float64,
    )
    return {
        "r1_r6_count": len(assignments),
        "assigned_count": len(assigned),
        "assigned_fraction": len(assigned) / len(assignments),
        "status_counts": dict(sorted(status_counts.items())),
        "unmapped_reason_counts": dict(
            sorted(
                Counter(
                    assignment.unmapped_reason
                    for assignment in assignments
                    if assignment.unmapped_reason is not None
                ).items()
            )
        ),
        "candidate_count_histogram": {
            str(key): value for key, value in sorted(candidate_histogram.items())
        },
        "side_mismatch_count": side_mismatch_count,
        "official_column_count": len(columns),
        "official_columns_by_side": dict(sorted(official_by_side.items())),
        "assigned_r1_r6_by_side": dict(sorted(assigned_by_side.items())),
        "covered_column_count": len(occupancy),
        "occupancy_histogram": {
            str(key): value for key, value in sorted(occupancy_histogram.items())
        },
        "complete_six_input_column_count": occupancy_histogram.get(expected, 0),
        "columns_above_expected_occupancy": sum(
            value for occupancy_value, value in occupancy_histogram.items()
            if occupancy_value > expected
        ),
        "best_contact_summary": _numeric_summary(mapped_contacts),
        "high_confidence_assignment_count": (
            status_counts.get("unique_high", 0) + status_counts.get("dominant", 0)
        ),
        "low_confidence_unique_count": status_counts.get("unique_low", 0),
        "soma_coordinates_used": bool(config["validation"]["use_soma_coordinates"]),
        "screen_orientation_resolved": bool(
            config["validation"]["screen_orientation_resolved"]
        ),
        "spatial_screen_encoding_enabled": False,
        "column_membership_inference_available": True,
    }




def summarize_official_source_consistency(
    graph: MemoryMappedConnectome,
    columns: tuple[OpticColumnRecord, ...],
) -> dict[str, Any]:
    """Cross-check official L1/R7/R8 IDs against MaleCNS annotations."""

    table = graph.catalog.table
    body_ids = np.asarray(
        table["body_id"].to_numpy(zero_copy_only=False), dtype=np.int64
    )
    body_to_index = {int(body_id): index for index, body_id in enumerate(body_ids)}
    types = table["type"].to_pylist()
    root_sides = table["root_side"].to_pylist()
    instances = table["instance"].to_pylist()

    def summarize(field: str, expected_prefix: str) -> dict[str, int]:
        official = [
            (getattr(column, field), column.medulla_side)
            for column in columns
            if getattr(column, field) is not None
        ]
        missing = 0
        side_mismatch = 0
        type_mismatch = 0
        for body_id, side in official:
            assert body_id is not None
            index = body_to_index.get(body_id)
            if index is None:
                missing += 1
                continue
            type_name = "" if types[index] is None else str(types[index])
            if expected_prefix == "L1":
                if type_name != "L1":
                    type_mismatch += 1
                instance = "" if instances[index] is None else str(instances[index])
                if instance != f"L1_{side}":
                    side_mismatch += 1
            else:
                if not type_name.startswith(expected_prefix):
                    type_mismatch += 1
                root_side = "" if root_sides[index] is None else str(root_sides[index])
                if root_side != side:
                    side_mismatch += 1
        return {
            "official_id_count": len(official),
            "present_count": len(official) - missing,
            "missing_count": missing,
            "side_mismatch_count": side_mismatch,
            "type_mismatch_count": type_mismatch,
        }

    result = {
        "L1": summarize("l1_body_id", "L1"),
        "R7": summarize("r7_body_id", "R7"),
        "R8": summarize("r8_body_id", "R8"),
    }
    if any(
        metrics["missing_count"]
        or metrics["side_mismatch_count"]
        or metrics["type_mismatch_count"]
        for metrics in result.values()
    ):
        raise RuntimeError(
            "official optic-column IDs disagree with MaleCNS annotations"
        )
    return result


def run_mapping(
    data_directory: Path,
    workbook_path: Path = DEFAULT_WORKBOOK,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    source_manifest = ROOT / str(config["source_manifest"])
    source = verify_official_workbook(workbook_path, source_manifest)
    columns = load_optic_columns(workbook_path)
    graph = MemoryMappedConnectome.load(data_directory)
    source_consistency = summarize_official_source_consistency(graph, columns)
    assignments = infer_r1r6_columns(graph, columns, config)
    summary = summarize_mapping(assignments, columns, config)
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Connectivity-grounded R1-R6 to official L1 column inference. "
            "Column grid labels are structural coordinates only; visual-field and "
            "screen orientation remain unresolved."
        ),
        "config": config,
        "config_sha256": file_sha256(config_path),
        "source": source,
        "official_source_consistency": source_consistency,
        "dataset": {
            "directory": str(graph.directory),
            "neuron_count": graph.neuron_count,
            "edge_count": graph.edge_count,
            "sign_policy": graph.sign_policy.name,
        },
        "summary": summary,
        "assignments": [asdict(assignment) for assignment in assignments],
    }


def _classify_assignment(
    *,
    neuron_index: int,
    body_id: int,
    root_side: str,
    candidates: tuple[ColumnCandidate, ...],
    any_l1: bool,
    unique_min_contacts: int,
    dominant_fraction: float,
    dominant_margin: float,
) -> R1R6ColumnAssignment:
    if not candidates:
        reason = "direct_l1_not_in_official_table" if any_l1 else "no_direct_l1"
        return R1R6ColumnAssignment(
            neuron_index=neuron_index,
            body_id=body_id,
            root_side=root_side,
            status="unmapped",
            unmapped_reason=reason,
            candidate_count=0,
            candidates=(),
            best_column=None,
            assigned_column=None,
            best_l1_body_id=None,
            best_contacts=0,
            second_contacts=0,
            total_candidate_contacts=0,
            best_fraction=None,
            margin_fraction=None,
            candidate_side_consistent=None,
        )

    best = candidates[0]
    second_contacts = candidates[1].contacts if len(candidates) > 1 else 0
    total = sum(candidate.contacts for candidate in candidates)
    best_fraction = best.contacts / total
    margin_fraction = (best.contacts - second_contacts) / total
    side_consistent = all(
        candidate.medulla_side == root_side for candidate in candidates
    )

    status: AssignmentStatus
    assigned_column: str | None
    if len(candidates) == 1:
        status = "unique_high" if best.contacts >= unique_min_contacts else "unique_low"
        assigned_column = best.column
    elif (
        best_fraction >= dominant_fraction
        and margin_fraction >= dominant_margin
    ):
        status = "dominant"
        assigned_column = best.column
    else:
        status = "ambiguous"
        assigned_column = None

    return R1R6ColumnAssignment(
        neuron_index=neuron_index,
        body_id=body_id,
        root_side=root_side,
        status=status,
        unmapped_reason=None,
        candidate_count=len(candidates),
        candidates=candidates,
        best_column=best.column,
        assigned_column=assigned_column,
        best_l1_body_id=best.l1_body_id,
        best_contacts=best.contacts,
        second_contacts=second_contacts,
        total_candidate_contacts=total,
        best_fraction=best_fraction,
        margin_fraction=margin_fraction,
        candidate_side_consistent=side_consistent,
    )


def _numeric_summary(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("data/processed/malecns-v1.0"),
    )
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    result = run_mapping(
        arguments.data_directory,
        arguments.workbook,
        arguments.config,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                **result["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
