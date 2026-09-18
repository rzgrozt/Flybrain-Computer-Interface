from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyarrow as pa

from flybrain_interface.connectome_data.optic_columns import OpticColumnRecord
from flybrain_interface.experiments.retinotopy import (
    infer_r1r6_columns,
    summarize_mapping,
    summarize_official_source_consistency,
)


def test_r1r6_column_inference_preserves_confidence_classes() -> None:
    graph = _graph()
    columns = (
        OpticColumnRecord(
            column="ME_R_col_10_06",
            medulla_side="R",
            grid_row=10,
            grid_column=6,
            l1_body_id=200,
            r7_body_id=None,
            r8_body_id=None,
            column_type="pale",
        ),
        OpticColumnRecord(
            column="ME_R_col_10_07",
            medulla_side="R",
            grid_row=10,
            grid_column=7,
            l1_body_id=201,
            r7_body_id=None,
            r8_body_id=None,
            column_type="yellow1",
        ),
    )
    config = _config()

    assignments = infer_r1r6_columns(graph, columns, config)  # type: ignore[arg-type]
    by_body = {assignment.body_id: assignment for assignment in assignments}

    assert by_body[100].status == "unique_high"
    assert by_body[100].assigned_column == "ME_R_col_10_06"
    assert by_body[101].status == "unique_low"
    assert by_body[101].assigned_column == "ME_R_col_10_06"

    assert by_body[102].status == "dominant"
    assert by_body[102].best_fraction == 9 / 11
    assert by_body[102].margin_fraction == 7 / 11
    assert by_body[102].assigned_column == "ME_R_col_10_06"

    assert by_body[103].status == "ambiguous"
    assert by_body[103].best_column == "ME_R_col_10_06"
    assert by_body[103].assigned_column is None

    assert by_body[104].status == "unmapped"
    assert by_body[104].unmapped_reason == "direct_l1_not_in_official_table"
    assert by_body[105].status == "unmapped"
    assert by_body[105].unmapped_reason == "no_direct_l1"

    summary = summarize_mapping(assignments, columns, config)
    assert summary["assigned_count"] == 3
    assert summary["side_mismatch_count"] == 0
    assert summary["status_counts"] == {
        "ambiguous": 1,
        "dominant": 1,
        "unique_high": 1,
        "unique_low": 1,
        "unmapped": 2,
    }
    assert not summary["spatial_screen_encoding_enabled"]
    assert summary["column_membership_inference_available"]


def _config() -> dict[str, object]:
    return {
        "visual_input_query": {
            "superclass": "ol_sensory",
            "class": "visual",
            "type": "R1-R6",
        },
        "confidence": {
            "unique_min_contacts": 5,
            "dominant_min_best_fraction": 0.80,
            "dominant_min_margin_fraction": 0.60,
        },
        "validation": {
            "expected_r1_r6_per_complete_column": 6,
            "require_side_consistency": True,
            "use_soma_coordinates": False,
            "screen_orientation_resolved": False,
        },
    }


def _graph() -> SimpleNamespace:
    table = pa.table(
        {
            "neuron_index": list(range(10)),
            "body_id": [100, 101, 102, 103, 104, 105, 200, 201, 202, 300],
            "superclass": [
                "ol_sensory",
                "ol_sensory",
                "ol_sensory",
                "ol_sensory",
                "ol_sensory",
                "ol_sensory",
                "optic",
                "optic",
                "optic",
                "other",
            ],
            "class": [
                "visual",
                "visual",
                "visual",
                "visual",
                "visual",
                "visual",
                None,
                None,
                None,
                None,
            ],
            "type": [
                "R1-R6",
                "R1-R6",
                "R1-R6",
                "R1-R6",
                "R1-R6",
                "R1-R6",
                "L1",
                "L1",
                "L1",
                "other",
            ],
            "root_side": ["R"] * 10,
        }
    )

    # Sources 0..5:
    # 0 -> official L1 with 10 contacts: unique_high
    # 1 -> official L1 with 2 contacts: unique_low
    # 2 -> official L1s 9/2: dominant
    # 3 -> official L1s 6/5: ambiguous
    # 4 -> non-official L1 only
    # 5 -> non-L1 only
    edges = {
        0: [(6, 10)],
        1: [(6, 2)],
        2: [(6, 9), (7, 2)],
        3: [(6, 6), (7, 5)],
        4: [(8, 8)],
        5: [(9, 7)],
    }
    targets: list[int] = []
    weights: list[int] = []
    indptr = [0]
    for source in range(10):
        for target, weight in edges.get(source, []):
            targets.append(target)
            weights.append(weight)
        indptr.append(len(targets))

    return SimpleNamespace(
        catalog=SimpleNamespace(table=table),
        outgoing_indptr=np.asarray(indptr, dtype=np.int64),
        target_indices=np.asarray(targets, dtype=np.int32),
        outgoing_synapse_counts=np.asarray(weights, dtype=np.int32),
    )


def test_official_source_consistency_uses_l1_instance_and_r7_r8_root_side() -> None:
    table = pa.table(
        {
            "body_id": [200, 300, 400],
            "type": ["L1", "R7p", "R8y"],
            "root_side": [None, "R", "R"],
            "instance": ["L1_R", "R7p_R", "R8y_R"],
        }
    )
    graph = SimpleNamespace(catalog=SimpleNamespace(table=table))
    columns = (
        OpticColumnRecord(
            column="ME_R_col_10_06",
            medulla_side="R",
            grid_row=10,
            grid_column=6,
            l1_body_id=200,
            r7_body_id=300,
            r8_body_id=400,
            column_type="pale",
        ),
    )

    result = summarize_official_source_consistency(
        graph, columns  # type: ignore[arg-type]
    )

    assert result["L1"]["present_count"] == 1
    assert result["L1"]["side_mismatch_count"] == 0
    assert result["R7"]["side_mismatch_count"] == 0
    assert result["R8"]["side_mismatch_count"] == 0
