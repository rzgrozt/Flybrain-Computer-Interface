from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from flybrain_interface.experiments.spatial_motor_hop2_dn_edges import (
    _edge_group_summary,
    _paired_asymmetry,
)
from flybrain_interface.experiments.spatial_motor_pathway import PathEdge


@dataclass
class _Catalog:
    table: pa.Table


@dataclass
class _Graph:
    catalog: _Catalog


def test_edge_group_summary_tracks_signed_contacts_and_categories() -> None:
    graph = _Graph(
        _Catalog(
            pa.table(
                {
                    "type": ["PS077", "LLPC1", "Other"],
                }
            )
        )
    )
    result = _edge_group_summary(
        graph,
        (
            PathEdge(source=0, target=9, synapse_count=5, sign=1),
            PathEdge(source=1, target=9, synapse_count=3, sign=-1),
            PathEdge(source=2, target=9, synapse_count=2, sign=1),
        ),
        {"PS077"},
        {"LLPC1"},
    )

    assert result["total_contacts"] == 10
    assert result["signed_contacts"] == 4
    assert result["categories"]["preserving"]["contacts"] == 5
    assert result["categories"]["preserving"]["signed_contacts"] == 5
    assert result["categories"]["unstable"]["contacts"] == 3
    assert result["categories"]["unstable"]["signed_contacts"] == -3


def test_paired_asymmetry_reports_right_minus_left_contacts() -> None:
    targets = [
        {
            "target_instance": "DNa02_L",
            "edge_summary": {
                "categories": {
                    "preserving": {"contacts": 3, "signed_contacts": 1},
                    "unstable": {"contacts": 4, "signed_contacts": -2},
                    "other": {"contacts": 8, "signed_contacts": 6},
                }
            },
        },
        {
            "target_instance": "DNa02_R",
            "edge_summary": {
                "categories": {
                    "preserving": {"contacts": 7, "signed_contacts": 5},
                    "unstable": {"contacts": 2, "signed_contacts": 2},
                    "other": {"contacts": 9, "signed_contacts": 7},
                }
            },
        },
    ]

    result = _paired_asymmetry(targets)
    row = result["paired_targets"][0]

    assert row["family"] == "DNa02"
    assert row["categories"]["preserving"]["right_minus_left_contacts"] == 4
    assert row["categories"]["unstable"]["right_minus_left_signed_contacts"] == 4
