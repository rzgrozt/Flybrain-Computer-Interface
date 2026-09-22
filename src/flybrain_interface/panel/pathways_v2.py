"""Bounded read-only presentation of verified anatomical sensory-to-DN paths.

No anatomy-only edge is treated as active, and contact counts are not
physiological transmission measurements.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PATHWAY_ARTIFACT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "spatial-motor-pathway-anatomy-v1.json"
)


@lru_cache(maxsize=1)
def _load_verified_artifact() -> dict[str, Any]:
    with PATHWAY_ARTIFACT.open(encoding="utf-8") as source:
        artifact: dict[str, Any] = json.load(source)
    if (
        artifact.get("mode") != "anatomy-only"
        or artifact.get("dataset", {}).get("neuron_count") != 166700
        or len(artifact.get("pathways", [])) > 64
    ):
        raise ValueError("unsupported anatomical pathway artifact")
    return artifact


def pathway_catalog() -> dict[str, Any]:
    """Tiny static catalog, safely independent of changing neural state."""
    if not PATHWAY_ARTIFACT.is_file():
        return {
            "schema_version": 2,
            "kind": "anatomical_pathway_catalog",
            "available": False,
            "targets": [],
        }
    artifact = _load_verified_artifact()
    return {
        "schema_version": 2,
        "kind": "anatomical_pathway_catalog",
        "source": "verified_connectome_anatomy",
        "available": True,
        "artifact_name": PATHWAY_ARTIFACT.name,
        "telemetry_available": False,
        "causal_contribution_available": False,
        "targets": [
            {
                "target_neuron_index": row["target_neuron_index"],
                "target_body_id": row["target_body_id"],
                "target_instance": row["target_instance"],
                "motor_population": row["motor_population"],
                "reachable": row["reachable"],
                "minimum_hops": row["distance"],
                "sampled_paths": min(12, len(row["top_paths"])),
            }
            for row in artifact["pathways"]
        ],
    }


def target_paths(target_index: int) -> dict[str, Any] | None:
    """Return at most 12 artifact-selected routes for one actual target."""
    artifact = _load_verified_artifact()
    target = next(
        (
            row
            for row in artifact["pathways"]
            if row["target_neuron_index"] == target_index
        ),
        None,
    )
    if target is None:
        return None
    return {
        "schema_version": 2,
        "kind": "anatomical_pathways",
        "source": "verified_connectome_anatomy",
        "target_neuron_index": target_index,
        "target_body_id": target["target_body_id"],
        "target_instance": target["target_instance"],
        "motor_population": target["motor_population"],
        "minimum_hops": target["distance"],
        "active_pathway_measurement": None,
        "causal_attribution": None,
        "paths": [
            {
                "path_index": index,
                "ranking_metric": "artifact_log_contact_score",
                "log_contact_score": entry["log_contact_score"],
                "neurons": [
                    {
                        "neuron_index": neuron["neuron_index"],
                        "body_id": neuron["body_id"],
                        "type": neuron.get("type"),
                        "instance": neuron.get("instance"),
                        "superclass": neuron.get("superclass"),
                        "transmitter": neuron.get("transmitter"),
                        "fast_sign": neuron.get("fast_sign"),
                        "measured_voltage_mv": None,
                        "measured_drive_mv": None,
                        "measured_spike_count": None,
                    }
                    for neuron in entry["neurons"]
                ],
                "synapse_counts": entry["synapse_counts"],
                "edge_activation": None,
            }
            for index, entry in enumerate(target["top_paths"][:12])
        ],
    }
