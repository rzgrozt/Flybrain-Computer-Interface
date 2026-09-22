"""Resolve artifact-selected routes for bounded live neural observation.

Anatomical contacts authorize which indices may be observed, but never indicate
whether an edge transmitted activity during a particular simulation chunk.
"""

from __future__ import annotations

from typing import Any

from flybrain_interface.panel.models import PathwayObservationConfig
from flybrain_interface.panel.pathways_v2 import target_paths


def resolve_observation(config: PathwayObservationConfig) -> dict[str, Any]:
    """Return <=4 verified indices for a two- or three-hop visual-to-DN route."""
    target = target_paths(config.target_neuron_index)
    if target is None or config.path_index >= len(target["paths"]):
        raise ValueError("unknown anatomical pathway selection")
    path = target["paths"][config.path_index]
    neurons = path["neurons"]
    if len(neurons) not in (3, 4) or len(path["synapse_counts"]) != len(neurons) - 1:
        raise ValueError("pathway is not a verified two- or three-hop route")
    indices = [int(neuron["neuron_index"]) for neuron in neurons]
    if len(set(indices)) != len(indices) or indices[-1] != config.target_neuron_index:
        raise ValueError("invalid verified route identifiers")
    return {
        "target_neuron_index": config.target_neuron_index,
        "path_index": config.path_index,
        "neuron_indices": indices,
        "body_ids": [str(neuron["body_id"]) for neuron in neurons],
        "hop_count": len(neurons) - 1,
        "source": "verified_connectome_anatomy",
    }
