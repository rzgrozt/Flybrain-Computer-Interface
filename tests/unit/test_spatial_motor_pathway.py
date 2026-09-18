from __future__ import annotations

import numpy as np

from flybrain_interface.connectome_data.runtime import (
    IncomingConnections,
    OutgoingConnections,
)
from flybrain_interface.experiments.spatial_motor_pathway import (
    _layer_response,
    shortest_effective_pathway,
    top_ranked_paths,
)


class TinyDirectedGraph:
    def __init__(self) -> None:
        self.neuron_count = 6
        self.presynaptic_signs = np.asarray([1, 1, 1, 1, 1, 0], dtype=np.int8)
        self._edges = {
            0: [(2, 5)],
            1: [(2, 3), (5, 100)],
            2: [(3, 4)],
            3: [(4, 2)],
            4: [],
            5: [(4, 100)],
        }

    def outgoing(self, source_index: int) -> OutgoingConnections:
        edges = self._edges[source_index]
        return OutgoingConnections(
            target_indices=np.asarray(
                [target for target, _ in edges],
                dtype=np.int32,
            ),
            synapse_counts=np.asarray(
                [count for _, count in edges],
                dtype=np.int32,
            ),
        )

    def incoming(self, target_index: int) -> IncomingConnections:
        sources: list[int] = []
        counts: list[int] = []
        for source, edges in self._edges.items():
            for target, count in edges:
                if target == target_index:
                    sources.append(source)
                    counts.append(count)
        return IncomingConnections(
            source_indices=np.asarray(sources, dtype=np.int32),
            synapse_counts=np.asarray(counts, dtype=np.int32),
        )


def test_shortest_pathway_excludes_modulatory_presynaptic_relay() -> None:
    graph = TinyDirectedGraph()

    pathway = shortest_effective_pathway(
        graph,
        {0, 1},
        4,
        max_hops=3,
    )

    assert pathway is not None
    assert pathway.distance == 3
    assert pathway.layers == ((0, 1), (2,), (3,), (4,))
    assert [[edge.synapse_count for edge in layer] for layer in pathway.edges] == [
        [5, 3],
        [4],
        [2],
    ]


def test_ranked_paths_prioritize_stronger_contact_chain() -> None:
    graph = TinyDirectedGraph()
    pathway = shortest_effective_pathway(graph, {0, 1}, 4, max_hops=3)

    assert pathway is not None
    ranked = top_ranked_paths(pathway, limit=2)

    assert ranked[0].neuron_indices == (0, 2, 3, 4)
    assert ranked[0].synapse_counts == (5, 4, 2)
    assert ranked[1].neuron_indices == (1, 2, 3, 4)


def test_layer_response_distinguishes_subthreshold_drive_from_spikes() -> None:
    samples = [
        {
            "voltage": np.asarray([0.4, -0.2]),
            "drive": np.asarray([0.5, -0.1]),
            "spike": np.asarray([0.0, 0.0]),
        },
        {
            "voltage": np.asarray([0.8, -0.3]),
            "drive": np.asarray([0.9, -0.2]),
            "spike": np.asarray([0.0, 0.0]),
        },
    ]

    result = _layer_response(
        (10, 11),
        samples,
        {10: 0, 11: 1},
        threshold_mv=0.25,
    )

    assert result["responsive_voltage_neuron_count"] == 2
    assert result["responsive_drive_neuron_count"] == 1
    assert result["nonzero_drive_neuron_count"] == 2
    assert result["spiking_neuron_count"] == 0
    assert result["max_abs_voltage_delta_mv"] == 0.8
