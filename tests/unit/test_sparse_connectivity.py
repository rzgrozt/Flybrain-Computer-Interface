"""Tests for the target-by-source sparse representation."""

import numpy as np

from flybrain_interface.connectome_data.sparse import SparseConnectivity


def test_sparse_connectivity_has_propagation_orientation() -> None:
    connectivity = SparseConnectivity.from_edges(
        neuron_count=3,
        sources=(0, 0),
        targets=(1, 1),
        signed_synapse_counts=(2.0, 3.0),
    )

    activity = connectivity.matrix @ np.asarray((1.0, 0.0, 0.0))

    assert connectivity.edge_count == 1
    assert activity.tolist() == [0.0, 5.0, 0.0]


def test_sparse_connectivity_round_trips_edges() -> None:
    connectivity = SparseConnectivity.from_edges(
        neuron_count=3,
        sources=(0, 1),
        targets=(1, 2),
        signed_synapse_counts=(5.0, -2.0),
    )

    source, target, weight = connectivity.edge_arrays()

    assert source.tolist() == [0, 1]
    assert target.tolist() == [1, 2]
    assert weight.tolist() == [5.0, -2.0]
