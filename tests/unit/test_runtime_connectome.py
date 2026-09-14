from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from flybrain_interface.connectome_data.runtime import (
    EXCITATORY_GLUTAMATE_POLICY,
    MemoryMappedConnectome,
)


def test_memory_mapped_graph_selects_and_propagates(tmp_path: Path) -> None:
    directory = _write_graph(tmp_path)
    graph = MemoryMappedConnectome.load(directory)

    assert isinstance(graph.indptr, np.memmap)
    assert isinstance(graph.source_indices, np.memmap)
    assert isinstance(graph.synapse_counts, np.memmap)
    assert isinstance(graph.target_indices, np.memmap)
    assert isinstance(graph.outgoing_synapse_counts, np.memmap)
    assert graph.edge_count == 4
    assert graph.sign_counts() == {
        "inhibitory": 2,
        "separated_or_unknown": 0,
        "excitatory": 1,
    }
    incoming = graph.incoming(2)
    assert incoming.source_indices.tolist() == [0, 1, 2]
    assert incoming.synapse_counts.tolist() == [2, 3, 1]
    assert np.shares_memory(incoming.source_indices, graph.source_indices)
    outgoing = graph.outgoing(0)
    assert outgoing.target_indices.tolist() == [1, 2]
    assert outgoing.synapse_counts.tolist() == [4, 2]
    assert np.shares_memory(outgoing.target_indices, graph.target_indices)

    # Sources 0 and 1 are active: +2 acetylcholine and -3 ground-truth GABA.
    result = graph.propagate([1.0, 1.0, 0.0])
    assert result.tolist() == [0.0, 4.0, -1.0]
    event_result = np.zeros(3, dtype=np.float64)
    visited = graph.accumulate_spikes([0, 1], event_result)
    assert visited == 3
    assert event_result.tolist() == result.tolist()
    scaled_result = np.zeros(3, dtype=np.float64)
    graph.accumulate_spikes([0], scaled_result, amplitudes=0.5)
    assert scaled_result.tolist() == [0.0, 2.0, 1.0]
    assert graph.catalog.select(superclass="sensory").tolist() == [0]
    assert graph.catalog.body_ids([2, 0]).tolist() == [30, 10]


def test_glutamate_assumption_is_swappable(tmp_path: Path) -> None:
    directory = _write_graph(tmp_path)
    inhibitory = MemoryMappedConnectome.load(directory)
    excitatory = MemoryMappedConnectome.load(
        directory, sign_policy=EXCITATORY_GLUTAMATE_POLICY
    )

    assert inhibitory.presynaptic_signs.tolist() == [1, -1, -1]
    assert excitatory.presynaptic_signs.tolist() == [1, -1, 1]


def test_runtime_graph_rejects_invalid_activity(tmp_path: Path) -> None:
    graph = MemoryMappedConnectome.load(_write_graph(tmp_path))

    with pytest.raises(ValueError, match="one value per neuron"):
        graph.propagate([1.0])
    with pytest.raises(IndexError, match="outside graph"):
        graph.incoming(3)
    with pytest.raises(IndexError, match="outside graph"):
        graph.outgoing(3)
    with pytest.raises(ValueError, match="unique"):
        graph.accumulate_spikes([0, 0], np.zeros(3, dtype=np.float64))
    with pytest.raises(TypeError, match="integers"):
        graph.accumulate_spikes([0.5], np.zeros(3, dtype=np.float64))
    with pytest.raises(KeyError, match="unknown neuron annotation"):
        graph.catalog.select(not_a_column="value")


def _write_graph(tmp_path: Path) -> Path:
    neurons = pa.table(
        {
            "neuron_index": [0, 1, 2],
            "body_id": [10, 20, 30],
            "superclass": ["sensory", "interneuron", "motor"],
            "consensus_nt": ["acetylcholine", "glutamate", "glutamate"],
            # Verified annotation overrides consensus for neuron 1.
            "ground_truth_nt": [None, "gaba", None],
        }
    )
    parquet.write_table(neurons, tmp_path / "neurons.parquet")
    np.save(tmp_path / "csr_indptr.npy", np.array([0, 0, 1, 4], dtype=np.int64))
    np.save(tmp_path / "csr_indices.npy", np.array([0, 0, 1, 2], dtype=np.int32))
    np.save(tmp_path / "csr_synapse_counts.npy", np.array([4, 2, 3, 1], dtype=np.int32))
    np.save(tmp_path / "outgoing_indptr.npy", np.array([0, 2, 3, 4], dtype=np.int64))
    np.save(
        tmp_path / "outgoing_target_indices.npy",
        np.array([1, 2, 2, 2], dtype=np.int32),
    )
    np.save(
        tmp_path / "outgoing_synapse_counts.npy",
        np.array([4, 2, 3, 1], dtype=np.int32),
    )
    return tmp_path
