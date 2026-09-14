"""Compact induced subgraphs for reference/runtime equivalence tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from flybrain_interface.connectome_data.runtime import MemoryMappedConnectome
from flybrain_interface.connectome_data.sparse import SparseConnectivity

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
IndexArray = npt.NDArray[np.int32]
CountArray = npt.NDArray[np.int32]
SignArray = npt.NDArray[np.int8]


@dataclass(frozen=True, slots=True)
class InducedConnectome:
    """In-memory source-major subgraph with local and released index identity."""

    original_indices: IntArray
    outgoing_indptr: IntArray
    target_indices: IndexArray
    synapse_counts: CountArray
    presynaptic_signs: SignArray

    @property
    def neuron_count(self) -> int:
        return int(self.original_indices.size)

    @property
    def edge_count(self) -> int:
        return int(self.target_indices.size)

    def accumulate_spikes(
        self,
        spiking_indices: npt.ArrayLike,
        destination: FloatArray,
        *,
        amplitudes: npt.ArrayLike | None = None,
    ) -> int:
        spikes = np.asarray(spiking_indices, dtype=np.int64)
        if amplitudes is None:
            event_amplitudes = np.ones(spikes.size, dtype=np.float64)
        else:
            event_amplitudes = np.asarray(amplitudes, dtype=np.float64)
            if event_amplitudes.ndim == 0:
                event_amplitudes = np.broadcast_to(event_amplitudes, spikes.shape)
        visited = 0
        for source, amplitude in zip(spikes, event_amplitudes, strict=True):
            sign = int(self.presynaptic_signs[source])
            if sign == 0 or amplitude == 0.0:
                continue
            start = int(self.outgoing_indptr[source])
            stop = int(self.outgoing_indptr[source + 1])
            destination[self.target_indices[start:stop]] += (
                self.synapse_counts[start:stop] * sign * amplitude
            )
            visited += stop - start
        return visited

    def as_sparse_connectivity(self) -> SparseConnectivity:
        sources = np.repeat(
            np.arange(self.neuron_count, dtype=np.int64),
            np.diff(self.outgoing_indptr),
        )
        signed_counts = (
            self.synapse_counts.astype(np.float64) * self.presynaptic_signs[sources]
        )
        return SparseConnectivity.from_edges(
            self.neuron_count,
            sources,
            self.target_indices,
            signed_counts,
        )


def induce_connectome(
    graph: MemoryMappedConnectome, original_indices: npt.ArrayLike
) -> InducedConnectome:
    """Extract the edges whose source and target are both in ``original_indices``."""

    selected = np.asarray(original_indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("original_indices must be a non-empty vector")
    if selected.min() < 0 or selected.max() >= graph.neuron_count:
        raise IndexError("original neuron index outside graph")
    if np.unique(selected).size != selected.size:
        raise ValueError("original_indices must be unique")

    original_to_local = np.full(graph.neuron_count, -1, dtype=np.int32)
    original_to_local[selected] = np.arange(selected.size, dtype=np.int32)
    target_parts: list[IndexArray] = []
    count_parts: list[CountArray] = []
    indptr = np.zeros(selected.size + 1, dtype=np.int64)

    for local_source, original_source in enumerate(selected):
        outgoing = graph.outgoing(int(original_source))
        local_targets = original_to_local[outgoing.target_indices]
        retained = local_targets >= 0
        target_parts.append(local_targets[retained].astype(np.int32, copy=False))
        count_parts.append(
            outgoing.synapse_counts[retained].astype(np.int32, copy=False)
        )
        indptr[local_source + 1] = indptr[local_source] + int(
            np.count_nonzero(retained)
        )

    targets = (
        np.concatenate(target_parts).astype(np.int32, copy=False)
        if target_parts
        else np.empty(0, dtype=np.int32)
    )
    counts = (
        np.concatenate(count_parts).astype(np.int32, copy=False)
        if count_parts
        else np.empty(0, dtype=np.int32)
    )
    signs = graph.presynaptic_signs[selected].astype(np.int8, copy=True)
    return InducedConnectome(
        original_indices=selected.copy(),
        outgoing_indptr=indptr,
        target_indices=targets,
        synapse_counts=counts,
        presynaptic_signs=signs,
    )


def strongest_outgoing_neighborhood(
    graph: MemoryMappedConnectome, *, neuron_count: int
) -> InducedConnectome:
    """Select a deterministic real subgraph around a strong excitatory edge."""

    if neuron_count < 2 or neuron_count > graph.neuron_count:
        raise ValueError("neuron_count must be between 2 and the full graph size")
    candidate_count = min(4096, graph.edge_count)
    candidate_positions = np.argpartition(
        graph.outgoing_synapse_counts, graph.edge_count - candidate_count
    )[graph.edge_count - candidate_count :]
    candidate_sources = (
        np.searchsorted(graph.outgoing_indptr, candidate_positions, side="right") - 1
    )
    candidate_targets = graph.target_indices[candidate_positions]
    candidate_weights = graph.outgoing_synapse_counts[candidate_positions]
    order = np.lexsort(
        (
            candidate_positions,
            candidate_targets,
            candidate_sources,
            -candidate_weights.astype(np.int64),
        )
    )
    candidate_sources = candidate_sources[order]
    candidate_targets = candidate_targets[order]
    valid = (graph.presynaptic_signs[candidate_sources] == 1) & (
        candidate_sources != candidate_targets
    )
    if not np.any(valid):
        raise ValueError("could not find an excitatory, non-self seed edge")
    seed = int(candidate_sources[np.flatnonzero(valid)[0]])
    outgoing = graph.outgoing(seed)
    order = np.lexsort((outgoing.target_indices, -outgoing.synapse_counts))
    neighbors = outgoing.target_indices[order]
    neighbors = neighbors[neighbors != seed]
    selected = np.concatenate(
        (np.asarray([seed], dtype=np.int32), neighbors[: neuron_count - 1])
    )
    return induce_connectome(graph, selected)
