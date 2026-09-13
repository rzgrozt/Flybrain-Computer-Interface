"""Sparse, CPU-first connectivity representation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SparseConnectivity:
    """Signed synapse counts stored as target-by-source CSR connectivity.

    ``matrix[target, source]`` is the signed number of synapses. This orientation
    makes future vector propagation a direct ``matrix @ presynaptic_activity``.
    """

    neuron_count: int
    matrix: csr_matrix

    def __post_init__(self) -> None:
        if self.neuron_count <= 0:
            raise ValueError("neuron_count must be positive")
        if self.matrix.shape != (self.neuron_count, self.neuron_count):
            raise ValueError("connectivity matrix shape must match neuron_count")
        if not np.isfinite(self.matrix.data).all():
            raise ValueError("connectivity weights must be finite")

    @classmethod
    def from_edges(
        cls,
        neuron_count: int,
        sources: Iterable[int],
        targets: Iterable[int],
        signed_synapse_counts: Iterable[float],
    ) -> SparseConnectivity:
        source = np.asarray(tuple(sources), dtype=np.int64)
        target = np.asarray(tuple(targets), dtype=np.int64)
        weight = np.asarray(tuple(signed_synapse_counts), dtype=np.float64)

        if not (source.size == target.size == weight.size):
            raise ValueError("source, target, and weight arrays must have equal length")
        if source.size and (
            source.min() < 0
            or target.min() < 0
            or source.max() >= neuron_count
            or target.max() >= neuron_count
        ):
            raise ValueError("edge index outside neuron range")

        matrix = csr_matrix(
            (weight, (target, source)),
            shape=(neuron_count, neuron_count),
            dtype=np.float64,
        )
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return cls(neuron_count=neuron_count, matrix=matrix)

    @property
    def edge_count(self) -> int:
        return int(self.matrix.nnz)

    def edge_arrays(self) -> tuple[IntArray, IntArray, FloatArray]:
        """Return source, target, and signed-count arrays for Brian2."""

        target, source = self.matrix.nonzero()
        weights = np.asarray(self.matrix[target, source]).reshape(-1)
        return (
            source.astype(np.int64, copy=False),
            target.astype(np.int64, copy=False),
            weights.astype(np.float64, copy=False),
        )
