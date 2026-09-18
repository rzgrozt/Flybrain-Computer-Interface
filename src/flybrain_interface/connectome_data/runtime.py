"""Read-only, memory-mapped access to a normalized MaleCNS graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as parquet
from scipy.sparse import csr_matrix

IntArray = npt.NDArray[np.int64]
IndexArray = npt.NDArray[np.int32]
CountArray = npt.NDArray[np.int32]
SignArray = npt.NDArray[np.int8]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TransmitterSignPolicy:
    """Explicit neuron-transmitter to fast-synaptic-sign assumptions.

    A zero sign means that the transmitter is not represented as an ordinary fast
    synaptic current. It does not mean that the biological neuron has no effect.
    """

    name: str
    signs: Mapping[str, int]
    unknown_sign: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("policy name cannot be empty")
        if self.unknown_sign not in (-1, 0, 1):
            raise ValueError("unknown_sign must be -1, 0, or 1")
        if any(sign not in (-1, 0, 1) for sign in self.signs.values()):
            raise ValueError("transmitter signs must be -1, 0, or 1")
        normalized = {
            transmitter.casefold(): sign for transmitter, sign in self.signs.items()
        }
        object.__setattr__(self, "signs", MappingProxyType(normalized))

    def sign_for(self, transmitter: str | None) -> int:
        if transmitter is None:
            return self.unknown_sign
        return self.signs.get(transmitter.casefold(), self.unknown_sign)


INHIBITORY_GLUTAMATE_POLICY = TransmitterSignPolicy(
    name="inhibitory-glutamate-modulators-separated-v1",
    signs={
        "acetylcholine": 1,
        "gaba": -1,
        "glutamate": -1,
        "histamine": -1,
        "dopamine": 0,
        "serotonin": 0,
        "octopamine": 0,
        "unclear": 0,
    },
)

EXCITATORY_GLUTAMATE_POLICY = TransmitterSignPolicy(
    name="excitatory-glutamate-modulators-separated-v1",
    signs={**INHIBITORY_GLUTAMATE_POLICY.signs, "glutamate": 1},
)


@dataclass(frozen=True, slots=True)
class IncomingConnections:
    """Zero-copy views of the connections arriving at one target neuron."""

    source_indices: IndexArray
    synapse_counts: CountArray


@dataclass(frozen=True, slots=True)
class OutgoingConnections:
    """Zero-copy views of the connections leaving one source neuron."""

    target_indices: IndexArray
    synapse_counts: CountArray


@dataclass(frozen=True, slots=True)
class SignedSourceProjection:
    """Sparse aggregate of signed outgoing contacts from a source population."""

    source_count: int
    anatomical_edge_count: int
    target_indices: IntArray
    signed_contact_weights: FloatArray

    def __post_init__(self) -> None:
        if self.source_count < 0 or self.anatomical_edge_count < 0:
            raise ValueError("projection counts cannot be negative")
        if self.target_indices.ndim != 1 or self.signed_contact_weights.ndim != 1:
            raise ValueError("projection arrays must be one-dimensional")
        if self.target_indices.shape != self.signed_contact_weights.shape:
            raise ValueError("projection target and weight arrays must align")
        if np.unique(self.target_indices).size != self.target_indices.size:
            raise ValueError("projection targets must be unique")
        if not np.isfinite(self.signed_contact_weights).all():
            raise ValueError("projection weights must be finite")
        if np.any(self.signed_contact_weights == 0.0):
            raise ValueError("projection weights must exclude zero entries")


@dataclass(frozen=True, slots=True)
class NeuronCatalog:
    """Small in-memory annotation table aligned with graph neuron indices."""

    table: pa.Table

    @property
    def neuron_count(self) -> int:
        return int(self.table.num_rows)

    def select(self, **exact_values: str) -> IntArray:
        """Select neuron indices by exact, case-sensitive annotation values."""

        mask: pa.Array | pa.ChunkedArray = pa.array(
            np.ones(self.neuron_count, dtype=np.bool_)
        )
        for column_name, value in exact_values.items():
            if column_name not in self.table.column_names:
                raise KeyError(f"unknown neuron annotation column: {column_name}")
            mask = pc.and_kleene(mask, pc.equal(self.table[column_name], value))
        selected = self.table.filter(pc.fill_null(mask, False))["neuron_index"]
        return np.asarray(selected.to_numpy(zero_copy_only=False), dtype=np.int64)

    def body_ids(self, neuron_indices: Sequence[int]) -> IntArray:
        indices = np.asarray(neuron_indices, dtype=np.int64)
        if indices.size and (indices.min() < 0 or indices.max() >= self.neuron_count):
            raise IndexError("neuron index outside catalog")
        body_ids = self.table["body_id"].take(pa.array(indices))
        return np.asarray(body_ids.to_numpy(zero_copy_only=False), dtype=np.int64)


@dataclass(frozen=True, slots=True)
class MemoryMappedConnectome:
    """Full bidirectional sparse graph without copying edge arrays into RAM."""

    directory: Path
    indptr: npt.NDArray[np.int64]
    source_indices: IndexArray
    synapse_counts: CountArray
    outgoing_indptr: npt.NDArray[np.int64]
    target_indices: IndexArray
    outgoing_synapse_counts: CountArray
    presynaptic_signs: SignArray
    transmitters: tuple[str | None, ...]
    catalog: NeuronCatalog
    sign_policy: TransmitterSignPolicy

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        sign_policy: TransmitterSignPolicy = INHIBITORY_GLUTAMATE_POLICY,
    ) -> MemoryMappedConnectome:
        directory = directory.resolve()
        indptr = np.load(directory / "csr_indptr.npy", mmap_mode="r")
        indices = np.load(directory / "csr_indices.npy", mmap_mode="r")
        counts = np.load(directory / "csr_synapse_counts.npy", mmap_mode="r")
        outgoing_indptr = np.load(directory / "outgoing_indptr.npy", mmap_mode="r")
        targets = np.load(directory / "outgoing_target_indices.npy", mmap_mode="r")
        outgoing_counts = np.load(
            directory / "outgoing_synapse_counts.npy", mmap_mode="r"
        )
        neurons = parquet.read_table(directory / "neurons.parquet")
        neuron_count = neurons.num_rows
        edge_count = int(indices.size)

        if indptr.dtype != np.int64 or indptr.shape != (neuron_count + 1,):
            raise ValueError("invalid CSR indptr dtype or shape")
        if indices.dtype != np.int32 or indices.shape != (edge_count,):
            raise ValueError("invalid CSR source-index dtype or shape")
        if counts.dtype != np.int32 or counts.shape != (edge_count,):
            raise ValueError("invalid CSR synapse-count dtype or shape")
        if int(indptr[0]) != 0 or int(indptr[-1]) != edge_count:
            raise ValueError("invalid CSR bounds")
        if outgoing_indptr.dtype != np.int64 or outgoing_indptr.shape != (
            neuron_count + 1,
        ):
            raise ValueError("invalid outgoing indptr dtype or shape")
        if targets.dtype != np.int32 or targets.shape != (edge_count,):
            raise ValueError("invalid outgoing target-index dtype or shape")
        if outgoing_counts.dtype != np.int32 or outgoing_counts.shape != (edge_count,):
            raise ValueError("invalid outgoing synapse-count dtype or shape")
        if int(outgoing_indptr[0]) != 0 or int(outgoing_indptr[-1]) != edge_count:
            raise ValueError("invalid outgoing index bounds")

        transmitters = _effective_transmitters(neurons)
        signs = np.fromiter(
            (sign_policy.sign_for(value) for value in transmitters),
            dtype=np.int8,
            count=neuron_count,
        )
        signs.flags.writeable = False
        return cls(
            directory=directory,
            indptr=indptr,
            source_indices=indices,
            synapse_counts=counts,
            outgoing_indptr=outgoing_indptr,
            target_indices=targets,
            outgoing_synapse_counts=outgoing_counts,
            presynaptic_signs=signs,
            transmitters=transmitters,
            catalog=NeuronCatalog(neurons),
            sign_policy=sign_policy,
        )

    @property
    def neuron_count(self) -> int:
        return self.catalog.neuron_count

    @property
    def edge_count(self) -> int:
        return int(self.source_indices.size)

    @property
    def mapped_edge_bytes(self) -> int:
        return int(
            self.source_indices.nbytes
            + self.synapse_counts.nbytes
            + self.target_indices.nbytes
            + self.outgoing_synapse_counts.nbytes
        )

    def incoming(self, target_index: int) -> IncomingConnections:
        if target_index < 0 or target_index >= self.neuron_count:
            raise IndexError("target neuron index outside graph")
        start = int(self.indptr[target_index])
        stop = int(self.indptr[target_index + 1])
        return IncomingConnections(
            source_indices=self.source_indices[start:stop],
            synapse_counts=self.synapse_counts[start:stop],
        )

    def outgoing(self, source_index: int) -> OutgoingConnections:
        if source_index < 0 or source_index >= self.neuron_count:
            raise IndexError("source neuron index outside graph")
        start = int(self.outgoing_indptr[source_index])
        stop = int(self.outgoing_indptr[source_index + 1])
        return OutgoingConnections(
            target_indices=self.target_indices[start:stop],
            synapse_counts=self.outgoing_synapse_counts[start:stop],
        )

    def unsigned_csr(self) -> csr_matrix:
        """Expose a zero-copy SciPy view of the unsigned contact matrix."""

        matrix = csr_matrix(
            (self.synapse_counts, self.source_indices, self.indptr),
            shape=(self.neuron_count, self.neuron_count),
            copy=False,
        )
        if not np.shares_memory(matrix.data, self.synapse_counts):
            raise RuntimeError("SciPy copied the memory-mapped synapse counts")
        if not np.shares_memory(matrix.indices, self.source_indices):
            raise RuntimeError("SciPy copied the memory-mapped source indices")
        return matrix

    def propagate(self, presynaptic_activity: npt.ArrayLike) -> FloatArray:
        """Compute signed contact input while leaving edge weights unsigned."""

        activity = np.asarray(presynaptic_activity, dtype=np.float64)
        if activity.shape != (self.neuron_count,):
            raise ValueError("presynaptic activity must have one value per neuron")
        if not np.isfinite(activity).all():
            raise ValueError("presynaptic activity must be finite")
        signed_activity = activity * self.presynaptic_signs
        return np.asarray(self.unsigned_csr() @ signed_activity, dtype=np.float64)

    def accumulate_spikes(
        self,
        spiking_indices: npt.ArrayLike,
        destination: FloatArray,
        *,
        amplitudes: npt.ArrayLike | None = None,
    ) -> int:
        """Accumulate signed outgoing contacts for one sparse spike event.

        The caller owns and may reuse ``destination``. The return value is the
        number of non-modulatory outgoing edges visited.
        """

        raw_spikes = np.asarray(spiking_indices)
        if not np.issubdtype(raw_spikes.dtype, np.integer):
            raise TypeError("spiking_indices must contain integers")
        spikes = raw_spikes.astype(np.int64, copy=False)
        if spikes.ndim != 1:
            raise ValueError("spiking_indices must be one-dimensional")
        if spikes.size and (spikes.min() < 0 or spikes.max() >= self.neuron_count):
            raise IndexError("spiking neuron index outside graph")
        if np.unique(spikes).size != spikes.size:
            raise ValueError("spiking_indices must be unique within an event")
        if destination.shape != (self.neuron_count,) or destination.dtype != np.float64:
            raise ValueError(
                "destination must be a float64 vector with one value per neuron"
            )
        if not destination.flags.writeable:
            raise ValueError("destination must be writeable")

        if amplitudes is None:
            event_amplitudes = np.ones(spikes.size, dtype=np.float64)
        else:
            event_amplitudes = np.asarray(amplitudes, dtype=np.float64)
            if event_amplitudes.ndim == 0:
                event_amplitudes = np.broadcast_to(event_amplitudes, spikes.shape)
            elif event_amplitudes.shape != spikes.shape:
                raise ValueError("amplitudes must match spiking_indices")
            if not np.isfinite(event_amplitudes).all():
                raise ValueError("amplitudes must be finite")

        visited_edges = 0
        for source, amplitude in zip(spikes, event_amplitudes, strict=True):
            sign = int(self.presynaptic_signs[source])
            if sign == 0 or amplitude == 0.0:
                continue
            start = int(self.outgoing_indptr[source])
            stop = int(self.outgoing_indptr[source + 1])
            targets = self.target_indices[start:stop]
            counts = self.outgoing_synapse_counts[start:stop]
            destination[targets] += counts * (sign * amplitude)
            visited_edges += stop - start
        return visited_edges


    def project_sources(self, source_indices: npt.ArrayLike) -> SignedSourceProjection:
        """Aggregate signed outgoing contacts from a fixed source population.

        This is intended for continuous/graded source activity. The topology and
        transmitter signs come from the same MaleCNS graph used by spike propagation;
        only the per-step source amplitude is supplied later by the caller.
        """

        raw = np.asarray(source_indices)
        if not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("source_indices must contain integers")
        sources = raw.astype(np.int64, copy=False)
        if sources.ndim != 1:
            raise ValueError("source_indices must be one-dimensional")
        if sources.size and (sources.min() < 0 or sources.max() >= self.neuron_count):
            raise IndexError("source neuron index outside graph")
        if np.unique(sources).size != sources.size:
            raise ValueError("source_indices must be unique")

        target_parts: list[IndexArray] = []
        weight_parts: list[FloatArray] = []
        anatomical_edge_count = 0
        for source in sources:
            sign = int(self.presynaptic_signs[source])
            if sign == 0:
                continue
            start = int(self.outgoing_indptr[source])
            stop = int(self.outgoing_indptr[source + 1])
            if start == stop:
                continue
            target_parts.append(self.target_indices[start:stop])
            weight_parts.append(
                self.outgoing_synapse_counts[start:stop].astype(np.float64)
                * float(sign)
            )
            anatomical_edge_count += stop - start

        if not target_parts:
            empty_indices = np.empty(0, dtype=np.int64)
            empty_weights = np.empty(0, dtype=np.float64)
            empty_indices.flags.writeable = False
            empty_weights.flags.writeable = False
            return SignedSourceProjection(
                source_count=int(sources.size),
                anatomical_edge_count=0,
                target_indices=empty_indices,
                signed_contact_weights=empty_weights,
            )

        targets = np.concatenate(target_parts).astype(np.int64, copy=False)
        weights = np.concatenate(weight_parts)
        order = np.argsort(targets, kind="stable")
        targets = targets[order]
        weights = weights[order]
        unique_targets, starts = np.unique(targets, return_index=True)
        aggregated = np.add.reduceat(weights, starts)
        keep = aggregated != 0.0
        unique_targets = unique_targets[keep].astype(np.int64, copy=False)
        aggregated = aggregated[keep].astype(np.float64, copy=False)
        unique_targets.flags.writeable = False
        aggregated.flags.writeable = False
        return SignedSourceProjection(
            source_count=int(sources.size),
            anatomical_edge_count=anatomical_edge_count,
            target_indices=unique_targets,
            signed_contact_weights=aggregated,
        )

    def transmitter_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for transmitter in self.transmitters:
            key = transmitter if transmitter is not None else "missing"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def sign_counts(self) -> dict[str, int]:
        return {
            "inhibitory": int(np.count_nonzero(self.presynaptic_signs == -1)),
            "separated_or_unknown": int(np.count_nonzero(self.presynaptic_signs == 0)),
            "excitatory": int(np.count_nonzero(self.presynaptic_signs == 1)),
        }


def _effective_transmitters(table: pa.Table) -> tuple[str | None, ...]:
    consensus = table["consensus_nt"].to_pylist()
    ground_truth = table["ground_truth_nt"].to_pylist()
    return tuple(
        str(verified) if verified is not None else _optional_string(predicted)
        for predicted, verified in zip(consensus, ground_truth, strict=True)
    )


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)
