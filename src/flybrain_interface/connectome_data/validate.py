"""Independent validation of normalized MaleCNS outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

import numpy as np
import pyarrow.parquet as parquet

from flybrain_interface.connectome_data.manifest import file_sha256

_CORE_OUTPUTS = {
    "neurons.parquet",
    "edges.parquet",
    "csr_indptr.npy",
    "csr_indices.npy",
    "csr_synapse_counts.npy",
}
_OUTGOING_OUTPUTS = {
    "outgoing_indptr.npy",
    "outgoing_target_indices.npy",
    "outgoing_synapse_counts.npy",
}


@dataclass(frozen=True, slots=True)
class ProcessedValidation:
    dataset: str
    neuron_count: int
    edge_count: int
    synaptic_contact_count: int
    self_edge_count: int
    verified_files: tuple[str, ...]


def validate_processed_dataset(
    lock_path: Path, output_directory: Path
) -> ProcessedValidation:
    lock: dict[str, Any] = json.loads(lock_path.read_text(encoding="utf-8"))
    schema_version = lock.get("schema_version")
    if schema_version not in (1, 2):
        raise ValueError("unsupported normalized-data lock schema")

    outputs: dict[str, dict[str, Any]] = lock["outputs"]
    expected_outputs = _CORE_OUTPUTS | (
        _OUTGOING_OUTPUTS if schema_version == 2 else set()
    )
    if set(outputs) != expected_outputs:
        raise ValueError("normalized-data lock outputs do not match its schema version")
    for filename, expected in outputs.items():
        if PurePath(filename).name != filename:
            raise ValueError("locked output filename must not contain a path")
        path = output_directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"size mismatch for normalized output {filename}")
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch for normalized output {filename}")

    neuron_count = int(lock["neuron_count"])
    edge_count = int(lock["edge_count"])
    contacts = int(lock["synaptic_contact_count"])
    self_edges = int(lock["self_edge_count"])
    indptr = np.load(output_directory / "csr_indptr.npy", mmap_mode="r")
    indices = np.load(output_directory / "csr_indices.npy", mmap_mode="r")
    weights = np.load(output_directory / "csr_synapse_counts.npy", mmap_mode="r")
    if indptr.shape != (neuron_count + 1,):
        raise ValueError("CSR indptr shape does not match neuron count")
    if indices.shape != (edge_count,) or weights.shape != (edge_count,):
        raise ValueError("CSR edge arrays do not match edge count")
    if int(indptr[0]) != 0 or int(indptr[-1]) != edge_count:
        raise ValueError("CSR indptr bounds are invalid")
    if np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("CSR indptr must be monotonic")
    if np.any(indices < 0) or np.any(indices >= neuron_count):
        raise ValueError("CSR source index outside neuron range")
    if np.any(weights <= 0):
        raise ValueError("CSR synapse counts must be positive")
    if int(np.sum(weights, dtype=np.int64)) != contacts:
        raise ValueError("CSR synaptic contact total does not match lock")
    observed_self_edges = sum(
        int(np.count_nonzero(indices[indptr[target] : indptr[target + 1]] == target))
        for target in range(neuron_count)
    )
    if observed_self_edges != self_edges:
        raise ValueError("CSR self-edge count does not match lock")

    if schema_version == 2:
        _validate_outgoing_arrays(
            output_directory,
            neuron_count=neuron_count,
            edge_count=edge_count,
            contacts=contacts,
            self_edges=self_edges,
            incoming_indptr=indptr,
            incoming_sources=indices,
        )

    neurons = parquet.ParquetFile(output_directory / "neurons.parquet")
    edges = parquet.ParquetFile(output_directory / "edges.parquet")
    if neurons.metadata.num_rows != neuron_count:
        raise ValueError("normalized neuron table row count does not match lock")
    if edges.metadata.num_rows != edge_count:
        raise ValueError("normalized edge table row count does not match lock")

    return ProcessedValidation(
        dataset=str(lock["dataset"]),
        neuron_count=neuron_count,
        edge_count=edge_count,
        synaptic_contact_count=contacts,
        self_edge_count=self_edges,
        verified_files=tuple(sorted(outputs)),
    )


def _validate_outgoing_arrays(
    output_directory: Path,
    *,
    neuron_count: int,
    edge_count: int,
    contacts: int,
    self_edges: int,
    incoming_indptr: np.ndarray[Any, Any],
    incoming_sources: np.ndarray[Any, Any],
) -> None:
    indptr = np.load(output_directory / "outgoing_indptr.npy", mmap_mode="r")
    targets = np.load(output_directory / "outgoing_target_indices.npy", mmap_mode="r")
    weights = np.load(output_directory / "outgoing_synapse_counts.npy", mmap_mode="r")
    if indptr.shape != (neuron_count + 1,):
        raise ValueError("outgoing indptr shape does not match neuron count")
    if targets.shape != (edge_count,) or weights.shape != (edge_count,):
        raise ValueError("outgoing edge arrays do not match edge count")
    if int(indptr[0]) != 0 or int(indptr[-1]) != edge_count:
        raise ValueError("outgoing indptr bounds are invalid")
    if np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("outgoing indptr must be monotonic")
    if np.any(targets < 0) or np.any(targets >= neuron_count):
        raise ValueError("outgoing target index outside neuron range")
    if np.any(weights <= 0):
        raise ValueError("outgoing synapse counts must be positive")
    if int(np.sum(weights, dtype=np.int64)) != contacts:
        raise ValueError("outgoing synaptic contact total does not match lock")
    observed_self_edges = sum(
        int(np.count_nonzero(targets[indptr[source] : indptr[source + 1]] == source))
        for source in range(neuron_count)
    )
    if observed_self_edges != self_edges:
        raise ValueError("outgoing self-edge count does not match lock")

    incoming_source_degrees = np.bincount(incoming_sources, minlength=neuron_count)
    if not np.array_equal(incoming_source_degrees, np.diff(indptr)):
        raise ValueError("incoming and outgoing source degrees disagree")
    outgoing_target_degrees = np.bincount(targets, minlength=neuron_count)
    if not np.array_equal(outgoing_target_degrees, np.diff(incoming_indptr)):
        raise ValueError("incoming and outgoing target degrees disagree")
