"""Independent validation of normalized MaleCNS outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

import numpy as np
import pyarrow.parquet as parquet

from flybrain_interface.connectome_data.manifest import file_sha256


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
    if lock.get("schema_version") != 1:
        raise ValueError("unsupported normalized-data lock schema")

    outputs: dict[str, dict[str, Any]] = lock["outputs"]
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
