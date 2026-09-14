"""Build a verified source-major index from normalized MaleCNS edges."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as parquet

from flybrain_interface.connectome_data.manifest import file_sha256

OUTGOING_FILENAMES = (
    "outgoing_indptr.npy",
    "outgoing_target_indices.npy",
    "outgoing_synapse_counts.npy",
)
_MEMORY_LIMIT_PATTERN = re.compile(r"^[1-9][0-9]*(?:MB|GB)$")


@dataclass(frozen=True, slots=True)
class OutgoingIndexReport:
    neuron_count: int
    edge_count: int
    synaptic_contact_count: int
    self_edge_count: int
    output_sha256: dict[str, str]


def build_outgoing_index(
    dataset_directory: Path,
    *,
    memory_limit: str = "4GB",
    threads: int = 4,
) -> OutgoingIndexReport:
    """Add source-major arrays to an existing normalized dataset safely."""

    dataset_directory = dataset_directory.resolve()
    edges_path = dataset_directory / "edges.parquet"
    neurons_path = dataset_directory / "neurons.parquet"
    if not edges_path.is_file() or not neurons_path.is_file():
        raise FileNotFoundError(
            "normalized edges.parquet and neurons.parquet are required"
        )
    existing = [
        name for name in OUTGOING_FILENAMES if (dataset_directory / name).exists()
    ]
    if existing:
        raise FileExistsError(f"outgoing index already exists: {', '.join(existing)}")

    partial = dataset_directory / ".outgoing-index.partial"
    if partial.exists():
        raise FileExistsError(
            f"partial outgoing index exists: {partial}; "
            "inspect and remove it explicitly"
        )
    partial.mkdir()
    neuron_count = parquet.ParquetFile(neurons_path).metadata.num_rows
    try:
        report = write_outgoing_arrays(
            edges_path,
            partial,
            int(neuron_count),
            memory_limit=memory_limit,
            threads=threads,
        )
        for filename in OUTGOING_FILENAMES:
            os.replace(partial / filename, dataset_directory / filename)
        partial.rmdir()
    except Exception:
        # Preserve partial work for diagnosis; never silently delete or overwrite it.
        raise
    return report


def write_outgoing_arrays(
    edges_path: Path,
    output_directory: Path,
    neuron_count: int,
    *,
    memory_limit: str,
    threads: int,
) -> OutgoingIndexReport:
    """Sort normalized edges source-first and stream them into NumPy arrays."""

    if not _MEMORY_LIMIT_PATTERN.fullmatch(memory_limit):
        raise ValueError("memory_limit must look like '4096MB' or '4GB'")
    if threads <= 0:
        raise ValueError("threads must be positive")
    if neuron_count <= 0:
        raise ValueError("neuron_count must be positive")

    sorted_edges = output_directory / ".edges-source-major.parquet"
    temporary = output_directory / ".duckdb-outgoing-tmp"
    temporary.mkdir()
    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit = '{memory_limit}'")
        connection.execute(f"SET threads = {threads}")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(f"SET temp_directory = '{_sql_path(temporary)}'")
        connection.execute(
            f"""
            COPY (
                SELECT source_index, target_index, synapse_count
                FROM read_parquet('{_sql_path(edges_path)}')
                ORDER BY source_index, target_index
            ) TO '{_sql_path(sorted_edges)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1048576)
            """
        )
    finally:
        connection.close()
        temporary.rmdir()

    edge_file = parquet.ParquetFile(sorted_edges)
    edge_count = edge_file.metadata.num_rows
    targets = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
        output_directory / "outgoing_target_indices.npy",
        mode="w+",
        dtype=np.int32,
        shape=(edge_count,),
    )
    weights = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
        output_directory / "outgoing_synapse_counts.npy",
        mode="w+",
        dtype=np.int32,
        shape=(edge_count,),
    )
    source_degrees = np.zeros(neuron_count, dtype=np.int64)
    offset = 0
    contacts = 0
    self_edges = 0
    previous_pair = (-1, -1)

    for batch in edge_file.iter_batches(
        columns=["source_index", "target_index", "synapse_count"],
        batch_size=1_048_576,
    ):
        source = np.asarray(
            batch.column(0).to_numpy(zero_copy_only=False), dtype=np.int64
        )
        target = np.asarray(
            batch.column(1).to_numpy(zero_copy_only=False), dtype=np.int64
        )
        weight = np.asarray(
            batch.column(2).to_numpy(zero_copy_only=False), dtype=np.int64
        )
        if source.size:
            first_pair = (int(source[0]), int(target[0]))
            if first_pair <= previous_pair:
                raise ValueError(
                    "source-major edges are not unique and strictly sorted"
                )
            if np.any(source[1:] < source[:-1]) or np.any(
                (source[1:] == source[:-1]) & (target[1:] <= target[:-1])
            ):
                raise ValueError(
                    "source-major edges are not unique and strictly sorted"
                )
            previous_pair = (int(source[-1]), int(target[-1]))
        if np.any(source < 0) or np.any(source >= neuron_count):
            raise ValueError("source index outside neuron range")
        if np.any(target < 0) or np.any(target >= neuron_count):
            raise ValueError("target index outside neuron range")
        if np.any(weight <= 0) or np.any(weight > np.iinfo(np.int32).max):
            raise ValueError("synapse counts must be positive int32 values")

        stop = offset + source.size
        targets[offset:stop] = target.astype(np.int32)
        weights[offset:stop] = weight.astype(np.int32)
        source_degrees += np.bincount(source, minlength=neuron_count)
        contacts += int(weight.sum())
        self_edges += int(np.count_nonzero(source == target))
        offset = stop

    if offset != edge_count:
        raise ValueError(f"edge row mismatch: {offset} != {edge_count}")
    indptr = np.empty(neuron_count + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(source_degrees, out=indptr[1:])
    np.save(output_directory / "outgoing_indptr.npy", indptr)
    targets.flush()
    weights.flush()
    sorted_edges.unlink()

    output_hashes = {
        filename: file_sha256(output_directory / filename)
        for filename in OUTGOING_FILENAMES
    }
    return OutgoingIndexReport(
        neuron_count=neuron_count,
        edge_count=int(edge_count),
        synaptic_contact_count=contacts,
        self_edge_count=self_edges,
        output_sha256=output_hashes,
    )


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")
