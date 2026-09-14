"""Memory-bounded MaleCNS annotation and connectivity normalization."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.feather as feather
import pyarrow.parquet as parquet

from flybrain_interface.connectome_data.manifest import (
    DatasetManifest,
    file_sha256,
    verify_dataset,
)

NORMALIZATION_SCHEMA_VERSION = 1
_MEMORY_LIMIT_PATTERN = re.compile(r"^[1-9][0-9]*(?:MB|GB)$")


@dataclass(frozen=True, slots=True)
class NormalizationReport:
    schema_version: int
    dataset: str
    selection_policy: str
    neuron_count: int
    edge_count: int
    synaptic_contact_count: int
    self_edge_count: int
    excluded_annotation_rows: int
    excluded_connectivity_rows: int
    neurons_without_consensus_nt: int
    source_rows: dict[str, int]
    superclass_counts: dict[str, int]
    consensus_nt_counts: dict[str, int]
    output_sha256: dict[str, str]


def normalize_dataset(
    manifest: DatasetManifest,
    raw_directory: Path,
    output_directory: Path,
    *,
    memory_limit: str = "4GB",
    threads: int = 4,
) -> NormalizationReport:
    if not _MEMORY_LIMIT_PATTERN.fullmatch(memory_limit):
        raise ValueError("memory_limit must look like '4096MB' or '4GB'")
    if threads <= 0:
        raise ValueError("threads must be positive")

    verified = verify_dataset(manifest, raw_directory)
    source_paths = {artifact.role: artifact.path for artifact in verified}
    if output_directory.exists():
        raise FileExistsError(
            f"output already exists: {output_directory}; "
            "remove it explicitly to rebuild"
        )

    partial = output_directory.with_name(f"{output_directory.name}.partial")
    if partial.exists():
        raise FileExistsError(
            f"partial output already exists: {partial}; "
            "inspect and remove it explicitly"
        )
    partial.mkdir(parents=True)
    temporary = partial / ".duckdb-tmp"
    temporary.mkdir()

    annotations = feather.read_table(source_paths["annotations"], memory_map=True)
    neurotransmitters = feather.read_table(
        source_paths["neurotransmitters"], memory_map=True
    )
    edge_dataset = pads.dataset(source_paths["connectivity"], format="ipc")
    _validate_source_schemas(annotations, neurotransmitters, edge_dataset.schema)

    source_rows = {
        "annotations": annotations.num_rows,
        "neurotransmitters": neurotransmitters.num_rows,
        "connectivity": _ipc_row_count(source_paths["connectivity"]),
    }
    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit = '{memory_limit}'")
        connection.execute(f"SET threads = {threads}")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(f"SET temp_directory = '{_sql_path(temporary)}'")
        connection.register("annotations", annotations)
        connection.register("neurotransmitters", neurotransmitters)
        connection.register("raw_edges", edge_dataset)
        connection.execute(
            """
            CREATE TEMP TABLE neuron_map AS
            SELECT
                CAST(
                    row_number() OVER (ORDER BY bodyId) - 1 AS INTEGER
                ) AS neuron_index,
                bodyId AS body_id
            FROM annotations
            WHERE superclass IS NOT NULL
            """
        )

        count_row = connection.execute("SELECT count(*) FROM neuron_map").fetchone()
        if count_row is None:
            raise RuntimeError("failed to count selected neurons")
        neuron_count = int(count_row[0])
        if neuron_count != manifest.selection.expected_neuron_count:
            raise ValueError(
                f"neuron census mismatch: {neuron_count} != "
                f"{manifest.selection.expected_neuron_count}"
            )

        neurons_path = partial / "neurons.parquet"
        connection.execute(
            f"""
            COPY (
                SELECT
                    m.neuron_index,
                    m.body_id,
                    a.status,
                    a.statusLabel AS status_label,
                    a.superclass,
                    a.class,
                    a.subclass,
                    a.type,
                    a.instance,
                    a.rootSide AS root_side,
                    a.somaSide AS soma_side,
                    a.somaNeuromere AS soma_neuromere,
                    a.entryNerve AS entry_nerve,
                    a.exitNerve AS exit_nerve,
                    a.receptorType AS receptor_type,
                    a.fruDsx AS fru_dsx,
                    n.consensus_nt,
                    n.predicted_nt,
                    n.predicted_nt_confidence,
                    n.ground_truth AS ground_truth_nt
                FROM neuron_map m
                JOIN annotations a ON a.bodyId = m.body_id
                LEFT JOIN neurotransmitters n ON n.body = m.body_id
                ORDER BY m.neuron_index
            ) TO '{_sql_path(neurons_path)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 262144)
            """
        )

        edges_path = partial / "edges.parquet"
        connection.execute(
            f"""
            COPY (
                SELECT
                    src.neuron_index AS source_index,
                    dst.neuron_index AS target_index,
                    e.weight AS synapse_count
                FROM raw_edges e
                JOIN neuron_map src ON src.body_id = e.body_pre
                JOIN neuron_map dst ON dst.body_id = e.body_post
                ORDER BY dst.neuron_index, src.neuron_index
            ) TO '{_sql_path(edges_path)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1048576)
            """
        )
    finally:
        connection.close()
        if temporary.exists():
            temporary.rmdir()

    edge_count, contact_count, self_edges = _write_csr_arrays(
        partial / "edges.parquet", partial, neuron_count
    )
    neuron_summary = parquet.read_table(
        partial / "neurons.parquet", columns=["superclass", "consensus_nt"]
    )
    consensus_column = neuron_summary.column("consensus_nt")
    superclass_counts = _string_counts(neuron_summary.column("superclass"))
    consensus_counts = _string_counts(consensus_column)

    output_names = (
        "neurons.parquet",
        "edges.parquet",
        "csr_indptr.npy",
        "csr_indices.npy",
        "csr_synapse_counts.npy",
    )
    output_hashes = {name: file_sha256(partial / name) for name in output_names}
    report = NormalizationReport(
        schema_version=NORMALIZATION_SCHEMA_VERSION,
        dataset=manifest.dataset,
        selection_policy=manifest.selection.description,
        neuron_count=neuron_count,
        edge_count=edge_count,
        synaptic_contact_count=contact_count,
        self_edge_count=self_edges,
        excluded_annotation_rows=source_rows["annotations"] - neuron_count,
        excluded_connectivity_rows=source_rows["connectivity"] - edge_count,
        neurons_without_consensus_nt=consensus_column.null_count,
        source_rows=source_rows,
        superclass_counts=superclass_counts,
        consensus_nt_counts=consensus_counts,
        output_sha256=output_hashes,
    )
    (partial / "report.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output_directory)
    return report


def _write_csr_arrays(
    edges_path: Path, output_directory: Path, neuron_count: int
) -> tuple[int, int, int]:
    edge_file = parquet.ParquetFile(edges_path)
    edge_count = edge_file.metadata.num_rows
    indices = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
        output_directory / "csr_indices.npy",
        mode="w+",
        dtype=np.int32,
        shape=(edge_count,),
    )
    weights = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
        output_directory / "csr_synapse_counts.npy",
        mode="w+",
        dtype=np.int32,
        shape=(edge_count,),
    )
    counts = np.zeros(neuron_count, dtype=np.int64)
    offset = 0
    contacts = 0
    self_edges = 0
    previous_pair = (-1, -1)

    for batch in edge_file.iter_batches(
        columns=["source_index", "target_index", "synapse_count"],
        batch_size=1_048_576,
    ):
        source = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
        target = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
        weight = batch.column(2).to_numpy(zero_copy_only=False).astype(np.int64)
        if np.any(weight <= 0) or np.any(weight > np.iinfo(np.int32).max):
            raise ValueError("synapse counts must be positive int32 values")
        if source.size:
            first_pair = (int(target[0]), int(source[0]))
            if first_pair <= previous_pair:
                raise ValueError("normalized edges are not unique and strictly sorted")
            if np.any(target[1:] < target[:-1]) or np.any(
                (target[1:] == target[:-1]) & (source[1:] <= source[:-1])
            ):
                raise ValueError("normalized edges are not unique and strictly sorted")
            previous_pair = (int(target[-1]), int(source[-1]))
        stop = offset + source.size
        indices[offset:stop] = source.astype(np.int32)
        weights[offset:stop] = weight.astype(np.int32)
        counts += np.bincount(target, minlength=neuron_count)
        contacts += int(weight.sum())
        self_edges += int(np.count_nonzero(source == target))
        offset = stop

    if offset != edge_count:
        raise ValueError(f"edge row mismatch: {offset} != {edge_count}")
    indptr = np.empty(neuron_count + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    np.save(output_directory / "csr_indptr.npy", indptr)
    indices.flush()
    weights.flush()
    return edge_count, contacts, self_edges


def _ipc_row_count(path: Path) -> int:
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        return sum(
            reader.get_batch(index).num_rows
            for index in range(reader.num_record_batches)
        )


def _validate_source_schemas(
    annotations: pa.Table,
    neurotransmitters: pa.Table,
    connectivity_schema: pa.Schema,
) -> None:
    required = {
        "annotations": {
            "bodyId",
            "status",
            "statusLabel",
            "superclass",
            "class",
            "subclass",
            "type",
            "instance",
            "rootSide",
            "somaSide",
            "somaNeuromere",
            "entryNerve",
            "exitNerve",
            "receptorType",
            "fruDsx",
        },
        "neurotransmitters": {
            "body",
            "consensus_nt",
            "predicted_nt",
            "predicted_nt_confidence",
            "ground_truth",
        },
        "connectivity": {"body_pre", "body_post", "weight"},
    }
    actual = {
        "annotations": set(annotations.column_names),
        "neurotransmitters": set(neurotransmitters.column_names),
        "connectivity": set(connectivity_schema.names),
    }
    for role, names in required.items():
        missing = names - actual[role]
        if missing:
            raise ValueError(f"{role} source is missing columns: {sorted(missing)}")

    annotation_ids = annotations.column("bodyId")
    if annotation_ids.null_count:
        raise ValueError("annotation bodyId cannot be null")
    if len(annotation_ids.unique()) != len(annotation_ids):
        raise ValueError("annotation bodyId must be unique")
    transmitter_ids = neurotransmitters.column("body")
    if transmitter_ids.null_count:
        raise ValueError("neurotransmitter body cannot be null")
    if len(transmitter_ids.unique()) != len(transmitter_ids):
        raise ValueError("neurotransmitter body must be unique")

    expected_types = {
        "annotation bodyId": (annotations.schema.field("bodyId").type, pa.int64()),
        "neurotransmitter body": (
            neurotransmitters.schema.field("body").type,
            pa.int64(),
        ),
        "connectivity body_pre": (
            connectivity_schema.field("body_pre").type,
            pa.int64(),
        ),
        "connectivity body_post": (
            connectivity_schema.field("body_post").type,
            pa.int64(),
        ),
        "connectivity weight": (
            connectivity_schema.field("weight").type,
            pa.int64(),
        ),
    }
    for label, (actual_type, expected_type) in expected_types.items():
        if actual_type != expected_type:
            raise ValueError(
                f"{label} type mismatch: {actual_type} != {expected_type}"
            )


def _string_counts(column: pa.ChunkedArray) -> dict[str, int]:
    return dict(sorted(Counter(value for value in column.to_pylist() if value).items()))


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")
