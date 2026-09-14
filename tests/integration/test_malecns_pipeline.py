"""End-to-end normalization tests using a tiny synthetic Arrow dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as parquet
import pytest

from flybrain_interface.connectome_data.manifest import file_sha256, load_manifest
from flybrain_interface.connectome_data.normalize import normalize_dataset
from flybrain_interface.connectome_data.validate import validate_processed_dataset


def test_normalize_dataset_filters_indexes_and_builds_csr(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    annotations = pa.table(
        {
            "bodyId": [30, 10, 20, 40],
            "status": ["Traced"] * 4,
            "statusLabel": ["Reviewed"] * 4,
            "superclass": ["vnc_motor", "ol_sensory", None, "descending_neuron"],
            "class": [None] * 4,
            "subclass": [None] * 4,
            "type": ["motor", "sensory", None, "descending"],
            "instance": ["m", "s", None, "d"],
            "rootSide": [None] * 4,
            "somaSide": ["L", "R", None, "R"],
            "somaNeuromere": [None] * 4,
            "entryNerve": [None] * 4,
            "exitNerve": [None] * 4,
            "receptorType": [None] * 4,
            "fruDsx": [None] * 4,
        }
    )
    neurotransmitters = pa.table(
        {
            "body": [10, 30, 40],
            "consensus_nt": ["acetylcholine", "glutamate", "gaba"],
            "predicted_nt": ["acetylcholine", "glutamate", "gaba"],
            "predicted_nt_confidence": [0.9, 0.8, 0.7],
            "ground_truth": [None, None, "gaba"],
        }
    )
    connectivity = pa.table(
        {
            "body_pre": [10, 10, 20, 30, 40, 40],
            "body_post": [30, 40, 30, 40, 10, 40],
            "weight": [5, 2, 9, 3, 4, 1],
        }
    )
    files = {
        "annotations.feather": annotations,
        "neurotransmitters.feather": neurotransmitters,
        "connectivity.feather": connectivity,
    }
    for filename, table in files.items():
        feather.write_feather(table, raw / filename)

    manifest_path = _write_manifest(raw, tmp_path / "manifest.json")
    output = tmp_path / "processed"
    report = normalize_dataset(
        load_manifest(manifest_path),
        raw,
        output,
        memory_limit="512MB",
        threads=1,
    )

    neurons = parquet.read_table(output / "neurons.parquet")
    assert neurons.column("body_id").to_pylist() == [10, 30, 40]
    assert report.neuron_count == 3
    assert report.edge_count == 5
    assert report.synaptic_contact_count == 15
    assert report.self_edge_count == 1
    assert report.excluded_annotation_rows == 1
    assert report.excluded_connectivity_rows == 1
    assert np.load(output / "csr_indptr.npy").tolist() == [0, 1, 2, 5]
    assert np.load(output / "csr_indices.npy").tolist() == [2, 0, 0, 1, 2]
    assert np.load(output / "csr_synapse_counts.npy").tolist() == [4, 5, 2, 3, 1]

    output_names = (
        "neurons.parquet",
        "edges.parquet",
        "csr_indptr.npy",
        "csr_indices.npy",
        "csr_synapse_counts.npy",
    )
    lock = {
        "schema_version": 1,
        "dataset": "test:v1",
        "neuron_count": report.neuron_count,
        "edge_count": report.edge_count,
        "synaptic_contact_count": report.synaptic_contact_count,
        "self_edge_count": report.self_edge_count,
        "outputs": {
            name: {
                "size_bytes": (output / name).stat().st_size,
                "sha256": report.output_sha256[name],
            }
            for name in output_names
        },
    }
    lock_path = tmp_path / "normalized-lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    validation = validate_processed_dataset(lock_path, output)

    assert validation.neuron_count == 3
    assert validation.edge_count == 5


def test_manifest_verification_rejects_corruption(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in (
        "annotations.feather",
        "neurotransmitters.feather",
        "connectivity.feather",
    ):
        (raw / name).write_bytes(name.encode())
    manifest_path = _write_manifest(raw, tmp_path / "manifest.json")
    (raw / "connectivity.feather").write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        normalize_dataset(load_manifest(manifest_path), raw, tmp_path / "processed")


def _write_manifest(raw: Path, destination: Path) -> Path:
    roles = {
        "annotations": "annotations.feather",
        "neurotransmitters": "neurotransmitters.feather",
        "connectivity": "connectivity.feather",
    }
    payload = {
        "schema_version": 1,
        "dataset": "test:v1",
        "dataset_uuid": "fixture",
        "release_date": "2026-01-01",
        "license": "CC0",
        "landing_page": "https://example.invalid/",
        "selection": {
            "description": "Fixture rows with an assigned superclass.",
            "required_superclass": True,
            "expected_neuron_count": 3,
        },
        "sources": [
            {
                "role": role,
                "filename": filename,
                "url": f"https://example.invalid/{filename}",
                "size_bytes": (raw / filename).stat().st_size,
                "gcs_generation": "fixture",
                "gcs_md5_base64": "fixture",
                "gcs_crc32c_base64": "fixture",
                "sha256": file_sha256(raw / filename),
            }
            for role, filename in roles.items()
        ],
    }
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination
