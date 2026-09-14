from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from flybrain_interface.connectome_data.outgoing import build_outgoing_index


def test_build_outgoing_index_is_sorted_and_non_overwriting(tmp_path: Path) -> None:
    parquet.write_table(
        pa.table(
            {
                "source_index": [2, 0, 0, 1],
                "target_index": [0, 2, 1, 2],
                "synapse_count": [4, 2, 5, 3],
            }
        ),
        tmp_path / "edges.parquet",
    )
    parquet.write_table(
        pa.table({"neuron_index": [0, 1, 2]}), tmp_path / "neurons.parquet"
    )

    report = build_outgoing_index(tmp_path, memory_limit="128MB", threads=1)

    assert report.neuron_count == 3
    assert report.edge_count == 4
    assert report.synaptic_contact_count == 14
    assert report.self_edge_count == 0
    assert np.load(tmp_path / "outgoing_indptr.npy").tolist() == [0, 2, 3, 4]
    assert np.load(tmp_path / "outgoing_target_indices.npy").tolist() == [1, 2, 2, 0]
    assert np.load(tmp_path / "outgoing_synapse_counts.npy").tolist() == [5, 2, 3, 4]
    assert set(report.output_sha256) == {
        "outgoing_indptr.npy",
        "outgoing_target_indices.npy",
        "outgoing_synapse_counts.npy",
    }

    with pytest.raises(FileExistsError, match="already exists"):
        build_outgoing_index(tmp_path, memory_limit="128MB", threads=1)
