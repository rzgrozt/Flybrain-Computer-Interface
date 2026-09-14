from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from flybrain_interface.panel.anatomy import (
    ATLAS_DIRECTORY,
    ATLAS_EXPORT_SHA256,
    AtlasJoin,
)
from flybrain_interface.panel.worker import build_activity_frame


def test_packaged_atlas_hashes_and_complete_id_join(tmp_path: Path) -> None:
    manifest = json.loads((ATLAS_DIRECTORY / "manifest.json").read_text())
    ids = np.fromfile(ATLAS_DIRECTORY / "ids.bin", dtype="<u4").astype(np.int64)
    body_ids = np.concatenate((ids, np.asarray([2**32 + 7], dtype=np.int64)))
    graph = tmp_path / "graph"
    graph.mkdir()
    parquet.write_table(
        pa.table(
            {
                "neuron_index": np.arange(body_ids.size, dtype=np.int32),
                "body_id": body_ids,
            }
        ),
        graph / "neurons.parquet",
    )
    np.save(graph / "outgoing_indptr.npy", np.zeros(body_ids.size + 1, np.int64))
    np.save(graph / "outgoing_target_indices.npy", np.empty(0, np.int32))
    np.save(graph / "outgoing_synapse_counts.npy", np.empty(0, np.int32))

    joined = AtlasJoin.load(ATLAS_DIRECTORY, graph)
    summary = joined.summary()

    assert summary["atlas_measured_soma_count"] == 140_024
    assert summary["graph_measured_soma_count"] == 140_024
    assert summary["graph_visible_soma_count"] == 124_289
    assert summary["graph_measured_but_view_omitted_count"] == 15_735
    assert summary["graph_without_measured_soma_count"] == 1
    assert manifest["brainCount"] == summary["graph_visible_soma_count"]
    for name, expected in ATLAS_EXPORT_SHA256.items():
        digest = hashlib.sha256((ATLAS_DIRECTORY / name).read_bytes()).hexdigest()
        assert digest == expected


def small_join() -> AtlasJoin:
    return AtlasJoin(
        atlas_directory=Path("atlas"),
        data_directory=Path("graph"),
        body_ids_by_index=np.asarray([101, 102, 103, 104], dtype=np.int64),
        atlas_row_by_index=np.asarray([0, 1, 2, -1], dtype=np.int32),
        visible_atlas_row_by_index=np.asarray([0, 1, 2, -1], dtype=np.int32),
        atlas_groups=np.asarray([0, 1, 2], dtype=np.uint8),
        manifest={"count": 3, "brainCount": 3},
        outgoing_indptr=np.asarray([0, 3, 3, 3, 3], dtype=np.int64),
        outgoing_targets=np.asarray([1, 2, 3], dtype=np.int32),
        outgoing_weights=np.asarray([4, 9, 20], dtype=np.int32),
    )


def test_activity_window_is_bounded_normalized_and_replaced() -> None:
    join = small_join()
    counts = np.asarray([1, 2, 0, 4], dtype=np.int64)

    frame = build_activity_frame([0, 1, 2, 3], counts, 0.02, join)
    cleared = build_activity_frame([0, 1, 2, 3], np.zeros(4, np.int64), 0.02, join)

    assert frame["signal"] == "emitted_simulated_spike_count"
    assert frame["values"] == [[0, "101", 1.0, 1], [1, "102", 1.0, 2]]
    assert frame["active_selected_count"] == 3
    assert frame["active_without_visible_soma_count"] == 1
    assert cleared["values"] == []
    assert cleared["active_selected_count"] == 0


def test_neighborhood_uses_true_weights_caps_and_missing_coordinate_counts() -> None:
    result = small_join().neighborhood(
        [0], max_nodes=3, max_edges=2, min_synapse_count=4
    )

    assert [edge["synapse_count"] for edge in result["edges"]] == [9, 4]
    relationships = {
        (edge["source_neuron_index"], edge["target_neuron_index"])
        for edge in result["edges"]
    }
    assert relationships == {(0, 1), (0, 2)}
    assert result["relationships_without_visible_soma_count"] == 1
    assert result["filter"]["direction"] == "outgoing_from_seed"
    assert "not anatomical neurites" in result["line_meaning"]


def test_neighborhood_rejects_duplicate_or_out_of_range_indices() -> None:
    join = small_join()
    with pytest.raises(ValueError, match="unique"):
        join.neighborhood([0, 0], max_nodes=3, max_edges=2, min_synapse_count=1)
    with pytest.raises(ValueError, match="below"):
        join.map_indices([4])
