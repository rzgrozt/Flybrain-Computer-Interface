"""Verified MaleCNS soma-atlas joins and bounded graph overlays."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as parquet

ATLAS_DATASET = "male-cns:v1.0"
ATLAS_SOURCE_REVISION = "38f55332055328d38c29e72474c4ad5b6876101f"
ATLAS_EXPORT_SHA256 = {
    "positions.bin": "c09e0180f49e4f7c40d99c60bab1273492a8c808a6e9afea1f5c4ca9cb6a3168",
    "ids.bin": "930402247442138d3f5fa5b5583357a96bc55f043719b6137a4994221702386a",
    "groups.bin": "51fbfba3f615125be95eb3302956c254626ae0b1825c423bb7cb80b8d169f171",
}
ATLAS_DIRECTORY = Path(__file__).with_name("static") / "brain-atlas"


@dataclass(frozen=True, slots=True)
class AtlasJoin:
    """One verified mapping from stable graph indices to measured soma rows."""

    atlas_directory: Path
    data_directory: Path
    body_ids_by_index: npt.NDArray[np.int64]
    atlas_row_by_index: npt.NDArray[np.int32]
    visible_atlas_row_by_index: npt.NDArray[np.int32]
    atlas_groups: npt.NDArray[np.uint8]
    manifest: dict[str, Any]
    outgoing_indptr: npt.NDArray[np.int64]
    outgoing_targets: npt.NDArray[np.int32]
    outgoing_weights: npt.NDArray[np.int32]

    @classmethod
    def load(cls, atlas_directory: Path, data_directory: Path) -> AtlasJoin:
        atlas_directory = atlas_directory.resolve()
        data_directory = data_directory.resolve()
        manifest = json.loads((atlas_directory / "manifest.json").read_text())
        _validate_atlas_manifest(manifest, atlas_directory)

        atlas_ids = np.fromfile(atlas_directory / "ids.bin", dtype="<u4")
        groups = np.fromfile(atlas_directory / "groups.bin", dtype="u1")
        if atlas_ids.size != int(manifest["count"]) or groups.size != atlas_ids.size:
            raise ValueError("atlas binary lengths do not match its manifest")
        if np.unique(atlas_ids).size != atlas_ids.size:
            raise ValueError("atlas body IDs must be unique")

        table = parquet.read_table(
            data_directory / "neurons.parquet", columns=["neuron_index", "body_id"]
        )
        neuron_indices = np.asarray(table["neuron_index"], dtype=np.int64)
        body_ids = np.asarray(table["body_id"], dtype=np.int64)
        expected = np.arange(body_ids.size, dtype=np.int64)
        if not np.array_equal(neuron_indices, expected):
            raise ValueError(
                "graph neuron indices are not contiguous stable row indices"
            )
        if np.unique(body_ids).size != body_ids.size:
            raise ValueError("graph body IDs must be unique")
        if atlas_ids.size and int(atlas_ids.max()) > np.iinfo(np.uint32).max:
            raise ValueError("atlas body ID exceeds its declared uint32 encoding")

        atlas_lookup = {int(body_id): row for row, body_id in enumerate(atlas_ids)}
        rows = np.fromiter(
            (atlas_lookup.get(int(body_id), -1) for body_id in body_ids),
            dtype=np.int32,
            count=body_ids.size,
        )
        visible_rows = rows.copy()
        visible = visible_rows >= 0
        visible[visible] &= groups[visible_rows[visible]] < 3
        visible_rows[~visible] = -1
        if int(np.count_nonzero(visible_rows >= 0)) != int(manifest["brainCount"]):
            raise ValueError(
                "graph-to-atlas visible join does not match atlas brainCount"
            )
        rows.flags.writeable = False
        visible_rows.flags.writeable = False
        body_ids.flags.writeable = False

        outgoing_indptr = np.load(
            data_directory / "outgoing_indptr.npy", mmap_mode="r"
        )
        outgoing_targets = np.load(
            data_directory / "outgoing_target_indices.npy", mmap_mode="r"
        )
        outgoing_weights = np.load(
            data_directory / "outgoing_synapse_counts.npy", mmap_mode="r"
        )
        if outgoing_indptr.dtype != np.int64 or outgoing_indptr.shape != (
            body_ids.size + 1,
        ):
            raise ValueError("invalid source-major graph indptr")
        if (
            outgoing_targets.dtype != np.int32
            or outgoing_weights.dtype != np.int32
            or outgoing_targets.shape != outgoing_weights.shape
            or int(outgoing_indptr[0]) != 0
            or int(outgoing_indptr[-1]) != outgoing_targets.size
        ):
            raise ValueError("invalid source-major graph edge arrays")
        if outgoing_targets.size and (
            int(outgoing_targets.min()) < 0
            or int(outgoing_targets.max()) >= body_ids.size
            or int(outgoing_weights.min()) <= 0
        ):
            raise ValueError("source-major graph contains invalid targets or weights")

        return cls(
            atlas_directory=atlas_directory,
            data_directory=data_directory,
            body_ids_by_index=body_ids,
            atlas_row_by_index=rows,
            visible_atlas_row_by_index=visible_rows,
            atlas_groups=groups,
            manifest=manifest,
            outgoing_indptr=outgoing_indptr,
            outgoing_targets=outgoing_targets,
            outgoing_weights=outgoing_weights,
        )

    @property
    def neuron_count(self) -> int:
        return int(self.body_ids_by_index.size)

    def summary(self) -> dict[str, Any]:
        measured = int(np.count_nonzero(self.atlas_row_by_index >= 0))
        mapped = int(np.count_nonzero(self.visible_atlas_row_by_index >= 0))
        return {
            "dataset": ATLAS_DATASET,
            "template_revision": ATLAS_SOURCE_REVISION,
            "graph_neuron_count": self.neuron_count,
            "atlas_measured_soma_count": int(self.manifest["count"]),
            "displayed_soma_count": int(self.manifest["brainCount"]),
            "graph_visible_soma_count": mapped,
            "graph_measured_soma_count": measured,
            "graph_visible_coverage_fraction": mapped / self.neuron_count,
            "graph_without_visible_soma_count": self.neuron_count - mapped,
            "graph_without_measured_soma_count": self.neuron_count - measured,
            "graph_measured_but_view_omitted_count": measured - mapped,
            "coordinate_units": self.manifest["coordinateUnits"],
            "display_transform": self.manifest["displayTransform"],
            "point_meaning": self.manifest["pointMeaning"],
            "brain_view": self.manifest["brainView"],
            "atlas_hashes": dict(ATLAS_EXPORT_SHA256),
        }

    def map_indices(self, indices: list[int]) -> dict[str, Any]:
        self._validate_indices(indices)
        entries = []
        for index in indices:
            row = int(self.atlas_row_by_index[index])
            visible_row = int(self.visible_atlas_row_by_index[index])
            entries.append(
                {
                    "neuron_index": index,
                    # Decimal strings cross JSON without losing integer precision.
                    "body_id": str(int(self.body_ids_by_index[index])),
                    "atlas_row": row if row >= 0 else None,
                    "visible_atlas_row": visible_row if visible_row >= 0 else None,
                    "visible_soma": visible_row >= 0,
                    "omission_reason": (
                        None
                        if visible_row >= 0
                        else "view_excludes_vnc_or_unclassified"
                        if row >= 0
                        else "no_measured_soma_in_atlas"
                    ),
                }
            )
        mapped = sum(bool(entry["visible_soma"]) for entry in entries)
        return {
            "entries": entries,
            "requested_count": len(entries),
            "visible_soma_count": mapped,
            "without_visible_soma_count": len(entries) - mapped,
        }

    def neighborhood(
        self,
        seed_indices: list[int],
        *,
        max_nodes: int,
        max_edges: int,
        min_synapse_count: int,
    ) -> dict[str, Any]:
        """Return strongest visible outgoing relationships from bounded seeds."""

        self._validate_indices(seed_indices)
        seed_set = set(seed_indices)
        candidates: list[tuple[int, int, int]] = []
        relationships_without_visible_soma = 0
        per_seed_limit = max_edges
        for source in seed_indices:
            start = int(self.outgoing_indptr[source])
            stop = int(self.outgoing_indptr[source + 1])
            targets = self.outgoing_targets[start:stop]
            weights = self.outgoing_weights[start:stop]
            eligible = np.flatnonzero(weights >= min_synapse_count)
            if int(self.visible_atlas_row_by_index[source]) < 0:
                relationships_without_visible_soma += int(eligible.size)
                continue
            visible_targets = self.visible_atlas_row_by_index[targets[eligible]] >= 0
            relationships_without_visible_soma += int(
                np.count_nonzero(~visible_targets)
            )
            eligible = eligible[visible_targets]
            if eligible.size > per_seed_limit:
                strongest = np.argpartition(weights[eligible], -per_seed_limit)[
                    -per_seed_limit:
                ]
                eligible = eligible[strongest]
            for local in eligible:
                target = int(targets[local])
                weight = int(weights[local])
                candidates.append((weight, source, target))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        nodes = set(seed_set)
        edges: list[dict[str, int]] = []
        for weight, source, target in candidates:
            additions = {source, target} - nodes
            if len(nodes) + len(additions) > max_nodes:
                continue
            nodes.update(additions)
            edges.append(
                {
                    "source_neuron_index": source,
                    "target_neuron_index": target,
                    "synapse_count": weight,
                }
            )
            if len(edges) >= max_edges:
                break

        visible_nodes = sorted(
            index
            for index in nodes
            if int(self.visible_atlas_row_by_index[index]) >= 0
        )
        node_records = [
            {
                "neuron_index": index,
                "body_id": str(int(self.body_ids_by_index[index])),
                "atlas_row": int(self.visible_atlas_row_by_index[index]),
                "seed": index in seed_set,
            }
            for index in visible_nodes
        ]
        return {
            "nodes": node_records,
            "edges": edges,
            "seed_count": len(seed_indices),
            "seed_without_visible_soma_count": sum(
                int(self.visible_atlas_row_by_index[index]) < 0
                for index in seed_indices
            ),
            "relationships_without_visible_soma_count": (
                relationships_without_visible_soma
            ),
            "candidate_relationship_count": len(candidates),
            "truncated": len(edges) == max_edges or len(visible_nodes) == max_nodes,
            "filter": {
                "direction": "outgoing_from_seed",
                "selection": "strongest_synapse_count_then_stable_index",
                "min_synapse_count": min_synapse_count,
                "max_nodes": max_nodes,
                "max_edges": max_edges,
            },
            "line_meaning": (
                "Directed graph relationship between measured soma markers; straight "
                "lines are not anatomical neurites or synapse locations."
            ),
        }

    def _validate_indices(self, indices: list[int]) -> None:
        if len(indices) != len(set(indices)):
            raise ValueError("neuron indices must be unique")
        if any(index < 0 or index >= self.neuron_count for index in indices):
            raise ValueError(f"neuron index must be below {self.neuron_count}")


def _validate_atlas_manifest(manifest: dict[str, Any], directory: Path) -> None:
    if manifest.get("dataset") != ATLAS_DATASET:
        raise ValueError("atlas dataset identity does not match MaleCNS v1.0")
    if manifest.get("exportSha256") != ATLAS_EXPORT_SHA256:
        raise ValueError(
            "atlas manifest hashes differ from the pinned upstream revision"
        )
    for name, expected in ATLAS_EXPORT_SHA256.items():
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"atlas asset hash mismatch: {name}")
