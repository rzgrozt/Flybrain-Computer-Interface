from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from flybrain_interface.connectome_data.optic_columns import OpticColumnRecord
from flybrain_interface.connectome_data.runtime import SignedSourceProjection
from flybrain_interface.experiments.spatial_graded_lamina import (
    CartridgeSources,
    _make_lamina_drive,
    load_config,
    resolve_cartridge_sources,
)
from flybrain_interface.sensory.vision import (
    GradedEarlyVisionConfig,
    simulate_graded_early_vision_channels,
)
from flybrain_interface.simulation.config import ShiuLIFConfig


def test_cartridge_resolution_uses_unique_direct_lamina_targets() -> None:
    graph = _cartridge_graph()
    columns = (
        OpticColumnRecord(
            column="ME_R_col_18_19",
            medulla_side="R",
            grid_row=18,
            grid_column=19,
            l1_body_id=200,
            r7_body_id=None,
            r8_body_id=None,
            column_type=None,
        ),
    )

    result = resolve_cartridge_sources(
        graph,  # type: ignore[arg-type]
        {"ME_R_col_18_19": (0, 1)},
        columns,
    )

    assert result == (
        CartridgeSources(
            column="ME_R_col_18_19",
            r1r6_indices=(0, 1),
            l1_index=2,
            l2_index=3,
            l3_index=4,
        ),
    )


def test_cartridge_resolution_rejects_competing_lamina_target() -> None:
    graph = _cartridge_graph(competing_l2=True)
    columns = (
        OpticColumnRecord(
            column="ME_R_col_18_19",
            medulla_side="R",
            grid_row=18,
            grid_column=19,
            l1_body_id=200,
            r7_body_id=None,
            r8_body_id=None,
            column_type=None,
        ),
    )

    with pytest.raises(RuntimeError, match="competing direct L2"):
        resolve_cartridge_sources(
            graph,  # type: ignore[arg-type]
            {"ME_R_col_18_19": (0, 1)},
            columns,
        )


def test_graded_lamina_drive_preserves_sign_policy_and_delay() -> None:
    config = load_config()
    model = config["graded_model"]
    graded_config = GradedEarlyVisionConfig(
        frame_rate_hz=float(config["timing"]["frame_rate_hz"]),
        neural_dt_ms=float(config["timing"]["neural_dt_ms"]),
        background_luminance=float(model["background_luminance"]),
        tonic_release_fraction=float(model["tonic_release_fraction"]),
        photoreceptor_tau_ms=float(model["photoreceptor_tau_ms"]),
        transient_fast_tau_ms=float(model["transient_fast_tau_ms"]),
        transient_slow_tau_ms=float(model["transient_slow_tau_ms"]),
        l3_tau_ms=float(model["l3_tau_ms"]),
        l1_gain_mv=float(model["l1_gain_mv"]),
        l2_gain_mv=float(model["l2_gain_mv"]),
        l2_slow_weight=float(model["l2_slow_weight"]),
        l3_gain_mv=float(model["l3_gain_mv"]),
    )
    intensity = np.asarray(
        [[0.5], [0.5], [0.0], [0.0], [0.5], [0.5]],
        dtype=np.float64,
    )
    graded = simulate_graded_early_vision_channels(intensity, graded_config)
    cartridge = CartridgeSources(
        column="ME_R_col_18_19",
        r1r6_indices=(0,),
        l1_index=2,
        l2_index=3,
        l3_index=4,
    )
    projections = {
        ("L1", cartridge.column): SignedSourceProjection(
            source_count=1,
            anatomical_edge_count=1,
            target_indices=np.asarray([10], dtype=np.int64),
            signed_contact_weights=np.asarray([-2.0], dtype=np.float64),
        ),
        ("L2", cartridge.column): SignedSourceProjection(
            source_count=1,
            anatomical_edge_count=1,
            target_indices=np.asarray([11], dtype=np.int64),
            signed_contact_weights=np.asarray([3.0], dtype=np.float64),
        ),
        ("L3", cartridge.column): SignedSourceProjection(
            source_count=1,
            anatomical_edge_count=1,
            target_indices=np.asarray([12], dtype=np.int64),
            signed_contact_weights=np.asarray([4.0], dtype=np.float64),
        ),
    }
    lif_config = ShiuLIFConfig(dt_ms=float(config["timing"]["neural_dt_ms"]))

    drive = _make_lamina_drive(
        graded,
        (cartridge,),
        projections,
        config,
        lif_config,
    )

    assert drive is not None
    assert {channel.label for channel in drive.channels} == {
        "l1:ME_R_col_18_19",
        "l2:ME_R_col_18_19",
        "l3:ME_R_col_18_19",
    }
    delay_steps = round(
        float(config["projected_drive"]["axonal_delay_ms"]) / lif_config.dt_ms
    )
    for channel in drive.channels:
        assert np.all(channel.amplitudes_mv[:delay_steps] == 0.0)
        assert np.any(channel.amplitudes_mv[delay_steps:] != 0.0)

    by_label = {channel.label: channel for channel in drive.channels}
    assert np.all(by_label["l1:ME_R_col_18_19"].signed_contact_weights < 0.0)
    assert np.all(by_label["l2:ME_R_col_18_19"].signed_contact_weights > 0.0)
    assert np.all(by_label["l3:ME_R_col_18_19"].signed_contact_weights > 0.0)


def _cartridge_graph(*, competing_l2: bool = False) -> SimpleNamespace:
    body_ids = [100, 101, 200, 300, 400, 301]
    types = ["R1-R6", "R1-R6", "L1", "L2", "L3", "L2"]
    table = pa.table(
        {
            "neuron_index": list(range(len(body_ids))),
            "body_id": body_ids,
            "type": types,
        }
    )
    edges: dict[int, list[tuple[int, int]]] = {
        0: [(2, 10), (3, 7), (4, 2)],
        1: [(2, 11), (3, 8), (4, 3)],
    }
    if competing_l2:
        edges[1].append((5, 1))

    targets: list[int] = []
    weights: list[int] = []
    indptr = [0]
    for source in range(len(body_ids)):
        for target, weight in edges.get(source, []):
            targets.append(target)
            weights.append(weight)
        indptr.append(len(targets))

    return SimpleNamespace(
        catalog=SimpleNamespace(table=table),
        outgoing_indptr=np.asarray(indptr, dtype=np.int64),
        target_indices=np.asarray(targets, dtype=np.int32),
        outgoing_synapse_counts=np.asarray(weights, dtype=np.int32),
    )
