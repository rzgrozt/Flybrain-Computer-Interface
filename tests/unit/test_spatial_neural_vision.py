from __future__ import annotations

import numpy as np

from flybrain_interface.connectome_data.runtime import SignedSourceProjection
from flybrain_interface.experiments.spatial_neural_vision import (
    _condition_frames,
    _conditions,
    _make_r1r6_drive,
    load_config,
)
from flybrain_interface.simulation.config import ShiuLIFConfig


def test_spatial_r1r6_drive_preserves_histamine_sign_and_delay() -> None:
    config = load_config()
    lif_config = ShiuLIFConfig(dt_ms=float(config["timing"]["neural_dt_ms"]))
    source_columns = ("ME_R_col_18_19", "ME_R_col_18_20")
    source_groups = {
        source_columns[0]: (1, 2),
        source_columns[1]: (3,),
    }
    projections = {
        source_columns[0]: SignedSourceProjection(
            source_count=2,
            anatomical_edge_count=2,
            target_indices=np.asarray([10], dtype=np.int64),
            signed_contact_weights=np.asarray([-5.0], dtype=np.float64),
        ),
        source_columns[1]: SignedSourceProjection(
            source_count=1,
            anatomical_edge_count=1,
            target_indices=np.asarray([11], dtype=np.int64),
            signed_contact_weights=np.asarray([-3.0], dtype=np.float64),
        ),
    }
    release_delta = np.zeros((40, 2), dtype=np.float64)
    release_delta[:, 0] = -0.45
    release_delta[:, 1] = 0.45

    drive = _make_r1r6_drive(
        release_delta,
        source_columns,
        source_groups,
        projections,
        config,
        lif_config,
    )

    assert drive is not None
    assert len(drive.channels) == 2
    delay_steps = round(
        float(config["projected_drive"]["axonal_delay_ms"]) / lif_config.dt_ms
    )
    dark, bright = drive.channels
    assert np.all(dark.amplitudes_mv[:delay_steps] == 0.0)
    assert np.all(bright.amplitudes_mv[:delay_steps] == 0.0)
    assert np.all(dark.amplitudes_mv[delay_steps:] < 0.0)
    assert np.all(bright.amplitudes_mv[delay_steps:] > 0.0)
    assert np.all(dark.signed_contact_weights < 0.0)
    assert np.all(bright.signed_contact_weights < 0.0)


def test_spatial_neural_protocol_uses_three_six_three_frames() -> None:
    config = load_config()
    conditions = _conditions(config)
    dark_center = next(
        condition
        for condition in conditions
        if condition["name"] == "dark_horizontal_0.500"
    )
    frames = _condition_frames(dark_center, config)

    background = float(config["screen"]["background_intensity"])
    baseline_frames = int(config["timing"]["baseline_frames"])
    stimulus_frames = int(config["timing"]["stimulus_frames"])
    recovery_start = baseline_frames + stimulus_frames

    assert frames.shape[0] == 12
    assert np.all(frames[:baseline_frames] == background)
    assert np.any(frames[baseline_frames:recovery_start] == 0.0)
    assert np.all(frames[recovery_start:] == background)
