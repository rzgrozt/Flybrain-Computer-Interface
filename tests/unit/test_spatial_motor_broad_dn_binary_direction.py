from __future__ import annotations

import numpy as np

from flybrain_interface.experiments.spatial_motor_broad_dn_binary_direction import (
    _accuracy,
    _fit_pipeline,
    _label,
    _score_pipeline,
)


def test_binary_label_is_center_relative() -> None:
    assert _label(0.2, 0.5) == 0
    assert _label(0.8, 0.5) == 1


def test_binary_pipeline_fits_simple_direction_state() -> None:
    features = np.asarray(
        [
            [-3.0, 0.0, 1.0],
            [-2.0, 0.1, 1.1],
            [-1.0, 0.2, 1.2],
            [1.0, 0.2, 2.0],
            [2.0, 0.1, 2.1],
            [3.0, 0.0, 2.2],
        ]
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    model = _fit_pipeline(
        features,
        labels,
        components=3,
        alpha=0.01,
        epsilon=1e-9,
    )
    scores = _score_pipeline(model, features)
    assert _accuracy(labels, scores) == 1.0
