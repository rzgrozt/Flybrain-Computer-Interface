from __future__ import annotations

import numpy as np

from flybrain_interface.experiments.spatial_motor_broad_dn_supervised_direction import (
    _accuracy,
    _feature_scores,
    _fit_model,
    _score_model,
)


def test_feature_scores_prioritize_directional_signal() -> None:
    features = np.asarray(
        [
            [-3.0, 0.0, 1.0],
            [-2.0, 0.1, 1.0],
            [-1.0, -0.1, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.1, 1.0],
            [3.0, -0.1, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    scores = _feature_scores(features, labels, epsilon=1e-9)
    assert int(np.argmax(scores)) == 0


def test_selected_ridge_model_fits_simple_directional_state() -> None:
    features = np.asarray(
        [
            [-3.0, 0.0, 1.0],
            [-2.0, 0.1, 1.0],
            [-1.0, -0.1, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.1, 1.0],
            [3.0, -0.1, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    model = _fit_model(
        features,
        labels,
        top_k=2,
        alpha=0.01,
        epsilon=1e-9,
    )
    assert _accuracy(labels, _score_model(model, features)) == 1.0
