from __future__ import annotations

import numpy as np

from flybrain_interface.experiments.spatial_motor_broad_dn_pca_classifier import (
    _accuracy,
    _fit_pipeline,
    _label,
    _score_pipeline,
)


def test_label_respects_neutral_band() -> None:
    assert _label(0.40, 0.5, 0.07) == 0
    assert _label(0.50, 0.5, 0.07) == 1
    assert _label(0.60, 0.5, 0.07) == 2


def test_pca_pipeline_fits_simple_three_class_structure() -> None:
    features = np.asarray(
        [
            [-3.0, 0.0, 1.0, 2.0],
            [-2.0, 0.1, 1.1, 2.1],
            [0.0, 3.0, 2.0, 1.0],
            [0.1, 2.0, 2.1, 1.1],
            [2.0, 0.0, 3.0, 0.0],
            [3.0, 0.1, 3.1, 0.1],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    model = _fit_pipeline(
        features,
        labels,
        components=3,
        alpha=0.01,
        epsilon=1e-9,
    )
    scores = _score_pipeline(model, features)
    assert _accuracy(labels, scores) == 1.0


def test_pca_pipeline_ignores_train_constant_feature() -> None:
    features = np.asarray(
        [
            [-2.0, 5.0],
            [-1.0, 5.0],
            [0.0, 5.0],
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    model = _fit_pipeline(
        features,
        labels,
        components=2,
        alpha=0.1,
        epsilon=1e-9,
    )
    scores_a = _score_pipeline(model, np.asarray([[1.5, 5.0]]))
    scores_b = _score_pipeline(model, np.asarray([[1.5, 999.0]]))
    assert np.allclose(scores_a, scores_b)
