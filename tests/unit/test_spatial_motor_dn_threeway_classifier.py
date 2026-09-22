from __future__ import annotations

import numpy as np

from flybrain_interface.experiments.spatial_motor_dn_threeway_classifier import (
    _accuracy,
    _fit_linear,
    _fit_rbf,
    _label,
    _scores_linear,
    _scores_rbf,
)


def test_threeway_labels_respect_neutral_band() -> None:
    assert _label(0.42, 0.5, 0.07) == 0
    assert _label(0.46, 0.5, 0.07) == 1
    assert _label(0.50, 0.5, 0.07) == 1
    assert _label(0.54, 0.5, 0.07) == 1
    assert _label(0.58, 0.5, 0.07) == 2


def test_linear_threeway_classifier_fits_simple_clusters() -> None:
    features = np.asarray(
        [
            [-3.0, 0.0],
            [-2.0, 0.1],
            [0.0, 2.0],
            [0.1, 3.0],
            [2.0, 0.0],
            [3.0, 0.1],
        ]
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    model = _fit_linear(features, labels, alpha=0.01, epsilon=1e-9)
    scores = _scores_linear(model, features)
    assert _accuracy(labels, scores) == 1.0


def test_rbf_threeway_classifier_fits_simple_clusters() -> None:
    features = np.asarray(
        [
            [-3.0, 0.0],
            [-2.0, 0.1],
            [0.0, 2.0],
            [0.1, 3.0],
            [2.0, 0.0],
            [3.0, 0.1],
        ]
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    model = _fit_rbf(
        features,
        labels,
        alpha=0.01,
        gamma=0.1,
        epsilon=1e-9,
    )
    scores = _scores_rbf(model, features)
    assert _accuracy(labels, scores) == 1.0
