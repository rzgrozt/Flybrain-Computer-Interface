from __future__ import annotations

import numpy as np

from flybrain_interface.experiments.spatial_motor_dn_direction_classifier import (
    _accuracy,
    _fit_linear_classifier,
    _fit_rbf_classifier,
    _label,
    _loocv,
    _score_linear,
    _score_rbf,
)


def test_direction_labels_are_center_relative() -> None:
    assert _label(0.2, 0.5) == -1
    assert _label(0.5, 0.5) == 0
    assert _label(0.8, 0.5) == 1


def test_accuracy_uses_score_sign() -> None:
    labels = np.asarray([-1, -1, 1, 1])
    scores = np.asarray([-0.5, 0.1, 0.2, 0.9])
    assert _accuracy(labels, scores) == 0.75


def test_linear_loocv_separates_simple_direction_state() -> None:
    features = np.asarray(
        [
            [-3.0, -1.0],
            [-2.0, -0.5],
            [-1.0, -0.2],
            [1.0, 0.2],
            [2.0, 0.5],
            [3.0, 1.0],
        ]
    )
    labels = np.asarray([-1, -1, -1, 1, 1, 1])
    result = _loocv(
        features,
        labels,
        fit=_fit_linear_classifier,
        score=_score_linear,
        params={"alpha": 0.1, "epsilon": 1e-9},
    )
    assert result["accuracy"] == 1.0


def test_rbf_classifier_scores_training_sides() -> None:
    features = np.asarray(
        [
            [-2.0, 0.0],
            [-1.0, 0.2],
            [1.0, -0.2],
            [2.0, 0.0],
        ]
    )
    labels = np.asarray([-1, -1, 1, 1])
    model = _fit_rbf_classifier(
        features,
        labels,
        alpha=0.01,
        gamma=0.1,
        epsilon=1e-9,
    )
    scores = _score_rbf(model, features)
    assert _accuracy(labels, scores) == 1.0
