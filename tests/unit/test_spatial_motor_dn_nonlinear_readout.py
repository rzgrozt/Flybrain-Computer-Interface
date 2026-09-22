from __future__ import annotations

import numpy as np

from flybrain_interface.experiments.spatial_motor_dn_nonlinear_readout import (
    _fit_rbf_ridge,
    _metrics,
    _predict_rbf,
    _rbf_kernel,
    _standardize_apply,
    _standardize_fit,
)


def test_standardization_suppresses_train_constant_features() -> None:
    train = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 10.0],
            [3.0, 10.0],
        ]
    )
    standardized, mean, scale = _standardize_fit(train, epsilon=1e-9)
    test = _standardize_apply(np.asarray([[2.5, 99.0]]), mean, scale)

    assert np.allclose(standardized[:, 1], 0.0)
    assert test[0, 1] == 0.0


def test_rbf_kernel_has_unit_diagonal() -> None:
    values = np.asarray([[0.0], [1.0], [2.0]])
    kernel = _rbf_kernel(values, values, gamma=0.5)

    assert np.allclose(np.diag(kernel), 1.0)
    assert kernel[0, 1] < 1.0
    assert kernel[0, 2] < kernel[0, 1]


def test_rbf_ridge_fits_training_positions() -> None:
    features = np.asarray(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    targets = np.asarray([0.1, 0.4, 0.7, 0.9])
    model = _fit_rbf_ridge(
        features,
        targets,
        alpha=1e-6,
        gamma=1.0,
        epsilon=1e-9,
    )
    predicted = _predict_rbf(model, features)

    assert np.max(np.abs(predicted - targets)) < 1e-4


def test_metrics_reports_side_accuracy() -> None:
    actual = np.asarray([0.2, 0.4, 0.6, 0.8])
    predicted = np.asarray([0.1, 0.7, 0.55, 0.9])
    result = _metrics(actual, predicted)

    assert result["side_accuracy"] == 0.75
