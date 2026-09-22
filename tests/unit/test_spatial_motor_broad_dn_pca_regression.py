from __future__ import annotations

import numpy as np

from flybrain_interface.experiments.spatial_motor_broad_dn_pca_regression import (
    _fit_pipeline,
    _metrics,
    _predict_pipeline,
)


def test_pca_regression_fits_simple_ordered_latent() -> None:
    features = np.asarray(
        [
            [-3.0, 0.0],
            [-2.0, 0.1],
            [-1.0, 0.2],
            [0.0, 0.3],
            [1.0, 0.4],
            [2.0, 0.5],
            [3.0, 0.6],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
    model = _fit_pipeline(
        features,
        targets,
        components=2,
        alpha=1e-6,
        epsilon=1e-9,
    )
    predicted = _predict_pipeline(model, features)
    assert np.mean(np.abs(predicted - targets)) < 0.03


def test_regression_metrics_report_side_accuracy() -> None:
    actual = np.asarray([0.2, 0.4, 0.6, 0.8])
    predicted = np.asarray([0.3, 0.45, 0.55, 0.7])
    result = _metrics(actual, predicted, center=0.5)
    assert result["side_accuracy"] == 1.0
