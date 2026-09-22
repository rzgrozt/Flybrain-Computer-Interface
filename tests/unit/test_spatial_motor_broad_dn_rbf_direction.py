from __future__ import annotations

import numpy as np

from flybrain_interface.experiments.spatial_motor_broad_dn_rbf_direction import (
    _accuracy,
    _fit_pipeline,
    _rbf_kernel,
    _score_pipeline,
)


def test_rbf_kernel_has_unit_diagonal() -> None:
    values = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
    kernel = _rbf_kernel(values, values, gamma=0.5)
    assert np.allclose(np.diag(kernel), 1.0)


def test_whitened_rbf_pipeline_fits_simple_direction_state() -> None:
    features = np.asarray(
        [
            [-3.0, 0.0],
            [-2.0, 0.2],
            [-1.0, -0.1],
            [1.0, 0.1],
            [2.0, -0.2],
            [3.0, 0.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    model = _fit_pipeline(
        features,
        labels,
        components=2,
        alpha=0.001,
        gamma=0.1,
        epsilon=1e-9,
    )
    scores = _score_pipeline(model, features)
    assert _accuracy(labels, scores) == 1.0
