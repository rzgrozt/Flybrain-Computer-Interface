from __future__ import annotations

import numpy as np
import pytest

from flybrain_interface.experiments.spatial_subthreshold_readout import (
    _prediction_metrics,
    _validate_positions,
    fit_ridge,
    predict_ridge,
)


def test_ridge_decodes_held_out_linear_position() -> None:
    train_position = np.asarray([0.15, 0.325, 0.5, 0.675, 0.85])
    train_features = np.column_stack(
        (
            train_position,
            2.0 * train_position,
            np.ones_like(train_position),
        )
    )
    test_position = np.asarray([0.2375, 0.4125, 0.5875, 0.7625])
    test_features = np.column_stack(
        (
            test_position,
            2.0 * test_position,
            np.ones_like(test_position),
        )
    )

    model = fit_ridge(
        train_features,
        train_position,
        alpha=1.0,
        epsilon=1e-12,
    )
    prediction = predict_ridge(model, test_features)
    metrics = _prediction_metrics(test_position, prediction)

    assert model.active_feature_count == 2
    assert metrics["mae"] < 0.04
    assert metrics["correlation"] is not None
    assert metrics["correlation"] > 0.99
    assert metrics["monotonic"]


def test_ridge_zero_variance_features_predict_training_mean() -> None:
    features = np.zeros((5, 3), dtype=np.float64)
    target = np.asarray([0.15, 0.325, 0.5, 0.675, 0.85])

    model = fit_ridge(features, target, alpha=1.0, epsilon=1e-12)
    prediction = predict_ridge(model, np.zeros((4, 3), dtype=np.float64))

    assert model.active_feature_count == 0
    np.testing.assert_allclose(prediction, 0.5)


def test_readout_positions_must_be_sorted_disjoint_and_in_bounds() -> None:
    _validate_positions(
        (0.15, 0.5, 0.85),
        (0.325, 0.675),
    )

    with pytest.raises(ValueError, match="disjoint"):
        _validate_positions((0.15, 0.5, 0.85), (0.5, 0.675))
    with pytest.raises(ValueError, match="inside"):
        _validate_positions((0.15, 0.5, 0.85), (0.675, 1.0))
    with pytest.raises(ValueError, match="sorted"):
        _validate_positions((0.5, 0.15, 0.85), (0.325, 0.675))
