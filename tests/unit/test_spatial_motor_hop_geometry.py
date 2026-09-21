from __future__ import annotations

import numpy as np

from flybrain_interface.experiments.spatial_motor_hop_geometry import (
    _decode_feature_block,
    _first_failed_hop,
    _side_accuracy,
)


def _config() -> dict[str, object]:
    return {
        "ridge": {
            "alpha": 0.0001,
            "standardization_epsilon": 1e-9,
        },
        "gates": {
            "maximum_test_mae": 0.2,
            "minimum_test_side_accuracy": 1.0,
        },
    }


def test_side_accuracy_checks_center_relative_direction() -> None:
    actual = np.asarray([0.325, 0.675])
    predicted = np.asarray([0.4, 0.8])
    assert _side_accuracy(actual, predicted) == 1.0
    assert _side_accuracy(actual, np.asarray([0.6, 0.4])) == 0.0


def test_decode_feature_block_generalizes_linear_position_feature() -> None:
    train = (0.15, 0.5, 0.85)
    test = (0.325, 0.675)
    ordered = (*train, *test)
    features = np.asarray([[position] for position in ordered], dtype=np.float64)

    result = _decode_feature_block(features, train, test, _config())

    assert result["active_feature_count"] == 1
    assert result["test"]["mae"] < 0.01
    assert result["test"]["side_accuracy"] == 1.0


def test_first_failed_hop_returns_earliest_generalization_break() -> None:
    layers = [
        {"hop": 1, "passes_generalization": True},
        {"hop": 2, "passes_generalization": False},
        {"hop": 3, "passes_generalization": False},
    ]
    assert _first_failed_hop(layers) == 2
