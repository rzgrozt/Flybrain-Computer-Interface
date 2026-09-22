"""Unit coverage for independent dense broad-DN direction validation."""

from __future__ import annotations

import numpy as np
import pytest

from flybrain_interface.experiments.spatial_motor_broad_dn_dense_direction import (
    _evaluate,
    _labels,
)


def test_dense_labels_are_center_relative() -> None:
    assert _labels((0.2, 0.5, 0.8), 0.5).tolist() == [0, 1, 1]


def test_dense_evaluation_uses_independent_validation_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flybrain_interface.experiments.spatial_motor_broad_dn_dense_direction._score_pipeline",
        lambda _model, _features: np.asarray([[0.8, 0.2], [0.1, 0.9]]),
    )
    result = _evaluate({}, np.zeros((2, 1)), (0.1, 0.9), 0.5)
    assert result["accuracy"] == 1.0
    assert [item["correct"] for item in result["pairs"]] == [True, True]
