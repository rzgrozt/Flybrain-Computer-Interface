from __future__ import annotations

import pytest

from flybrain_interface.experiments.spatial_motor_hop2_causal import (
    _variant_indices,
    _variant_summary,
)


def test_variant_indices_apply_only_requested_changes() -> None:
    baseline = {1, 2, 3, 4}
    remove = {2, 8}
    add = {5, 6}

    assert _variant_indices(
        baseline,
        remove_indices=remove,
        add_indices=add,
        variant="baseline",
    ) == (1, 2, 3, 4)
    assert _variant_indices(
        baseline,
        remove_indices=remove,
        add_indices=add,
        variant="remove_disruptive_visual_projection",
    ) == (1, 3, 4)
    assert _variant_indices(
        baseline,
        remove_indices=remove,
        add_indices=add,
        variant="add_preserving_cb_intrinsic",
    ) == (1, 2, 3, 4, 5, 6)
    assert _variant_indices(
        baseline,
        remove_indices=remove,
        add_indices=add,
        variant="combined",
    ) == (1, 3, 4, 5, 6)


def test_variant_summary_counts_hop2_and_target_passes() -> None:
    targets = [
        {
            "hop2": {"passes_generalization": True, "best_test_mae": 0.1},
            "target": {"passes_generalization": False, "best_test_mae": 0.3},
        },
        {
            "hop2": {"passes_generalization": False, "best_test_mae": 0.2},
            "target": {"passes_generalization": True, "best_test_mae": 0.1},
        },
    ]
    result = _variant_summary(targets)

    assert result["target_count"] == 2
    assert result["hop2_pass_count"] == 1
    assert result["target_pass_count"] == 1
    assert result["mean_hop2_best_test_mae"] == pytest.approx(0.15)
    assert result["mean_target_best_test_mae"] == pytest.approx(0.2)
