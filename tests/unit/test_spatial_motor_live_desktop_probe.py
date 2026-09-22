from __future__ import annotations

import pytest

from flybrain_interface.experiments.spatial_motor_live_desktop_probe import (
    _expected_action,
    _rgb_triplet,
)


def test_expected_action_maps_delta_sign_to_direction_class() -> None:
    assert _expected_action(-0.2) == 0
    assert _expected_action(0.2) == 1


def test_expected_action_rejects_zero_delta() -> None:
    with pytest.raises(ValueError):
        _expected_action(0.0)


def test_rgb_triplet_requires_exactly_three_channels() -> None:
    assert _rgb_triplet([255, 0, 255]) == (255, 0, 255)
    with pytest.raises(ValueError, match="exactly three"):
        _rgb_triplet([255, 0])
