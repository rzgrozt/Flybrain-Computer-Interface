from __future__ import annotations

import pytest

from flybrain_interface.experiments.spatial_motor_live_desktop_probe import (
    _expected_action,
)


def test_expected_action_maps_delta_sign_to_direction_class() -> None:
    assert _expected_action(-0.2) == 0
    assert _expected_action(0.2) == 1


def test_expected_action_rejects_zero_delta() -> None:
    with pytest.raises(ValueError):
        _expected_action(0.0)
