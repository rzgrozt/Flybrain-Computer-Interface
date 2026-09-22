from __future__ import annotations

import numpy as np
import pytest

from flybrain_interface.experiments.spatial_motor_virtual_cursor_closed_loop import (
    _axis_command,
    _choose_action,
    _class_names,
    _episode_specs,
    _relative_position,
)


def test_relative_position_recenters_target_around_cursor() -> None:
    spec = {"center": 0.5, "minimum": 0.1, "maximum": 0.9}
    assert _relative_position(0.7, 0.5, spec) == 0.7
    assert _relative_position(0.7, 0.6, spec) == 0.6
    assert _relative_position(0.3, 0.4, spec) == pytest.approx(0.4)


def test_relative_position_clamps_to_visual_range() -> None:
    spec = {"center": 0.5, "minimum": 0.1, "maximum": 0.9}
    assert _relative_position(0.0, 0.9, spec) == 0.1
    assert _relative_position(1.0, 0.1, spec) == 0.9


def test_hysteresis_blocks_low_margin_reversal() -> None:
    scores = np.asarray([0.47, 0.53])
    assert _choose_action(
        scores,
        0,
        reversal_margin_threshold=0.15,
    ) == 0


def test_hysteresis_allows_confident_reversal() -> None:
    assert _choose_action(
        np.asarray([0.1, 0.9]),
        0,
        reversal_margin_threshold=0.15,
    ) == 1


def test_episode_specs_support_explicit_start_target_pairs() -> None:
    config = {
        "cursor": {"step_scale": 0.1},
        "episodes": [
            {"start_x": 0.25, "target_x": 0.8},
            {"start_x": 0.75, "target_x": 0.2},
        ],
    }
    assert _episode_specs(config) == ((0.25, 0.8), (0.75, 0.2))


def test_episode_specs_keep_legacy_target_list_behavior() -> None:
    config = {
        "cursor": {"start_x": 0.5},
        "target_positions": [0.2, 0.8],
    }
    assert _episode_specs(config) == ((0.5, 0.2), (0.5, 0.8))


def test_episode_specs_support_vertical_coordinates() -> None:
    config = {
        "cursor": {"step_scale": 0.1},
        "episodes": [
            {"start_y": 0.25, "target_y": 0.8},
            {"start_y": 0.75, "target_y": 0.2},
        ],
    }
    assert _episode_specs(config, "vertical") == ((0.25, 0.8), (0.75, 0.2))


def test_vertical_axis_command_and_names() -> None:
    command = _axis_command("vertical", -1.0)
    assert command.delta_x == 0.0
    assert command.delta_y == -1.0
    assert _class_names("vertical") == ("UP", "DOWN")
