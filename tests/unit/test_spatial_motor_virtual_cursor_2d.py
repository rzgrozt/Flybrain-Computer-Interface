from __future__ import annotations

import pytest

from flybrain_interface.experiments.spatial_motor_virtual_cursor_2d import (
    _classifier_load_config,
    _distance,
    _episode_specs,
    _pointer_request,
    _success,
)


def test_episode_specs_parse_full_2d_coordinates() -> None:
    config = {
        "episodes": [
            {
                "start_x": 0.2,
                "start_y": 0.8,
                "target_x": 0.7,
                "target_y": 0.3,
            }
        ]
    }
    assert _episode_specs(config) == ((0.2, 0.8, 0.7, 0.3),)


def test_distance_is_euclidean() -> None:
    assert _distance(0.0, 0.0, 0.3, 0.4) == pytest.approx(0.5)


def test_success_requires_both_axes_within_tolerance() -> None:
    assert _success(0.48, 0.54, 0.5, 0.5, 0.05)
    assert not _success(0.44, 0.54, 0.5, 0.5, 0.05)


def test_pointer_request_requires_explicit_live_permission() -> None:
    spec = {"backend": "uinput", "enabled": True, "pixel_step": 24}
    with pytest.raises(PermissionError):
        _pointer_request(spec, allow_os_pointer=False)
    assert _pointer_request(spec, allow_os_pointer=True) == ("uinput", 24, True)


def test_classifier_load_config_keeps_only_loader_fields() -> None:
    spec = {
        "classifier_backend": "rbf_pca",
        "classifier_config": "config.json",
        "classifier_artifact": "artifact.json",
        "extra": "ignored",
    }
    assert _classifier_load_config(spec) == {
        "classifier_backend": "rbf_pca",
        "classifier_config": "config.json",
        "classifier_artifact": "artifact.json",
    }
