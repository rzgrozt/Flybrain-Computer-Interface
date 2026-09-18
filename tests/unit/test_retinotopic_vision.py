from __future__ import annotations

import math

import numpy as np
import pytest

from flybrain_interface.connectome_data.optic_columns import OpticColumnRecord
from flybrain_interface.sensory.vision.retinotopic import (
    MeasuredColumnDirection,
    ScreenProjection,
    VirtualScreenConfig,
    column_name_from_pq,
    column_pq,
    encode_screen_sequence,
    expand_screen_encoding_to_neurons,
    project_direction_to_screen,
    sample_screen_intensity,
)


def test_column_pq_roundtrip_matches_official_name() -> None:
    column = OpticColumnRecord(
        column="ME_R_col_18_19",
        medulla_side="R",
        grid_row=18,
        grid_column=19,
        l1_body_id=1,
        r7_body_id=None,
        r8_body_id=None,
        column_type=None,
    )

    assert column_pq(column) == (0, 0)
    assert column_name_from_pq("R", *column_pq(column)) == column.column


def test_virtual_screen_projection_has_expected_orientation() -> None:
    config = VirtualScreenConfig(horizontal_fov_deg=90.0, vertical_fov_deg=90.0)

    center = project_direction_to_screen(
        _direction("ME_R_col_18_19", forward=1.0, right=0.0, up=0.0),
        config,
    )
    right = project_direction_to_screen(
        _direction("ME_R_col_18_20", forward=1.0, right=0.5, up=0.0),
        config,
    )
    up = project_direction_to_screen(
        _direction("ME_R_col_19_19", forward=1.0, right=0.0, up=0.5),
        config,
    )

    assert center.visible
    assert center.u == pytest.approx(0.5)
    assert center.v == pytest.approx(0.5)

    assert right.visible
    assert right.u is not None and right.u > 0.5
    assert right.v == pytest.approx(0.5)

    assert up.visible
    assert up.u == pytest.approx(0.5)
    assert up.v is not None and up.v < 0.5


def test_virtual_screen_rejects_backward_and_outside_rays() -> None:
    config = VirtualScreenConfig(horizontal_fov_deg=90.0, vertical_fov_deg=90.0)

    backward = project_direction_to_screen(
        _direction("ME_R_col_18_19", forward=-1.0, right=0.0, up=0.0),
        config,
    )
    outside = project_direction_to_screen(
        _direction("ME_R_col_18_20", forward=1.0, right=2.0, up=0.0),
        config,
    )

    assert not backward.visible
    assert backward.u is None and backward.v is None
    assert not outside.visible
    assert outside.u is None and outside.v is None


def test_screen_sampling_uses_bilinear_interpolation_and_channel_mean() -> None:
    center = ScreenProjection(column="ME_R_col_18_19", visible=True, u=0.5, v=0.5)
    grayscale = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    rgb = np.stack((grayscale, grayscale, grayscale), axis=2)

    assert sample_screen_intensity(grayscale, center) == pytest.approx(0.5)
    assert sample_screen_intensity(rgb, center) == pytest.approx(0.5)


def test_screen_sampling_returns_none_for_invisible_projection() -> None:
    invisible = ScreenProjection(
        column="ME_R_col_18_19",
        visible=False,
        u=None,
        v=None,
    )

    assert sample_screen_intensity(np.ones((2, 2)), invisible) is None


@pytest.mark.parametrize(
    "horizontal,vertical",
    [
        (0.0, 90.0),
        (180.0, 90.0),
        (90.0, 0.0),
        (90.0, 180.0),
        (math.nan, 90.0),
    ],
)
def test_virtual_screen_config_rejects_invalid_fov(
    horizontal: float,
    vertical: float,
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        VirtualScreenConfig(
            horizontal_fov_deg=horizontal,
            vertical_fov_deg=vertical,
        )


def _direction(
    column: str,
    *,
    forward: float,
    right: float,
    up: float,
) -> MeasuredColumnDirection:
    norm = math.sqrt(forward * forward + right * right + up * up)
    forward /= norm
    right /= norm
    up /= norm
    side = "L" if "_L_" in column else "R"
    parts = column.split("_")
    q = int(parts[-2]) - 18
    p = int(parts[-1]) - 19
    return MeasuredColumnDirection(
        side=side,
        column=column,
        p=p,
        q=q,
        forward=forward,
        right=right,
        up=up,
        elevation_deg=math.degrees(math.asin(up)),
        azimuth_deg=math.degrees(math.atan2(right, forward)),
        source_raw_p=p,
        source_raw_q=q,
    )


def test_screen_sequence_encoding_is_deterministic_and_spatial() -> None:
    directions = (
        _direction("ME_R_col_18_19", forward=1.0, right=0.0, up=0.0),
        _direction("ME_R_col_18_20", forward=1.0, right=0.5, up=0.0),
    )
    config = VirtualScreenConfig(horizontal_fov_deg=90.0, vertical_fov_deg=90.0)
    frames = np.asarray(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=np.float64,
    )

    first = encode_screen_sequence(
        frames,
        directions,
        config,
        frame_rate_hz=60.0,
        start_s=0.1,
    )
    second = encode_screen_sequence(
        frames.copy(),
        directions,
        config,
        frame_rate_hz=60.0,
        start_s=0.1,
    )

    assert first.columns == second.columns
    assert first.frame_sha256 == second.frame_sha256
    np.testing.assert_array_equal(first.intensity_by_frame, second.intensity_by_frame)
    assert first.frame_timestamps_s == pytest.approx((0.1, 0.1 + 1 / 60))
    assert first.intensity_by_frame.shape == (2, 2)
    assert first.intensity_by_frame[0, 1] > first.intensity_by_frame[0, 0]
    assert first.intensity_by_frame[1, 1] < first.intensity_by_frame[1, 0]
    assert not first.intensity_by_frame.flags.writeable


def test_column_encoding_expands_to_neurons_with_adapted_offscreen_background() -> None:
    direction = _direction(
        "ME_R_col_18_19",
        forward=1.0,
        right=0.0,
        up=0.0,
    )
    config = VirtualScreenConfig(horizontal_fov_deg=90.0, vertical_fov_deg=90.0)
    frames = np.asarray([np.ones((2, 2)), np.zeros((2, 2))], dtype=np.float64)
    encoding = encode_screen_sequence(
        frames,
        (direction,),
        config,
        frame_rate_hz=60.0,
    )

    neurons = expand_screen_encoding_to_neurons(
        encoding,
        (
            (10, "ME_R_col_18_19"),
            (4, "ME_R_col_99_99"),
        ),
        background_intensity=0.5,
    )

    assert neurons.neuron_indices == (4, 10)
    assert neurons.neuron_columns == ("ME_R_col_99_99", "ME_R_col_18_19")
    np.testing.assert_allclose(neurons.intensity_by_frame[:, 0], [0.5, 0.5])
    np.testing.assert_allclose(neurons.intensity_by_frame[:, 1], [1.0, 0.0])
    assert not neurons.intensity_by_frame.flags.writeable
