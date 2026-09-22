from __future__ import annotations

import numpy as np
import pytest

from flybrain_interface.environment.screen import (
    ColorTargetDetector,
    ScreenRegion,
    SpectacleScreenCapture,
    crop_frame,
    spectacle_capability,
)


def test_crop_frame_returns_requested_region() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[3:7, 5:11] = 123
    cropped = crop_frame(frame, ScreenRegion(x=5, y=3, width=6, height=4))
    assert cropped.shape == (4, 6, 3)
    assert np.all(cropped == 123)


def test_crop_frame_rejects_out_of_bounds_region() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop_frame(frame, ScreenRegion(x=18, y=8, width=4, height=4))


def test_color_target_detector_finds_magenta_centroid() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[20:40, 120:160] = np.asarray([255, 0, 255], dtype=np.uint8)
    detector = ColorTargetDetector(
        target_rgb=(255, 0, 255),
        tolerance=0,
        minimum_pixels=10,
    )
    target = detector.detect(frame)
    assert target is not None
    assert target.pixel_count == 800
    assert target.bounding_box == (120, 20, 159, 39)
    assert target.x_normalized == pytest.approx(139.5 / 199.0)
    assert target.y_normalized == pytest.approx(29.5 / 99.0)


def test_color_target_detector_returns_none_when_too_small() -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[5:7, 5:7] = 255
    detector = ColorTargetDetector(
        target_rgb=(255, 255, 255),
        tolerance=0,
        minimum_pixels=5,
    )
    assert detector.detect(frame) is None


def test_spectacle_capability_can_report_missing_binary() -> None:
    result = spectacle_capability("definitely-not-a-real-spectacle-binary")
    assert not result.available


def test_capture_rejects_nonpositive_timeout() -> None:
    with pytest.raises(ValueError):
        SpectacleScreenCapture(timeout_s=0.0)
