"""Local desktop capture and deterministic target detection."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ScreenCaptureCapability:
    backend: str
    available: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ScreenRegion:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("screen-region origin must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("screen-region size must be positive")


@dataclass(frozen=True, slots=True)
class DetectedTarget:
    x_normalized: float
    y_normalized: float
    pixel_count: int
    bounding_box: tuple[int, int, int, int]


def spectacle_capability(
    executable: str = "spectacle",
) -> ScreenCaptureCapability:
    path = shutil.which(executable)
    if path is None:
        return ScreenCaptureCapability(
            backend="spectacle",
            available=False,
            detail=f"{executable} executable not found",
        )
    return ScreenCaptureCapability(
        backend="spectacle",
        available=True,
        detail=f"{path} available for background capture",
    )


class SpectacleScreenCapture:
    """Capture the local KDE desktop without opening the Spectacle GUI."""

    def __init__(
        self,
        *,
        executable: str = "spectacle",
        include_pointer: bool = False,
        timeout_s: float = 5.0,
    ) -> None:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        capability = spectacle_capability(executable)
        if not capability.available:
            raise RuntimeError(capability.detail)
        self.executable = executable
        self.include_pointer = include_pointer
        self.timeout_s = timeout_s

    def capture(self, region: ScreenRegion | None = None) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "Pillow is required for desktop capture decoding"
            ) from error

        with tempfile.TemporaryDirectory(prefix="flybrain-capture-") as temporary:
            path = Path(temporary) / "frame.png"
            command = [
                self.executable,
                "--background",
                "--nonotify",
                "--fullscreen",
                "--output",
                str(path),
            ]
            if self.include_pointer:
                command.append("--pointer")
            completed = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or "unknown capture error"
                raise RuntimeError(
                    f"Spectacle capture failed ({completed.returncode}): {detail}"
                )
            if not path.is_file():
                raise RuntimeError("Spectacle did not create the requested frame")

            with Image.open(path) as image:
                rgb = image.convert("RGB")
                frame = np.asarray(rgb, dtype=np.uint8).copy()

        return crop_frame(frame, region) if region is not None else frame


@dataclass(frozen=True, slots=True)
class ColorTargetDetector:
    """Find a compact target by RGB color distance."""

    target_rgb: tuple[int, int, int] = (255, 0, 255)
    tolerance: int = 24
    minimum_pixels: int = 16

    def __post_init__(self) -> None:
        if any(channel < 0 or channel > 255 for channel in self.target_rgb):
            raise ValueError("target RGB values must stay inside [0, 255]")
        if self.tolerance < 0 or self.tolerance > 255:
            raise ValueError("color tolerance must stay inside [0, 255]")
        if self.minimum_pixels <= 0:
            raise ValueError("minimum_pixels must be positive")

    def detect(self, frame: np.ndarray) -> DetectedTarget | None:
        _validate_frame(frame)
        target = np.asarray(self.target_rgb, dtype=np.int16)
        delta = np.abs(frame.astype(np.int16) - target)
        mask = np.max(delta, axis=2) <= self.tolerance
        ys, xs = np.nonzero(mask)
        if xs.size < self.minimum_pixels:
            return None

        height, width, _ = frame.shape
        x_denominator = max(1, width - 1)
        y_denominator = max(1, height - 1)
        return DetectedTarget(
            x_normalized=float(np.mean(xs) / x_denominator),
            y_normalized=float(np.mean(ys) / y_denominator),
            pixel_count=int(xs.size),
            bounding_box=(
                int(np.min(xs)),
                int(np.min(ys)),
                int(np.max(xs)),
                int(np.max(ys)),
            ),
        )


def crop_frame(
    frame: np.ndarray,
    region: ScreenRegion,
) -> np.ndarray:
    _validate_frame(frame)
    height, width, _ = frame.shape
    x_stop = region.x + region.width
    y_stop = region.y + region.height
    if x_stop > width or y_stop > height:
        raise ValueError("screen region exceeds captured frame bounds")
    return frame[region.y:y_stop, region.x:x_stop].copy()


def _validate_frame(frame: np.ndarray) -> None:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3)")
    if frame.dtype != np.uint8:
        raise ValueError("frame must use uint8 RGB values")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError("frame dimensions must be positive")
