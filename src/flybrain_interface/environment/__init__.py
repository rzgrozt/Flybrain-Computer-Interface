"""Local environment interfaces."""

from flybrain_interface.environment.cursor import CursorState, VirtualCursorEnvironment
from flybrain_interface.environment.kwin_cursor import CursorPosition, KWinCursorBridge
from flybrain_interface.environment.pointer import (
    DryRunPointerAdapter,
    PointerCapability,
    PointerMove,
    UInputPointerAdapter,
    create_pointer_adapter,
    uinput_capability,
)
from flybrain_interface.environment.screen import (
    ColorTargetDetector,
    DetectedTarget,
    ScreenCaptureCapability,
    ScreenRegion,
    SpectacleScreenCapture,
    crop_frame,
    spectacle_capability,
)

__all__ = [
    "ColorTargetDetector",
    "CursorPosition",
    "CursorState",
    "DetectedTarget",
    "DryRunPointerAdapter",
    "KWinCursorBridge",
    "PointerCapability",
    "PointerMove",
    "ScreenCaptureCapability",
    "ScreenRegion",
    "SpectacleScreenCapture",
    "UInputPointerAdapter",
    "VirtualCursorEnvironment",
    "create_pointer_adapter",
    "crop_frame",
    "spectacle_capability",
    "uinput_capability",
]
