"""Sandbox-local OS pointer adapters for computer-use experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from flybrain_interface.contracts import MotorCommand


@dataclass(frozen=True, slots=True)
class PointerCapability:
    backend: str
    available: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PointerMove:
    dx_pixels: int
    dy_pixels: int


class PointerAdapter(Protocol):
    def move(self, command: MotorCommand) -> PointerMove: ...

    def close(self) -> None: ...


def uinput_capability(device: Path = Path("/dev/uinput")) -> PointerCapability:
    if not device.exists():
        return PointerCapability(
            backend="uinput",
            available=False,
            detail=f"{device} does not exist",
        )
    try:
        with device.open("wb", buffering=0):
            pass
    except OSError as error:
        return PointerCapability(
            backend="uinput",
            available=False,
            detail=f"{device} is not writable: {error}",
        )
    try:
        import evdev  # noqa: F401
    except ImportError:
        return PointerCapability(
            backend="uinput",
            available=False,
            detail="python evdev package is not installed",
        )
    return PointerCapability(
        backend="uinput",
        available=True,
        detail=f"{device} writable and evdev importable",
    )


@dataclass(slots=True)
class DryRunPointerAdapter:
    """Convert motor commands to pixel deltas without touching the OS pointer."""

    pixel_step: int = 20

    def __post_init__(self) -> None:
        if self.pixel_step <= 0:
            raise ValueError("pixel_step must be positive")

    def move(self, command: MotorCommand) -> PointerMove:
        return _command_to_pixels(command, self.pixel_step)

    def close(self) -> None:
        return None


class UInputPointerAdapter:
    """Emit relative pointer motion through Linux uinput.

    Construction is intentionally gated by enabled=True. This prevents accidental
    OS-level movement when experiments are imported or tests are executed.
    """

    def __init__(
        self,
        *,
        pixel_step: int = 20,
        enabled: bool = False,
        device_name: str = "FlyBrain Virtual Pointer",
    ) -> None:
        if not enabled:
            raise PermissionError(
                "OS pointer injection is disabled; pass enabled=True explicitly"
            )
        if pixel_step <= 0:
            raise ValueError("pixel_step must be positive")
        capability = uinput_capability()
        if not capability.available:
            raise RuntimeError(capability.detail)

        from evdev import UInput, ecodes

        self.pixel_step = pixel_step
        self._ecodes = ecodes
        self._device: Any = UInput(
            {
                ecodes.EV_KEY: [ecodes.BTN_LEFT, ecodes.BTN_RIGHT],
                ecodes.EV_REL: [ecodes.REL_X, ecodes.REL_Y],
            },
            name=device_name,
        )

    def move(self, command: MotorCommand) -> PointerMove:
        move = _command_to_pixels(command, self.pixel_step)
        if move.dx_pixels:
            self._device.write(
                self._ecodes.EV_REL,
                self._ecodes.REL_X,
                move.dx_pixels,
            )
        if move.dy_pixels:
            self._device.write(
                self._ecodes.EV_REL,
                self._ecodes.REL_Y,
                move.dy_pixels,
            )
        self._device.syn()
        return move

    def close(self) -> None:
        device = getattr(self, "_device", None)
        if device is not None:
            device.close()
            self._device = None

    def __enter__(self) -> UInputPointerAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def create_pointer_adapter(
    *,
    backend: str,
    pixel_step: int,
    enabled: bool,
) -> PointerAdapter:
    if backend == "dry_run":
        return DryRunPointerAdapter(pixel_step=pixel_step)
    if backend == "uinput":
        return UInputPointerAdapter(
            pixel_step=pixel_step,
            enabled=enabled,
        )
    raise ValueError(f"unsupported pointer backend: {backend}")


def _command_to_pixels(command: MotorCommand, pixel_step: int) -> PointerMove:
    if pixel_step <= 0:
        raise ValueError("pixel_step must be positive")
    return PointerMove(
        dx_pixels=int(round(command.delta_x * pixel_step)),
        dy_pixels=int(round(command.delta_y * pixel_step)),
    )
