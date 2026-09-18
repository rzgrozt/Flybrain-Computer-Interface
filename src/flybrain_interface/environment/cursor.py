"""Deterministic normalized cursor environment for closed-loop experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from flybrain_interface.contracts import MotorCommand


@dataclass(frozen=True, slots=True)
class CursorState:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not (isfinite(self.x) and isfinite(self.y)):
            raise ValueError("cursor coordinates must be finite")
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ValueError("cursor coordinates must stay inside [0, 1]")


@dataclass(slots=True)
class VirtualCursorEnvironment:
    """Apply neural motor commands to a sandbox-only normalized cursor."""

    x: float = 0.5
    y: float = 0.5
    step_scale: float = 0.02

    def __post_init__(self) -> None:
        if not isfinite(self.step_scale) or self.step_scale <= 0.0:
            raise ValueError("step_scale must be finite and positive")
        self._validate_position()

    @property
    def state(self) -> CursorState:
        return CursorState(self.x, self.y)

    def reset(self, *, x: float = 0.5, y: float = 0.5) -> CursorState:
        self.x = x
        self.y = y
        self._validate_position()
        return self.state

    def apply(self, command: MotorCommand) -> CursorState:
        self.x = self._clamp(self.x + command.delta_x * self.step_scale)
        self.y = self._clamp(self.y + command.delta_y * self.step_scale)
        return self.state

    def _validate_position(self) -> None:
        if not (isfinite(self.x) and isfinite(self.y)):
            raise ValueError("cursor coordinates must be finite")
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ValueError("cursor coordinates must stay inside [0, 1]")

    @staticmethod
    def _clamp(value: float) -> float:
        return min(1.0, max(0.0, value))
