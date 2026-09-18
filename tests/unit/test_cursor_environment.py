from __future__ import annotations

import pytest

from flybrain_interface.contracts import MotorCommand
from flybrain_interface.environment.cursor import (
    CursorState,
    VirtualCursorEnvironment,
)


def test_virtual_cursor_applies_normalized_motor_command() -> None:
    environment = VirtualCursorEnvironment(step_scale=0.1)

    state = environment.apply(MotorCommand(delta_x=0.5, delta_y=-0.25))

    assert state == CursorState(x=0.55, y=0.475)


def test_virtual_cursor_clamps_to_screen_bounds() -> None:
    environment = VirtualCursorEnvironment(x=0.99, y=0.01, step_scale=0.1)

    state = environment.apply(MotorCommand(delta_x=1.0, delta_y=-1.0))

    assert state == CursorState(x=1.0, y=0.0)


def test_virtual_cursor_rejects_invalid_initial_state() -> None:
    with pytest.raises(ValueError, match="inside"):
        VirtualCursorEnvironment(x=1.1)
