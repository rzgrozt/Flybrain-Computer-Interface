from __future__ import annotations

from pathlib import Path

import pytest

from flybrain_interface.contracts import MotorCommand
from flybrain_interface.environment.pointer import (
    DryRunPointerAdapter,
    UInputPointerAdapter,
    create_pointer_adapter,
    uinput_capability,
)


def test_dry_run_pointer_converts_motor_command_to_pixels() -> None:
    adapter = DryRunPointerAdapter(pixel_step=25)
    move = adapter.move(MotorCommand(delta_x=-1.0, delta_y=0.5))
    assert move.dx_pixels == -25
    assert move.dy_pixels == 12


def test_uinput_adapter_requires_explicit_enable() -> None:
    with pytest.raises(PermissionError):
        UInputPointerAdapter(enabled=False)


def test_factory_defaults_can_remain_dry_run() -> None:
    adapter = create_pointer_adapter(
        backend="dry_run",
        pixel_step=10,
        enabled=False,
    )
    move = adapter.move(MotorCommand(delta_x=1.0, delta_y=-1.0))
    assert move.dx_pixels == 10
    assert move.dy_pixels == -10


def test_uinput_capability_reports_missing_device(tmp_path: Path) -> None:
    result = uinput_capability(tmp_path / "missing-uinput")
    assert not result.available
    assert result.backend == "uinput"
