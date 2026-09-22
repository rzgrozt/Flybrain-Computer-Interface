from __future__ import annotations

import pytest

from flybrain_interface.environment.kwin_cursor import (
    CursorPosition,
    KWinCursorBridge,
)


def test_cursor_position_normalizes_workspace_coordinates() -> None:
    position = CursorPosition(
        x=500,
        y=250,
        workspace_width=1001,
        workspace_height=501,
    )
    assert position.x_normalized == pytest.approx(0.5)
    assert position.y_normalized == pytest.approx(0.5)


def test_cursor_position_clamps_out_of_range_coordinates() -> None:
    position = CursorPosition(
        x=9999,
        y=-10,
        workspace_width=100,
        workspace_height=100,
    )
    assert position.x_normalized == 1.0
    assert position.y_normalized == 0.0


def test_kwin_script_reports_cursor_and_workspace_size() -> None:
    bridge = KWinCursorBridge()
    source = bridge._script_source()
    assert "workspace.cursorPos" in source
    assert "workspace.virtualScreenSize" in source
    assert "ReportCursor" in source
    assert bridge.service_name in source


def test_cursor_position_requires_positive_workspace() -> None:
    with pytest.raises(ValueError):
        CursorPosition(
            x=0,
            y=0,
            workspace_width=0,
            workspace_height=100,
        )
