"""The panel must report pointer capability without enabling OS actions."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.routing import APIRoute

from flybrain_interface.environment.pointer import PointerCapability
from flybrain_interface.panel.app import create_app


def test_capability_endpoint_is_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unavailable_pointer() -> PointerCapability:
        return PointerCapability("uinput", False, "disabled for unit test")

    monkeypatch.setattr(
        "flybrain_interface.panel.app.uinput_capability", unavailable_pointer
    )
    app = create_app(tmp_path, tmp_path / "outputs")
    route = next(
        item
        for item in app.routes
        if isinstance(item, APIRoute)
        and item.path == "/api/computer-use/capabilities"
    )
    result = asyncio.run(route.endpoint())
    assert result == {
        "os_pointer": {
            "backend": "uinput",
            "available": False,
            "detail": "disabled for unit test",
            "default_enabled": False,
            "requires_explicit_cli_allow": True,
        }
    }
