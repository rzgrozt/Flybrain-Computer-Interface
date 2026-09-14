from __future__ import annotations

from pathlib import Path
from queue import Queue
from typing import Any

from flybrain_interface.panel.controller import PanelController
from flybrain_interface.panel.models import ExperimentConfig


class AliveProcess:
    def is_alive(self) -> bool:
        return True


def test_command_marks_transition_synchronously() -> None:
    controller = PanelController(Path("data"), Path("runs"))
    controller._process = AliveProcess()  # type: ignore[assignment]
    controller._commands = Queue(maxsize=16)

    controller.command("start", ExperimentConfig())

    assert controller.latest["status"] == "starting"
    queued: dict[str, Any] = controller._commands.get_nowait()
    assert queued["action"] == "start"
    assert queued["config"]["subnormal_drive_policy"] == "preserve"
