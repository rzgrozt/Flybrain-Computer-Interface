from __future__ import annotations

from pathlib import Path
from queue import Queue
from typing import Any

import pytest

from flybrain_interface.panel.controller import PanelController
from flybrain_interface.panel.models import ExperimentConfig


class AliveProcess:
    def is_alive(self) -> bool:
        return True


class DeadProcess:
    exitcode = 9

    def __init__(self) -> None:
        self.join_timeout: float | None = None
        self.closed = False

    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        self.join_timeout = timeout

    def close(self) -> None:
        self.closed = True


class ReplacementProcess:
    exitcode = None

    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started


class TrackingQueue(Queue[Any]):
    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)
        self.closed = False
        self.cancelled = False

    def close(self) -> None:
        self.closed = True

    def cancel_join_thread(self) -> None:
        self.cancelled = True


class ReplacementContext:
    def __init__(self) -> None:
        self.queues: list[TrackingQueue] = []
        self.process = ReplacementProcess()

    def Queue(self, maxsize: int) -> TrackingQueue:  # noqa: N802
        channel = TrackingQueue(maxsize)
        self.queues.append(channel)
        return channel

    def Process(self, **kwargs: object) -> ReplacementProcess:  # noqa: N802
        return self.process


def test_command_marks_transition_synchronously() -> None:
    controller = PanelController(Path("data"), Path("runs"))
    controller._process = AliveProcess()  # type: ignore[assignment]
    controller._commands = Queue(maxsize=16)

    controller.command("start", ExperimentConfig())

    assert controller.latest["status"] == "starting"
    queued: dict[str, Any] = controller._commands.get_nowait()
    assert queued["action"] == "start"
    assert queued["config"]["subnormal_drive_policy"] == "preserve"


@pytest.mark.parametrize("action", ["start", "reset"])
def test_explicit_start_or_reset_recovers_dead_worker(action: str) -> None:
    controller = PanelController(Path("data"), Path("runs"))
    dead = DeadProcess()
    old_commands = TrackingQueue()
    old_telemetry = TrackingQueue()
    replacement = ReplacementContext()
    controller._process = dead  # type: ignore[assignment]
    controller._commands = old_commands
    controller._telemetry = old_telemetry
    controller._context = replacement  # type: ignore[assignment]
    controller.latest = {"kind": "error", "status": "failed"}

    recovered = controller.command(
        action, ExperimentConfig() if action == "start" else None
    )

    assert recovered is True
    assert dead.join_timeout == 0.2
    assert dead.closed
    assert old_commands.closed and old_commands.cancelled
    assert old_telemetry.closed and old_telemetry.cancelled
    assert replacement.process.started
    if action == "start":
        assert replacement.queues[0].get_nowait()["action"] == "start"
        assert controller.latest["status"] == "starting"
    else:
        assert replacement.queues[0].empty()
        assert controller.latest["status"] == "idle"
