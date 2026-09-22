"""Process lifecycle and bounded local command channel."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

from flybrain_interface.panel.models import ExperimentConfig
from flybrain_interface.panel.worker import worker_main


class PanelController:
    """Own exactly one worker and recover cleanly from worker failure."""

    def __init__(self, data_directory: Path, output_directory: Path) -> None:
        self.data_directory = data_directory.resolve()
        self.output_directory = output_directory.resolve()
        self._context = mp.get_context("spawn")
        self._commands: Any = None
        self._telemetry: Any = None
        self._process: BaseProcess | None = None
        self.latest: dict[str, Any] = {"kind": "status", "status": "offline"}

    def start_worker(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._dispose_worker()
        self._commands = self._context.Queue(maxsize=16)
        self._telemetry = self._context.Queue(maxsize=1)
        self._process = self._context.Process(
            target=worker_main,
            args=(
                self._commands,
                self._telemetry,
                str(self.data_directory),
                str(self.output_directory),
            ),
            name="flybrain-simulation-worker",
        )
        process = self._process
        assert process is not None
        process.start()
        self.latest = {"kind": "status", "status": "idle"}

    def command(self, action: str, config: ExperimentConfig | None = None) -> bool:
        recovered = False
        if self._process is None or not self._process.is_alive():
            if action not in {"start", "reset"}:
                self.latest = {
                    "kind": "error",
                    "status": "failed",
                    "error": "simulation worker is not running",
                }
                raise RuntimeError("simulation worker is not running")
            self.start_worker()
            recovered = True
            if action == "reset":
                return recovered
        payload: dict[str, Any] = {"action": action}
        if config is not None:
            payload["config"] = config.model_dump(mode="json")
        try:
            self._commands.put_nowait(payload)
        except queue.Full as error:
            raise RuntimeError("worker command channel is full") from error
        transitional = {
            "start": "starting",
            "pause": "pausing",
            "resume": "resuming",
            "reset": "resetting",
        }
        if action in transitional:
            self.latest = {
                **self.latest,
                "kind": "status",
                "status": transitional[action],
            }
        return recovered

    def observe_pathway(self, observation: dict[str, Any] | None) -> None:
        """Queue a server-resolved, bounded observation; no numerical state mutation."""
        if self._process is None or not self._process.is_alive():
            raise RuntimeError("simulation worker is not running")
        try:
            self._commands.put_nowait(
                {"action": "observe_pathway", "observation": observation}
            )
        except queue.Full as error:
            raise RuntimeError("worker command channel is full") from error

    def poll_latest(self) -> dict[str, Any] | None:
        if self._process is not None and not self._process.is_alive():
            exit_code = self._process.exitcode
            if self.latest.get("status") not in {"stopped", "failed"}:
                self.latest = {
                    "kind": "error",
                    "status": "failed",
                    "error": f"simulation worker exited unexpectedly ({exit_code})",
                }
                return self.latest
        result = None
        while self._telemetry is not None:
            try:
                result = self._telemetry.get_nowait()
            except queue.Empty:
                break
        if result is not None:
            self.latest = result
        return result

    def wait_for_status(self, status: str, timeout_s: float = 10.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            event = self.poll_latest()
            if event is not None and event.get("status") == status:
                return event
            time.sleep(0.01)
        raise TimeoutError(f"worker did not reach {status}")

    def shutdown(self) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive():
            try:
                self.command("shutdown")
                process.join(timeout=5.0)
            except RuntimeError:
                pass
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
        if process.is_alive():
            self.latest = {
                "kind": "error",
                "status": "failed",
                "error": "simulation worker did not stop",
            }
            return
        self._dispose_worker()
        self.latest = {"kind": "status", "status": "stopped"}

    def _dispose_worker(self) -> None:
        process = self._process
        if process is not None:
            if process.is_alive():
                return
            process.join(timeout=0.2)
            process.close()
        for channel in (self._commands, self._telemetry):
            if channel is not None:
                channel.close()
                channel.cancel_join_thread()
        self._commands = None
        self._telemetry = None
        self._process = None
