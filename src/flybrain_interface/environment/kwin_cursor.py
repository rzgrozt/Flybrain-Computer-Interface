"""KWin cursor-position bridge for Plasma Wayland sessions."""

# ruff: noqa: F821

import asyncio
import os
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CursorPosition:
    x: int
    y: int
    workspace_width: int
    workspace_height: int

    def __post_init__(self) -> None:
        if self.workspace_width <= 0 or self.workspace_height <= 0:
            raise ValueError("workspace dimensions must be positive")

    @property
    def x_normalized(self) -> float:
        return min(
            1.0,
            max(0.0, self.x / max(1, self.workspace_width - 1)),
        )

    @property
    def y_normalized(self) -> float:
        return min(
            1.0,
            max(0.0, self.y / max(1, self.workspace_height - 1)),
        )


class KWinCursorBridge:
    """Receive workspace.cursorPos from a temporary KWin script over local DBus."""

    def __init__(self) -> None:
        token = uuid.uuid4().hex[:10]
        self.service_name = f"org.flybrain.CursorBridge.p{os.getpid()}.t{token}"
        self.object_path = "/CursorBridge"
        self.interface_name = "org.flybrain.CursorBridge"
        self.plugin_name = f"flybrain-cursor-{os.getpid()}-{token}"
        self._latest: CursorPosition | None = None
        self._lock = threading.Lock()
        self._position_ready = threading.Event()
        self._service_ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_stop: asyncio.Event | None = None
        self._service_error: BaseException | None = None
        self._script_path: Path | None = None
        self._script_id: int | None = None

    @property
    def latest(self) -> CursorPosition | None:
        with self._lock:
            return self._latest

    def start(self, *, timeout_s: float = 5.0) -> CursorPosition:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if self._thread is not None:
            raise RuntimeError("KWin cursor bridge is already started")

        self._thread = threading.Thread(
            target=self._thread_main,
            name="flybrain-kwin-cursor-dbus",
            daemon=True,
        )
        self._thread.start()
        if not self._service_ready.wait(timeout_s):
            self.close()
            raise TimeoutError("cursor DBus service did not become ready")
        if self._service_error is not None:
            error = self._service_error
            self.close()
            raise RuntimeError("cursor DBus service failed") from error

        try:
            self._load_kwin_script()
        except Exception:
            self.close()
            raise

        if not self._position_ready.wait(timeout_s):
            self.close()
            raise TimeoutError("KWin script did not report cursor position")
        position = self.latest
        if position is None:
            self.close()
            raise RuntimeError("cursor position report was empty")
        return position

    def wait_for_position(self, *, timeout_s: float = 1.0) -> CursorPosition:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if not self._position_ready.wait(timeout_s):
            raise TimeoutError("cursor position is not available")
        position = self.latest
        if position is None:
            raise RuntimeError("cursor position report was empty")
        return position

    def close(self) -> None:
        if self._script_id is not None:
            subprocess.run(
                [
                    "qdbus6",
                    "org.kde.KWin",
                    "/Scripting",
                    "org.kde.kwin.Scripting.unloadScript",
                    self.plugin_name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                check=False,
            )
            self._script_id = None

        loop = self._loop
        stop = self._async_stop
        if loop is not None and stop is not None:
            loop.call_soon_threadsafe(stop.set)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None
        self._loop = None
        self._async_stop = None

        path = self._script_path
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._script_path = None

    def __enter__(self) -> "KWinCursorBridge":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            self._service_error = error
            self._service_ready.set()

    async def _serve(self) -> None:
        try:
            from dbus_next.aio import MessageBus  # type: ignore[attr-defined]
            from dbus_next.service import ServiceInterface, method
        except ImportError as error:
            raise RuntimeError(
                "dbus-next is required for the KWin cursor bridge"
            ) from error

        owner = self

        class CursorInterface(ServiceInterface):
            def __init__(self) -> None:
                super().__init__(owner.interface_name)

            @method()  # type: ignore[untyped-decorator]
            def ReportCursor(
                self,
                x: "i",  # type: ignore[name-defined]
                y: "i",  # type: ignore[name-defined]
                width: "i",  # type: ignore[name-defined]
                height: "i",  # type: ignore[name-defined]
            ) -> None:
                position = CursorPosition(
                    x=int(x),
                    y=int(y),
                    workspace_width=int(width),
                    workspace_height=int(height),
                )
                with owner._lock:
                    owner._latest = position
                owner._position_ready.set()

        bus = await MessageBus().connect()
        await bus.request_name(self.service_name)
        bus.export(self.object_path, CursorInterface())

        self._loop = asyncio.get_running_loop()
        self._async_stop = asyncio.Event()
        self._service_ready.set()
        await self._async_stop.wait()
        bus.disconnect()  # type: ignore[no-untyped-call]

    def _load_kwin_script(self) -> None:
        script = self._script_source()
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".js",
            prefix="flybrain-cursor-",
            delete=False,
            encoding="utf-8",
        )
        try:
            handle.write(script)
            handle.flush()
        finally:
            handle.close()
        self._script_path = Path(handle.name)

        load = subprocess.run(
            [
                "qdbus6",
                "org.kde.KWin",
                "/Scripting",
                "org.kde.kwin.Scripting.loadScript",
                str(self._script_path),
                self.plugin_name,
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if load.returncode != 0:
            raise RuntimeError(
                f"KWin loadScript failed: {load.stderr.strip()}"
            )
        try:
            script_id = int(load.stdout.strip())
        except ValueError as error:
            raise RuntimeError(
                f"KWin loadScript returned invalid id: {load.stdout!r}"
            ) from error
        if script_id < 0:
            raise RuntimeError(f"KWin loadScript failed with id {script_id}")
        self._script_id = script_id

        run = subprocess.run(
            [
                "qdbus6",
                "org.kde.KWin",
                f"/Scripting/Script{script_id}",
                "org.kde.kwin.Script.run",
            ],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if run.returncode != 0:
            raise RuntimeError(
                f"KWin Script.run failed: {run.stderr.strip()}"
            )

    def _script_source(self) -> str:
        return (
            "function reportCursor() {\n"
            "  const pos = workspace.cursorPos;\n"
            "  const size = workspace.virtualScreenSize;\n"
            f'  callDBus("{self.service_name}", '
            f'"{self.object_path}", "{self.interface_name}", '
            '"ReportCursor", Math.round(pos.x), Math.round(pos.y), '
            "Math.round(size.width), Math.round(size.height));\n"
            "}\n"
            "workspace.cursorPosChanged.connect(reportCursor);\n"
            "workspace.virtualScreenSizeChanged.connect(reportCursor);\n"
            "reportCursor();\n"
        )
