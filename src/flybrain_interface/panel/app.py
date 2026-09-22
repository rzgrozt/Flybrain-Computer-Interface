"""FastAPI presentation layer for the local FlyBrain experiment panel."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from flybrain_interface.environment.pointer import uinput_capability
from flybrain_interface.experiments.sensory_descending import panel_presets
from flybrain_interface.experiments.visual_pathway import visual_panel_preset
from flybrain_interface.panel.anatomy import ATLAS_DIRECTORY, AtlasJoin
from flybrain_interface.panel.controller import PanelController
from flybrain_interface.panel.models import (
    ActionResponse,
    AnatomyMapRequest,
    ExperimentConfig,
    NeighborhoodRequest,
    PathwayObserveRequest,
)
from flybrain_interface.panel.pathway_observation import resolve_observation
from flybrain_interface.panel.pathways_v2 import pathway_catalog, target_paths
from flybrain_interface.panel.recordings import (
    RECORDINGS,
    cached_recording,
    recording_catalog,
)
from flybrain_interface.panel.telemetry_v2 import project_telemetry

STATIC_DIRECTORY = Path(__file__).with_name("static")


class TelemetryHub:
    """Fan out latest values through one-slot client queues."""

    def __init__(self) -> None:
        self.clients: set[asyncio.Queue[dict[str, Any]]] = set()
        self.browser_frames_dropped = 0

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        target: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        self.clients.add(target)
        return target

    def unsubscribe(self, target: asyncio.Queue[dict[str, Any]]) -> None:
        self.clients.discard(target)

    def publish(self, payload: dict[str, Any]) -> None:
        for target in tuple(self.clients):
            if target.full():
                try:
                    target.get_nowait()
                    self.browser_frames_dropped += 1
                except asyncio.QueueEmpty:
                    pass
            enriched = dict(payload)
            enriched["browser_frames_dropped"] = self.browser_frames_dropped
            try:
                target.put_nowait(enriched)
            except asyncio.QueueFull:
                self.browser_frames_dropped += 1


def create_app(
    data_directory: Path | None = None, output_directory: Path | None = None
) -> FastAPI:
    data_directory = data_directory or Path("data/processed/malecns-v1.0")
    output_directory = output_directory or Path("runs/panel")
    controller = PanelController(data_directory, output_directory)
    hub = TelemetryHub()
    anatomy: AtlasJoin | None = None
    anatomy_lock = threading.Lock()

    def get_anatomy() -> AtlasJoin:
        nonlocal anatomy
        if anatomy is None:
            with anatomy_lock:
                if anatomy is None:
                    anatomy = AtlasJoin.load(ATLAS_DIRECTORY, controller.data_directory)
        return anatomy

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        controller.start_worker()
        task = asyncio.create_task(_relay(controller, hub))
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            controller.shutdown()

    app = FastAPI(
        title="FlyBrain Experiment Panel",
        description="Local laboratory control and bounded observation surface",
        lifespan=lifespan,
    )
    app.state.controller = controller
    app.state.telemetry_hub = hub
    app.mount("/assets", StaticFiles(directory=STATIC_DIRECTORY), name="assets")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html")

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        controller.poll_latest()
        return controller.latest

    @app.get("/api/v2/status")
    async def status_v2() -> dict[str, Any]:
        controller.poll_latest()
        return project_telemetry(controller.latest)

    @app.get("/api/v2/sandbox/status")
    async def sandbox_status() -> dict[str, Any]:
        """Honest status until a local QEMU display adapter is configured."""
        return {
            "schema_version": 2,
            "kind": "sandbox_status",
            "connection": "not_configured",
            "backend": None,
            "vm_frame_available": False,
            "guest_pointer_available": False,
            "guest_actions_enabled": False,
            "host_pointer_control": False,
            "fallback": "recorded_virtual_cursor",
            "fallback_available": any(
                item["available"] for item in recording_catalog()["recordings"]
            ),
        }

    @app.get("/api/v2/pathways")
    async def pathways_catalog_v2() -> dict[str, Any]:
        try:
            return await asyncio.to_thread(pathway_catalog)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise HTTPException(503, "anatomical pathways unavailable") from error

    @app.get("/api/v2/pathways/{target_index}")
    async def pathways_target_v2(target_index: int) -> dict[str, Any]:
        if target_index < 0 or target_index >= 166700:
            raise HTTPException(404, "unknown anatomical target")
        try:
            result = await asyncio.to_thread(target_paths, target_index)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise HTTPException(503, "anatomical pathways unavailable") from error
        if result is None:
            raise HTTPException(404, "unknown anatomical target")
        return result

    @app.post(
        "/api/v2/pathways/observe", response_model=ActionResponse, status_code=202
    )
    async def observe_pathway(request: PathwayObserveRequest) -> ActionResponse:
        """Attach a validated route to the active neural worker at its next chunk."""
        current = str(controller.latest.get("status", "idle"))
        if current not in {"starting", "running", "pausing", "paused", "resuming"}:
            raise HTTPException(
                409, "start a live simulation before observing a pathway"
            )
        try:
            selection = (
                await asyncio.to_thread(resolve_observation, request.observation)
                if request.observation is not None
                else None
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except (OSError, KeyError, TypeError) as error:
            raise HTTPException(503, "anatomical pathways unavailable") from error
        try:
            controller.observe_pathway(selection)
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        return ActionResponse(
            accepted=True,
            status=current,
            detail="observation queued; measurements start at the next simulated chunk",
        )

    @app.get("/api/v2/recordings")
    async def recordings() -> dict[str, Any]:
        return recording_catalog()

    @app.get("/api/v2/recordings/{recording_id}")
    async def recorded_experiment(recording_id: str) -> dict[str, Any]:
        if recording_id not in RECORDINGS:
            raise HTTPException(404, "unknown recording")
        try:
            return await asyncio.to_thread(cached_recording, recording_id)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise HTTPException(
                503, "recorded artifact unavailable or invalid"
            ) from error

    @app.get("/api/computer-use/capabilities")
    async def computer_use_capabilities() -> dict[str, Any]:
        pointer = await asyncio.to_thread(uinput_capability)
        return {
            "os_pointer": {
                "backend": pointer.backend,
                "available": pointer.available,
                "detail": pointer.detail,
                "default_enabled": False,
                "requires_explicit_cli_allow": True,
            }
        }

    @app.get("/api/presets/sensory-descending")
    async def sensory_descending_presets() -> dict[str, Any]:
        return await asyncio.to_thread(panel_presets, controller.data_directory)

    @app.get("/api/presets/visual-pathway")
    async def visual_pathway_preset() -> dict[str, Any]:
        return await asyncio.to_thread(visual_panel_preset, controller.data_directory)

    @app.get("/api/anatomy")
    async def anatomy_summary() -> dict[str, Any]:
        try:
            joined = await asyncio.to_thread(get_anatomy)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(503, str(error)) from error
        return joined.summary()

    @app.post("/api/anatomy/map")
    async def anatomy_map(request: AnatomyMapRequest) -> dict[str, Any]:
        try:
            joined = await asyncio.to_thread(get_anatomy)
            return joined.map_indices(request.neuron_indices)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/anatomy/neighborhood")
    async def anatomy_neighborhood(request: NeighborhoodRequest) -> dict[str, Any]:
        try:
            joined = await asyncio.to_thread(get_anatomy)
            return joined.neighborhood(
                request.seed_indices,
                max_nodes=request.max_nodes,
                max_edges=request.max_edges,
                min_synapse_count=request.min_synapse_count,
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/experiments", response_model=ActionResponse, status_code=202)
    async def start_experiment(config: ExperimentConfig) -> ActionResponse:
        current = controller.latest.get("status")
        if current in {"starting", "running", "pausing", "paused", "resuming"}:
            raise HTTPException(409, "an experiment is already active")
        try:
            controller.command("start", config)
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        return ActionResponse(
            accepted=True, status="starting", detail="worker accepted start"
        )

    @app.post("/api/experiments/{action}", response_model=ActionResponse)
    async def experiment_action(action: str) -> ActionResponse:
        allowed = {
            "pause": {"running"},
            "resume": {"paused"},
            "reset": {"running", "paused", "completed", "failed"},
        }
        if action not in allowed:
            raise HTTPException(404, "unknown experiment action")
        current = str(controller.latest.get("status", "idle"))
        if current not in allowed[action]:
            raise HTTPException(409, f"cannot {action} while status is {current}")
        try:
            recovered = controller.command(action)
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        if recovered:
            return ActionResponse(
                accepted=True,
                status="idle",
                detail="worker recovered to idle; no experiment was replayed",
            )
        return ActionResponse(accepted=True, status=current, detail=f"{action} queued")

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry(websocket: WebSocket) -> None:
        await websocket.accept()
        target = hub.subscribe()
        try:
            await websocket.send_json(controller.latest)
            while True:
                try:
                    payload = await asyncio.wait_for(target.get(), timeout=0.5)
                except TimeoutError:
                    payload = {**controller.latest, "kind": "heartbeat"}
                await websocket.send_json(payload)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.unsubscribe(target)

    @app.websocket("/ws/v2/telemetry")
    async def websocket_telemetry_v2(websocket: WebSocket) -> None:
        """Independent one-slot observer for versioned, bounded neural events."""
        await websocket.accept()
        target = hub.subscribe()
        try:
            await websocket.send_json(project_telemetry(controller.latest))
            while True:
                try:
                    payload = await asyncio.wait_for(target.get(), timeout=0.5)
                except TimeoutError:
                    payload = {**controller.latest, "kind": "heartbeat"}
                await websocket.send_json(project_telemetry(payload))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.unsubscribe(target)

    return app


async def _relay(controller: PanelController, hub: TelemetryHub) -> None:
    while True:
        event = await asyncio.to_thread(controller.poll_latest)
        if event is not None:
            hub.publish(event)
        await asyncio.sleep(0.01)


app = create_app()
