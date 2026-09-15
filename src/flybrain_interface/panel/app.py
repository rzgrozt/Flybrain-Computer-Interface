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

from flybrain_interface.experiments.sensory_descending import panel_presets
from flybrain_interface.experiments.visual_pathway import visual_panel_preset
from flybrain_interface.panel.anatomy import ATLAS_DIRECTORY, AtlasJoin
from flybrain_interface.panel.controller import PanelController
from flybrain_interface.panel.models import (
    ActionResponse,
    AnatomyMapRequest,
    ExperimentConfig,
    NeighborhoodRequest,
)

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
                    anatomy = AtlasJoin.load(
                        ATLAS_DIRECTORY, controller.data_directory
                    )
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

    return app


async def _relay(controller: PanelController, hub: TelemetryHub) -> None:
    while True:
        event = await asyncio.to_thread(controller.poll_latest)
        if event is not None:
            hub.publish(event)
        await asyncio.sleep(0.01)


app = create_app()
