"""Panel v2 recording contracts preserve experimental provenance and safety."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute

from flybrain_interface.panel.app import create_app
from flybrain_interface.panel.recordings import (
    ARTIFACT_DIRECTORY,
    load_recording,
    recording_catalog,
)


@pytest.mark.parametrize(
    ("recording_id", "axis_count"),
    [("factorized-2d", 2), ("horizontal-1d", 1), ("vertical-1d", 1)],
)
def test_actual_recordings_are_bounded_and_explicit(
    recording_id: str, axis_count: int
) -> None:
    if not ARTIFACT_DIRECTORY.exists():
        pytest.skip("validated local artifacts are not installed")
    result = load_recording(recording_id)
    assert result["schema_version"] == 2
    assert result["vm_frames_available"] is False
    assert result["timestamps_available"] is False
    assert result["per_neuron_telemetry_available"] is False
    assert result["descending_neuron_count"] == 1310
    assert result["episodes"]
    first = result["episodes"][0]["steps"][0]
    assert first["episode_id"] == result["episodes"][0]["episode_id"]
    assert first["timestamp_utc"] is None
    assert first["simulated_time_s"] is None
    assert len(first["probes"]) == axis_count
    assert first["neural_trace_available"] is False
    for probe in first["probes"]:
        assert probe["scores_are_probabilities"] is False
        assert probe["observation_mode"] == "separate_axis_probe"
        assert probe["simulated_time_s"] is None
        assert len(probe["scores"]) == len(probe["score_labels"]) == 2
    if axis_count == 2:
        assert {probe["axis"] for probe in first["probes"]} == {
            "horizontal", "vertical"
        }
        assert first["motor_command"]["dx"] != 0
        assert first["motor_command"]["dy"] != 0
    else:
        zero_axis = "dy" if recording_id == "horizontal-1d" else "dx"
        assert first["motor_command"][zero_axis] == 0


def test_recording_catalog_uses_only_known_files(tmp_path: Path) -> None:
    catalog = recording_catalog(tmp_path)
    assert catalog["schema_version"] == 2
    assert len(catalog["recordings"]) == 3
    assert all(not row["available"] for row in catalog["recordings"])


def test_sandbox_is_explicitly_disconnected_and_read_only(tmp_path: Path) -> None:
    app = create_app(tmp_path, tmp_path / "runs")
    routes = {
        route.path: route.endpoint
        for route in app.routes
        if isinstance(route, APIRoute)
    }
    status = asyncio.run(routes["/api/v2/sandbox/status"]())
    assert status["connection"] == "not_configured"
    assert status["vm_frame_available"] is False
    assert status["guest_actions_enabled"] is False
    assert status["host_pointer_control"] is False
    assert status["backend"] is None
    catalog = asyncio.run(routes["/api/v2/recordings"]())
    assert catalog["schema_version"] == 2
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes["/api/v2/recordings/{recording_id}"]("../secret"))
    assert error.value.status_code == 404


def test_recording_http_contract_without_optional_http_client(tmp_path: Path) -> None:
    """Exercise HTTP ASGI routes without loading an extra testing dependency."""
    app = create_app(tmp_path, tmp_path / "runs")

    async def get(path: str) -> tuple[int, dict[str, object]]:
        messages: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8000),
        }
        await app(scope, receive, send)
        status = next(
            message["status"] for message in messages
            if message["type"] == "http.response.start"
        )
        data = b"".join(
            message["body"] for message in messages
            if message["type"] == "http.response.body"
        )
        return int(status), json.loads(data)

    status_code, status = asyncio.run(get("/api/v2/sandbox/status"))
    assert status_code == 200
    assert status["guest_actions_enabled"] is False
    status_code, catalog = asyncio.run(get("/api/v2/recordings"))
    assert status_code == 200
    assert len(catalog["recordings"]) == 3
    if ARTIFACT_DIRECTORY.exists():
        status_code, episode = asyncio.run(
            get("/api/v2/recordings/factorized-2d")
        )
        assert status_code == 200
        assert episode["simulation_mode"] == (
            "separate_horizontal_vertical_probes"
        )
    status_code, _ = asyncio.run(get("/api/v2/recordings/unknown"))
    assert status_code == 404
