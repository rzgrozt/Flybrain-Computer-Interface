"""Versioned telemetry keeps scientific provenance and bounded state explicit."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.routing import APIRoute, APIWebSocketRoute

from flybrain_interface.panel.app import create_app
from flybrain_interface.panel.telemetry_v2 import project_telemetry


def test_project_measured_worker_frame_preserves_clocks_and_unknowns() -> None:
    payload = {
        "kind": "telemetry",
        "experiment_id": "run-01",
        "status": "running",
        "chunks": 8,
        "simulated_time_s": 0.16,
        "chunk_spikes": 21,
        "total_spikes": 340,
        "population_rates_hz": {"sample": 4.5},
        "watch_indices": [12],
        "voltage_mv": [-53.4],
        "synaptic_drive_mv": [0.002],
        "brain_activity": {"bin_duration_s": 0.02, "selected_neuron_count": 1},
        "browser_frames_dropped": 2,
    }
    result = project_telemetry(payload, received_at_utc="2026-09-22T15:00:00+00:00")
    assert result["schema_version"] == 2
    assert result["session_id"] == "run-01"
    assert result["episode_id"] is None
    assert result["step_id"] == 8
    assert result["neural_tick"] is None
    assert result["simulated_time_s"] == 0.16
    assert result["received_at_utc"] == "2026-09-22T15:00:00+00:00"
    assert result["acquired_at_utc"] is None
    assert result["activity"]["chunk_spikes"] == 21
    assert result["activity"]["brain_activity"]["bin_duration_s"] == 0.02
    assert result["selected_neurons"]["voltage_mv"] == [-53.4]
    assert result["selected_neurons"]["synaptic_drive_mv"] == [0.002]
    assert result["selected_neurons"]["spike_times_s"] is None
    assert result["motor"] is None
    assert result["pathway_activation"] is None
    assert result["observed_pathway"] is None
    assert result["reward"] is None
    assert result["sandbox"] is None
    assert result["runtime"]["browser_frames_dropped"] == 2
    assert result["snapshot_continuity_guaranteed"] is False


def test_idle_or_heartbeat_is_not_presented_as_measured_activity() -> None:
    idle = project_telemetry({"kind": "status", "status": "idle"})
    assert idle["source"] == "session_status"
    assert idle["activity"]["chunk_spikes"] is None
    assert idle["selected_neurons"]["voltage_mv"] is None
    assert idle["simulated_time_s"] is None

    heartbeat = project_telemetry(
        {"kind": "heartbeat", "status": "running", "chunk_spikes": 200}
    )
    assert heartbeat["source"] == "session_status"
    assert heartbeat["activity"]["chunk_spikes"] is None


def test_pathway_projection_requires_real_simulated_measurement() -> None:
    sample = {
        "source": "simulated_neural_measurement",
        "target_neuron_index": 725,
        "path_index": 0,
        "neuron_indices": [1, 2, 725],
        "voltage_mv": [-55.0, -54.0, -53.0],
        "synaptic_drive_mv": [0.001, 0.002, 0.003],
        "spike_counts": [0, 1, 0],
        "sampled_chunk": 2,
        "bin_duration_s": 0.02,
    }
    selected = {"target_neuron_index": 725, "path_index": 0}
    measured = project_telemetry(
        {
            "kind": "telemetry",
            "chunks": 2,
            "pathway_measurement": sample,
            "pathway_selection": selected,
        }
    )
    assert measured["observed_pathway"] == sample
    assert measured["pathway_selection"] == selected
    assert measured["pathway_activation"] is None
    nonmeasurement = project_telemetry(
        {
            "kind": "heartbeat",
            "pathway_measurement": sample,
            "pathway_selection": selected,
        }
    )
    assert nonmeasurement["observed_pathway"] is None
    assert nonmeasurement["pathway_activation"] is None


def test_v2_routes_preserve_legacy_websocket(tmp_path: Path) -> None:
    app = create_app(tmp_path, tmp_path / "outputs")
    routes = {
        item.path: item.endpoint for item in app.routes if isinstance(item, APIRoute)
    }
    websocket_routes = {
        item.path for item in app.routes if isinstance(item, APIWebSocketRoute)
    }
    assert "/ws/telemetry" in websocket_routes
    assert "/ws/v2/telemetry" in websocket_routes
    status = asyncio.run(routes["/api/v2/status"]())
    assert status["kind"] == "simulation_telemetry"
    assert status["source"] == "session_status"
    assert status["status"] == "offline"
