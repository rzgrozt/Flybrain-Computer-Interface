"""Versioned, presentation-neutral envelopes for bounded worker telemetry."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def project_telemetry(
    payload: dict[str, Any], *, received_at_utc: str | None = None
) -> dict[str, Any]:
    """Project a latest-value frame without inferring missing measurements."""
    measured = payload.get("kind") == "telemetry"
    return {
        "schema_version": 2,
        "kind": "simulation_telemetry",
        "source": "simulated_neural_measurement" if measured else "session_status",
        "session_id": payload.get("experiment_id"),
        "episode_id": None,
        "step_id": payload.get("chunks") if measured else None,
        "neural_tick": None,
        "simulated_time_s": payload.get("simulated_time_s"),
        "received_at_utc": received_at_utc or datetime.now(UTC).isoformat(),
        "acquired_at_utc": None,
        "status": payload.get("status", "unknown"),
        "reason": payload.get("reason"),
        "activity": {
            "chunk_spikes": payload.get("chunk_spikes") if measured else None,
            "total_spikes": payload.get("total_spikes") if measured else None,
            "population_rates_hz": (
                payload.get("population_rates_hz") if measured else None
            ),
            "brain_activity": payload.get("brain_activity") if measured else None,
        },
        "selected_neurons": {
            "indices": payload.get("watch_indices") if measured else None,
            "voltage_mv": payload.get("voltage_mv") if measured else None,
            "synaptic_drive_mv": payload.get("synaptic_drive_mv") if measured else None,
            "spike_times_s": None,
        },
        "motor": None,
        "observed_pathway": payload.get("pathway_measurement") if measured else None,
        "pathway_selection": payload.get("pathway_selection"),
        "pathway_activation": None,
        "reward": None,
        "sandbox": None,
        "runtime": {
            "speed_ratio": payload.get("speed_ratio"),
            "worker_loop_seconds": payload.get("worker_loop_seconds"),
            "visited_edges": payload.get("visited_edges"),
            "worker_ipc_frames_dropped": payload.get("ipc_telemetry_dropped_total"),
            "browser_frames_dropped": payload.get("browser_frames_dropped"),
        },
        "snapshot_continuity_guaranteed": False,
    }
