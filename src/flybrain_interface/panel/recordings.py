"""Read-only adapters for validated virtual-cursor recording artifacts.

This module presents recorded environmental outcomes and decoder output without
fabricating timestamps, per-neuron observations or a live guest display.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ARTIFACT_DIRECTORY = Path(__file__).resolve().parents[3] / "artifacts"

RECORDINGS: dict[str, tuple[str, str]] = {
    "factorized-2d": (
        "spatial-motor-virtual-cursor-2d-generalization-v1.json",
        "Factorized 2D · independent horizontal and vertical probes",
    ),
    "horizontal-1d": (
        "spatial-motor-virtual-cursor-rbf-generalization-v1.json",
        "Horizontal 1D · temporal-voltage RBF",
    ),
    "vertical-1d": (
        "spatial-motor-virtual-cursor-rbf-vertical-generalization-v1.json",
        "Vertical 1D · temporal-voltage/drive RBF",
    ),
}


def recording_catalog(directory: Path = ARTIFACT_DIRECTORY) -> dict[str, Any]:
    """Enumerate only approved, bundled artifact names, without filesystem access."""
    return {
        "schema_version": 2,
        "kind": "recording_catalog",
        "recordings": [
            {
                "recording_id": recording_id,
                "label": label,
                "available": (directory / filename).is_file(),
                "source": "recorded_experiment",
            }
            for recording_id, (filename, label) in RECORDINGS.items()
        ],
    }


def _point(x: float, y: float) -> dict[str, float]:
    return {"x": float(x), "y": float(y)}


def _axis_probe(axis: str, value: dict[str, Any], step: int) -> dict[str, Any]:
    labels = ["LEFT", "RIGHT"] if axis == "horizontal" else ["UP", "DOWN"]
    return {
        "probe_id": f"{axis}-{step}",
        "axis": axis,
        "source": "recorded_neural_simulation",
        "observation_mode": "separate_axis_probe",
        "simulated_time_s": None,
        "timestamp_utc": None,
        "feature_modality": (
            "temporal_voltage" if axis == "horizontal" else "temporal_concat"
        ),
        "decoder": "whitened_pca_rbf",
        "score_labels": labels,
        "scores": [float(score) for score in value["scores"]],
        "scores_are_probabilities": False,
        "selected_action": str(value["action"]),
        "raw_action": str(value.get("raw_action", value["action"])),
        "total_network_spikes": value.get("network_spikes"),
        "per_neuron_telemetry": False,
    }


def _normalize_step(
    recording_id: str, episode_id: str, step: dict[str, Any], index: int
) -> dict[str, Any]:
    two_dimensional = recording_id == "factorized-2d"
    if two_dimensional:
        before = _point(**step["cursor_before"])
        after = _point(**step["cursor_after"])
        target = _point(**step["target"])
        probes = [
            _axis_probe("horizontal", step["horizontal"], index),
            _axis_probe("vertical", step["vertical"], index),
        ]
    else:
        horizontal = recording_id == "horizontal-1d"
        axis = "horizontal" if horizontal else "vertical"
        before_value, after_value = step["cursor_before"], step["cursor_after"]
        target_value = step["target_x"] if horizontal else step["target_y"]
        before = _point(before_value, 0.5) if horizontal else _point(0.5, before_value)
        after = _point(after_value, 0.5) if horizontal else _point(0.5, after_value)
        target = _point(target_value, 0.5) if horizontal else _point(0.5, target_value)
        probes = [_axis_probe(axis, step, index)]
    return {
        "event_id": f"{episode_id}:step-{index}",
        "episode_id": episode_id,
        "step": int(step.get("step", index)),
        "source": "recorded_experiment",
        "timestamp_utc": None,
        "simulated_time_s": None,
        "probes": probes,
        "motor_command": {
            "dx": after["x"] - before["x"],
            "dy": after["y"] - before["y"],
            "source": "recorded_environment_delta",
            "units": "normalized_cursor_position",
            "click": None,
        },
        "cursor_before": before,
        "cursor_after": after,
        "target": target,
        "distance_before": float(step["distance_before"]),
        "distance_after": float(step["distance_after"]),
        "distance_delta": (
            float(step["distance_before"]) - float(step["distance_after"])
        ),
        "improved": bool(step["improved"]),
        "reward": None,
        "screen_frame_available": False,
        "neural_trace_available": False,
    }


def load_recording(
    recording_id: str, directory: Path = ARTIFACT_DIRECTORY
) -> dict[str, Any]:
    """Read a whitelisted artifact and normalize only the bounded display fields."""
    filename, label = RECORDINGS[recording_id]
    with (directory / filename).open(encoding="utf-8") as source:
        raw = json.load(source)
    episodes = raw.get("episodes", [])
    if not isinstance(episodes, list) or len(episodes) > 64:
        raise ValueError("invalid or unbounded recorded episode list")
    output: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes):
        steps = episode.get("steps", [])
        if not isinstance(steps, list) or len(steps) > 4096:
            raise ValueError("invalid or unbounded recorded step list")
        episode_id = f"{recording_id}:episode-{index + 1}"
        output.append(
            {
                "episode_id": episode_id,
                "episode_index": index,
                "source": "recorded_experiment",
                "success": bool(episode["success"]),
                "initial_distance": float(episode["initial_distance"]),
                "final_distance": float(episode["final_distance"]),
                "distance_reduction": float(episode["distance_reduction"]),
                "steps": [
                    _normalize_step(recording_id, episode_id, step, step_index)
                    for step_index, step in enumerate(steps)
                ],
            }
        )
    return {
        "schema_version": 2,
        "kind": "recorded_experiment",
        "recording_id": recording_id,
        "label": label,
        "source": "recorded_validation_artifact",
        "artifact_name": filename,
        "simulation_mode": (
            "separate_horizontal_vertical_probes"
            if recording_id == "factorized-2d"
            else "single_axis_probe"
        ),
        "vm_frames_available": False,
        "timestamps_available": False,
        "per_neuron_telemetry_available": False,
        "descending_neuron_count": int(raw["descending_neuron_count"]),
        "summary": raw.get("summary"),
        "episodes": output,
    }


@lru_cache(maxsize=3)
def cached_recording(recording_id: str) -> dict[str, Any]:
    """Cache immutable reviewed artifacts; never scan or serve arbitrary paths."""
    return load_recording(recording_id)
