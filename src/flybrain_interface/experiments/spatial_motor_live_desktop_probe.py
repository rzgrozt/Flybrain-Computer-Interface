"""Run one live-desktop MaleCNS sensory-to-motor probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from flybrain_interface.contracts import MotorCommand
from flybrain_interface.environment.kwin_cursor import KWinCursorBridge
from flybrain_interface.environment.pointer import create_pointer_adapter
from flybrain_interface.environment.screen import (
    ColorTargetDetector,
    SpectacleScreenCapture,
)
from flybrain_interface.experiments.spatial_motor_broad_dn_pca_classifier import (
    _feature_vector,
)
from flybrain_interface.experiments.spatial_motor_virtual_cursor_2d import (
    _classifier_load_config,
    _pointer_request,
)
from flybrain_interface.experiments.spatial_motor_virtual_cursor_closed_loop import (
    _load_classifier,
    _prepare_runtime,
    _relative_position,
    _sample_state,
    _score_classifier,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-live-desktop-probe-v1.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported live-desktop probe schema")
    return payload


def _expected_action(delta: float) -> int:
    if delta == 0.0:
        raise ValueError("target delta cannot be zero for directional gate")
    return 0 if delta < 0.0 else 1


def _axis_probe(
    *,
    axis: str,
    relative_position: float,
    model: dict[str, Any],
    modality: str,
    backend: str,
    setup: Any,
    simulator: Any,
    watch_indices: tuple[int, ...],
    temporal_bins: int,
) -> tuple[int, np.ndarray, dict[str, Any]]:
    sample = _sample_state(
        axis,
        relative_position,
        setup,
        simulator,
        watch_indices,
        temporal_bins,
    )
    features = _feature_vector(sample, modality).reshape(1, -1)
    scores = _score_classifier(backend, model, features)[0]
    action = int(np.argmax(scores))
    return action, scores, sample


def run_live_probe(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
    *,
    allow_os_pointer: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)

    horizontal_model, horizontal_modality, horizontal_cfg, horizontal_backend = (
        _load_classifier(_classifier_load_config(config["horizontal"]))
    )
    vertical_model, vertical_modality, vertical_cfg, vertical_backend = (
        _load_classifier(_classifier_load_config(config["vertical"]))
    )
    if str(horizontal_cfg["axis"]) != "horizontal":
        raise ValueError("horizontal classifier must use horizontal axis")
    if str(vertical_cfg["axis"]) != "vertical":
        raise ValueError("vertical classifier must use vertical axis")

    capture_spec = config["capture"]
    if str(capture_spec["backend"]) != "spectacle":
        raise ValueError("only spectacle capture is supported in this probe")
    capture = SpectacleScreenCapture(
        include_pointer=bool(capture_spec["include_pointer"]),
        timeout_s=float(capture_spec["timeout_s"]),
    )
    frame = capture.capture()

    target_spec = config["target"]
    target = ColorTargetDetector(
        target_rgb=tuple(int(v) for v in target_spec["rgb"]),
        tolerance=int(target_spec["tolerance"]),
        minimum_pixels=int(target_spec["minimum_pixels"]),
    ).detect(frame)
    if target is None:
        raise RuntimeError("configured desktop target was not detected")

    cursor_bridge = KWinCursorBridge()
    try:
        cursor = cursor_bridge.start(timeout_s=5.0)
    finally:
        cursor_bridge.close()

    relative_spec = config["relative_visualization"]
    relative_x = _relative_position(
        target.x_normalized,
        cursor.x_normalized,
        relative_spec,
    )
    relative_y = _relative_position(
        target.y_normalized,
        cursor.y_normalized,
        relative_spec,
    )

    runtime_config = {"broad_state_config": config["broad_state_config"]}
    setup, simulator, watch_indices, temporal_bins = _prepare_runtime(
        data_directory,
        runtime_config,
    )

    horizontal_action, horizontal_scores, horizontal_sample = _axis_probe(
        axis="horizontal",
        relative_position=relative_x,
        model=horizontal_model,
        modality=horizontal_modality,
        backend=horizontal_backend,
        setup=setup,
        simulator=simulator,
        watch_indices=watch_indices,
        temporal_bins=temporal_bins,
    )
    vertical_action, vertical_scores, vertical_sample = _axis_probe(
        axis="vertical",
        relative_position=relative_y,
        model=vertical_model,
        modality=vertical_modality,
        backend=vertical_backend,
        setup=setup,
        simulator=simulator,
        watch_indices=watch_indices,
        temporal_bins=temporal_bins,
    )

    delta_x = target.x_normalized - cursor.x_normalized
    delta_y = target.y_normalized - cursor.y_normalized
    command = MotorCommand(
        delta_x=-1.0 if horizontal_action == 0 else 1.0,
        delta_y=-1.0 if vertical_action == 0 else 1.0,
    )

    pointer_spec = config["os_pointer"]
    pointer_backend, pointer_pixel_step, pointer_enabled = _pointer_request(
        pointer_spec,
        allow_os_pointer=allow_os_pointer,
    )
    pointer = create_pointer_adapter(
        backend=pointer_backend,
        pixel_step=pointer_pixel_step,
        enabled=pointer_enabled,
    )
    try:
        pointer_move = pointer.move(command)
    finally:
        pointer.close()

    horizontal_correct = horizontal_action == _expected_action(delta_x)
    vertical_correct = vertical_action == _expected_action(delta_y)
    require_toward = bool(
        config["gates"]["require_action_toward_target"]
    )
    checks = {
        "horizontal_action_toward_target": (
            horizontal_correct if require_toward else True
        ),
        "vertical_action_toward_target": (
            vertical_correct if require_toward else True
        ),
    }

    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "A live KDE Wayland desktop frame is captured locally, a configured "
            "high-contrast target is detected in pixel space, and the current "
            "KWin cursor position is read through workspace.cursorPos. Their "
            "normalized relative geometry is then passed through the same MaleCNS "
            "visual and descending-state pipeline used by the validated virtual "
            "controller. No target coordinate is exposed directly to the motor "
            "readouts."
        ),
        "frame": {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        },
        "target": {
            "x_normalized": target.x_normalized,
            "y_normalized": target.y_normalized,
            "pixel_count": target.pixel_count,
            "bounding_box": target.bounding_box,
        },
        "cursor": {
            "x": cursor.x,
            "y": cursor.y,
            "workspace_width": cursor.workspace_width,
            "workspace_height": cursor.workspace_height,
            "x_normalized": cursor.x_normalized,
            "y_normalized": cursor.y_normalized,
        },
        "relative_visual_position": {
            "horizontal": relative_x,
            "vertical": relative_y,
        },
        "horizontal": {
            "action": ("LEFT", "RIGHT")[horizontal_action],
            "scores": [float(value) for value in horizontal_scores],
            "network_spikes": int(horizontal_sample["total_network_spikes"]),
        },
        "vertical": {
            "action": ("UP", "DOWN")[vertical_action],
            "scores": [float(value) for value in vertical_scores],
            "network_spikes": int(vertical_sample["total_network_spikes"]),
        },
        "motor_command": {
            "delta_x": command.delta_x,
            "delta_y": command.delta_y,
        },
        "os_pointer": {
            "backend": pointer_backend,
            "enabled": pointer_enabled,
            "move": {
                "dx_pixels": pointer_move.dx_pixels,
                "dy_pixels": pointer_move.dy_pixels,
            },
        },
        "gates": {
            "checks": checks,
            "passed": all(checks.values()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("data/processed/malecns-v1.0"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-os-pointer",
        action="store_true",
        help="explicitly allow uinput pointer motion when config enables it",
    )
    args = parser.parse_args()

    result = run_live_probe(
        args.data_directory,
        args.config,
        allow_os_pointer=args.allow_os_pointer,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "target": result["target"],
                "cursor": result["cursor"],
                "relative_visual_position": result[
                    "relative_visual_position"
                ],
                "horizontal": result["horizontal"],
                "vertical": result["vertical"],
                "motor_command": result["motor_command"],
                "os_pointer": result["os_pointer"],
                "gates": result["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
