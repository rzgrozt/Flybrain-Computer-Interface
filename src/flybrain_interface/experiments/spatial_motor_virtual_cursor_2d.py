"""Factorized 2D MaleCNS-driven virtual cursor target seeking."""

from __future__ import annotations

import argparse
import json
from math import hypot
from pathlib import Path
from typing import Any

import numpy as np

from flybrain_interface.contracts import MotorCommand
from flybrain_interface.environment.cursor import VirtualCursorEnvironment
from flybrain_interface.environment.pointer import create_pointer_adapter
from flybrain_interface.experiments.spatial_motor_broad_dn_pca_classifier import (
    _feature_vector,
)
from flybrain_interface.experiments.spatial_motor_virtual_cursor_closed_loop import (
    _choose_action,
    _load_classifier,
    _prepare_runtime,
    _relative_position,
    _sample_state,
    _score_classifier,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = (
    ROOT / "configs" / "spatial-motor-virtual-cursor-2d-generalization-v1.json"
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported 2D virtual cursor schema")
    return payload


def _classifier_load_config(
    axis_spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "classifier_backend": axis_spec["classifier_backend"],
        "classifier_config": axis_spec["classifier_config"],
        "classifier_artifact": axis_spec["classifier_artifact"],
    }


def _episode_specs(
    config: dict[str, Any],
) -> tuple[tuple[float, float, float, float], ...]:
    rows: list[tuple[float, float, float, float]] = []
    for episode in config["episodes"]:
        row = (
            float(episode["start_x"]),
            float(episode["start_y"]),
            float(episode["target_x"]),
            float(episode["target_y"]),
        )
        if any(value < 0.0 or value > 1.0 for value in row):
            raise ValueError("2D episode coordinates must stay inside [0, 1]")
        rows.append(row)
    if not rows:
        raise ValueError("2D episodes cannot be empty")
    return tuple(rows)


def _distance(
    x: float,
    y: float,
    target_x: float,
    target_y: float,
) -> float:
    return hypot(target_x - x, target_y - y)


def _success(
    x: float,
    y: float,
    target_x: float,
    target_y: float,
    tolerance: float,
) -> bool:
    return (
        abs(target_x - x) <= tolerance
        and abs(target_y - y) <= tolerance
    )


def _axis_action(
    *,
    axis: str,
    relative_position: float,
    model: dict[str, Any],
    modality: str,
    backend: str,
    previous_action: int | None,
    reversal_margin_threshold: float,
    setup: Any,
    simulator: Any,
    watch_indices: tuple[int, ...],
    temporal_bins: int,
) -> tuple[int, int, np.ndarray, dict[str, Any]]:
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
    raw = int(np.argmax(scores))
    action = _choose_action(
        scores,
        previous_action,
        reversal_margin_threshold=reversal_margin_threshold,
    )
    return raw, action, scores, sample


def _pointer_request(
    pointer_spec: dict[str, Any],
    *,
    allow_os_pointer: bool,
) -> tuple[str, int, bool]:
    backend = str(pointer_spec["backend"])
    pixel_step = int(pointer_spec["pixel_step"])
    enabled = bool(pointer_spec["enabled"])
    if enabled and backend != "uinput":
        raise ValueError("enabled OS pointer requires the uinput backend")
    if enabled and not allow_os_pointer:
        raise PermissionError(
            "OS pointer injection requires explicit --allow-os-pointer"
        )
    return backend, pixel_step, enabled


def run_closed_loop_2d(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
    *,
    allow_os_pointer: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)

    horizontal_model, horizontal_modality, horizontal_classifier, horizontal_backend = (
        _load_classifier(_classifier_load_config(config["horizontal"]))
    )
    vertical_model, vertical_modality, vertical_classifier, vertical_backend = (
        _load_classifier(_classifier_load_config(config["vertical"]))
    )
    if str(horizontal_classifier["axis"]) != "horizontal":
        raise ValueError("horizontal classifier must use horizontal axis")
    if str(vertical_classifier["axis"]) != "vertical":
        raise ValueError("vertical classifier must use vertical axis")

    runtime_config = {
        "broad_state_config": config["broad_state_config"],
    }
    setup, simulator, watch_indices, temporal_bins = _prepare_runtime(
        data_directory,
        runtime_config,
    )

    cursor_spec = config["cursor"]
    relative_spec = config["relative_visualization"]
    step_scale = float(cursor_spec["step_scale"])
    tolerance = float(cursor_spec["axis_success_tolerance"])
    max_steps = int(cursor_spec["max_steps"])
    reversal_margin_threshold = float(cursor_spec["reversal_margin_threshold"])
    pointer_spec = config.get(
        "os_pointer",
        {"backend": "dry_run", "enabled": False, "pixel_step": 24},
    )
    pointer_backend, pointer_pixel_step, pointer_enabled = _pointer_request(
        pointer_spec,
        allow_os_pointer=allow_os_pointer,
    )
    pointer_adapter = create_pointer_adapter(
        backend=pointer_backend,
        pixel_step=pointer_pixel_step,
        enabled=pointer_enabled,
    )

    episodes: list[dict[str, Any]] = []
    for start_x, start_y, target_x, target_y in _episode_specs(config):
        cursor = VirtualCursorEnvironment(
            x=start_x,
            y=start_y,
            step_scale=step_scale,
        )
        initial_distance = _distance(
            cursor.state.x,
            cursor.state.y,
            target_x,
            target_y,
        )
        previous_horizontal: int | None = None
        previous_vertical: int | None = None
        steps: list[dict[str, Any]] = []
        success = _success(
            cursor.state.x,
            cursor.state.y,
            target_x,
            target_y,
            tolerance,
        )

        for step_index in range(max_steps):
            before = cursor.state
            distance_before = _distance(
                before.x,
                before.y,
                target_x,
                target_y,
            )
            if _success(
                before.x,
                before.y,
                target_x,
                target_y,
                tolerance,
            ):
                success = True
                break

            relative_x = _relative_position(
                target_x,
                before.x,
                relative_spec,
            )
            relative_y = _relative_position(
                target_y,
                before.y,
                relative_spec,
            )
            (
                raw_horizontal,
                horizontal_action,
                horizontal_scores,
                horizontal_sample,
            ) = _axis_action(
                axis="horizontal",
                relative_position=relative_x,
                model=horizontal_model,
                modality=horizontal_modality,
                backend=horizontal_backend,
                previous_action=previous_horizontal,
                reversal_margin_threshold=reversal_margin_threshold,
                setup=setup,
                simulator=simulator,
                watch_indices=watch_indices,
                temporal_bins=temporal_bins,
            )
            (
                raw_vertical,
                vertical_action,
                vertical_scores,
                vertical_sample,
            ) = _axis_action(
                axis="vertical",
                relative_position=relative_y,
                model=vertical_model,
                modality=vertical_modality,
                backend=vertical_backend,
                previous_action=previous_vertical,
                reversal_margin_threshold=reversal_margin_threshold,
                setup=setup,
                simulator=simulator,
                watch_indices=watch_indices,
                temporal_bins=temporal_bins,
            )

            delta_x = -1.0 if horizontal_action == 0 else 1.0
            delta_y = -1.0 if vertical_action == 0 else 1.0
            command = MotorCommand(delta_x=delta_x, delta_y=delta_y)
            after = cursor.apply(command)
            pointer_move = pointer_adapter.move(command)
            distance_after = _distance(
                after.x,
                after.y,
                target_x,
                target_y,
            )
            steps.append(
                {
                    "step": step_index,
                    "cursor_before": {"x": before.x, "y": before.y},
                    "cursor_after": {"x": after.x, "y": after.y},
                    "target": {"x": target_x, "y": target_y},
                    "relative_visual_position": {
                        "horizontal": relative_x,
                        "vertical": relative_y,
                    },
                    "horizontal": {
                        "raw_action": ("LEFT", "RIGHT")[raw_horizontal],
                        "action": ("LEFT", "RIGHT")[horizontal_action],
                        "scores": [
                            float(value) for value in horizontal_scores
                        ],
                        "network_spikes": int(
                            horizontal_sample["total_network_spikes"]
                        ),
                    },
                    "vertical": {
                        "raw_action": ("UP", "DOWN")[raw_vertical],
                        "action": ("UP", "DOWN")[vertical_action],
                        "scores": [
                            float(value) for value in vertical_scores
                        ],
                        "network_spikes": int(
                            vertical_sample["total_network_spikes"]
                        ),
                    },
                    "os_pointer_move": {
                        "dx_pixels": pointer_move.dx_pixels,
                        "dy_pixels": pointer_move.dy_pixels,
                    },
                    "distance_before": distance_before,
                    "distance_after": distance_after,
                    "improved": bool(distance_after < distance_before),
                    "nonincreasing": bool(
                        distance_after <= distance_before + 1e-12
                    ),
                }
            )
            previous_horizontal = horizontal_action
            previous_vertical = vertical_action

            if _success(
                after.x,
                after.y,
                target_x,
                target_y,
                tolerance,
            ):
                success = True
                break

        final = cursor.state
        final_distance = _distance(
            final.x,
            final.y,
            target_x,
            target_y,
        )
        episodes.append(
            {
                "start": {"x": start_x, "y": start_y},
                "target": {"x": target_x, "y": target_y},
                "final": {"x": final.x, "y": final.y},
                "initial_distance": initial_distance,
                "final_distance": final_distance,
                "distance_reduction": initial_distance - final_distance,
                "success": success,
                "steps": steps,
            }
        )

    success_fraction = float(
        sum(bool(episode["success"]) for episode in episodes)
        / len(episodes)
    )
    mean_distance_reduction = float(
        np.mean(
            [
                float(episode["distance_reduction"])
                for episode in episodes
            ]
        )
    )
    all_steps_nonincreasing = all(
        bool(step["nonincreasing"])
        for episode in episodes
        for step in episode["steps"]
    )
    gates = config["gates"]
    checks = {
        "episode_success_fraction": success_fraction
        >= float(gates["minimum_episode_success_fraction"]),
        "mean_distance_reduction": (
            mean_distance_reduction > 0.0
            if bool(gates["require_mean_distance_reduction"])
            else True
        ),
        "all_steps_nonincreasing": (
            all_steps_nonincreasing
            if bool(gates["require_all_steps_nonincreasing"])
            else True
        ),
    }
    pointer_adapter.close()
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "This controller composes two independently validated MaleCNS "
            "descending-state readouts. Horizontal and vertical cursor-relative "
            "probes are simulated separately at each step, then their frozen RBF "
            "actions are combined into one 2D motor command. This validates "
            "factorized 2D control, not yet a single unified 2D retinal frame."
        ),
        "descending_neuron_count": len(watch_indices),
        "horizontal_readout": {
            "backend": horizontal_backend,
            "modality": horizontal_modality,
        },
        "vertical_readout": {
            "backend": vertical_backend,
            "modality": vertical_modality,
        },
        "os_pointer": {
            "backend": pointer_backend,
            "enabled": pointer_enabled,
            "pixel_step": pointer_pixel_step,
        },
        "episodes": episodes,
        "summary": {
            "success_fraction": success_fraction,
            "mean_distance_reduction": mean_distance_reduction,
            "all_steps_nonincreasing": all_steps_nonincreasing,
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

    result = run_closed_loop_2d(
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
                "summary": result["summary"],
                "gates": result["gates"],
                "episodes": [
                    {
                        "target": episode["target"],
                        "success": episode["success"],
                        "steps": len(episode["steps"]),
                        "final_distance": episode["final_distance"],
                    }
                    for episode in result["episodes"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
