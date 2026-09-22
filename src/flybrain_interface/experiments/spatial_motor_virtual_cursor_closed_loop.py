"""Run online MaleCNS-driven virtual-cursor target-seeking episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from flybrain_interface.contracts import MotorCommand
from flybrain_interface.environment.cursor import VirtualCursorEnvironment
from flybrain_interface.experiments.spatial_motor_broad_descending_state import (
    _descending_watch,
    _graded_relay_indices,
)
from flybrain_interface.experiments.spatial_motor_broad_dn_binary_direction import (
    _fit_pipeline,
    _score_pipeline,
)
from flybrain_interface.experiments.spatial_motor_broad_dn_pca_classifier import (
    _feature_vector,
    _matrix,
    _samples_by_position,
)
from flybrain_interface.experiments.spatial_motor_broad_dn_rbf_direction import (
    _collect_samples as _collect_rbf_samples,
)
from flybrain_interface.experiments.spatial_motor_broad_dn_rbf_direction import (
    _fit_pipeline as _fit_rbf_pipeline,
)
from flybrain_interface.experiments.spatial_motor_broad_dn_rbf_direction import (
    _matrix as _rbf_matrix,
)
from flybrain_interface.experiments.spatial_motor_broad_dn_rbf_direction import (
    _score_pipeline as _score_rbf_pipeline,
)
from flybrain_interface.experiments.spatial_motor_pathway import _setup_visual_system
from flybrain_interface.experiments.spatial_subthreshold_readout import _run_position
from flybrain_interface.simulation.runtime import (
    RuntimeBackend,
    SparseLIFSimulator,
    SubnormalDrivePolicy,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-virtual-cursor-closed-loop-v1.json"
CLASS_NAMES = ("LEFT", "RIGHT")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported virtual cursor closed-loop schema")
    return payload


def _load_classifier(
    config: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    classifier_config = json.loads(
        (ROOT / str(config["classifier_config"])).read_text(encoding="utf-8")
    )
    classifier_artifact = json.loads(
        (ROOT / str(config["classifier_artifact"])).read_text(encoding="utf-8")
    )
    winner = classifier_artifact["benchmark"]["winner"]
    modality = str(winner["modality"])
    train_positions = tuple(
        float(value) for value in classifier_config["train_positions"]
    )
    center = float(classifier_config["center_position"])
    labels = np.asarray(
        [0 if position < center else 1 for position in train_positions],
        dtype=np.int64,
    )
    backend = str(config.get("classifier_backend", "linear_pca"))

    if backend == "linear_pca":
        training_artifact = json.loads(
            (ROOT / str(config["training_state_artifact"])).read_text(
                encoding="utf-8"
            )
        )
        samples = _samples_by_position(
            training_artifact,
            str(classifier_config["axis"]),
        )
        features = _matrix(samples, train_positions, modality)
        model = _fit_pipeline(
            features,
            labels,
            components=int(winner["components"]),
            alpha=float(winner["alpha"]),
            epsilon=float(classifier_config["standardization_epsilon"]),
        )
    elif backend == "rbf_pca":
        training_artifacts = [
            json.loads((ROOT / str(path)).read_text(encoding="utf-8"))
            for path in classifier_config["training_artifacts"]
        ]
        samples = _collect_rbf_samples(
            training_artifacts,
            str(classifier_config["axis"]),
        )
        features = _rbf_matrix(samples, train_positions, modality)
        model = _fit_rbf_pipeline(
            features,
            labels,
            components=int(winner["components"]),
            alpha=float(winner["alpha"]),
            gamma=float(winner["gamma"]),
            epsilon=float(classifier_config["standardization_epsilon"]),
        )
    else:
        raise ValueError(f"unsupported classifier backend: {backend}")
    return model, modality, classifier_config, backend


def _score_classifier(
    backend: str,
    model: dict[str, Any],
    features: np.ndarray,
) -> np.ndarray:
    if backend == "linear_pca":
        return _score_pipeline(model, features)
    if backend == "rbf_pca":
        return _score_rbf_pipeline(model, features)
    raise ValueError(f"unsupported classifier backend: {backend}")


def _choose_action(
    scores: np.ndarray,
    previous_action: int | None,
    *,
    reversal_margin_threshold: float,
) -> int:
    predicted = int(np.argmax(scores))
    if previous_action not in {0, 1} or predicted not in {0, 1}:
        return predicted
    if predicted == previous_action:
        return predicted
    margin = float(scores[predicted] - scores[previous_action])
    if margin < reversal_margin_threshold:
        return previous_action
    return predicted


def _relative_position(
    target_x: float,
    cursor_x: float,
    spec: dict[str, Any],
) -> float:
    center = float(spec["center"])
    value = center + (target_x - cursor_x)
    return min(float(spec["maximum"]), max(float(spec["minimum"]), value))


def _prepare_runtime(
    data_directory: Path,
    config: dict[str, Any],
) -> tuple[Any, SparseLIFSimulator, tuple[int, ...], int]:
    broad_config = json.loads(
        (ROOT / str(config["broad_state_config"])).read_text(encoding="utf-8")
    )
    motor_config_path = ROOT / str(broad_config["motor_config"])
    motor_config = json.loads(motor_config_path.read_text(encoding="utf-8"))
    base_config_path = ROOT / str(motor_config["base_config"])
    setup = _setup_visual_system(data_directory, base_config_path)
    watch_indices = _descending_watch(
        setup,
        max_hops=int(broad_config["maximum_descending_hops"]),
    )
    relay_indices = _graded_relay_indices(setup, motor_config)
    relay_spec = motor_config["graded_relay"]
    runtime = setup.base["runtime"]
    simulator = SparseLIFSimulator(
        setup.graph,
        clamped_indices=setup.clamped,
        config=setup.lif_config,
        backend=cast(RuntimeBackend, str(runtime["backend"])),
        subnormal_drive_policy=cast(
            SubnormalDrivePolicy,
            str(runtime["subnormal_drive_policy"]),
        ),
        graded_relay_indices=relay_indices,
        graded_relay_gain=float(relay_spec["gain"]),
        graded_relay_activation_scale_mv=float(
            relay_spec.get("activation_scale_mv", 7.0)
        ),
    )
    simulator.prepare()
    return setup, simulator, watch_indices, int(broad_config["temporal_bins"])


def _sample_state(
    axis: str,
    position: float,
    setup: Any,
    simulator: SparseLIFSimulator,
    watch_indices: tuple[int, ...],
    temporal_bins: int,
) -> dict[str, Any]:
    if axis not in {"horizontal", "vertical"}:
        raise ValueError(f"unsupported control axis: {axis}")
    return _run_position(
        axis,
        position,
        setup.graph,
        simulator,
        setup.base,
        setup.directions,
        setup.screen,
        setup.source_columns,
        setup.source_groups,
        setup.cartridges,
        setup.projections,
        watch_indices,
        setup.graded_config,
        setup.lif_config,
        setup.duration_s,
        temporal_bins=temporal_bins,
    )


def _episode_specs(
    config: dict[str, Any],
    axis: str = "horizontal",
) -> tuple[tuple[float, float], ...]:
    if axis not in {"horizontal", "vertical"}:
        raise ValueError(f"unsupported control axis: {axis}")
    cursor_spec = config["cursor"]
    coordinate = "x" if axis == "horizontal" else "y"
    start_key = f"start_{coordinate}"
    target_key = f"target_{coordinate}"
    if "episodes" in config:
        rows: list[tuple[float, float]] = []
        for episode in config["episodes"]:
            start = float(episode[start_key])
            target = float(episode[target_key])
            if not (0.0 <= start <= 1.0 and 0.0 <= target <= 1.0):
                raise ValueError("episode coordinates must stay inside [0, 1]")
            rows.append((start, target))
        if not rows:
            raise ValueError("episodes cannot be empty")
        return tuple(rows)

    start = float(cursor_spec[start_key])
    return tuple(
        (start, float(target)) for target in config["target_positions"]
    )


def _axis_coordinate(cursor: VirtualCursorEnvironment, axis: str) -> float:
    return cursor.state.x if axis == "horizontal" else cursor.state.y


def _axis_command(axis: str, value: float) -> MotorCommand:
    if axis == "horizontal":
        return MotorCommand(delta_x=value, delta_y=0.0)
    if axis == "vertical":
        return MotorCommand(delta_x=0.0, delta_y=value)
    raise ValueError(f"unsupported control axis: {axis}")


def _class_names(axis: str) -> tuple[str, str]:
    if axis == "horizontal":
        return ("LEFT", "RIGHT")
    if axis == "vertical":
        return ("UP", "DOWN")
    raise ValueError(f"unsupported control axis: {axis}")


def run_closed_loop(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    model, modality, classifier_config, classifier_backend = _load_classifier(config)
    control_axis = str(config.get("control_axis", classifier_config["axis"]))
    if control_axis != str(classifier_config["axis"]):
        raise ValueError("control axis must match classifier axis")
    class_names = _class_names(control_axis)
    coordinate = "x" if control_axis == "horizontal" else "y"
    setup, simulator, watch_indices, temporal_bins = _prepare_runtime(
        data_directory,
        config,
    )
    cursor_spec = config["cursor"]
    relative_spec = config["relative_visualization"]
    step_scale = float(cursor_spec["step_scale"])
    tolerance = float(cursor_spec["success_tolerance"])
    max_steps = int(cursor_spec["max_steps"])
    reversal_margin_threshold = float(cursor_spec["reversal_margin_threshold"])
    command_values = (-1.0, 1.0)

    episodes: list[dict[str, Any]] = []
    for start_position, target_position in _episode_specs(config, control_axis):
        cursor = VirtualCursorEnvironment(
            x=start_position if control_axis == "horizontal" else 0.5,
            y=start_position if control_axis == "vertical" else 0.5,
            step_scale=step_scale,
        )
        initial_distance = abs(
            target_position - _axis_coordinate(cursor, control_axis)
        )
        steps: list[dict[str, Any]] = []
        success = initial_distance <= tolerance
        previous_action: int | None = None
        for step_index in range(max_steps):
            current_position = _axis_coordinate(cursor, control_axis)
            distance_before = abs(target_position - current_position)
            if distance_before <= tolerance:
                success = True
                break
            relative_position = _relative_position(
                target_position,
                current_position,
                relative_spec,
            )
            neural_sample = _sample_state(
                control_axis,
                relative_position,
                setup,
                simulator,
                watch_indices,
                temporal_bins,
            )
            features = _feature_vector(neural_sample, modality).reshape(1, -1)
            scores = _score_classifier(classifier_backend, model, features)[0]
            raw_prediction = int(np.argmax(scores))
            predicted = _choose_action(
                scores,
                previous_action,
                reversal_margin_threshold=reversal_margin_threshold,
            )
            before = _axis_coordinate(cursor, control_axis)
            cursor.apply(
                _axis_command(control_axis, command_values[predicted])
            )
            after = _axis_coordinate(cursor, control_axis)
            distance_after = abs(target_position - after)
            step_record = {
                "step": step_index,
                "control_axis": control_axis,
                "cursor_before": before,
                "cursor_after": after,
                "target_position": target_position,
                f"target_{coordinate}": target_position,
                "relative_visual_position": relative_position,
                "raw_action": class_names[raw_prediction],
                "action": class_names[predicted],
                "hysteresis_applied": bool(predicted != raw_prediction),
                "scores": [float(value) for value in scores],
                "distance_before": distance_before,
                "distance_after": distance_after,
                "improved": bool(distance_after < distance_before),
                "network_spikes": int(neural_sample["total_network_spikes"]),
            }
            steps.append(step_record)
            previous_action = predicted
            if distance_after <= tolerance:
                success = True
                break

        final_distance = abs(
            target_position - _axis_coordinate(cursor, control_axis)
        )
        episode_record = {
            "control_axis": control_axis,
            "start_position": start_position,
            "target_position": target_position,
            f"start_{coordinate}": start_position,
            f"target_{coordinate}": target_position,
            "initial_distance": initial_distance,
            "final_distance": final_distance,
            "distance_reduction": initial_distance - final_distance,
            "success": success,
            "steps": steps,
        }
        episodes.append(episode_record)

    success_fraction = float(
        sum(bool(episode["success"]) for episode in episodes) / len(episodes)
    )
    mean_reduction = float(
        np.mean([float(episode["distance_reduction"]) for episode in episodes])
    )
    gates = config["gates"]
    checks = {
        "episode_success_fraction": success_fraction
        >= float(gates["minimum_episode_success_fraction"]),
        "mean_distance_reduction": (
            mean_reduction > 0.0
            if bool(gates["require_mean_distance_reduction"])
            else True
        ),
    }
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "At each step the target is re-expressed in a cursor-relative "
            "visual frame, then passed through the same MaleCNS visual and "
            "graded-relay dynamics. The action classifier observes only broad "
            "descending-neuron temporal state. "
            "No screen coordinate or target coordinate is exposed to the classifier."
        ),
        "classifier_winner": {
            "backend": classifier_backend,
            "modality": modality,
            **classifier_config,
        },
        "control_axis": control_axis,
        "descending_neuron_count": len(watch_indices),
        "episodes": episodes,
        "summary": {
            "success_fraction": success_fraction,
            "mean_distance_reduction": mean_reduction,
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
    args = parser.parse_args()

    result = run_closed_loop(args.data_directory, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": result["summary"],
                "gates": result["gates"],
                "episodes": [
                    {
                        "target_position": episode["target_position"],
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
