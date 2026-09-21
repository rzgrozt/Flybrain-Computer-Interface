"""Calibrate anatomically grounded motor state into virtual cursor motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from flybrain_interface.contracts import MotorCommand
from flybrain_interface.environment.cursor import VirtualCursorEnvironment
from flybrain_interface.experiments.spatial_motor_readout import run_characterization
from flybrain_interface.experiments.spatial_subthreshold_readout import (
    fit_ridge,
    predict_ridge,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-cursor-calibration-v1.json"
FEATURE_KEYS = (
    "steer_left",
    "steer_right",
    "steer_left_secondary",
    "steer_right_secondary",
    "forward",
    "backward",
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported motor cursor calibration schema")
    return payload


def _features(run: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(
                run["populations"][key][
                    "mean_signed_peak_voltage_delta_mv"
                ]
            )
            for key in FEATURE_KEYS
        ],
        dtype=np.float64,
    )


def _axis_calibration(
    axis: str,
    runs: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_position = {float(run["position"]): run for run in runs}
    train_positions = tuple(float(v) for v in config["train_positions"])
    test_positions = tuple(float(v) for v in config["test_positions"])
    if set(train_positions) & set(test_positions):
        raise ValueError("train and test positions must be disjoint")
    requested = set(train_positions) | set(test_positions)
    if requested - set(by_position):
        raise ValueError("motor characterization is missing calibration positions")

    x_train = np.vstack([_features(by_position[p]) for p in train_positions])
    y_train = np.asarray(train_positions, dtype=np.float64)
    x_test = np.vstack([_features(by_position[p]) for p in test_positions])
    y_test = np.asarray(test_positions, dtype=np.float64)
    ridge = config["ridge"]
    model = fit_ridge(
        x_train,
        y_train,
        alpha=float(ridge["alpha"]),
        epsilon=float(ridge["standardization_epsilon"]),
    )
    predicted = predict_ridge(model, x_test)
    error = predicted - y_test
    start = float(config["cursor"]["start"])
    gain = float(config["cursor"]["gain"])
    max_delta = float(config["cursor"]["max_delta"])
    step_scale = float(config["cursor"]["step_scale"])

    trials: list[dict[str, Any]] = []
    for actual, estimate in zip(y_test, predicted, strict=True):
        raw_delta = gain * (float(estimate) - start)
        delta = max(-max_delta, min(max_delta, raw_delta))
        cursor = VirtualCursorEnvironment(
            x=start,
            y=start,
            step_scale=step_scale,
        )
        if axis == "horizontal":
            command = MotorCommand(delta_x=delta, delta_y=0.0)
            before = cursor.state.x
            after = cursor.apply(command).x
        elif axis == "vertical":
            command = MotorCommand(delta_x=0.0, delta_y=delta)
            before = cursor.state.y
            after = cursor.apply(command).y
        else:
            raise ValueError(f"unsupported cursor axis: {axis}")
        side_correct = bool(
            (actual < start and estimate < start and after < before)
            or (actual > start and estimate > start and after > before)
        )
        trials.append(
            {
                "actual_position": float(actual),
                "predicted_position": float(estimate),
                "cursor_before": before,
                "cursor_after": after,
                "motor_delta": delta,
                "side_correct": side_correct,
            }
        )

    mae = float(np.mean(np.abs(error)))
    return {
        "active_feature_count": model.active_feature_count,
        "feature_keys": FEATURE_KEYS,
        "train_positions": train_positions,
        "test_positions": test_positions,
        "predictions": trials,
        "mae": mae,
        "all_test_sides_correct": all(
            bool(trial["side_correct"]) for trial in trials
        ),
    }


def run_validation(
    data_directory: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    motor_config = ROOT / str(config["motor_config"])
    characterization = run_characterization(data_directory, motor_config)
    axes = {
        axis: _axis_calibration(
            axis,
            characterization["axes"][axis]["runs"],
            config,
        )
        for axis in ("horizontal", "vertical")
    }
    gates = config["gates"]
    checks: dict[str, bool] = {}
    for axis, result in axes.items():
        checks[f"{axis}_mae"] = (
            float(result["mae"]) <= float(gates["maximum_test_mae"])
        )
        if bool(gates["require_test_side_correct"]):
            checks[f"{axis}_side_correct"] = bool(
                result["all_test_sides_correct"]
            )
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "A fixed ridge calibration uses only anatomically selected descending "
            "motor-population voltage state. Training and evaluation positions are "
            "disjoint. Cursor motion is produced only from the calibrated motor state; "
            "no visual/topographic features enter the motor decoder."
        ),
        "config": config,
        "motor_characterization": characterization,
        "axes": axes,
        "gates": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": gates,
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
    result = run_validation(args.data_directory, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "gates": result["gates"],
                "axes": result["axes"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
