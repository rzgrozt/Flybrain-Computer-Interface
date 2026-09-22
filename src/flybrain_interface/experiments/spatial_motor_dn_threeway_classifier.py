"""Three-way LEFT/NEUTRAL/RIGHT classifier over descending-neuron state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from flybrain_interface.contracts import MotorCommand
from flybrain_interface.environment.cursor import VirtualCursorEnvironment
from flybrain_interface.experiments.spatial_motor_dn_nonlinear_readout import (
    _dn_features,
    _rbf_kernel,
    _standardize_apply,
    _standardize_fit,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-dn-threeway-classifier-v1.json"
CLASS_NAMES = ("LEFT", "NEUTRAL", "RIGHT")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported DN three-way classifier schema")
    return payload


def _label(position: float, center: float, neutral_half_width: float) -> int:
    if position < center - neutral_half_width:
        return 0
    if position > center + neutral_half_width:
        return 2
    return 1


def _collect_runs(
    artifacts: list[dict[str, Any]],
    axis: str,
) -> dict[float, dict[str, Any]]:
    runs: dict[float, dict[str, Any]] = {}
    for artifact in artifacts:
        characterization = artifact.get("motor_characterization", artifact)
        for run in characterization["axes"][axis]["runs"]:
            runs[float(run["position"])] = run
    return runs


def _matrix(
    runs: dict[float, dict[str, Any]],
    positions: tuple[float, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    rows: list[np.ndarray] = []
    names: tuple[str, ...] | None = None
    for position in positions:
        if position not in runs:
            raise ValueError(f"missing DN state for position {position}")
        row, row_names = _dn_features(runs[position])
        if names is None:
            names = row_names
        elif names != row_names:
            raise RuntimeError("DN feature ordering changed across runs")
        rows.append(row)
    assert names is not None
    return np.vstack(rows), names


def _one_hot(labels: np.ndarray, class_count: int = 3) -> np.ndarray:
    result = np.zeros((labels.size, class_count), dtype=np.float64)
    result[np.arange(labels.size), labels] = 1.0
    return result


def _fit_linear(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float,
    epsilon: float,
) -> dict[str, Any]:
    standardized, mean, scale = _standardize_fit(features, epsilon=epsilon)
    design = np.column_stack((np.ones(standardized.shape[0]), standardized))
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        design.T @ design + alpha * penalty,
        design.T @ _one_hot(labels),
    )
    return {"mean": mean, "scale": scale, "weights": weights}


def _scores_linear(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    standardized = _standardize_apply(
        features,
        np.asarray(model["mean"], dtype=np.float64),
        np.asarray(model["scale"], dtype=np.float64),
    )
    design = np.column_stack((np.ones(standardized.shape[0]), standardized))
    return design @ np.asarray(model["weights"], dtype=np.float64)


def _fit_rbf(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float,
    gamma: float,
    epsilon: float,
) -> dict[str, Any]:
    standardized, mean, scale = _standardize_fit(features, epsilon=epsilon)
    kernel = _rbf_kernel(standardized, standardized, gamma=gamma)
    weights = np.linalg.solve(
        kernel + alpha * np.eye(kernel.shape[0], dtype=np.float64),
        _one_hot(labels),
    )
    return {
        "mean": mean,
        "scale": scale,
        "centers": standardized,
        "weights": weights,
        "gamma": gamma,
    }


def _scores_rbf(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    standardized = _standardize_apply(
        features,
        np.asarray(model["mean"], dtype=np.float64),
        np.asarray(model["scale"], dtype=np.float64),
    )
    kernel = _rbf_kernel(
        standardized,
        np.asarray(model["centers"], dtype=np.float64),
        gamma=float(model["gamma"]),
    )
    return kernel @ np.asarray(model["weights"], dtype=np.float64)


def _accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean(np.argmax(scores, axis=1) == labels))


def _loocv(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    kind: str,
    alpha: float,
    gamma: float | None,
    epsilon: float,
) -> float:
    predictions: list[int] = []
    for held_out in range(features.shape[0]):
        mask = np.ones(features.shape[0], dtype=bool)
        mask[held_out] = False
        if kind == "linear":
            model = _fit_linear(
                features[mask],
                labels[mask],
                alpha=alpha,
                epsilon=epsilon,
            )
            scores = _scores_linear(model, features[held_out : held_out + 1])
        elif kind == "rbf":
            assert gamma is not None
            model = _fit_rbf(
                features[mask],
                labels[mask],
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
            )
            scores = _scores_rbf(model, features[held_out : held_out + 1])
        else:
            raise ValueError(f"unsupported classifier kind: {kind}")
        predictions.append(int(np.argmax(scores[0])))
    return float(np.mean(np.asarray(predictions) == labels))


def _select_linear(
    features: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    epsilon = float(config["standardization_epsilon"])
    rows = [
        {
            "alpha": float(alpha),
            "loocv_accuracy": _loocv(
                features,
                labels,
                kind="linear",
                alpha=float(alpha),
                gamma=None,
                epsilon=epsilon,
            ),
        }
        for alpha in config["linear_alpha_grid"]
    ]
    rows.sort(key=lambda row: (-row["loocv_accuracy"], row["alpha"]))
    return {"winner": rows[0], "candidates": rows}


def _select_rbf(
    features: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    epsilon = float(config["standardization_epsilon"])
    rows: list[dict[str, Any]] = []
    for alpha in config["rbf_alpha_grid"]:
        for gamma in config["rbf_gamma_grid"]:
            rows.append(
                {
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    "loocv_accuracy": _loocv(
                        features,
                        labels,
                        kind="rbf",
                        alpha=float(alpha),
                        gamma=float(gamma),
                        epsilon=epsilon,
                    ),
                }
            )
    rows.sort(
        key=lambda row: (
            -row["loocv_accuracy"],
            row["gamma"],
            row["alpha"],
        )
    )
    return {"winner": rows[0], "candidates": rows}


def _evaluate(
    scores: np.ndarray,
    positions: tuple[float, ...],
    labels: np.ndarray,
    cursor_config: dict[str, Any],
) -> dict[str, Any]:
    predicted = np.argmax(scores, axis=1)
    start = float(cursor_config["start"])
    step_scale = float(cursor_config["step_scale"])
    magnitude = float(cursor_config["command_magnitude"])
    commands = (-magnitude, 0.0, magnitude)
    trials: list[dict[str, Any]] = []
    for position, expected, prediction, row_scores in zip(
        positions,
        labels,
        predicted,
        scores,
        strict=True,
    ):
        cursor = VirtualCursorEnvironment(x=start, y=0.5, step_scale=step_scale)
        command = MotorCommand(delta_x=commands[int(prediction)], delta_y=0.0)
        before = cursor.state.x
        after = cursor.apply(command).x
        trials.append(
            {
                "position": float(position),
                "expected": CLASS_NAMES[int(expected)],
                "predicted": CLASS_NAMES[int(prediction)],
                "scores": [float(value) for value in row_scores],
                "correct": bool(expected == prediction),
                "cursor_before": before,
                "cursor_after": after,
            }
        )
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "trials": trials,
    }


def benchmark(
    artifacts: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    axis = str(config["axis"])
    center = float(config["center_position"])
    neutral_half_width = float(config["neutral_half_width"])
    train_positions = tuple(float(v) for v in config["train_positions"])
    test_positions = tuple(float(v) for v in config["test_positions"])
    if set(train_positions) & set(test_positions):
        raise ValueError("train and test positions must be disjoint")

    runs = _collect_runs(artifacts, axis)
    x_train, feature_names = _matrix(runs, train_positions)
    x_test, test_names = _matrix(runs, test_positions)
    if feature_names != test_names:
        raise RuntimeError("DN feature ordering mismatch")
    y_train = np.asarray(
        [_label(position, center, neutral_half_width) for position in train_positions]
    )
    y_test = np.asarray(
        [_label(position, center, neutral_half_width) for position in test_positions]
    )
    if set(y_train.tolist()) != {0, 1, 2}:
        raise ValueError("training set must include LEFT, NEUTRAL and RIGHT")

    linear_selection = _select_linear(x_train, y_train, config)
    linear_model = _fit_linear(
        x_train,
        y_train,
        alpha=float(linear_selection["winner"]["alpha"]),
        epsilon=float(config["standardization_epsilon"]),
    )
    linear_eval = _evaluate(
        _scores_linear(linear_model, x_test),
        test_positions,
        y_test,
        config["cursor"],
    )

    rbf_selection = _select_rbf(x_train, y_train, config)
    rbf_winner = rbf_selection["winner"]
    rbf_model = _fit_rbf(
        x_train,
        y_train,
        alpha=float(rbf_winner["alpha"]),
        gamma=float(rbf_winner["gamma"]),
        epsilon=float(config["standardization_epsilon"]),
    )
    rbf_eval = _evaluate(
        _scores_rbf(rbf_model, x_test),
        test_positions,
        y_test,
        config["cursor"],
    )

    threshold = float(config["gates"]["minimum_test_accuracy"])
    linear_eval["passes"] = bool(linear_eval["accuracy"] >= threshold)
    rbf_eval["passes"] = bool(rbf_eval["accuracy"] >= threshold)
    return {
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "train_positions": train_positions,
        "test_positions": test_positions,
        "neutral_half_width": neutral_half_width,
        "linear": {
            "selection": linear_selection,
            "evaluation": linear_eval,
        },
        "rbf": {
            "selection": rbf_selection,
            "evaluation": rbf_eval,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts = [
        json.loads((ROOT / str(path)).read_text(encoding="utf-8"))
        for path in config["source_artifacts"]
    ]
    result = benchmark(artifacts, config)
    payload = {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "LEFT/NEUTRAL/RIGHT action is decoded only from individual descending-"
            "neuron voltage, drive and spike state. Hyperparameters are selected by "
            "training-only leave-one-out validation. Held-out positions are used once "
            "for the final virtual-cursor action gate."
        ),
        "config": config,
        "benchmark": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "linear": result["linear"],
                "rbf": result["rbf"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
