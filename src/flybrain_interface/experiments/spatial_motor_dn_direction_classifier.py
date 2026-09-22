"""Classify LEFT/RIGHT directly from individual descending-neuron state."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
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
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-dn-direction-classifier-v1.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported DN direction-classifier schema")
    return payload


def _label(position: float, center: float) -> int:
    if position < center:
        return -1
    if position > center:
        return 1
    return 0


def _runs_by_position(
    characterization: dict[str, Any],
    axis: str,
) -> dict[float, dict[str, Any]]:
    return {
        float(run["position"]): run
        for run in characterization["axes"][axis]["runs"]
    }


def _matrix(
    runs: dict[float, dict[str, Any]],
    positions: tuple[float, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    rows: list[np.ndarray] = []
    names: tuple[str, ...] | None = None
    for position in positions:
        if position not in runs:
            raise ValueError(f"missing motor-state position: {position}")
        row, row_names = _dn_features(runs[position])
        if names is None:
            names = row_names
        elif names != row_names:
            raise RuntimeError("DN feature ordering changed across positions")
        rows.append(row)
    assert names is not None
    return np.vstack(rows), names


def _fit_linear_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha: float,
    epsilon: float,
) -> dict[str, Any]:
    x_standard, mean, scale = _standardize_fit(x_train, epsilon=epsilon)
    design = np.column_stack((np.ones(x_standard.shape[0]), x_standard))
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        design.T @ design + alpha * penalty,
        design.T @ y_train,
    )
    return {"mean": mean, "scale": scale, "weights": weights}


def _score_linear(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    standardized = _standardize_apply(
        features,
        np.asarray(model["mean"], dtype=np.float64),
        np.asarray(model["scale"], dtype=np.float64),
    )
    design = np.column_stack((np.ones(standardized.shape[0]), standardized))
    return design @ np.asarray(model["weights"], dtype=np.float64)


def _fit_rbf_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha: float,
    gamma: float,
    epsilon: float,
) -> dict[str, Any]:
    if gamma <= 0.0:
        raise ValueError("RBF gamma must be positive")
    standardized, mean, scale = _standardize_fit(x_train, epsilon=epsilon)
    kernel = _rbf_kernel(standardized, standardized, gamma=gamma)
    weights = np.linalg.solve(
        kernel + alpha * np.eye(kernel.shape[0], dtype=np.float64),
        y_train,
    )
    return {
        "mean": mean,
        "scale": scale,
        "centers": standardized,
        "weights": weights,
        "gamma": gamma,
    }


def _score_rbf(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
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
    predicted = np.where(scores >= 0.0, 1, -1)
    return float(np.mean(predicted == labels))


def _loocv(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    fit: Callable[..., dict[str, Any]],
    score: Callable[[dict[str, Any], np.ndarray], np.ndarray],
    params: dict[str, Any],
) -> dict[str, Any]:
    predictions: list[float] = []
    for held_out in range(features.shape[0]):
        mask = np.ones(features.shape[0], dtype=bool)
        mask[held_out] = False
        model = fit(features[mask], labels[mask], **params)
        predictions.append(float(score(model, features[held_out : held_out + 1])[0]))
    scores = np.asarray(predictions, dtype=np.float64)
    return {
        "accuracy": _accuracy(labels, scores),
        "mean_abs_margin": float(np.mean(np.abs(scores))),
        "scores": [float(value) for value in scores],
    }


def _select_linear(
    features: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    epsilon = float(config["standardization_epsilon"])
    rows: list[dict[str, Any]] = []
    for alpha in config["linear_alpha_grid"]:
        value = float(alpha)
        cv = _loocv(
            features,
            labels,
            fit=_fit_linear_classifier,
            score=_score_linear,
            params={"alpha": value, "epsilon": epsilon},
        )
        rows.append({"alpha": value, **cv})
    rows.sort(
        key=lambda row: (
            -float(row["accuracy"]),
            -float(row["mean_abs_margin"]),
            float(row["alpha"]),
        )
    )
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
            alpha_value = float(alpha)
            gamma_value = float(gamma)
            cv = _loocv(
                features,
                labels,
                fit=_fit_rbf_classifier,
                score=_score_rbf,
                params={
                    "alpha": alpha_value,
                    "gamma": gamma_value,
                    "epsilon": epsilon,
                },
            )
            rows.append(
                {
                    "alpha": alpha_value,
                    "gamma": gamma_value,
                    **cv,
                }
            )
    rows.sort(
        key=lambda row: (
            -float(row["accuracy"]),
            -float(row["mean_abs_margin"]),
            float(row["gamma"]),
            float(row["alpha"]),
        )
    )
    return {"winner": rows[0], "candidates": rows}


def _evaluate(
    name: str,
    model: dict[str, Any],
    score_fn: Callable[[dict[str, Any], np.ndarray], np.ndarray],
    x_test: np.ndarray,
    test_positions: tuple[float, ...],
    center: float,
    cursor_config: dict[str, Any],
) -> dict[str, Any]:
    labels = np.asarray([_label(position, center) for position in test_positions])
    scores = score_fn(model, x_test)
    predicted = np.where(scores >= 0.0, 1, -1)
    start = float(cursor_config["start"])
    step_scale = float(cursor_config["step_scale"])
    magnitude = float(cursor_config["command_magnitude"])
    trials: list[dict[str, Any]] = []
    for position, label, prediction, score_value in zip(
        test_positions,
        labels,
        predicted,
        scores,
        strict=True,
    ):
        cursor = VirtualCursorEnvironment(x=start, y=0.5, step_scale=step_scale)
        command = MotorCommand(
            delta_x=float(prediction) * magnitude,
            delta_y=0.0,
        )
        before = cursor.state.x
        after = cursor.apply(command).x
        trials.append(
            {
                "position": position,
                "expected": "LEFT" if label < 0 else "RIGHT",
                "predicted": "LEFT" if prediction < 0 else "RIGHT",
                "score": float(score_value),
                "correct": bool(prediction == label),
                "cursor_before": before,
                "cursor_after": after,
            }
        )
    return {
        "name": name,
        "accuracy": float(np.mean(predicted == labels)),
        "mean_abs_margin": float(np.mean(np.abs(scores))),
        "trials": trials,
    }


def benchmark_from_artifact(
    artifact: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    axis = str(config["axis"])
    center = float(config["center_position"])
    characterization = artifact["motor_characterization"]
    runs = _runs_by_position(characterization, axis)
    train_positions = tuple(float(v) for v in config["train_positions"])
    test_positions = tuple(float(v) for v in config["test_positions"])
    if set(train_positions) & set(test_positions):
        raise ValueError("train and test positions must be disjoint")
    if center not in runs:
        raise ValueError("center calibration position missing from artifact")

    x_train, feature_names = _matrix(runs, train_positions)
    x_test, test_names = _matrix(runs, test_positions)
    if feature_names != test_names:
        raise RuntimeError("train/test DN feature ordering mismatch")
    labels = np.asarray([_label(position, center) for position in train_positions])
    if np.any(labels == 0):
        raise ValueError("directional training positions cannot include center")
    if set(labels.tolist()) != {-1, 1}:
        raise ValueError("training set must contain both LEFT and RIGHT examples")

    linear_selection = _select_linear(x_train, labels, config)
    linear_alpha = float(linear_selection["winner"]["alpha"])
    linear_model = _fit_linear_classifier(
        x_train,
        labels,
        alpha=linear_alpha,
        epsilon=float(config["standardization_epsilon"]),
    )
    linear_eval = _evaluate(
        "linear",
        linear_model,
        _score_linear,
        x_test,
        test_positions,
        center,
        config["cursor"],
    )

    rbf_selection = _select_rbf(x_train, labels, config)
    rbf_winner = rbf_selection["winner"]
    rbf_model = _fit_rbf_classifier(
        x_train,
        labels,
        alpha=float(rbf_winner["alpha"]),
        gamma=float(rbf_winner["gamma"]),
        epsilon=float(config["standardization_epsilon"]),
    )
    rbf_eval = _evaluate(
        "rbf",
        rbf_model,
        _score_rbf,
        x_test,
        test_positions,
        center,
        config["cursor"],
    )

    center_features, center_names = _matrix(runs, (center,))
    if center_names != feature_names:
        raise RuntimeError("center DN feature ordering mismatch")
    center_scores = {
        "linear": float(_score_linear(linear_model, center_features)[0]),
        "rbf": float(_score_rbf(rbf_model, center_features)[0]),
    }

    gates = config["gates"]
    minimum_accuracy = float(gates["minimum_test_accuracy"])
    for result in (linear_eval, rbf_eval):
        result["passes"] = bool(result["accuracy"] >= minimum_accuracy)

    return {
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "train_positions": train_positions,
        "test_positions": test_positions,
        "center_position": center,
        "center_scores": center_scores,
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
    artifact_path = ROOT / str(config["source_artifact"])
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    result = benchmark_from_artifact(artifact, config)
    payload = {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "LEFT/RIGHT action is classified only from individual DNa02/DNg13/DNp09/"
            "MDN voltage, drive and spike state. Hyperparameters are selected by "
            "leave-one-out cross-validation on directional training positions only. "
            "Held-out positions are used once for the final cursor-direction gate."
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
                "linear": result["linear"]["evaluation"],
                "rbf": result["rbf"]["evaluation"],
                "center_scores": result["center_scores"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
