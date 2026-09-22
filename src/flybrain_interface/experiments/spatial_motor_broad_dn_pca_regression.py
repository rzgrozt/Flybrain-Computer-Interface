"""Train-only PCA plus ridge regression over broad descending-neuron state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from flybrain_interface.experiments.spatial_motor_broad_dn_pca_classifier import (
    _apply_pca,
    _apply_standardizer,
    _fit_pca,
    _fit_standardizer,
    _matrix,
    _samples_by_position,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-broad-dn-pca-regression-v1.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported broad-DN PCA regression schema")
    return payload


def _fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    design = np.column_stack((np.ones(features.shape[0]), features))
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        design.T @ design + alpha * penalty,
        design.T @ targets,
    )


def _predict(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(features.shape[0]), features))
    return design @ weights


def _fit_pipeline(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    components: int,
    alpha: float,
    epsilon: float,
) -> dict[str, Any]:
    standardized, standard_mean, standard_scale = _fit_standardizer(
        features,
        epsilon=epsilon,
    )
    projected, pca_mean, basis = _fit_pca(
        standardized,
        components=components,
    )
    weights = _fit_ridge(projected, targets, alpha=alpha)
    return {
        "standard_mean": standard_mean,
        "standard_scale": standard_scale,
        "pca_mean": pca_mean,
        "basis": basis,
        "weights": weights,
    }


def _predict_pipeline(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    standardized = _apply_standardizer(
        features,
        np.asarray(model["standard_mean"], dtype=np.float64),
        np.asarray(model["standard_scale"], dtype=np.float64),
    )
    projected = _apply_pca(
        standardized,
        np.asarray(model["pca_mean"], dtype=np.float64),
        np.asarray(model["basis"], dtype=np.float64),
    )
    return _predict(np.asarray(model["weights"], dtype=np.float64), projected)


def _metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    center: float,
) -> dict[str, Any]:
    error = predicted - actual
    side_correct = np.asarray(
        [
            (expected < center and estimate < center)
            or (expected > center and estimate > center)
            or (
                expected == center
                and abs(float(estimate) - center) <= 1e-12
            )
            for expected, estimate in zip(actual, predicted, strict=True)
        ],
        dtype=bool,
    )
    correlation = None
    if (
        actual.size >= 2
        and float(np.std(actual)) > 0.0
        and float(np.std(predicted)) > 1e-12
    ):
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "side_accuracy": float(np.mean(side_correct)),
        "correlation": correlation,
        "pairs": [
            {
                "actual": float(expected),
                "predicted": float(estimate),
                "side_correct": bool(correct),
            }
            for expected, estimate, correct in zip(
                actual,
                predicted,
                side_correct,
                strict=True,
            )
        ],
    }


def _loocv_mae(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    components: int,
    alpha: float,
    epsilon: float,
) -> float:
    predictions: list[float] = []
    for held_out in range(features.shape[0]):
        mask = np.ones(features.shape[0], dtype=bool)
        mask[held_out] = False
        model = _fit_pipeline(
            features[mask],
            targets[mask],
            components=components,
            alpha=alpha,
            epsilon=epsilon,
        )
        prediction = _predict_pipeline(
            model,
            features[held_out : held_out + 1],
        )[0]
        predictions.append(float(prediction))
    return float(
        np.mean(np.abs(np.asarray(predictions, dtype=np.float64) - targets))
    )


def _select_candidate(
    samples: dict[float, dict[str, Any]],
    train_positions: tuple[float, ...],
    config: dict[str, Any],
) -> dict[str, Any]:
    targets = np.asarray(train_positions, dtype=np.float64)
    epsilon = float(config["standardization_epsilon"])
    rows: list[dict[str, Any]] = []
    for modality in config["modalities"]:
        features = _matrix(samples, train_positions, str(modality))
        for components in config["pca_components_grid"]:
            component_count = int(components)
            if component_count >= features.shape[0]:
                continue
            for alpha in config["alpha_grid"]:
                alpha_value = float(alpha)
                rows.append(
                    {
                        "modality": str(modality),
                        "components": component_count,
                        "alpha": alpha_value,
                        "loocv_mae": _loocv_mae(
                            features,
                            targets,
                            components=component_count,
                            alpha=alpha_value,
                            epsilon=epsilon,
                        ),
                    }
                )
    rows.sort(
        key=lambda row: (
            float(row["loocv_mae"]),
            int(row["components"]),
            float(row["alpha"]),
            str(row["modality"]),
        )
    )
    return {"winner": rows[0], "candidates": rows}


def benchmark(artifact: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    axis = str(config["axis"])
    center = float(config["center_position"])
    train_positions = tuple(float(v) for v in config["train_positions"])
    test_positions = tuple(float(v) for v in config["test_positions"])
    samples = _samples_by_position(artifact, axis)

    selection = _select_candidate(samples, train_positions, config)
    winner = selection["winner"]
    modality = str(winner["modality"])
    x_train = _matrix(samples, train_positions, modality)
    x_test = _matrix(samples, test_positions, modality)
    y_train = np.asarray(train_positions, dtype=np.float64)
    y_test = np.asarray(test_positions, dtype=np.float64)
    model = _fit_pipeline(
        x_train,
        y_train,
        components=int(winner["components"]),
        alpha=float(winner["alpha"]),
        epsilon=float(config["standardization_epsilon"]),
    )
    train_prediction = _predict_pipeline(model, x_train)
    test_prediction = _predict_pipeline(model, x_test)
    train_metrics = _metrics(y_train, train_prediction, center=center)
    test_metrics = _metrics(y_test, test_prediction, center=center)
    gates = config["gates"]
    test_metrics["passes"] = bool(
        float(test_metrics["mae"]) <= float(gates["maximum_test_mae"])
        and float(test_metrics["side_accuracy"])
        >= float(gates["minimum_side_accuracy"])
    )
    return {
        "winner": winner,
        "selection": selection,
        "train": train_metrics,
        "test": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    artifact = json.loads(
        (ROOT / str(config["source_artifact"])).read_text(encoding="utf-8")
    )
    result = benchmark(artifact, config)
    payload = {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "Relative horizontal position is regressed only from broad descending-"
            "neuron state. PCA and ridge hyperparameters are selected by training-only "
            "leave-one-out MAE. Held-out positions are used once for the static "
            "generalization gate."
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
                "winner": result["winner"],
                "train": result["train"],
                "test": result["test"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
