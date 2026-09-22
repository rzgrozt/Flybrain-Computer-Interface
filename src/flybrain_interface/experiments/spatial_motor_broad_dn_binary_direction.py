"""Binary LEFT/RIGHT PCA classifier over broad descending-neuron temporal state."""

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
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-broad-dn-binary-direction-v1.json"
CLASS_NAMES = ("LEFT", "RIGHT")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported broad-DN binary classifier schema")
    return payload


def _label(position: float, center: float) -> int:
    return 0 if position < center else 1


def _one_hot(labels: np.ndarray) -> np.ndarray:
    result = np.zeros((labels.size, 2), dtype=np.float64)
    result[np.arange(labels.size), labels] = 1.0
    return result


def _fit_ridge(features: np.ndarray, labels: np.ndarray, *, alpha: float) -> np.ndarray:
    design = np.column_stack((np.ones(features.shape[0]), features))
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        design.T @ design + alpha * penalty,
        design.T @ _one_hot(labels),
    )


def _scores(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(features.shape[0]), features))
    return design @ weights


def _fit_pipeline(
    features: np.ndarray,
    labels: np.ndarray,
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
    weights = _fit_ridge(projected, labels, alpha=alpha)
    return {
        "standard_mean": standard_mean,
        "standard_scale": standard_scale,
        "pca_mean": pca_mean,
        "basis": basis,
        "weights": weights,
    }


def _score_pipeline(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
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
    return _scores(np.asarray(model["weights"], dtype=np.float64), projected)


def _accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean(np.argmax(scores, axis=1) == labels))


def _loocv_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    components: int,
    alpha: float,
    epsilon: float,
) -> float:
    predictions: list[int] = []
    for held_out in range(features.shape[0]):
        mask = np.ones(features.shape[0], dtype=bool)
        mask[held_out] = False
        model = _fit_pipeline(
            features[mask],
            labels[mask],
            components=components,
            alpha=alpha,
            epsilon=epsilon,
        )
        predictions.append(
            int(
                np.argmax(
                    _score_pipeline(
                        model,
                        features[held_out : held_out + 1],
                    )[0]
                )
            )
        )
    return float(np.mean(np.asarray(predictions) == labels))


def _select_candidate(
    samples: dict[float, dict[str, Any]],
    train_positions: tuple[float, ...],
    labels: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    epsilon = float(config["standardization_epsilon"])
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
                        "loocv_accuracy": _loocv_accuracy(
                            features,
                            labels,
                            components=component_count,
                            alpha=alpha_value,
                            epsilon=epsilon,
                        ),
                    }
                )
    rows.sort(
        key=lambda row: (
            -float(row["loocv_accuracy"]),
            int(row["components"]),
            float(row["alpha"]),
            str(row["modality"]),
        )
    )
    return {"winner": rows[0], "candidates": rows}


def _evaluate(
    model: dict[str, Any],
    features: np.ndarray,
    positions: tuple[float, ...],
    center: float,
) -> dict[str, Any]:
    labels = np.asarray([_label(position, center) for position in positions])
    scores = _score_pipeline(model, features)
    predicted = np.argmax(scores, axis=1)
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "pairs": [
            {
                "position": float(position),
                "expected": CLASS_NAMES[int(expected)],
                "predicted": CLASS_NAMES[int(prediction)],
                "scores": [float(value) for value in row_scores],
                "correct": bool(expected == prediction),
            }
            for position, expected, prediction, row_scores in zip(
                positions,
                labels,
                predicted,
                scores,
                strict=True,
            )
        ],
    }


def benchmark(artifact: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    axis = str(config["axis"])
    center = float(config["center_position"])
    train_positions = tuple(float(v) for v in config["train_positions"])
    test_positions = tuple(float(v) for v in config["test_positions"])
    diagnostic_positions = tuple(
        float(v) for v in config["diagnostic_near_center_positions"]
    )
    samples = _samples_by_position(artifact, axis)
    y_train = np.asarray([_label(position, center) for position in train_positions])
    if set(y_train.tolist()) != {0, 1}:
        raise ValueError("binary training set must contain LEFT and RIGHT")

    selection = _select_candidate(samples, train_positions, y_train, config)
    winner = selection["winner"]
    modality = str(winner["modality"])
    x_train = _matrix(samples, train_positions, modality)
    model = _fit_pipeline(
        x_train,
        y_train,
        components=int(winner["components"]),
        alpha=float(winner["alpha"]),
        epsilon=float(config["standardization_epsilon"]),
    )
    test = _evaluate(
        model,
        _matrix(samples, test_positions, modality),
        test_positions,
        center,
    )
    diagnostic = _evaluate(
        model,
        _matrix(samples, diagnostic_positions, modality),
        diagnostic_positions,
        center,
    )
    test["passes"] = bool(
        test["accuracy"] >= float(config["gates"]["minimum_test_accuracy"])
    )
    return {
        "winner": winner,
        "selection": selection,
        "test": test,
        "near_center_diagnostic": diagnostic,
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
            "The motor policy learns only LEFT versus RIGHT from broad descending-"
            "neuron temporal state. Stopping is intentionally delegated to the closed-"
            "loop environment tolerance, so near-center points are diagnostic only and "
            "do not influence model selection or the directional test gate."
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
                "test": result["test"],
                "near_center_diagnostic": result["near_center_diagnostic"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
