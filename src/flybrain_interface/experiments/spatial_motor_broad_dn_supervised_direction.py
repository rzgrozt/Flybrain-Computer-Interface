"""Train-only supervised feature selection over broad descending-neuron state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from flybrain_interface.experiments.spatial_motor_broad_dn_pca_classifier import (
    _feature_vector,
    _samples_by_position,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = (
    ROOT / "configs" / "spatial-motor-broad-dn-supervised-direction-v1.json"
)
CLASS_NAMES = ("LEFT", "RIGHT")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported broad-DN supervised-direction schema")
    return payload


def _matrix(
    samples: dict[float, dict[str, Any]],
    positions: tuple[float, ...],
    modality: str,
) -> np.ndarray:
    return np.vstack(
        [_feature_vector(samples[position], modality) for position in positions]
    )


def _labels(positions: tuple[float, ...], center: float) -> np.ndarray:
    return np.asarray(
        [0 if position < center else 1 for position in positions],
        dtype=np.int64,
    )


def _feature_scores(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    left = features[labels == 0]
    right = features[labels == 1]
    if left.size == 0 or right.size == 0:
        raise ValueError("both directional classes are required")
    mean_delta = np.mean(right, axis=0) - np.mean(left, axis=0)
    pooled = np.sqrt(
        0.5 * (np.var(left, axis=0) + np.var(right, axis=0))
    )
    return np.abs(mean_delta) / np.maximum(pooled, epsilon)


def _select_indices(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    top_k: int,
    epsilon: float,
) -> np.ndarray:
    scores = _feature_scores(features, labels, epsilon=epsilon)
    count = min(int(top_k), scores.size)
    order = np.argsort(-scores, kind="stable")
    return order[:count].astype(np.int64, copy=False)


def _fit_standardizer(
    features: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    safe_scale = np.where(scale > epsilon, scale, np.inf)
    return (features - mean) / safe_scale, mean, safe_scale


def _apply_standardizer(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (features - mean) / scale


def _one_hot(labels: np.ndarray) -> np.ndarray:
    result = np.zeros((labels.size, 2), dtype=np.float64)
    result[np.arange(labels.size), labels] = 1.0
    return result


def _fit_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    top_k: int,
    alpha: float,
    epsilon: float,
) -> dict[str, Any]:
    indices = _select_indices(
        features,
        labels,
        top_k=top_k,
        epsilon=epsilon,
    )
    selected = features[:, indices]
    standardized, mean, scale = _fit_standardizer(
        selected,
        epsilon=epsilon,
    )
    design = np.column_stack((np.ones(standardized.shape[0]), standardized))
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(
        design.T @ design + alpha * penalty,
        design.T @ _one_hot(labels),
    )
    return {
        "indices": indices,
        "mean": mean,
        "scale": scale,
        "weights": weights,
    }


def _score_model(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    indices = np.asarray(model["indices"], dtype=np.int64)
    selected = features[:, indices]
    standardized = _apply_standardizer(
        selected,
        np.asarray(model["mean"], dtype=np.float64),
        np.asarray(model["scale"], dtype=np.float64),
    )
    design = np.column_stack((np.ones(standardized.shape[0]), standardized))
    return design @ np.asarray(model["weights"], dtype=np.float64)


def _accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean(np.argmax(scores, axis=1) == labels))


def _loocv(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    top_k: int,
    alpha: float,
    epsilon: float,
) -> dict[str, float]:
    predictions: list[int] = []
    margins: list[float] = []
    for held_out in range(features.shape[0]):
        mask = np.ones(features.shape[0], dtype=bool)
        mask[held_out] = False
        model = _fit_model(
            features[mask],
            labels[mask],
            top_k=top_k,
            alpha=alpha,
            epsilon=epsilon,
        )
        scores = _score_model(model, features[held_out : held_out + 1])[0]
        prediction = int(np.argmax(scores))
        predictions.append(prediction)
        margins.append(float(abs(scores[1] - scores[0])))
    return {
        "accuracy": float(np.mean(np.asarray(predictions) == labels)),
        "mean_abs_margin": float(np.mean(margins)),
    }


def _select_candidate(
    samples: dict[float, dict[str, Any]],
    train_positions: tuple[float, ...],
    labels: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    epsilon = float(config["standardization_epsilon"])
    rows: list[dict[str, Any]] = []
    for modality in config["modalities"]:
        features = _matrix(samples, train_positions, str(modality))
        for top_k in config["top_k_grid"]:
            for alpha in config["alpha_grid"]:
                cv = _loocv(
                    features,
                    labels,
                    top_k=int(top_k),
                    alpha=float(alpha),
                    epsilon=epsilon,
                )
                rows.append(
                    {
                        "modality": str(modality),
                        "top_k": int(top_k),
                        "alpha": float(alpha),
                        "loocv_accuracy": cv["accuracy"],
                        "loocv_mean_abs_margin": cv["mean_abs_margin"],
                    }
                )
    rows.sort(
        key=lambda row: (
            -float(row["loocv_accuracy"]),
            int(row["top_k"]),
            -float(row["loocv_mean_abs_margin"]),
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
    labels = _labels(positions, center)
    scores = _score_model(model, features)
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
    labels = _labels(train_positions, center)
    selection = _select_candidate(samples, train_positions, labels, config)
    winner = selection["winner"]
    modality = str(winner["modality"])
    x_train = _matrix(samples, train_positions, modality)
    model = _fit_model(
        x_train,
        labels,
        top_k=int(winner["top_k"]),
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
        "selected_feature_indices": [
            int(value) for value in np.asarray(model["indices"])
        ],
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
            "Feature ranking, model selection and regularization are performed only on "
            "directional training positions. The held-out directional positions are "
            "used once for the final gate. Features remain broad descending-neuron "
            "state only; no screen coordinates are exposed to the motor model."
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
                "near_center_diagnostic": result[
                    "near_center_diagnostic"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
