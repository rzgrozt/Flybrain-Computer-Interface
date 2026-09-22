"""Train-only PCA plus ridge classifier over broad descending-neuron state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from flybrain_interface.contracts import MotorCommand
from flybrain_interface.environment.cursor import VirtualCursorEnvironment

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-broad-dn-pca-classifier-v1.json"
CLASS_NAMES = ("LEFT", "NEUTRAL", "RIGHT")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported broad-DN PCA classifier schema")
    return payload


def _label(position: float, center: float, neutral_half_width: float) -> int:
    if position < center - neutral_half_width:
        return 0
    if position > center + neutral_half_width:
        return 2
    return 1


def _samples_by_position(
    artifact: dict[str, Any],
    axis: str,
) -> dict[float, dict[str, Any]]:
    return {
        float(sample["position"]): sample
        for sample in artifact["axes"][axis]["samples"]
    }


def _feature_vector(sample: dict[str, Any], modality: str) -> np.ndarray:
    if modality in {"voltage", "drive", "spike"}:
        return np.asarray(sample[modality], dtype=np.float64)
    if modality == "concat":
        return np.concatenate(
            [
                np.asarray(sample["voltage"], dtype=np.float64),
                np.asarray(sample["drive"], dtype=np.float64),
                np.asarray(sample["spike"], dtype=np.float64),
            ]
        )
    if modality in {"temporal_voltage", "temporal_drive"}:
        temporal = np.asarray(sample[modality], dtype=np.float64)
        if temporal.ndim != 2 or temporal.shape[0] == 0:
            raise ValueError(f"missing temporal feature matrix: {modality}")
        return temporal.reshape(-1)
    if modality == "temporal_concat":
        temporal_voltage = np.asarray(sample["temporal_voltage"], dtype=np.float64)
        temporal_drive = np.asarray(sample["temporal_drive"], dtype=np.float64)
        if temporal_voltage.shape != temporal_drive.shape or temporal_voltage.ndim != 2:
            raise ValueError("temporal voltage/drive matrices must match")
        return np.concatenate(
            (
                temporal_voltage.reshape(-1),
                temporal_drive.reshape(-1),
            )
        )
    raise ValueError(f"unsupported modality: {modality}")


def _matrix(
    samples: dict[float, dict[str, Any]],
    positions: tuple[float, ...],
    modality: str,
) -> np.ndarray:
    rows = []
    for position in positions:
        if position not in samples:
            raise ValueError(f"missing broad-DN state for position {position}")
        rows.append(_feature_vector(samples[position], modality))
    return np.vstack(rows)


def _fit_standardizer(
    features: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    safe_scale = np.where(scale > epsilon, scale, np.inf)
    standardized = (features - mean) / safe_scale
    return standardized, mean, safe_scale


def _apply_standardizer(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return np.asarray((features - mean) / scale, dtype=np.float64)


def _fit_pca(
    features: np.ndarray,
    *,
    components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if components <= 0:
        raise ValueError("PCA components must be positive")
    mean = np.mean(features, axis=0)
    centered = features - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    max_components = min(components, vt.shape[0])
    basis = vt[:max_components].T
    projected = centered @ basis
    return projected, mean, basis


def _apply_pca(
    features: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    return np.asarray((features - mean) @ basis, dtype=np.float64)


def _one_hot(labels: np.ndarray) -> np.ndarray:
    result = np.zeros((labels.size, 3), dtype=np.float64)
    result[np.arange(labels.size), labels] = 1.0
    return result


def _fit_ridge_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    design = np.column_stack((np.ones(features.shape[0]), features))
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        design.T @ design + alpha * penalty,
        design.T @ _one_hot(labels),
    )


def _scores(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(features.shape[0]), features))
    return np.asarray(design @ weights, dtype=np.float64)


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
    weights = _fit_ridge_classifier(projected, labels, alpha=alpha)
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
    return _scores(
        np.asarray(model["weights"], dtype=np.float64),
        projected,
    )


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
        scores = _score_pipeline(
            model,
            features[held_out : held_out + 1],
        )
        predictions.append(int(np.argmax(scores[0])))
    return float(np.mean(np.asarray(predictions) == labels))


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
        for components in config["pca_components_grid"]:
            component_count = int(components)
            if component_count >= features.shape[0]:
                continue
            for alpha in config["alpha_grid"]:
                alpha_value = float(alpha)
                accuracy = _loocv_accuracy(
                    features,
                    labels,
                    components=component_count,
                    alpha=alpha_value,
                    epsilon=epsilon,
                )
                rows.append(
                    {
                        "modality": str(modality),
                        "components": component_count,
                        "alpha": alpha_value,
                        "loocv_accuracy": accuracy,
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
    if not rows:
        raise RuntimeError("no PCA classifier candidates were evaluated")
    return {"winner": rows[0], "candidates": rows}


def _evaluate(
    model: dict[str, Any],
    x_test: np.ndarray,
    positions: tuple[float, ...],
    labels: np.ndarray,
    cursor_config: dict[str, Any],
) -> dict[str, Any]:
    scores = _score_pipeline(model, x_test)
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
        before = cursor.state.x
        after = cursor.apply(
            MotorCommand(
                delta_x=commands[int(prediction)],
                delta_y=0.0,
            )
        ).x
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
    artifact: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    axis = str(config["axis"])
    center = float(config["center_position"])
    neutral_half_width = float(config["neutral_half_width"])
    train_positions = tuple(float(v) for v in config["train_positions"])
    test_positions = tuple(float(v) for v in config["test_positions"])
    if set(train_positions) & set(test_positions):
        raise ValueError("train and test positions must be disjoint")

    samples = _samples_by_position(artifact, axis)
    y_train = np.asarray(
        [_label(position, center, neutral_half_width) for position in train_positions]
    )
    y_test = np.asarray(
        [_label(position, center, neutral_half_width) for position in test_positions]
    )
    if set(y_train.tolist()) != {0, 1, 2}:
        raise ValueError("training set must contain all three direction classes")

    selection = _select_candidate(
        samples,
        train_positions,
        y_train,
        config,
    )
    winner = selection["winner"]
    modality = str(winner["modality"])
    x_train = _matrix(samples, train_positions, modality)
    x_test = _matrix(samples, test_positions, modality)
    model = _fit_pipeline(
        x_train,
        y_train,
        components=int(winner["components"]),
        alpha=float(winner["alpha"]),
        epsilon=float(config["standardization_epsilon"]),
    )
    evaluation = _evaluate(
        model,
        x_test,
        test_positions,
        y_test,
        config["cursor"],
    )
    evaluation["passes"] = bool(
        evaluation["accuracy"]
        >= float(config["gates"]["minimum_test_accuracy"])
    )
    return {
        "raw_descending_neuron_count": int(artifact["descending_neuron_count"]),
        "winner": winner,
        "selection": selection,
        "evaluation": evaluation,
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
            "The classifier observes only anatomically reachable descending-neuron "
            "state. Modality, PCA dimensionality and ridge regularization are selected "
            "using training-only leave-one-out validation. PCA and standardization are "
            "refit inside every fold, so held-out test positions do not influence the "
            "representation or hyperparameter selection."
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
                "evaluation": result["evaluation"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
