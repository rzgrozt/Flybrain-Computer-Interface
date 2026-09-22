"""Whitened PCA plus RBF direction readout over broad descending-neuron state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from flybrain_interface.experiments.spatial_motor_broad_dn_pca_classifier import (
    _apply_pca,
    _apply_standardizer,
    _feature_vector,
    _fit_pca,
    _fit_standardizer,
)

ROOT = Path(__file__).parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "spatial-motor-broad-dn-rbf-direction-v1.json"
CLASS_NAMES = ("LEFT", "RIGHT")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported broad-DN RBF direction schema")
    return payload


def _collect_samples(
    artifacts: list[dict[str, Any]],
    axis: str,
) -> dict[float, dict[str, Any]]:
    samples: dict[float, dict[str, Any]] = {}
    for artifact in artifacts:
        for sample in artifact["axes"][axis]["samples"]:
            samples[float(sample["position"])] = sample
    return samples


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


def _one_hot(labels: np.ndarray) -> np.ndarray:
    result = np.zeros((labels.size, 2), dtype=np.float64)
    result[np.arange(labels.size), labels] = 1.0
    return result


def _rbf_kernel(
    left: np.ndarray,
    right: np.ndarray,
    *,
    gamma: float,
) -> np.ndarray:
    delta = left[:, None, :] - right[None, :, :]
    squared = np.sum(delta * delta, axis=2)
    return np.exp(-gamma * squared)


def _fit_pipeline(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    components: int,
    alpha: float,
    gamma: float,
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
    component_scale = np.std(projected, axis=0)
    safe_component_scale = np.where(
        component_scale > epsilon,
        component_scale,
        np.inf,
    )
    whitened = projected / safe_component_scale
    kernel = _rbf_kernel(whitened, whitened, gamma=gamma)
    weights = np.linalg.solve(
        kernel + alpha * np.eye(kernel.shape[0], dtype=np.float64),
        _one_hot(labels),
    )
    return {
        "standard_mean": standard_mean,
        "standard_scale": standard_scale,
        "pca_mean": pca_mean,
        "basis": basis,
        "component_scale": safe_component_scale,
        "centers": whitened,
        "weights": weights,
        "gamma": gamma,
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
    whitened = projected / np.asarray(
        model["component_scale"],
        dtype=np.float64,
    )
    kernel = _rbf_kernel(
        whitened,
        np.asarray(model["centers"], dtype=np.float64),
        gamma=float(model["gamma"]),
    )
    return kernel @ np.asarray(model["weights"], dtype=np.float64)


def _accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(np.mean(np.argmax(scores, axis=1) == labels))


def _loocv_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    components: int,
    alpha: float,
    gamma: float,
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
            gamma=gamma,
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
            count = int(components)
            if count >= features.shape[0]:
                continue
            for alpha in config["alpha_grid"]:
                for gamma in config["gamma_grid"]:
                    rows.append(
                        {
                            "modality": str(modality),
                            "components": count,
                            "alpha": float(alpha),
                            "gamma": float(gamma),
                            "loocv_accuracy": _loocv_accuracy(
                                features,
                                labels,
                                components=count,
                                alpha=float(alpha),
                                gamma=float(gamma),
                                epsilon=epsilon,
                            ),
                        }
                    )
    rows.sort(
        key=lambda row: (
            -float(row["loocv_accuracy"]),
            int(row["components"]),
            float(row["gamma"]),
            float(row["alpha"]),
            str(row["modality"]),
        )
    )
    return {"winner": rows[0], "candidates": rows}


def fit_from_config(
    artifacts: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    axis = str(config["axis"])
    center = float(config["center_position"])
    train_positions = tuple(float(v) for v in config["train_positions"])
    samples = _collect_samples(artifacts, axis)
    labels = _labels(train_positions, center)
    selection = _select_candidate(samples, train_positions, labels, config)
    winner = selection["winner"]
    modality = str(winner["modality"])
    features = _matrix(samples, train_positions, modality)
    model = _fit_pipeline(
        features,
        labels,
        components=int(winner["components"]),
        alpha=float(winner["alpha"]),
        gamma=float(winner["gamma"]),
        epsilon=float(config["standardization_epsilon"]),
    )
    return model, modality, selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts = [
        json.loads((ROOT / str(path)).read_text(encoding="utf-8"))
        for path in config["training_artifacts"]
    ]
    model, modality, selection = fit_from_config(artifacts, config)
    winner = selection["winner"]
    passed = bool(
        float(winner["loocv_accuracy"])
        >= float(config["gates"]["minimum_loocv_accuracy"])
    )
    payload = {
        "schema_version": 1,
        "experiment": config["name"],
        "interpretation": (
            "The final direction model is selected and fit only from broad descending-"
            "neuron temporal state. PCA latent dimensions are whitened before the RBF "
            "kernel to keep distance scales numerically stable. No screen coordinates "
            "are exposed to inference."
        ),
        "config": config,
        "benchmark": {
            "winner": winner,
            "selection": selection,
            "modality": modality,
            "passes": passed,
            "training_sample_count": len(config["train_positions"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "winner": winner,
                "passes": passed,
                "training_sample_count": len(config["train_positions"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
